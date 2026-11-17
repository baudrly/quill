"""quill-xl, gene side: the whole Gene Ontology, not a slim.

Every human gene with GO annotation is given its full set of terms, propagated up the
ontology graph through is_a and part_of. A table of log-odds links peptide k-mers to
each term that has enough training genes; reading a protein through that table gives a
score for thousands of terms at once, and a memory of every training gene lets a query
recall the genes it resembles. Everything is counted, never trained by gradient: entries
are closed-form functions of co-occurrence counts on the training chromosomes. Two
scalars are chosen by grid search on three calibration chromosomes.

    python3 qgox.py data     # graph, propagated annotations, train/calibration/held sets
    python3 qgox.py build    # models/quill-xl-go.npz (evaluation) and -full.npz (release)
    python3 qgox.py dense    # the unfiltered dense table, kept as an ablation arm
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

HS = Path('/home/claude/data/hs')
GO = Path('/home/claude/data/go')
GOX = Path('/home/claude/data/gox')
MODELS = HERE / 'models'
OBO = GOX / 'go-basic.obo'
TRAIN = [str(i) for i in range(1, 15)] + ['X', 'Y', 'MT']
CALIB = ['15', '16', '18']
HELD = ['17', '19', '20', '21', '22']
ROOTS = {'0008150': 'BP', '0003674': 'MF', '0005575': 'CC'}
NS = {'biological_process': 'BP', 'molecular_function': 'MF', 'cellular_component': 'CC'}
AA = 'ACDEFGHIKLMNPQRSTVWY'
MIN_GENES = 20            # a term needs this many training genes to get a row
N3 = 20 ** 3
HASH5 = 1 << 16
NFEAT = N3 + HASH5
ALPHA = 0.5               # additive smoothing of the co-occurrence counts
N5 = 20 ** 5
Z_MIN = 4.0               # significance an association needs to enter the dictionary


# ---------------------------------------------------------------- ontology graph
def parse_obo(path=OBO):
    """Terms of go-basic.obo: name, namespace, parents (is_a + part_of), alt ids."""
    terms, alt, cur = {}, {}, None
    with open(path) as fh:
        for line in fh:
            line = line.rstrip('\n')
            if line == '[Term]':
                cur = dict(parents=set(), obsolete=False)
                continue
            if line.startswith('[') or not line:
                if cur and 'id' in cur and not cur['obsolete']:
                    terms[cur['id']] = cur
                if line.startswith('['):
                    cur = None
                continue
            if cur is None:
                continue
            if line.startswith('id: GO:'):
                cur['id'] = line[7:]
            elif line.startswith('name: '):
                cur['name'] = line[6:]
            elif line.startswith('namespace: '):
                cur['ns'] = NS[line[11:]]
            elif line.startswith('alt_id: GO:'):
                alt[line[11:]] = cur.get('id')
            elif line.startswith('is_a: GO:'):
                cur['parents'].add(line[9:16])
            elif line.startswith('relationship: part_of GO:'):
                cur['parents'].add(line[25:32])
            elif line.startswith('is_obsolete: true'):
                cur['obsolete'] = True
    if cur and 'id' in cur and not cur['obsolete']:
        terms[cur['id']] = cur
    return terms, alt


def ancestors(terms):
    """Every term's ancestor set, itself included, by memoised walk up the graph."""
    memo = {}

    def up(t):
        if t in memo:
            return memo[t]
        out = {t}
        for p in terms[t]['parents']:
            if p in terms:
                out |= up(p)
        memo[t] = out
        return out

    sys.setrecursionlimit(10000)
    for t in terms:
        up(t)
    return memo


# ---------------------------------------------------------------- features
_AIDX = np.full(256, -1, np.int64)
for _i, _a in enumerate(AA):
    _AIDX[ord(_a)] = _i


def features(prot):
    """Indices of the distinct peptide 3-mers and hashed 5-mers present in a protein."""
    a = _AIDX[np.frombuffer(prot.encode('ascii', 'replace'), np.uint8)]
    n = len(a)
    if n < 5:
        return np.zeros(0, np.int64)
    ok3 = (a[:-2] >= 0) & (a[1:-1] >= 0) & (a[2:] >= 0)
    k3 = (a[:-2] * 400 + a[1:-1] * 20 + a[2:])[ok3]
    ok5 = ok3[:-2] & (a[3:-1] >= 0) & (a[4:] >= 0)
    k5 = (((a[:-4] * 20 + a[1:-3]) * 20 + a[2:-2]) * 20 + a[3:-1]) * 20 + a[4:]
    k5 = k5[ok5]
    h5 = ((k5 * 2654435761) >> 7) & (HASH5 - 1)
    return np.unique(np.concatenate([k3, N3 + h5]))


def exact5(prot):
    """Distinct exact 5-mers, for the homology baseline."""
    a = _AIDX[np.frombuffer(prot.encode('ascii', 'replace'), np.uint8)]
    if len(a) < 5:
        return np.zeros(0, np.int64)
    ok = (a[:-4] >= 0) & (a[1:-3] >= 0) & (a[2:-2] >= 0) & (a[3:-1] >= 0) & (a[4:] >= 0)
    k5 = (((a[:-4] * 20 + a[1:-3]) * 20 + a[2:-2]) * 20 + a[3:-1]) * 20 + a[4:]
    return np.unique(k5[ok])


def matrix(prots, fn, ncol):
    """Binary genes x features sparse matrix."""
    rows, cols = [], []
    for i, p in enumerate(prots):
        f = fn(p)
        rows.append(np.full(len(f), i, np.int64))
        cols.append(f)
    r = np.concatenate(rows) if rows else np.zeros(0, np.int64)
    c = np.concatenate(cols) if cols else np.zeros(0, np.int64)
    return sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(prots), ncol))


# ---------------------------------------------------------------- dataset
def assemble():
    terms, alt = parse_obo()
    anc = ancestors(terms)
    out, stats = {}, dict(obsolete_or_unknown=0, direct=0)
    for split, chroms in (('train', TRAIN), ('calib', CALIB), ('held', HELD)):
        recs = []
        for c in chroms:
            gp, hp = GO / f'{c}.pkl', HS / f'{c}.pkl'
            if not (gp.exists() and hp.exists()):
                continue
            g = pickle.load(open(gp, 'rb'))
            h = pickle.load(open(hp, 'rb'))
            best = {}
            for tid, strand, segs, gid, prot in h['cds']:
                if gid not in best or len(prot) > len(best[gid]):
                    best[gid] = prot
            for gid, prot in best.items():
                direct = g['gene_go'].get(gid)
                if not direct or len(prot) < 30:
                    continue
                full = set()
                for t in direct:
                    t = t if t in terms else alt.get(t)
                    if t is None or t not in terms:
                        stats['obsolete_or_unknown'] += 1
                        continue
                    stats['direct'] += 1
                    full |= anc[t]
                full -= set(ROOTS)
                if not full:
                    continue
                recs.append(dict(gene=gid, symbol=h['genes'].get(gid, ('',))[0], chrom=c,
                                 prot=prot.replace('*', ''), terms=sorted(full)))
        out[split] = recs
        print(f'{split}: {len(recs)} genes, '
              f'{np.mean([len(r["terms"]) for r in recs]):.1f} propagated terms each',
              flush=True)
    meta = {t: dict(name=v.get('name', ''), ns=v.get('ns', ''),
                    parents=sorted(p for p in v['parents'] if p in terms))
            for t, v in terms.items()}
    GOX.mkdir(parents=True, exist_ok=True)
    with open(GOX / 'sets.pkl', 'wb') as fh:
        pickle.dump(out, fh, protocol=4)
    (GOX / 'terms.json').write_text(json.dumps(meta))
    print(f'{len(meta)} live terms; {stats["direct"]} direct annotations kept, '
          f'{stats["obsolete_or_unknown"]} dropped as obsolete or unknown')
    return out, meta


def load_sets():
    with open(GOX / 'sets.pkl', 'rb') as fh:
        return pickle.load(fh)


def load_terms():
    return json.loads((GOX / 'terms.json').read_text())


def label_matrix(recs, index):
    rows, cols = [], []
    for i, r in enumerate(recs):
        for t in r['terms']:
            j = index.get(t)
            if j is not None:
                rows.append(i)
                cols.append(j)
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                         shape=(len(recs), len(index)))


# ---------------------------------------------------------------- the table
def count_table(recs, min_genes=MIN_GENES):
    """Log-odds of each feature for each term, from co-occurrence counts alone."""
    names = model_terms(recs, min_genes)
    index = {t: i for i, t in enumerate(names)}
    X = matrix([r['prot'] for r in recs], features, NFEAT)
    Y = label_matrix(recs, index)
    n = len(recs)
    nt = np.asarray(Y.sum(0)).ravel()                       # genes per term
    nf = np.asarray(X.sum(0)).ravel()                       # genes per feature
    C = np.asarray((Y.T @ X).todense(), np.float32)         # terms x features
    W = np.empty_like(C)
    step = 256
    for a in range(0, len(names), step):
        b = min(a + step, len(names))
        pos = (C[a:b] + ALPHA) / (nt[a:b, None] + 2 * ALPHA)
        neg = (nf[None, :] - C[a:b] + ALPHA) / (n - nt[a:b, None] + 2 * ALPHA)
        W[a:b] = np.log2(pos / neg)
    prior = np.log2((nt + ALPHA) / (n - nt + ALPHA))
    return names, W, prior.astype(np.float32), nt.astype(np.int64)


def quantise(W, lim=6.0):
    """One signed byte per weight: log-odds clipped to +-lim bits in steps of lim/127."""
    return np.clip(np.rint(W * (127.0 / lim)), -127, 127).astype(np.int8), lim


def model_terms(recs, min_genes=MIN_GENES):
    from collections import Counter
    freq = Counter(t for r in recs for t in r['terms'])
    return sorted(t for t, n in freq.items() if n >= min_genes)


def build_dictionary(recs, names, z_min=Z_MIN, min_support=3):
    """Sparse k-mer -> GO term dictionary: only associations that are significant.

    For every exact peptide 5-mer seen in at least `min_support` training genes and every
    term, the co-occurrence count c is compared with its expectation e under
    independence. An entry is kept when c >= min_support and the z-score clears z_min,
    and its weight is the log2 lift. Nothing else is stored, so the table holds the
    k-mers that actually mark a function and none of the noise a dense table carries.
    """
    index = {t: i for i, t in enumerate(names)}
    T, N = len(names), len(recs)
    Y = label_matrix(recs, index).tocsc()
    nt = np.asarray(Y.sum(0)).ravel()
    X5 = matrix([r['prot'] for r in recs], exact5, N5).tocsc()
    nf_all = np.asarray(X5.sum(0)).ravel()
    keep = np.flatnonzero(nf_all >= min_support)
    X = X5[:, keep].tocsr()
    nf = nf_all[keep].astype(np.float64)
    order = np.argsort(-nt)
    rows, cols, vals = [], [], []
    i = 0
    while i < T:                      # broad terms have dense rows: chunk by gene budget
        budget, j = 0, i
        while j < T and (j == i or budget + nt[order[j]] < 60000):
            budget += nt[order[j]]
            j += 1
        ch = order[i:j]
        C = (Y[:, ch].T @ X).tocoo()
        c = C.data
        e = nt[ch][C.row] * nf[C.col] / N
        z = (c - e) / np.sqrt(np.maximum(e * (1.0 - nt[ch][C.row] / N), 1e-9))
        ok = (c >= min_support) & (z >= z_min)
        rows.append(ch[C.row[ok]])
        cols.append(C.col[ok])
        vals.append(np.log2((c[ok] + ALPHA) / (e[ok] + ALPHA)))
        i = j
    D = sp.csr_matrix((np.concatenate(vals).astype(np.float32),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(T, len(keep)))
    return D, keep.astype(np.int64), (nt / N).astype(np.float32), nt.astype(np.int64)


class GoxModel:
    """quill-xl, gene side: dictionary + gene memory + prior, tied by the GO graph.

    dictionary  sparse log2-lift of significant peptide 5-mers for each GO term
    memory      every training gene as its set of 5-mers, with its symbol and terms;
                a query protein recalls the genes it shares 5-mers with
    prior       how often each term occurs among annotated genes
    """

    def __init__(self, path=None):
        z = np.load(path or MODELS / 'quill-xl-go.npz', allow_pickle=True)
        self.names = json.loads(str(z['names']))
        self.index = {t: i for i, t in enumerate(self.names)}
        T = len(self.names)
        self.keep = z['keep']
        self.col_of = None
        self.D = sp.csr_matrix((z['D_data'].astype(np.float32), z['D_indices'],
                                z['D_indptr']), shape=(T, len(self.keep)))
        self.prior = z['prior'].astype(np.float32)
        self.tau, self.sigma = float(z['tau']), float(z['sigma'])
        self.order, self.parent_ptr, self.parent_idx = (z['order'], z['parent_ptr'],
                                                        z['parent_idx'])
        G = len(z['mem_n'])
        self.mem = sp.csr_matrix((np.ones(len(z['mem_indices']), np.float32),
                                  z['mem_indices'], z['mem_indptr']), shape=(G, N5))
        self.mem_n = z['mem_n'].astype(np.float32)
        self.mem_terms = sp.csr_matrix((np.ones(len(z['memY_indices']), np.float32),
                                        z['memY_indices'], z['memY_indptr']),
                                       shape=(G, T))
        self.symbols = json.loads(str(z['symbols']))
        self.nparam = int(self.D.nnz + self.mem.nnz + self.mem_terms.nnz +
                          self.prior.size + 2)
        self.parts = dict(dictionary=int(self.D.nnz), memory_kmers=int(self.mem.nnz),
                          memory_terms=int(self.mem_terms.nnz), prior=int(T),
                          scalars=2)

    def recall(self, X5, k=20):
        """Nearest remembered genes: (indices, containment similarity), best first."""
        S = (X5 @ self.mem.T).toarray()
        nq = np.maximum(np.asarray(X5.sum(1)).ravel(), 1.0)
        S /= np.minimum(nq[:, None], self.mem_n[None, :])
        idx = np.argpartition(-S, min(k, S.shape[1] - 1), axis=1)[:, :k]
        sim = np.take_along_axis(S, idx, 1)
        o = np.argsort(-sim, 1)
        return np.take_along_axis(idx, o, 1), np.take_along_axis(sim, o, 1)

    def predict(self, prots, parts=('dictionary', 'memory', 'prior'), batch=256,
                exclude_self=False):
        """Probability of every model term for every protein, graph-consistent."""
        T = len(self.names)
        out = np.zeros((len(prots), T), np.float32)
        for a in range(0, len(prots), batch):
            chunk = prots[a:a + batch]
            X5 = matrix(chunk, exact5, N5).tocsr()
            P = np.zeros((len(chunk), T), np.float32)
            if 'dictionary' in parts:
                S = np.asarray((X5[:, self.keep] @ self.D.T).todense(), np.float32)
                P = np.maximum(P, 1.0 - np.exp(-S / self.tau))
            if 'memory' in parts:
                idx, sim = self.recall(X5)
                for i in range(len(chunk)):
                    for j, s in zip(idx[i], sim[i]):
                        if s <= 0.02 or (exclude_self and s > 0.999):
                            continue
                        cols = self.mem_terms.indices[self.mem_terms.indptr[j]:
                                                      self.mem_terms.indptr[j + 1]]
                        p = 1.0 - np.exp(-s / self.sigma)
                        P[i, cols] = np.maximum(P[i, cols], p)
            if 'prior' in parts:
                P = np.maximum(P, self.prior[None, :] * 0.999)
            out[a:a + len(chunk)] = P
        return self.propagate(out)

    def nearest(self, prot, k=3):
        X5 = matrix([prot], exact5, N5).tocsr()
        idx, sim = self.recall(X5, k=max(k, 2))
        return [(self.symbols[j], float(s)) for j, s in zip(idx[0][:k], sim[0][:k])]

    def propagate(self, P):
        """A parent is at least as likely as its most likely child."""
        for i in self.order:
            for k in range(self.parent_ptr[i], self.parent_ptr[i + 1]):
                j = self.parent_idx[k]
                np.maximum(P[:, j], P[:, i], out=P[:, j])
        return P


def save_model(path, recs, names, meta, D, keep, prior, tau, sigma):
    order, ptr, idx = graph_arrays(names, meta)
    index = {t: i for i, t in enumerate(names)}
    mem = matrix([r['prot'] for r in recs], exact5, N5).tocsr()
    memY = label_matrix(recs, index).tocsr()
    np.savez_compressed(
        path, names=json.dumps(names), keep=keep, D_data=D.data.astype(np.float16),
        D_indices=D.indices.astype(np.int32), D_indptr=D.indptr.astype(np.int64),
        prior=prior, tau=tau, sigma=sigma, order=order, parent_ptr=ptr, parent_idx=idx,
        mem_indices=mem.indices.astype(np.int32), mem_indptr=mem.indptr.astype(np.int64),
        mem_n=np.asarray(mem.sum(1)).ravel().astype(np.float32),
        memY_indices=memY.indices.astype(np.int32),
        memY_indptr=memY.indptr.astype(np.int64),
        symbols=json.dumps([r['symbol'] or r['gene'] for r in recs]))


def graph_arrays(names, meta):
    """Child-before-parent order and CSR parent lists restricted to the model's terms."""
    index = {t: i for i, t in enumerate(names)}
    parents = [[index[p] for p in meta[t]['parents'] if p in index] for t in names]
    depth = {}

    def d(i):
        if i in depth:
            return depth[i]
        depth[i] = 0 if not parents[i] else 1 + max(d(j) for j in parents[i])
        return depth[i]

    for i in range(len(names)):
        d(i)
    order = np.array(sorted(range(len(names)), key=lambda i: -depth[i]), np.int64)
    ptr = np.zeros(len(names) + 1, np.int64)
    for i, p in enumerate(parents):
        ptr[i + 1] = ptr[i] + len(p)
    idx = np.array([j for p in parents for j in p], np.int64)
    return order, ptr, idx


def fmax(P, recs, names, meta, ns=None, thresholds=None):
    """CAFA protein-centric Fmax: truth is every propagated term, model or not."""
    thresholds = np.linspace(0.01, 0.99, 50) if thresholds is None else thresholds
    cols = np.array([i for i, t in enumerate(names) if ns is None or meta[t]['ns'] == ns])
    index = {names[i]: k for k, i in enumerate(cols)}
    truth_n, T = [], sp.lil_matrix((len(recs), len(cols)), dtype=np.float32)
    for i, r in enumerate(recs):
        tt = [t for t in r['terms'] if ns is None or meta[t]['ns'] == ns]
        truth_n.append(len(tt))
        for t in tt:
            j = index.get(t)
            if j is not None:
                T[i, j] = 1.0
    T = np.asarray(T.todense(), bool)
    truth_n = np.array(truth_n, np.float64)
    keep = truth_n > 0
    Pn, T, truth_n = P[:, cols][keep], T[keep], truth_n[keep]
    best = (0.0, 0.0, 0.0, 0.0)
    for th in thresholds:
        pred = Pn >= th
        npred = pred.sum(1)
        tp = (pred & T).sum(1)
        has = npred > 0
        if not has.any():
            continue
        prec = float((tp[has] / npred[has]).mean())
        rec = float((tp / truth_n).mean())
        f = 2 * prec * rec / max(prec + rec, 1e-12)
        if f > best[0]:
            best = (f, prec, rec, float(th))
    return dict(fmax=round(best[0], 4), precision=round(best[1], 4),
                recall=round(best[2], 4), threshold=round(best[3], 3),
                n_genes=int(keep.sum()), n_terms=int(len(cols)))


def cmd_data(_):
    assemble()


def cmd_build(a):
    sets, meta = load_sets(), load_terms()
    MODELS.mkdir(exist_ok=True)
    recs = sets['train']
    names = model_terms(recs)
    D, keep, prior, nt = build_dictionary(recs, names)
    path = MODELS / 'quill-xl-go.npz'
    save_model(path, recs, names, meta, D, keep, prior, 40.0, 0.1)
    # two fitted scalars, by grid search on the calibration chromosomes: how fast
    # dictionary evidence and recalled similarity each turn into probability
    m = GoxModel(path)
    cal = sets['calib']
    prots = [r['prot'] for r in cal]
    best_tau = (0.0, 40.0)
    for tau in (10.0, 20.0, 40.0, 80.0, 160.0):
        m.tau = tau
        f = fmax(m.predict(prots, parts=('dictionary', 'prior')), cal, names, meta)['fmax']
        print(f'  tau {tau:<6}: calibration Fmax {f:.4f}', flush=True)
        if f > best_tau[0]:
            best_tau = (f, tau)
    m.tau = best_tau[1]
    best_sig = (0.0, 0.1)
    for sigma in (0.03, 0.06, 0.12, 0.25, 0.5):
        m.sigma = sigma
        f = fmax(m.predict(prots), cal, names, meta)['fmax']
        print(f'  sigma {sigma:<5}: calibration Fmax {f:.4f}', flush=True)
        if f > best_sig[0]:
            best_sig = (f, sigma)
    save_model(path, recs, names, meta, D, keep, prior, best_tau[1], best_sig[1])
    m = GoxModel(path)
    ns = {k: sum(1 for t in names if meta[t]['ns'] == k) for k in ('BP', 'MF', 'CC')}
    print(f'{path.name}: {len(names)} GO terms {ns}; {m.parts}; '
          f'{m.nparam:,} parameters, {path.stat().st_size/1e6:.1f} MB; '
          f'tau {best_tau[1]}, sigma {best_sig[1]}')

    # the release model remembers every annotated gene, held-out chromosomes included;
    # it is what ships, and it is never used for a held-out number
    allrecs = sets['train'] + sets['calib'] + sets['held']
    names_all = model_terms(allrecs)
    D2, keep2, prior2, _ = build_dictionary(allrecs, names_all)
    full = MODELS / 'quill-xl-go-full.npz'
    save_model(full, allrecs, names_all, meta, D2, keep2, prior2, best_tau[1],
               best_sig[1])
    mf = GoxModel(full)
    print(f'{full.name}: {len(names_all)} GO terms, {len(allrecs)} genes remembered; '
          f'{mf.nparam:,} parameters, {full.stat().st_size/1e6:.1f} MB')


def cmd_dense(_):
    """The ablation arm: a dense table of every k-mer x term log-odds, unfiltered."""
    sets, meta = load_sets(), load_terms()
    names, W, prior, nt = count_table(sets['train'])
    q, lim = quantise(W)
    np.savez(MODELS / 'quill-xl-go-dense.npz', names=json.dumps(names), W=q, lim=lim,
             prior=prior)
    print(f'dense table: {q.size:,} parameters')


def dense_predict(prots, temp, batch=256):
    """Probabilities from the dense ablation table."""
    z = np.load(MODELS / 'quill-xl-go-dense.npz', allow_pickle=True)
    Wf = z['W'].astype(np.float32) * (float(z['lim']) / 127.0)
    prior = z['prior'].astype(np.float32)
    out = np.empty((len(prots), Wf.shape[0]), np.float32)
    for a in range(0, len(prots), batch):
        X = matrix(prots[a:a + batch], features, NFEAT)
        nfe = np.maximum(np.asarray(X.sum(1)).ravel(), 1.0)
        S = np.asarray(X @ Wf.T) / nfe[:, None]
        out[a:a + X.shape[0]] = 1.0 / (1.0 + np.exp2(-(S / temp + prior[None, :])))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['data', 'build', 'dense'])
    a = ap.parse_args()
    {'data': cmd_data, 'build': cmd_build, 'dense': cmd_dense}[a.cmd](a)
