"""quill-xl v2: many experts, several ontologies, four sizes. No gradient descent.

Experts, each turning evidence into a probability for every Gene Ontology term:
  kmer        significant peptide 5-mer -> term associations (counted)
  memory      recall of the most similar training genes (counted)
  protein     written protein features of proteome.py: physicochemistry, signal peptide,
              transmembrane topology, targeting, disorder, coiled coils, short linear
              motifs, modification sites -> term associations (counted)
  prosite     PROSITE pattern hits (external, curated) -> term associations (counted)
  rna         written mRNA features of transcriptome.py: UTR elements, RNA-binding-protein
              sites, microRNA seed matches, folding, codon adaptation, NMD -> term
              associations (counted)
  signatures  the 67 written protein signatures of domains.py at their written confidence
  prior       term frequency

The GO graph makes every output consistent. Enzyme Commission numbers come from the EC
cross-references in the ontology file and are scored as their own ontology, and the
generic GO slim is reported as a coarse view. Every count is taken on the training
chromosomes; the handful of scalars (one rate per expert and how experts are combined)
are chosen by grid search on three calibration chromosomes. All numbers in results/xl.json
are on held-out chromosomes 17, 19, 20, 21, 22.

    python3 qxl.py build      # all four sizes -> models/quill-xl2-{nano,small,medium,large}.npz
    python3 qxl.py bench      # held-out evaluation and ablations -> results/xl.json
"""
import json
import pickle
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import qdomains
import qgox
import qprot
import qrna

MODELS = HERE / 'models'
RES = HERE / 'results'
OBO = qgox.OBO
SIZES = {
    # terms need this many training genes; which experts; dictionary threshold and cap
    'nano': dict(min_genes=50, experts=('protein', 'prosite', 'rna', 'signatures'),
                 z=4.0, kmer_cap=None, no_folding=True),
    'small': dict(min_genes=20, experts=('kmer', 'protein', 'prosite', 'rna',
                                         'signatures'), z=4.0, kmer_cap=60),
    'medium': dict(min_genes=20, experts=('kmer', 'memory', 'protein', 'prosite', 'rna',
                                          'signatures'), z=4.0, kmer_cap=None),
    'large': dict(min_genes=10, experts=('kmer', 'memory', 'protein', 'prosite', 'rna',
                                         'signatures'), z=3.0, kmer_cap=None),
}
FEATURE_Z, FEATURE_SUPPORT = 3.0, 5
TAUS = (1, 2, 4, 8, 16, 32, 64, 128)


# ---------------------------------------------------------------- data
def load_all():
    sets, meta = qgox.load_sets(), qgox.load_terms()
    with open(qprot.CACHE, 'rb') as fh:
        prot = pickle.load(fh)
    rna = qrna.load_cache()
    return sets, meta, prot, rna


def feature_lists(recs, prot, rna):
    """Per gene: protein+PROSITE feature names and RNA feature names."""
    pf, rf, pr = [], [], []
    for r in recs:
        d, hits = prot.get(r['gene'], (None, []))
        p = qprot.features(d, ()) if d else []
        pf.append(p)
        pr.append([f'prosite:{a}' for a in hits])
        rd = rna.get(r['gene'])
        rf.append(qrna.features(rd) if rd else [])
    return pf, pr, rf


def binarise(lists, vocab):
    rows, cols = [], []
    for i, fs in enumerate(lists):
        for f in fs:
            j = vocab.get(f)
            if j is not None:
                rows.append(i)
                cols.append(j)
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                         shape=(len(lists), len(vocab)))


def association(X, Y, z_min=FEATURE_Z, min_support=FEATURE_SUPPORT):
    """Sparse term x feature log2-lift, kept where co-occurrence is significant."""
    N = X.shape[0]
    nf = np.asarray(X.sum(0)).ravel().astype(np.float64)
    nt = np.asarray(Y.sum(0)).ravel().astype(np.float64)
    C = (Y.T.tocsr() @ X.tocsc()).tocoo()
    c = C.data
    e = nt[C.row] * nf[C.col] / N
    z = (c - e) / np.sqrt(np.maximum(e * (1 - nt[C.row] / N), 1e-9))
    ok = (c >= min_support) & (z >= z_min)
    pos = sp.csr_matrix((np.log2((c[ok] + .5) / (e[ok] + .5)).astype(np.float32),
                         (C.row[ok], C.col[ok])), shape=(Y.shape[1], X.shape[1]))
    # depletion: a common feature that a term's genes avoid. Pairs never seen together
    # are absent from C, so depletion is checked over the dense block of common features.
    common = np.flatnonzero(nf >= 50)
    if len(common):
        Cd = np.asarray((Y.T.tocsr() @ X.tocsc()[:, common]).todense(), np.float64)
        E = nt[:, None] * nf[None, common] / N
        Z = (Cd - E) / np.sqrt(np.maximum(E * (1 - nt[:, None] / N), 1e-9))
        r, k = np.nonzero((Z <= -z_min) & (E >= min_support))
        neg = sp.csr_matrix((np.log2((Cd[r, k] + .5) / (E[r, k] + .5)).astype(np.float32),
                             (r, common[k])), shape=pos.shape)
        pos = pos + neg
    return pos.tocsr()


def ec_map(names):
    """GO term -> EC numbers it cross-references, with EC ancestors added."""
    ec, cur = {}, None
    with open(OBO) as fh:
        for line in fh:
            if line.startswith('id: GO:'):
                cur = line[7:14]
            elif line.startswith('xref: EC:') and cur:
                ec.setdefault(cur, set()).add(line[9:].split()[0])

    def up(e):
        p = e.split('.')
        out = {e}
        for k in range(1, len(p)):
            if p[k] not in ('-', ''):
                out.add('.'.join(p[:k] + ['-'] * (len(p) - k)))
        return out

    return {t: set().union(*(up(e) for e in v)) for t, v in ec.items()}


# ---------------------------------------------------------------- the model
class XL:
    FEATURE_KEYS = ('protein', 'prosite', 'rna')

    def __init__(self, path):
        z = np.load(path, allow_pickle=True)
        self.cfg = json.loads(str(z['cfg']))
        self.names = json.loads(str(z['names']))
        self.index = {t: i for i, t in enumerate(self.names)}
        self.T = len(self.names)
        self.vocab = {k: {f: i for i, f in enumerate(json.loads(str(z[f'vocab_{k}'])))}
                      for k in self.FEATURE_KEYS}
        self.D = {}
        for k in self.FEATURE_KEYS + ('kmer',):
            if f'D_{k}_data' in z.files:
                self.D[k] = sp.csr_matrix((z[f'D_{k}_data'].astype(np.float32),
                                           z[f'D_{k}_indices'], z[f'D_{k}_indptr']),
                                          shape=tuple(z[f'D_{k}_shape']))
        self.keep = z['keep'] if 'keep' in z.files else None
        self.prior = z['prior'].astype(np.float32)
        self.tau = json.loads(str(z['tau']))
        self.combine = str(z['combine'])
        self.ns = json.loads(str(z['ns'])) if 'ns' in z.files else None
        self.weights = json.loads(str(z['weights'])) if 'weights' in z.files else {}
        self.wvec = {}
        if self.weights and self.ns:
            nsarr = np.array(self.ns)
            for e in self.cfg['experts']:
                v = np.zeros(self.T, np.float32)
                for k in ('BP', 'MF', 'CC'):
                    v[nsarr == k] = self.weights.get(k, {}).get(e, 0.0)
                self.wvec[e] = v
        self.order, self.parent_ptr, self.parent_idx = (z['order'], z['parent_ptr'],
                                                        z['parent_idx'])
        if 'mem_indices' in z.files:
            G = len(z['mem_n'])
            self.mem = sp.csr_matrix((np.ones(len(z['mem_indices']), np.float32),
                                      z['mem_indices'], z['mem_indptr']), shape=(G, qgox.N5))
            self.mem_n = z['mem_n']
            self.mem_terms = sp.csr_matrix((np.ones(len(z['memY_indices']), np.float32),
                                            z['memY_indices'], z['memY_indptr']),
                                           shape=(G, self.T))
            self.symbols = json.loads(str(z['symbols']))
        else:
            self.mem = None
        self.rules = None
        sizes = {k: int(v.nnz) for k, v in self.D.items()}
        if self.mem is not None:
            sizes['memory'] = int(self.mem.nnz + self.mem_terms.nnz)
        sizes['prior'] = self.T
        sizes['scalars'] = len(self.tau) + 3 * len(self.weights.get('MF', {}))
        self.parts = sizes
        self.nparam = int(sum(sizes.values()))

    # -- experts --
    def expert(self, name, prots, flists=None, X5=None):
        n = len(prots)
        if name in self.FEATURE_KEYS:
            # naive Bayes: the term's prior log-odds plus the evidence of every feature
            # present, positive or negative, damped by one temperature for correlation
            X = binarise(flists[name], self.vocab[name])
            S = np.asarray((X @ self.D[name].T).todense(), np.float32)
            lo = np.log2(self.prior / np.maximum(1 - self.prior, 1e-6))
            z = lo[None, :] + S / self.tau.get(name, 4.0)
            return (1.0 / (1.0 + np.exp2(-np.clip(z, -30, 30)))).astype(np.float32)
        if name == 'kmer':
            S = np.asarray((X5[:, self.keep] @ self.D['kmer'].T).todense(), np.float32)
            return 1.0 - np.exp(-np.maximum(S, 0) / self.tau.get('kmer', 32.0))
        if name == 'memory':
            P = np.zeros((n, self.T), np.float32)
            S = (X5 @ self.mem.T).toarray()
            nq = np.maximum(np.asarray(X5.sum(1)).ravel(), 1.0)
            S /= np.minimum(nq[:, None], self.mem_n[None, :])
            k = min(20, S.shape[1] - 1)
            idx = np.argpartition(-S, k, axis=1)[:, :k]
            sig = self.tau.get('memory', 0.06)
            for i in range(n):
                for j in idx[i]:
                    s = S[i, j]
                    if s <= 0.02:
                        continue
                    cols = self.mem_terms.indices[self.mem_terms.indptr[j]:
                                                  self.mem_terms.indptr[j + 1]]
                    P[i, cols] = np.maximum(P[i, cols], 1.0 - np.exp(-s / sig))
            return P
        if name == 'signatures':
            if self.rules is None:
                self.rules = qdomains.resolve(qgox.load_terms())[0]
            P = np.zeros((n, self.T), np.float32)
            for i, p in enumerate(prots):
                for _, tid, conf in qdomains.scan(p, self.rules):
                    j = self.index.get(tid)
                    if j is not None:
                        P[i, j] = max(P[i, j], conf)
            return P
        raise KeyError(name)

    def predict(self, prots, flists, experts=None, batch=512, combine=None):
        experts = list(self.cfg['experts']) if experts is None else list(experts)
        combine = combine or self.combine
        out = np.empty((len(prots), self.T), np.float32)
        for a in range(0, len(prots), batch):
            sl = slice(a, a + batch)
            ch = prots[sl]
            fl = {k: v[sl] for k, v in flists.items()}
            X5 = qgox.matrix(ch, qgox.exact5, qgox.N5).tocsr() \
                if {'kmer', 'memory'} & set(experts) else None
            Ps = {e: self.expert(e, ch, fl, X5) for e in experts}
            if combine == 'weighted' and self.weights:
                out[sl] = self.weighted(Ps)
            else:
                out[sl] = combine_probs(list(Ps.values()), self.prior, combine)
        return self.propagate(out)

    def weighted(self, Ps):
        """Damped noisy-OR with one weight per expert per GO namespace."""
        n = next(iter(Ps.values())).shape[0] if Ps else 1
        L = np.zeros((n, self.T), np.float32)
        for e, P in Ps.items():
            w = self.wvec.get(e)
            if w is None:
                continue
            L += w[None, :] * np.log1p(-np.minimum(P, 0.999))
        return np.maximum(1.0 - np.exp(L), self.prior[None, :] * 0.999).astype(np.float32)

    def propagate(self, P):
        for i in self.order:
            for k in range(self.parent_ptr[i], self.parent_ptr[i + 1]):
                j = self.parent_idx[k]
                np.maximum(P[:, j], P[:, i], out=P[:, j])
        return P


def combine_probs(Ps, prior, how):
    """Max of the experts, or a damped noisy-OR; never below the prior."""
    if how == 'max' or len(Ps) == 1:
        P = np.maximum.reduce(Ps) if Ps else np.zeros_like(prior)[None, :]
    else:
        g = float(how.split(':')[1])
        L = sum(np.log1p(-np.minimum(p, 0.999)) for p in Ps)
        P = 1.0 - np.exp(g * L)
    return np.maximum(P, prior[None, :] * 0.999).astype(np.float32)


def kmer_dictionary(tr, names, cfg):
    """The significant peptide 5-mer dictionary, optionally capped per term."""
    D, keep, _, _ = qgox.build_dictionary(tr, names, z_min=cfg['z'])
    if not cfg['kmer_cap']:
        return D, keep
    # keep each term's strongest associations only
    rows = []
    for i in range(D.shape[0]):
        s, e = D.indptr[i], D.indptr[i + 1]
        top = np.argsort(-D.data[s:e])[:cfg['kmer_cap']]
        rows.append((np.full(len(top), i), D.indices[s:e][top], D.data[s:e][top]))
    r, c, v = (np.concatenate([x[k] for x in rows]) for k in range(3))
    used = np.unique(c)
    remap = np.full(D.shape[1], -1, np.int64)
    remap[used] = np.arange(len(used))
    return sp.csr_matrix((v, (r, remap[c])), shape=(len(names), len(used))), keep[used]


def store_sparse(d, key, D):
    for part in ('data', 'indices', 'indptr'):
        d[f'D_{key}_{part}'] = getattr(D, part).astype(
            np.float16 if part == 'data' else np.int32)
    d[f'D_{key}_shape'] = np.array(D.shape)


# ---------------------------------------------------------------- build
def build(size, sets, meta, prot, rna, flist_cache):
    cfg = dict(SIZES[size])
    tr = sets['train']
    names = qgox.model_terms(tr, cfg['min_genes'])
    index = {t: i for i, t in enumerate(names)}
    Y = qgox.label_matrix(tr, index)
    pf, pr, rf = flist_cache['train']
    d = {}
    vocabs = {}
    for key, lists in (('protein', pf), ('prosite', pr), ('rna', rf)):
        cnt = Counter(f for fs in lists for f in fs)
        vocab = sorted(f for f, c in cnt.items() if c >= FEATURE_SUPPORT)
        if cfg.get('no_folding'):
            # the browser has no folding engine, so the browser model does without;
            # nor does a pasted sequence come with its exon structure
            vocab = [f for f in vocab if not f.startswith(
                ('rna:utr5_mfe', 'rna:utr5_fold', 'rna:exons', 'rna:nmd'))]
        vocabs[key] = vocab
        d[f'vocab_{key}'] = json.dumps(vocab)
        if key in cfg['experts']:
            X = binarise(lists, {f: i for i, f in enumerate(vocab)})
            D = association(X, Y)
            store_sparse(d, key, D)
    if 'kmer' in cfg['experts']:
        D, keep = kmer_dictionary(tr, names, cfg)
        store_sparse(d, 'kmer', D)
        d['keep'] = keep.astype(np.int32)
    if 'memory' in cfg['experts']:
        mem = qgox.matrix([r['prot'] for r in tr], qgox.exact5, qgox.N5).tocsr()
        d.update(mem_indices=mem.indices.astype(np.int32),
                 mem_indptr=mem.indptr.astype(np.int64),
                 mem_n=np.asarray(mem.sum(1)).ravel().astype(np.float32),
                 memY_indices=Y.tocsr().indices.astype(np.int32),
                 memY_indptr=Y.tocsr().indptr.astype(np.int64),
                 symbols=json.dumps([r['symbol'] or r['gene'] for r in tr]))
    order, ptr, idx = qgox.graph_arrays(names, meta)
    nt = np.asarray(Y.sum(0)).ravel()
    d.update(cfg=json.dumps(cfg), names=json.dumps(names),
             prior=(nt / len(tr)).astype(np.float32), tau=json.dumps({}),
             combine='max', order=order, parent_ptr=ptr, parent_idx=idx)
    path = MODELS / f'quill-xl2-{size}.npz'
    np.savez_compressed(path, **d)

    # calibration: one rate per expert, then how to combine, by grid search
    m = XL(path)
    cal = sets['calib']
    cp = [r['prot'] for r in cal]
    cf = dict(zip(('protein', 'prosite', 'rna'), flist_cache['calib']))
    tau = {}
    for e in cfg['experts']:
        if e == 'signatures':
            continue
        grid = (0.02, 0.04, 0.06, 0.1, 0.2) if e == 'memory' else \
            ((10, 20, 40, 80, 160) if e == 'kmer' else TAUS)
        best = (-1, None)
        for t in grid:
            m.tau = dict(tau, **{e: t})
            f = qgox.fmax(m.predict(cp, cf, experts=[e]), cal, names, meta)['fmax']
            if f > best[0]:
                best = (f, t)
        tau[e] = best[1]
        print(f'  [{size}] {e:10s} rate {best[1]:<6} calibration Fmax {best[0]:.4f}',
              flush=True)
    m.tau = tau
    # one weight per expert per namespace, by coordinate search on calibration Fmax
    Pc = {e: m.predict(cp, cf, experts=[e], combine='max') for e in cfg['experts']}
    nsarr = np.array([meta[t]['ns'] for t in names])
    grid = (0.0, 0.25, 0.5, 1.0)
    weights = {}
    for ns in ('BP', 'MF', 'CC'):
        w = {e: 0.5 for e in cfg['experts']}

        def score(wd):
            L = sum(wd[e] * np.log1p(-np.minimum(Pc[e], 0.999)) for e in Pc)
            P = np.maximum(1.0 - np.exp(L), m.prior[None, :] * 0.999).astype(np.float32)
            return qgox.fmax(m.propagate(P), cal, names, meta, ns)['fmax']

        cur = score(w)
        for _ in range(2):
            for e in cfg['experts']:
                for g in grid:
                    if g == w[e]:
                        continue
                    trial = dict(w, **{e: g})
                    f = score(trial)
                    if f > cur + 1e-4:
                        cur, w = f, trial
        weights[ns] = w
        print(f'  [{size}] {ns} weights {w}  calibration Fmax {cur:.4f}', flush=True)
    d['tau'] = json.dumps(tau)
    d['combine'] = 'weighted'
    d['weights'] = json.dumps(weights)
    d['ns'] = json.dumps(nsarr.tolist())
    np.savez_compressed(path, **d)
    m = XL(path)
    print(f'quill-xl2-{size}: {len(names)} terms, experts {cfg["experts"]}, '
          f'{m.nparam:,} parameters, {path.stat().st_size/1e6:.1f} MB, '
          f'combined per namespace', flush=True)
    return path


def flists_for(sets, prot, rna):
    out = {}
    for s in ('train', 'calib', 'held'):
        pf, pr, rf = feature_lists(sets[s], prot, rna)
        out[s] = (pf, pr, rf)
    return out


def cmd_build(sizes):
    t0 = time.time()
    sets, meta, prot, rna = load_all()
    fc = flists_for(sets, prot, rna)
    for s in sizes:
        build(s, sets, meta, prot, rna, fc)
    print(f'built in {time.time() - t0:.0f} s')


# ---------------------------------------------------------------- evaluation
def ec_fmax(P, recs, names, ecm, th):
    """Fmax over EC numbers implied by GO MF terms; prediction is the max over terms."""
    ecs = sorted(set().union(*ecm.values()))
    eidx = {e: i for i, e in enumerate(ecs)}
    M = sp.lil_matrix((len(names), len(ecs)), dtype=np.float32)
    for j, t in enumerate(names):
        for e in ecm.get(t, ()):
            M[j, eidx[e]] = 1.0
    M = M.tocsc()
    Pe = np.zeros((P.shape[0], len(ecs)), np.float32)
    for k in range(len(ecs)):
        rows = M.indices[M.indptr[k]:M.indptr[k + 1]]
        if len(rows):
            Pe[:, k] = P[:, rows].max(1)
    truth = [set().union(*(ecm.get(t, set()) for t in r['terms'])) for r in recs]
    keep = [i for i, t in enumerate(truth) if t]
    if not keep:
        return dict(fmax=0.0, n_genes=0)
    T = np.zeros((len(keep), len(ecs)), bool)
    for a, i in enumerate(keep):
        for e in truth[i]:
            if e in eidx:
                T[a, eidx[e]] = True
    Pk = Pe[keep]
    nt = np.array([len(truth[i]) for i in keep], float)
    best = (0, 0, 0, 0)
    for t in th:
        pred = Pk >= t
        npred = pred.sum(1)
        tp = (pred & T).sum(1)
        has = npred > 0
        if not has.any():
            continue
        prec = float((tp[has] / npred[has]).mean())
        rec = float((tp / nt).mean())
        f = 2 * prec * rec / max(prec + rec, 1e-12)
        if f > best[0]:
            best = (f, prec, rec, t)
    return dict(fmax=round(best[0], 4), precision=round(best[1], 4),
                recall=round(best[2], 4), n_genes=len(keep), n_ec=len(ecs))


def slim_terms(meta):
    out = set()
    cur = None
    with open(OBO) as fh:
        for line in fh:
            if line.startswith('id: GO:'):
                cur = line[7:14]
            elif line.strip() == 'subset: goslim_generic' and cur:
                out.add(cur)
    return out


def evaluate(m, recs, flists, meta, ecm, remote, slim, experts=None, combine=None):
    th = np.linspace(0.005, 0.99, 80)
    P = m.predict([r['prot'] for r in recs], flists, experts=experts, combine=combine)
    res = {}
    sub = [r for r, k in zip(recs, remote) if k]
    for ns in ('BP', 'MF', 'CC'):
        res[ns] = qgox.fmax(P.copy(), recs, m.names, meta, ns, th)['fmax']
        res[ns + '_remote'] = qgox.fmax(P[remote].copy(), sub, m.names, meta, ns,
                                        th)['fmax']
    res['EC'] = ec_fmax(P, recs, m.names, ecm, th)['fmax']
    res['EC_remote'] = ec_fmax(P[remote], sub, m.names, ecm, th)['fmax']
    # generic GO slim view: truth and prediction restricted to slim terms
    cols = [i for i, t in enumerate(m.names) if t in slim]
    sl_names = [m.names[i] for i in cols]
    sl_recs = [dict(r, terms=[t for t in r['terms'] if t in slim]) for r in recs]
    res['slim'] = qgox.fmax(P[:, cols].copy(), sl_recs, sl_names, meta, None, th)['fmax']
    return res


def cmd_bench():
    sets, meta, prot, rna = load_all()
    held = sets['held']
    pf, pr, rf = feature_lists(held, prot, rna)
    fl = {'protein': pf, 'prosite': pr, 'rna': rf}
    ecm = ec_map(None)
    slim = slim_terms(meta)
    ref = qgox.GoxModel(qgox.MODELS / 'quill-xl-go.npz')
    _, sim = ref.recall(qgox.matrix([r['prot'] for r in held], qgox.exact5,
                                    qgox.N5).tocsr(), k=2)
    remote = sim[:, 0] < 0.10
    out = dict(n_held=len(held), n_remote=int(remote.sum()),
               n_ec_genes=sum(1 for r in held if any(t in ecm for t in r['terms'])),
               n_slim_terms=len(slim), models={}, ablation={})
    # the naive baseline at each model's term set is the 'prior' expert alone
    for size in SIZES:
        path = MODELS / f'quill-xl2-{size}.npz'
        if not path.exists():
            continue
        m = XL(path)
        t0 = time.time()
        r = evaluate(m, held, fl, meta, ecm, remote, slim)
        dt = time.time() - t0
        base = evaluate(m, held, fl, meta, ecm, remote, slim, experts=[])
        out['models'][size] = dict(parameters=m.nparam, parts=m.parts,
                                   mb=round(path.stat().st_size / 1e6, 2),
                                   terms=m.T, experts=m.cfg['experts'], tau=m.tau,
                                   combine=m.combine, held_out=r, naive=base,
                                   seconds_per_1000_genes=round(1000 * dt / len(held), 1))
        print(f'{size:7s} {m.nparam:>11,} params {path.stat().st_size/1e6:6.1f} MB  ' +
              '  '.join(f'{k} {v:.3f}' for k, v in r.items()), flush=True)
        print(f'{"naive":7s} ' + ' ' * 31 +
              '  '.join(f'{k} {v:.3f}' for k, v in base.items()), flush=True)
    # what each expert contributes to the medium model: alone, and left out
    m = XL(MODELS / 'quill-xl2-medium.npz')
    for e in m.cfg['experts']:
        alone = evaluate(m, held, fl, meta, ecm, remote, slim, experts=[e])
        without = evaluate(m, held, fl, meta, ecm, remote, slim,
                           experts=[x for x in m.cfg['experts'] if x != e])
        out['ablation'][e] = dict(alone=alone, without=without)
        print(f'  {e:10s} alone   ' + '  '.join(f'{k} {v:.3f}' for k, v in alone.items()),
              flush=True)
        print(f'  {e:10s} without ' + '  '.join(f'{k} {v:.3f}' for k, v in without.items()),
              flush=True)
    RES.mkdir(exist_ok=True)
    (RES / 'xl.json').write_text(json.dumps(out, indent=1))


def cmd_extra(sizes):
    """Per-size frequency baselines, each expert on its own, and re-evaluated sizes."""
    sets, meta, prot, rna = load_all()
    held = sets['held']
    pf, pr, rf = feature_lists(held, prot, rna)
    fl = {'protein': pf, 'prosite': pr, 'rna': rf}
    ecm, slim = ec_map(None), slim_terms(meta)
    ref = qgox.GoxModel(qgox.MODELS / 'quill-xl-go.npz')
    _, sim = ref.recall(qgox.matrix([r['prot'] for r in held], qgox.exact5,
                                    qgox.N5).tocsr(), k=2)
    remote = sim[:, 0] < 0.10
    out = json.loads((RES / 'xl.json').read_text())
    for size in SIZES:
        path = MODELS / f'quill-xl2-{size}.npz'
        m = XL(path)
        if size in sizes:
            t0 = time.time()
            r = evaluate(m, held, fl, meta, ecm, remote, slim)
            out['models'][size].update(parameters=m.nparam, parts=m.parts, terms=m.T,
                                       mb=round(path.stat().st_size / 1e6, 2), held_out=r,
                                       tau=m.tau, weights=m.weights,
                                       seconds_per_1000_genes=round(
                                           1000 * (time.time() - t0) / len(held), 1))
            print(f'{size:7s} ' + '  '.join(f'{k} {v:.3f}' for k, v in r.items()), flush=True)
        out['models'][size]['weights'] = m.weights
        b = evaluate(m, held, fl, meta, ecm, remote, slim, experts=[])
        out['models'][size]['naive'] = b
        print(f'{size:7s} naive ' + '  '.join(f'{k} {v:.3f}' for k, v in b.items()), flush=True)
    m = XL(MODELS / 'quill-xl2-medium.npz')
    out['standalone'] = {}
    for e in m.cfg['experts']:
        r = evaluate(m, held, fl, meta, ecm, remote, slim, experts=[e], combine='max')
        out['standalone'][e] = r
        print(f'  {e:10s} on its own ' + '  '.join(f'{k} {v:.3f}' for k, v in r.items()),
              flush=True)
    (RES / 'xl.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'build':
        cmd_build(sys.argv[2:] or list(SIZES))
    elif cmd == 'bench':
        cmd_bench()
    elif cmd == 'extra':
        cmd_extra(sys.argv[2:])
