"""quill-xl3: the final-push function models. Still no gradient anywhere.

On top of the quill-xl2 experts (peptide dictionary, k-mer gene memory, protein, PROSITE,
RNA and written signatures) this adds
  align       Smith-Waterman alignment memory with BLOSUM62 (qalign.py)
  dna         gene architecture, promoter content and genomic neighbourhood (qdna.py),
              a naive-Bayes expert like the protein and RNA ones
  crosstalk   a second pass: counted lifts between GO terms of different namespaces
              let a predicted process or location support a function and vice versa

    python3 qxl3.py build [sizes]     # -> models/quill-xl3-*.npz (calibrated on chr 15,16,18)
    python3 qxl3.py bench             # held-out chromosomes -> results/xl3.json
"""
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import qalign
import qdna
import qgox
import qxl

MODELS, RES = qxl.MODELS, qxl.RES
FEATURE_KEYS = ('protein', 'prosite', 'rna', 'dna')
BROWSER_SKIP = ('rna:utr5_mfe', 'rna:utr5_fold', 'rna:exons', 'rna:nmd')
SIZES = {
    'nano': dict(min_genes=50, experts=('protein', 'prosite', 'rna', 'signatures'),
                 z=4.0, kmer_cap=None, no_folding=True),
    'small': dict(min_genes=20, experts=('kmer', 'protein', 'prosite', 'rna', 'dna',
                                         'signatures'), z=4.0, kmer_cap=60),
    'medium': dict(min_genes=20, experts=('kmer', 'memory', 'align', 'protein', 'prosite',
                                          'rna', 'dna', 'signatures'), z=4.0, kmer_cap=None),
    'large': dict(min_genes=10, experts=('kmer', 'memory', 'align', 'protein', 'prosite',
                                         'rna', 'dna', 'signatures'), z=3.0, kmer_cap=None),
}
RATE_GRID = dict(memory=(0.02, 0.04, 0.06, 0.1, 0.2), kmer=(10, 20, 40, 80, 160),
                 align=(0.02, 0.05, 0.1, 0.2, 0.4))
WEIGHT_GRID = (0.0, 0.25, 0.5, 1.0, 1.5)
CROSS_Z, CROSS_SUPPORT, CROSS_GAIN = 4.0, 10, 5e-3
NAMESPACES = ('BP', 'MF', 'CC')


def load_all():
    sets, meta, prot, rna = qxl.load_all()
    return sets, meta, prot, rna, qdna.load_cache()


def feature_lists(recs, prot, rna, dna):
    pf, pr, rf = qxl.feature_lists(recs, prot, rna)
    return dict(protein=pf, prosite=pr, rna=rf, dna=[dna.get(r['gene'], []) for r in recs])


def crosstalk(Y, ns):
    """Sparse term x term log2-lift between terms of different namespaces."""
    L = qxl.association(Y, Y, z_min=CROSS_Z, min_support=CROSS_SUPPORT).tocoo()
    ok = ns[L.row] != ns[L.col]
    return sp.csr_matrix((L.data[ok], (L.row[ok], L.col[ok])), shape=L.shape)


class XL3(qxl.XL):
    FEATURE_KEYS = FEATURE_KEYS

    def __init__(self, path):
        super().__init__(path)
        z = np.load(path, allow_pickle=True)
        self.cross = self.aligner = None
        self.cross_w = json.loads(str(z['cross_w'])) if 'cross_w' in z.files else {}
        if 'cross_data' in z.files:
            self.cross = sp.csr_matrix((z['cross_data'].astype(np.float32), z['cross_indices'],
                                        z['cross_indptr']), shape=(self.T, self.T))
            self.parts['crosstalk'] = int(self.cross.nnz)
        if 'align_prots' in z.files:
            self.align_prots = str(z['align_prots']).split('|')
            self.align_self = z['align_self']
            if self.mem is None:
                G = len(self.align_prots)
                self.mem_terms = sp.csr_matrix(
                    (np.ones(len(z['memY_indices']), np.float32), z['memY_indices'],
                     z['memY_indptr']), shape=(G, self.T))
                self.symbols = json.loads(str(z['symbols']))
                self.parts['memory'] = int(self.mem_terms.nnz)
            self.parts['align'] = int(sum(map(len, self.align_prots)) + len(self.align_self))
        self.parts['scalars'] = len(self.tau) + 3 * (len(self.weights.get('MF', {})) +
                                                    bool(self.cross_w))
        self.nparam = int(sum(self.parts.values()))

    def expert(self, name, prots, flists=None, X5=None):
        if name != 'align':
            return super().expert(name, prots, flists, X5)
        if self.aligner is None:
            self.aligner = qalign.Aligner(self.align_prots, self.align_self)
        return self.aligner.transfer(prots, self.mem_terms, self.tau.get('align', 0.05))

    def second_pass(self, P):
        """Propagated first-pass probabilities -> crosstalk probabilities."""
        lo = np.log2(self.prior / np.maximum(1 - self.prior, 1e-6))
        S = np.asarray((self.cross @ sp.csr_matrix(P * (P > 2 * self.prior[None, :])).T)
                       .T.todense(), np.float32)
        z = lo[None, :] + S / self.tau.get('crosstalk', 8.0)
        return (1.0 / (1.0 + np.exp2(-np.clip(z, -30, 30)))).astype(np.float32)

    def with_crosstalk(self, P):
        if self.cross is None or not any(self.cross_w.values()):
            return P
        w = np.zeros(self.T, np.float32)
        nsarr = np.array(self.ns)
        for k, v in self.cross_w.items():
            w[nsarr == k] = v
        L = np.log1p(-np.minimum(P, 0.999)) + w[None, :] * np.log1p(
            -np.minimum(self.second_pass(P), 0.999))
        return self.propagate((1.0 - np.exp(L)).astype(np.float32))

    def predict(self, prots, flists, experts=None, batch=512, combine=None):
        P = super().predict(prots, flists, experts, batch, combine)
        if experts is None:
            for a in range(0, len(prots), batch):
                P[a:a + batch] = self.with_crosstalk(P[a:a + batch])
        if self.aligner is not None:
            self.aligner.save()
        return P


def best_rate(m, expert, grid, score):
    best = (-1.0, None)
    for t in grid:
        m.tau[expert] = t
        f = score()
        if f > best[0]:
            best = (f, t)
    m.tau[expert] = best[1]
    return best


def build(size, sets, meta, fl):
    cfg = dict(SIZES[size])
    tr, cal = sets['train'], sets['calib']
    names = qgox.model_terms(tr, cfg['min_genes'])
    index = {t: i for i, t in enumerate(names)}
    Y = qgox.label_matrix(tr, index).tocsr()
    nsarr = np.array([meta[t]['ns'] for t in names])
    d = {}
    for key in FEATURE_KEYS:
        cnt = Counter(f for fs in fl['train'][key] for f in fs)
        vocab = sorted(f for f, c in cnt.items() if c >= qxl.FEATURE_SUPPORT and not (
            cfg.get('no_folding') and f.startswith(BROWSER_SKIP)))
        d[f'vocab_{key}'] = json.dumps(vocab)
        if key in cfg['experts']:
            qxl.store_sparse(d, key, qxl.association(
                qxl.binarise(fl['train'][key], {f: i for i, f in enumerate(vocab)}), Y))
    if 'kmer' in cfg['experts']:
        D, keep = qxl.kmer_dictionary(tr, names, cfg)
        qxl.store_sparse(d, 'kmer', D)
        d['keep'] = keep.astype(np.int32)
    prots = [r['prot'] for r in tr]
    if {'memory', 'align'} & set(cfg['experts']):
        d.update(memY_indices=Y.indices.astype(np.int32), memY_indptr=Y.indptr.astype(np.int64),
                 symbols=json.dumps([r['symbol'] or r['gene'] for r in tr]))
    if 'memory' in cfg['experts']:
        mem = qgox.matrix(prots, qgox.exact5, qgox.N5).tocsr()
        d.update(mem_indices=mem.indices.astype(np.int32),
                 mem_indptr=mem.indptr.astype(np.int64),
                 mem_n=np.asarray(mem.sum(1)).ravel().astype(np.float32))
    if 'align' in cfg['experts']:
        with open(qalign.CACHE.with_name('align_self.pkl'), 'rb') as fh:
            d.update(align_prots='|'.join(prots), align_self=pickle.load(fh).astype(np.float32))
    C = crosstalk(Y, nsarr)
    d.update(cross_data=C.data.astype(np.float16), cross_indices=C.indices.astype(np.int32),
             cross_indptr=C.indptr.astype(np.int32))
    order, ptr, idx = qgox.graph_arrays(names, meta)
    d.update(cfg=json.dumps(cfg), names=json.dumps(names), tau=json.dumps({}), combine='max',
             prior=(np.asarray(Y.sum(0)).ravel() / len(tr)).astype(np.float32),
             order=order, parent_ptr=ptr, parent_idx=idx, ns=json.dumps(nsarr.tolist()))
    path = MODELS / f'quill-xl3-{size}.npz'
    np.savez_compressed(path, **d)

    # calibration on chromosomes 15, 16 and 18: one rate per expert, one weight per expert
    # per namespace, then the rate and per-namespace weight of the second pass
    m = XL3(path)
    cp, cf = [r['prot'] for r in cal], fl['calib']
    m.tau = {}
    for e in cfg['experts']:
        if e != 'signatures':
            f, t = best_rate(m, e, RATE_GRID.get(e, qxl.TAUS), lambda: qgox.fmax(
                m.predict(cp, cf, experts=[e]), cal, names, meta)['fmax'])
            print(f'  [{size}] {e:10s} rate {t:<6} calibration Fmax {f:.4f}', flush=True)
    Pc = {e: np.log1p(-np.minimum(m.predict(cp, cf, experts=[e], combine='max'), 0.999))
          for e in cfg['experts']}

    def first(w):
        L = sum(w[e] * Pc[e] for e in Pc)
        return m.propagate(np.maximum(1.0 - np.exp(L), m.prior[None, :] * 0.999)
                           .astype(np.float32))

    weights = {}
    for ns in NAMESPACES:
        w = {e: 0.5 for e in cfg['experts']}
        cur = qgox.fmax(first(w), cal, names, meta, ns)['fmax']
        for _ in range(2):
            for e in cfg['experts']:
                for g in WEIGHT_GRID:
                    if g != w[e]:
                        trial = dict(w, **{e: g})
                        f = qgox.fmax(first(trial), cal, names, meta, ns)['fmax']
                        if f > cur + 1e-4:
                            cur, w = f, trial
        weights[ns] = w
        print(f'  [{size}] {ns} weights {w}  calibration Fmax {cur:.4f}', flush=True)
    m.weights = weights
    m.wvec = {e: np.select([nsarr == k for k in NAMESPACES],
                           [weights[k][e] for k in NAMESPACES]).astype(np.float32)
              for e in cfg['experts']}
    P1 = first({e: m.wvec[e][None, :] for e in cfg['experts']})
    base = {ns: qgox.fmax(P1.copy(), cal, names, meta, ns)['fmax'] for ns in NAMESPACES}
    best = (sum(base.values()), None, {ns: 0.0 for ns in NAMESPACES})
    for t in (2, 4, 8, 16, 32):
        m.tau['crosstalk'] = t
        L1, L2 = np.log1p(-np.minimum(P1, 0.999)), np.log1p(-np.minimum(m.second_pass(P1), 0.999))
        cw, tot = {}, 0.0
        for ns in NAMESPACES:
            f = {g: qgox.fmax(m.propagate((1.0 - np.exp(L1 + g * L2)).astype(np.float32)),
                              cal, names, meta, ns)['fmax'] for g in (0.0, 0.1, 0.25, 0.5, 1.0)}
            g = max(f, key=f.get)
            cw[ns] = g if f[g] > base[ns] + CROSS_GAIN else 0.0
            tot += f[cw[ns]]
        if tot > best[0] + 1e-4:
            best = (tot, t, cw)
    m.tau['crosstalk'] = best[1] or 8
    print(f'  [{size}] first pass {base}; crosstalk rate {best[1]} weights {best[2]} '
          f'-> sum {best[0]:.4f} (from {sum(base.values()):.4f})', flush=True)
    d.update(tau=json.dumps(m.tau), combine='weighted', weights=json.dumps(weights),
             cross_w=json.dumps(best[2]))
    if not any(best[2].values()):
        for k in ('cross_data', 'cross_indices', 'cross_indptr'):
            del d[k]
    np.savez_compressed(path, **d)
    if m.aligner is not None:
        m.aligner.save()
    m = XL3(path)
    print(f'quill-xl3-{size}: {m.T} terms, {m.nparam:,} parameters {m.parts}, '
          f'{path.stat().st_size / 1e6:.1f} MB', flush=True)


def cmd_build(sizes):
    t0 = time.time()
    sets, meta, prot, rna, dna = load_all()
    fl = {s: feature_lists(sets[s], prot, rna, dna) for s in ('train', 'calib')}
    for s in sizes:
        build(s, sets, meta, fl)
    print(f'built in {time.time() - t0:.0f} s')


def reliability(P, Y, bins=10):
    """Expected calibration error over every gene x term pair, and the bin table."""
    edges = np.linspace(0, 1, bins + 1)
    k = np.minimum((P * bins).astype(np.int64), bins - 1).ravel()
    n = np.bincount(k, minlength=bins)
    conf = np.bincount(k, P.ravel(), bins) / np.maximum(n, 1)
    freq = np.bincount(k, np.asarray(Y.todense()).ravel(), bins) / np.maximum(n, 1)
    return dict(ece=round(float((n * np.abs(conf - freq)).sum() / n.sum()), 5),
                ece_above_0p1=round(float((n[1:] * np.abs(conf - freq)[1:]).sum() /
                                          max(n[1:].sum(), 1)), 4),
                bins=[dict(lo=round(float(edges[b]), 1), n=int(n[b]),
                           predicted=round(float(conf[b]), 4),
                           observed=round(float(freq[b]), 4)) for b in range(bins)])


def cmd_bench():
    sets, meta, prot, rna, dna = load_all()
    held = sets['held']
    fl = feature_lists(held, prot, rna, dna)
    ecm, slim = qxl.ec_map(None), qxl.slim_terms(meta)
    ref = qgox.GoxModel(qgox.MODELS / 'quill-xl-go.npz')
    remote = ref.recall(qgox.matrix([r['prot'] for r in held], qgox.exact5,
                                    qgox.N5).tocsr(), k=2)[1][:, 0] < 0.10
    ev = lambda m, **kw: qxl.evaluate(m, held, fl, meta, ecm, remote, slim, **kw)
    show = lambda tag, r: print(f'{tag:28s}' + '  '.join(f'{k} {v:.3f}' for k, v in r.items()),
                                flush=True)
    out = dict(n_held=len(held), n_remote=int(remote.sum()), models={}, standalone={},
               without={}, previous={})
    for size in SIZES:
        path = MODELS / f'quill-xl3-{size}.npz'
        m = XL3(path)
        t0 = time.time()
        r = ev(m)
        dt = time.time() - t0
        first = ev(m, experts=list(m.cfg['experts']))
        out['models'][size] = dict(
            parameters=m.nparam, parts=m.parts, mb=round(path.stat().st_size / 1e6, 2),
            terms=m.T, experts=m.cfg['experts'], tau=m.tau, weights=m.weights,
            crosstalk=m.cross_w, held_out=r, first_pass=first, naive=ev(m, experts=[]),
            seconds_per_1000_genes=round(1000 * dt / len(held), 1))
        show(f'{size} {m.nparam:,}', r)
        show(f'{size} first pass', first)
        old = qxl.XL(MODELS / f'quill-xl2-{size}.npz')
        out['previous'][size] = dict(parameters=old.nparam, held_out=qxl.evaluate(
            old, held, fl, meta, ecm, remote, slim))
        show(f'{size} xl2', out['previous'][size]['held_out'])
    m = XL3(MODELS / 'quill-xl3-medium.npz')
    for e in m.cfg['experts']:
        out['standalone'][e] = ev(m, experts=[e], combine='max')
        show(f'  {e} alone', out['standalone'][e])
        out['without'][e] = ev(m, experts=[x for x in m.cfg['experts'] if x != e])
        show(f'  without {e}', out['without'][e])
    out['reliability'] = reliability(m.predict([r['prot'] for r in held], fl),
                                     qgox.label_matrix(held, m.index))
    print('reliability', out['reliability'], flush=True)
    (RES / 'xl3.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    if sys.argv[1] == 'build':
        cmd_build(sys.argv[2:] or list(SIZES))
    else:
        cmd_bench()
