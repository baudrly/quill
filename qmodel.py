"""Assemble a Quill model: emission tables, matrices, measured transitions,
and EM over the mixture weights. Saves one .npz per size.
"""
import json, pickle, sys, time
import numpy as np
from numba import njit
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import qanno as A
import qstates as S
from qbuild import quant, dequant_table
import qinfer as I

T = Path('/home/claude/data/tables')
HS = Path('/home/claude/data/hs')
MODELS = Path('/home/claude/quill/models'); MODELS.mkdir(exist_ok=True)
DQ = dequant_table()
NCLS, NGC = 10, 3
EM_CHROMS = ['1', '2']


# ------------------------------------------------------------------ emission tables
def backoff_class(cls2, cls1, cls0):
    """Absolute-discount interpolation cls2 -> cls1 -> cls0 -> uniform, per class and GC bin."""
    out2 = np.zeros((NCLS, NGC, 16, 4)); out0 = np.zeros((NCLS, NGC, 4))
    for c in range(NCLS):
        for g in range(NGC):
            c0 = cls0[c, g].astype(np.float64); n0 = c0.sum()
            p0 = (c0 + 0.5) / (n0 + 2.0)
            out0[c, g] = p0
            c1 = cls1[c, g].astype(np.float64)
            p1 = np.empty((4, 4))
            for a in range(4):
                n = c1[a].sum(); u = (c1[a] > 0).sum()
                d = 0.5
                p1[a] = (np.maximum(c1[a] - d, 0) + d * u * p0) / n if n > 0 else p0
            c2 = cls2[c, g].astype(np.float64)
            for ab in range(16):
                n = c2[ab].sum(); u = (c2[ab] > 0).sum()
                d = 0.5
                lo = p1[ab // 4]
                p = (np.maximum(c2[ab] - d, 0) + d * u * lo) / n if n > 0 else lo
                out2[c, g, ab] = p / p.sum()
    return out2, out0


def build_pwms(z):
    """donor(9) acceptor(23) atg(3) stop(3) pas(6), each a normalised position matrix."""
    rows = []
    meta = {}

    def push(name, mat):
        meta[name] = (len(rows), len(mat))
        for r in mat:
            rows.append(r)

    def norm(c):
        c = c.astype(np.float64) + 1.0
        return c / c.sum(1, keepdims=True)

    don = norm(z['pwm_donor'])[:9]
    acc = norm(z['pwm_acceptor'])[:23]
    koz = norm(z['pwm_kozak'])
    stp = norm(z['pwm_stop'])[:3]
    push('donor', don); push('acceptor', acc); push('atg', koz[8:11]); push('stop', stp)
    # poly(A): hexamers enriched near the 3' end relative to further upstream
    hx = z['hexam'].astype(np.float64)
    near = hx[:, A.PAS_W - 45:A.PAS_W - 10].sum(1)
    far = hx[:, :A.PAS_W - 80].sum(1) + 1.0
    enr = (near / near.sum()) / (far / far.sum())
    top = np.argsort(-np.where(near > 200, enr, 0))[:6]
    w = near[top]
    pas = np.zeros((6, 4))
    for k, wk in zip(top, w):
        for j in range(6):
            pas[j, (k >> (2 * (5 - j))) & 3] += wk
    pas = (pas + 1.0); pas /= pas.sum(1, keepdims=True)
    push('pas', pas)
    names = [''.join('ACGT'[(k >> (2 * (5 - j))) & 3] for j in range(6)) for k in top]
    return np.array(rows), meta, list(zip(names, (w / w.sum()).round(3).tolist()))


def build_codon(z):
    """P(base | phase, partial codon, previous codon) from codon-pair counts, stops removed."""
    cp = z['codon_pair'].astype(np.float64).copy()
    stops = [48, 50, 56]                      # TAA TGA TAG with A=0,C=1,G=2,T=3 -> T=3
    stops = [3 * 16 + 0 * 4 + 0, 3 * 16 + 2 * 4 + 0, 3 * 16 + 0 * 4 + 2]
    cp[:, stops] = 0.0
    tot = cp.sum(1, keepdims=True)
    glob = cp.sum(0); glob = glob / glob.sum()
    pri = 12.0
    P = (cp + pri * glob[None, :]) / (tot + pri)          # P(codon | prev codon)
    c1 = np.zeros((65, 4)); c2 = np.zeros((65, 4, 4)); c3 = np.zeros((65, 16, 4))
    for p in range(65):
        m = P[p].reshape(4, 4, 4)
        c1[p] = m.sum((1, 2))
        c2[p] = m.sum(2) / np.maximum(m.sum((1, 2))[:, None], 1e-12)
        c3[p] = (m / np.maximum(m.sum(2, keepdims=True), 1e-12)).reshape(16, 4)
    for a in (c1, c2, c3):
        a /= np.maximum(a.sum(-1, keepdims=True), 1e-12)
    return c1, c2, c3, P


# ------------------------------------------------------------------ transitions
@njit(cache=True)
def _tally(p, npair, occ):
    for t in range(len(p)):
        a = p[t]
        if a < 0:
            continue
        occ[a] += 1
        if t + 1 < len(p):
            b = p[t + 1]
            if b >= 0:
                npair[a, b] += 1


def intron_mixture(inlen, iters=200):
    """Two-component geometric mixture for intron length; returns the crossover point."""
    L = np.arange(1, len(inlen)); c = inlen[1:].astype(np.float64)
    keep = c > 0; L, c = L[keep], c[keep]
    pi = np.array([0.6, 0.4]); mu = np.array([800.0, 8000.0])
    for _ in range(iters):
        lp = np.stack([np.log(pi[k]) - np.log(mu[k]) - L / mu[k] for k in range(2)])
        lp -= lp.max(0)
        r = np.exp(lp); r /= r.sum(0)
        w = r * c
        pi = w.sum(1) / w.sum()
        mu = (w * L).sum(1) / w.sum(1)
    cross = float((np.log(pi[0] / mu[0]) - np.log(pi[1] / mu[1])) / (1 / mu[0] - 1 / mu[1]))
    return pi, mu, int(np.clip(cross, 200, 20000))


def pas_hexamers(z, k=6):
    hx = z['hexam'].astype(np.float64)
    near = hx[:, A.PAS_W - 45:A.PAS_W - 10].sum(1)
    far = hx[:, :A.PAS_W - 80].sum(1) + 1.0
    enr = (near / near.sum()) / (far / far.sum())
    return set(int(v) for v in np.argsort(-np.where(near > 200, enr, 0))[:k])


def measure_transitions(chroms, report=True, pas=(), ):
    npair = np.zeros((S.G.nstate, S.G.nstate), np.int64)
    occ = np.zeros(S.G.nstate, np.int64)
    bad = ok = 0
    allowed = set(zip(S.G.efrom.tolist(), S.G.eto.tolist()))
    for name in chroms:
        seq = np.load(HS / f'{name}.npy')
        meta = pickle.load(open(HS / f'{name}.pkl', 'rb'))
        cgi = A.find_cgi(seq); tand = A.find_tandem(seq)
        for strand in ('+', '-'):
            p = S.state_path(len(seq), meta, cgi, tand, strand, seq, pas)
            if strand == '-':
                p = p[::-1]
            _tally(p, npair, occ)
        del seq
    for i in range(S.G.nstate):
        for j in range(S.G.nstate):
            if npair[i, j]:
                if (i, j) in allowed:
                    ok += npair[i, j]
                else:
                    bad += npair[i, j]
    if report:
        print(f'  path transitions inside the grammar: {ok/(ok+bad):.5%} '
              f'({bad} of {ok+bad} fell outside)')
    elog = np.zeros(len(S.G.efrom), np.float32)
    for s in range(S.G.nstate):
        e = np.flatnonzero(S.G.efrom == s)
        if len(e) == 0:
            continue
        c = np.array([npair[s, S.G.eto[k]] for k in e], np.float64)
        fams = [S.G.efam[k] for k in e]
        if c.sum() < 20:                       # unobserved: spread by family convention
            c = np.where(np.array(fams) == 'fixed', 1.0, 0.0)
            if c.sum() == 0:
                c = np.ones(len(e))
        c = c + 0.5
        elog[e] = np.log2(c / c.sum())
    init = np.full(S.G.nstate, -30.0, np.float32)
    f = occ.astype(np.float64) + 1.0
    init[:] = np.log2(f / f.sum())
    return elog, init, occ, npair


# ------------------------------------------------------------------ mixture EM
@njit(cache=True)
def expert_probs(seq, gcb, pos, slot, bb_tab, bb_off, bb_k, dq, cls_tab, cls0_tab,
                 pwm_tab, cod1, cod2, cod3, slot_kind, slot_cls, slot_row, out, conf_out):
    pbb = np.zeros(4); prim = np.zeros(4); cod = np.zeros(4)
    for n in range(len(pos)):
        t = pos[n]; s = slot[n]
        b = seq[t]
        run = 0
        while t - 1 - run >= 0 and seq[t - 1 - run] <= 3 and run < 32:
            run += 1
        ctx = np.uint64(0)
        m = bb_k if run > bb_k else run
        for j in range(m, 0, -1):
            ctx = (ctx << np.uint64(2)) | np.uint64(seq[t - j])
        I._bb_lookup(bb_tab, bb_off, ctx, run, bb_k, dq, pbb)
        mx = 0.0
        for x in range(4):
            if pbb[x] > mx:
                mx = pbb[x]
        conf_out[n] = 0 if mx < 0.35 else (1 if mx < 0.5 else (2 if mx < 0.7 else 3))
        out[n, 0] = pbb[b]
        g = gcb[t]
        if slot_kind[s] == 0:
            c = slot_cls[s]
            if run >= 2:
                c2 = int(seq[t - 2]) * 4 + int(seq[t - 1])
                sp = 0.0
                for x in range(4):
                    prim[x] = dq[cls_tab[c, g, c2, x]]; sp += prim[x]
            else:
                sp = 0.0
                for x in range(4):
                    prim[x] = dq[cls0_tab[c, g, x]]; sp += prim[x]
            out[n, 1] = prim[b] / sp
            out[n, 2] = -1.0
            if 3 <= c <= 5:
                ph = c - 3
                if run >= ph + 3:
                    ok = True; pc = 0
                    for j in range(3):
                        bq = seq[t - ph - 3 + j]
                        if bq > 3:
                            ok = False
                        pc = pc * 4 + int(bq if bq < 4 else 0)
                    if ok:
                        sc = 0.0
                        if ph == 0:
                            for x in range(4):
                                cod[x] = dq[cod1[pc, x]]; sc += cod[x]
                        elif ph == 1:
                            c0 = int(seq[t - 1])
                            for x in range(4):
                                cod[x] = dq[cod2[pc, c0, x]]; sc += cod[x]
                        else:
                            c01 = int(seq[t - 2]) * 4 + int(seq[t - 1])
                            for x in range(4):
                                cod[x] = dq[cod3[pc, c01, x]]; sc += cod[x]
                        out[n, 2] = cod[b] / sc
        else:
            r = slot_row[s]
            sp = 0.0
            for x in range(4):
                prim[x] = dq[pwm_tab[r, x]]; sp += prim[x]
            out[n, 1] = prim[b] / sp
            out[n, 2] = -1.0
        out[n, 3] = 0.25


def run_em(M, slot, conf, iters=40, cap=400000):
    """EM over mixture weights per (slot, conf). M[:, j] < 0 marks an unavailable expert."""
    nslot = S.NSLOT
    w = np.full((nslot, I.NCONF, I.NEXP), 0.25, np.float64)
    key = slot.astype(np.int64) * I.NCONF + conf
    order = np.argsort(key, kind='stable')
    key, M, = key[order], M[order]
    bounds = np.searchsorted(key, np.arange(nslot * I.NCONF + 1))
    for it in range(iters):
        for s in range(nslot):
            for c in range(I.NCONF):
                k = s * I.NCONF + c
                lo, hi = bounds[k], bounds[k + 1]
                if hi - lo < 50:
                    continue
                m = M[lo:min(hi, lo + cap)]
                av = m >= 0
                ww = w[s, c] * av
                num = ww * np.where(av, m, 0.0)
                den = num.sum(1, keepdims=True)
                r = num / np.maximum(den, 1e-300)
                new = r.mean(0)
                w[s, c] = new / new.sum()
    for s in range(nslot):
        lo, hi = bounds[s * I.NCONF], bounds[(s + 1) * I.NCONF]
        if hi - lo < 50:
            continue
        pooled = None
        for c in range(I.NCONF):
            k = s * I.NCONF + c
            if bounds[k + 1] - bounds[k] >= 50:
                pooled = w[s, c] if pooled is None else pooled
        if pooled is None:
            m = M[lo:hi]
            av = m >= 0
            ww = np.full(I.NEXP, 0.25) * av.any(0)
            for _ in range(iters):
                num = ww * np.where(av, m, 0.0)
                r = num / np.maximum(num.sum(1, keepdims=True), 1e-300)
                ww = r.mean(0); ww = ww / ww.sum()
            pooled = ww
        for c in range(I.NCONF):
            k = s * I.NCONF + c
            if bounds[k + 1] - bounds[k] < 50:
                w[s, c] = pooled
    return w.astype(np.float32), bounds


def gather_em_sample(model, chroms, per_class=250000, seed=0):
    rng = np.random.default_rng(seed)
    pos_all, slot_all, seq_all, gcb_all = [], [], [], []
    M_list, conf_list = [], []
    for name in chroms:
        seq = np.load(HS / f'{name}.npy')
        meta = pickle.load(open(HS / f'{name}.pkl', 'rb'))
        cgi = A.find_cgi(seq); tand = A.find_tandem(seq)
        gcb = A.gc_bins(seq, A.GC_WIN, *A.GC_EDGES)
        rseq = A.revcomp(seq); rgcb = A.gc_bins(rseq, A.GC_WIN, *A.GC_EDGES)
        for strand, sq, gb in (('+', seq, gcb), ('-', rseq, rgcb)):
            p = S.state_path(len(seq), meta, cgi, tand, strand, sq if strand == '+' else seq, model.pas_set)
            if strand == '-':
                p = p[::-1].copy()
            slots = np.where(p >= 0, S.G.slot[np.maximum(p, 0)], -1)
            valid = (p >= 0) & (sq <= 3)
            keep = []
            for sl in range(S.NSLOT):
                idx = np.flatnonzero(valid & (slots == sl))
                if len(idx) == 0:
                    continue
                lim = per_class if sl < NCLS else 4 * per_class
                if len(idx) > lim:
                    idx = rng.choice(idx, lim, replace=False)
                keep.append(idx)
            if not keep:
                continue
            idx = np.concatenate(keep)
            idx = idx[(idx > 40) & (idx < len(sq) - 8)]
            out = np.zeros((len(idx), I.NEXP)); cf = np.zeros(len(idx), np.int64)
            expert_probs(sq, gb, idx.astype(np.int64), slots[idx].astype(np.int64),
                         *model.bb_args(), DQ, model.cls, model.cls0, model.pwm,
                         model.cod1, model.cod2, model.cod3,
                         model.slot_kind, model.slot_cls, model.slot_row, out, cf)
            M_list.append(out); conf_list.append(cf); slot_all.append(slots[idx].astype(np.int64))
        del seq, rseq
    return np.concatenate(M_list), np.concatenate(slot_all), np.concatenate(conf_list)


# ------------------------------------------------------------------ the model object
class Quill:
    def __init__(self, order, name):
        self.order, self.name = order, name
        self.bb_tab, self.bb_off = self._load_bb(order)
        z = np.load(T / 'anno_train.npz')
        p2, p0 = backoff_class(z['cls2'], z['cls1'], z['cls0'])
        self.cls, self.cls0 = quant(p2), quant(p0)
        rows, pm, pas = build_pwms(z)
        self.pwm, self.pwm_meta, self.pas_top = quant(rows), pm, pas
        c1, c2, c3, self.codonP = build_codon(z)
        self.pas_set = pas_hexamers(z)
        self.cod1, self.cod2, self.cod3 = quant(c1), quant(c2), quant(c3)
        self.slot_kind = np.array([0 if S.G.cls[s] >= 0 and S.G.pwm[s] == 0 else 1
                                   for s in range(S.G.nstate)], np.int64)
        kind = np.zeros(S.NSLOT, np.int64); scls = np.zeros(S.NSLOT, np.int64)
        srow = np.zeros(S.NSLOT, np.int64)
        for s in range(S.G.nstate):
            sl = S.G.slot[s]
            if S.G.pwm[s]:
                kind[sl] = 1
                srow[sl] = pm[S.PWMNAME[int(S.G.pwm[s])]][0] + int(S.G.pwmpos[s])
            else:
                kind[sl] = 0; scls[sl] = int(S.G.cls[s])
        self.slot_kind, self.slot_cls, self.slot_row = kind, scls, srow
        self.mix = np.full((S.NSLOT, I.NCONF, I.NEXP), 0.25, np.float32)
        self.elog = np.zeros(len(S.G.efrom), np.float32)
        self.init = np.zeros(S.G.nstate, np.float32)

    def _load_bb(self, order):
        keep = [k for k in range(order + 1) if k <= 8 or k == order]
        offs = np.full(13, -1, np.int64); parts = []; cur = 0
        for k in keep:
            a = np.load(T / 'baked' / f'bb{k}.npy').reshape(-1)
            offs[k] = cur; parts.append(a); cur += len(a)
        return np.concatenate(parts), offs

    def bb_args(self):
        return self.bb_tab, self.bb_off, self.order

    def infer_args(self):
        return (*self.bb_args(), DQ, self.cls, self.cls0, self.pwm,
                self.cod1, self.cod2, self.cod3, self.mix,
                self.slot_kind, self.slot_cls, self.slot_row,
                S.G.efrom, S.G.eto, self.elog, S.G.slot.astype(np.int64), self.init)

    def nparam(self):
        n = dict(backbone=int(len(self.bb_tab)), classes=int(self.cls.size + self.cls0.size),
                 pwm=int(self.pwm.size), codon=int(self.cod1.size + self.cod2.size + self.cod3.size),
                 mixture=int(self.mix.size), transitions=int(self.elog.size + self.init.size))
        n['total'] = sum(n.values())
        return n

    def save(self, path):
        np.savez_compressed(path, order=self.order, bb_tab=self.bb_tab, bb_off=self.bb_off,
                            cls=self.cls, cls0=self.cls0, pwm=self.pwm, cod1=self.cod1,
                            cod2=self.cod2, cod3=self.cod3, mix=self.mix, elog=self.elog,
                            init=self.init, slot_kind=self.slot_kind, slot_cls=self.slot_cls,
                            slot_row=self.slot_row, pwm_meta=json.dumps(self.pwm_meta),
                            pas_top=json.dumps(self.pas_top), nparam=json.dumps(self.nparam()))


def main():
    t0 = time.time()
    print('measuring transitions from annotation paths on chr1+chr2', flush=True)
    zz = np.load(T / 'anno_train.npz')
    pi, mu, cross = intron_mixture(zz['inlen'])
    S.INTRON_SPLIT[0] = cross
    print(f'  intron length mixture: pi={pi.round(3).tolist()} mean={mu.round(0).tolist()} '
          f'split at {cross} bp', flush=True)
    pas = pas_hexamers(zz)
    print('  poly(A) hexamers used in the path:',
          [''.join('ACGT'[(k >> (2 * (5 - j))) & 3] for j in range(6)) for k in sorted(pas)], flush=True)
    elog, init, occ, npair = measure_transitions(EM_CHROMS, pas=pas)
    import os
    only = os.environ.get('ONLY')
    for order, name in ((9, 'quill-s'), (12, 'quill-l')):
        if only and name != only:
            continue
        print(f'--- {name} (backbone order {order})', flush=True)
        q = Quill(order, name)
        q.elog, q.init = elog, init
        print('  poly(A) hexamers:', q.pas_top, flush=True)
        M, slot, conf = gather_em_sample(q, EM_CHROMS)
        print(f'  EM sample: {len(M)/1e6:.2f}M positions', flush=True)
        q.mix, _ = run_em(M, slot, conf)
        q.save(MODELS / f'{name}.npz')
        print(f'  params: {q.nparam()}', flush=True)
        for sl, nm in ((S.IG, 'IG'), (S.CDS0, 'CDS0'), (S.SLOT_PWM[1] + 3, 'donor+1'),
                       (S.SLOT_PWM[2] + 18, 'acceptor-2')):
            print(f'    mix[{nm}] conf0..3 =', np.round(q.mix[sl], 3).tolist(), flush=True)
    print(f'done in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
