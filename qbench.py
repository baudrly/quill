"""Quill benchmarks: language modelling, annotation, generation, BRCA1 variant
effects, Genomic Benchmarks. Results land in results/*.json.
"""
import gzip, json, pickle, sys, time
import numpy as np
from numba import njit
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import qanno as A
import qstates as S
import qinfer as I
from qbuild import dequant_table

HS = Path('/home/claude/data/hs'); MM = Path('/home/claude/data/mm')
T = Path('/home/claude/data/tables')
MODELS = Path('/home/claude/quill/models')
RES = Path('/home/claude/quill/results'); RES.mkdir(exist_ok=True)
GB = Path('/home/claude/data/repos/genomic_benchmarks/datasets')
DQ = dequant_table()


class Model:
    def __init__(self, name):
        z = np.load(MODELS / f'{name}.npz')
        self.name = name
        for k in ('bb_tab', 'bb_off', 'cls', 'cls0', 'pwm', 'cod1', 'cod2', 'cod3',
                  'mix', 'elog', 'init', 'slot_kind', 'slot_cls', 'slot_row'):
            setattr(self, k, z[k])
        self.order = int(z['order'])
        self.nparam = json.loads(str(z['nparam']))
        self.pwm_meta = json.loads(str(z['pwm_meta']))
        self.state_slot = S.G.slot.astype(np.int64)

    def args(self):
        return (self.bb_tab, self.bb_off, self.order, DQ, self.cls, self.cls0, self.pwm,
                self.cod1, self.cod2, self.cod3, self.mix, self.slot_kind,
                self.slot_cls, self.slot_row, S.G.efrom, S.G.eto, self.elog,
                self.state_slot, self.init)

    def logp(self, seq, gcb=None):
        if gcb is None:
            gcb = A.gc_bins(seq, A.GC_WIN, *A.GC_EDGES)
        lp, n, _ = I.forward(seq, gcb, *self.args(), False)
        return lp, n

    def logp_sym(self, seq):
        a, n = self.logp(seq)
        b, _ = self.logp(A.revcomp(seq))
        m = max(a, b)
        return m + np.log2(2.0 ** (a - m) + 2.0 ** (b - m)) - 1.0, n

    def viterbi(self, seq, gcb=None):
        if gcb is None:
            gcb = A.gc_bins(seq, A.GC_WIN, *A.GC_EDGES)
        return I.viterbi(seq, gcb, *self.args())

    def posterior(self, seq, gcb=None):
        if gcb is None:
            gcb = A.gc_bins(seq, A.GC_WIN, *A.GC_EDGES)
        lp, n, post = I.forward(seq, gcb, *self.args(), True)
        return lp, post


# ------------------------------------------------------------------ 1. language modelling
def bench_lm(models, nwin=40, wlen=100000, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    srcs = [('human_heldout_chr19_22', [HS / f'{c}.npy' for c in ('19', '20', '21', '22')]),
            ('human_chr17_heldout', [HS / '17.npy']),
            ('mouse_chr19_heldout_species', [MM / '19.npy'])]
    for tag, paths in srcs:
        wins = []
        per = int(np.ceil(nwin / len(paths)))
        for p in paths:
            seq = np.load(p)
            got = 0
            for _ in range(per * 50):
                if got >= per:
                    break
                st = int(rng.integers(0, len(seq) - wlen))
                w = seq[st:st + wlen].copy()
                if (w > 3).mean() < 0.01:
                    wins.append(w); got += 1
            del seq
        for m in models:
            tb = sb = tn = 0.0
            t0 = time.time()
            for w in wins:
                lp, n = m.logp(w)
                lps, _ = m.logp_sym(w)
                tb += -lp; sb += -lps; tn += n
            dt = time.time() - t0
            out.setdefault(tag, {})[m.name] = dict(
                bits_per_base=round(tb / tn, 4), bits_per_base_sym=round(sb / tn, 4),
                nwin=len(wins), mbase=round(tn / 1e6, 2), sec=round(dt, 1),
                kb_per_sec=round(tn / 1000 / dt * 3, 1))
            print(f'  {tag} {m.name}: {tb/tn:.4f} bits/base (sym {sb/tn:.4f}) '
                  f'over {tn/1e6:.1f} Mb', flush=True)
    return out


# ------------------------------------------------------------------ 2. annotation
CODING_STATES = np.array([S.G.idx[f'CDS{i}'] for i in range(3)] +
                         [S.G.idx[f'ATG{i}'] for i in range(3)] +
                         [S.G.idx[f'STOP{i}'] for i in range(3)] +
                         [S.G.idx[f'DON{e}_{j}'] for e in range(3) for j in range(3)] +
                         [S.G.idx[f'ACC{e}_{j}'] for e in range(3) for j in range(20, 23)])
INTRON_STATES = np.array([S.G.idx[f'INTRON{e}_{d}'] for e in range(3) for d in 'SL'] +
                         [S.G.idx[f'INTRONnc_{d}'] for d in 'SL'] +
                         [S.G.idx[f'DON{e}_{j}'] for e in range(3) for j in range(3, 9)] +
                         [S.G.idx[f'ACC{e}_{j}'] for e in range(3) for j in range(20)] +
                         [S.G.idx[f'DONnc_{j}'] for j in range(3, 9)] +
                         [S.G.idx[f'ACCnc_{j}'] for j in range(20)])


def three_class(path):
    """Viterbi state path -> 0 intergenic/other, 1 coding, 2 intron."""
    out = np.zeros(len(path), np.uint8)
    out[np.isin(path, INTRON_STATES)] = 2
    out[np.isin(path, CODING_STATES)] = 1
    return out


def truth_three_class_human(name):
    lab = np.load(T / f'lab_{name}.npy')
    out = np.zeros(len(lab), np.uint8)
    out[lab == A.INTRON] = 2
    out[(lab >= A.CDS0) & (lab <= 5)] = 1
    return out, lab


def truth_three_class_mouse(gtf, n):
    out = np.zeros(n, np.uint8); gene = np.zeros(n, np.uint8)
    with open(gtf) as fh:
        for line in fh:
            if line[0] == '#':
                continue
            f = line.split('\t')
            if f[0] != 'chr19':
                continue
            a, b = int(f[3]) - 1, int(f[4])
            if f[2] == 'gene':
                gene[a:b] = 1
            elif f[2] == 'CDS':
                out[a:b] = 1
    out[(gene == 1) & (out == 0)] = 2
    return out


def mcc(tp, fp, fn, tn):
    d = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return ((tp * tn - fp * fn) / d) if d > 0 else 0.0


STATE_GROUP3 = np.zeros(S.G.nstate, np.int64)
STATE_GROUP3[INTRON_STATES] = 2
STATE_GROUP3[CODING_STATES] = 1


def bench_anno(models, chunk=500000, pad=25000, span_mb=20):
    """Ab initio structural annotation on contiguous held-out sequence (no region selection)."""
    out = {}
    tasks = [('human_chr20_heldout', HS / '20.npy', lambda: truth_three_class_human('20')[0]),
             ('human_chr21_heldout', HS / '21.npy', lambda: truth_three_class_human('21')[0]),
             ('mouse_chr19_heldout_species', MM / '19.npy',
              lambda: truth_three_class_mouse(
                  '/home/claude/data/mouse/gencode.vM19.annotation.chr19.gtf',
                  len(np.load(MM / '19.npy'))))]
    for tag, path, tfun in tasks:
        seq = np.load(path); truth = tfun()
        asm = np.flatnonzero(seq <= 3)
        c0 = int(asm[0]) + 1000000
        c1 = min(int(asm[-1]), c0 + int(span_mb * 1e6))
        for m in models:
            t0 = time.time()
            pv = np.zeros(c1 - c0, np.uint8); pp = np.zeros(c1 - c0, np.uint8)
            for st in range(c0, c1, chunk):
                lo, hi = max(0, st - pad), min(len(seq), st + chunk + pad)
                sq = seq[lo:hi].copy()
                gcb = A.gc_bins(sq, A.GC_WIN, *A.GC_EDGES)
                a, b = st - lo, min(st + chunk, c1) - lo
                pv[st - c0:st - c0 + (b - a)] = three_class(m.viterbi(sq, gcb))[a:b]
                pp[st - c0:st - c0 + (b - a)] = I.group_posterior(
                    sq, gcb, *m.args(), STATE_GROUP3, 3).argmax(1).astype(np.uint8)[a:b]
            tr = truth[c0:c1]; ok = seq[c0:c1] <= 3
            res = {'mbase': round(float(ok.sum()) / 1e6, 2), 'sec': round(time.time() - t0, 1),
                   'region': [c0, c1]}
            for dec, pred in (('posterior', pp), ('viterbi', pv)):
                r = {}
                for k, nm in ((1, 'coding'), (2, 'intron')):
                    P, Tt = (pred == k) & ok, (tr == k) & ok
                    tp = int((P & Tt).sum()); fp = int((P & ~Tt).sum())
                    fn = int((~P & Tt).sum()); tn = int(ok.sum()) - tp - fp - fn
                    r[nm] = dict(precision=round(tp / max(tp + fp, 1), 4),
                                 recall=round(tp / max(tp + fn, 1), 4),
                                 f1=round(2 * tp / max(2 * tp + fp + fn, 1), 4),
                                 mcc=round(mcc(tp, fp, fn, tn), 4),
                                 pred_frac=round(float(P.sum() / max(ok.sum(), 1)), 5),
                                 truth_frac=round(float(Tt.sum() / max(ok.sum(), 1)), 5))
                r['accuracy3'] = round(float((pred[ok] == tr[ok]).mean()), 4)
                res[dec] = r
            out.setdefault(tag, {})[m.name] = res
            print(f'  {tag} {m.name}: posterior coding MCC {res["posterior"]["coding"]["mcc"]:.3f} '
                  f'(P {res["posterior"]["coding"]["precision"]:.3f} '
                  f'R {res["posterior"]["coding"]["recall"]:.3f}), '
                  f'intron MCC {res["posterior"]["intron"]["mcc"]:.3f}; '
                  f'viterbi coding MCC {res["viterbi"]["coding"]["mcc"]:.3f}', flush=True)
        del seq, truth
    return out


# ------------------------------------------------------------------ 3. generation
GENETIC = 'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'
IDX = {b: i for i, b in enumerate('ACGT')}
TCGA = [IDX[c] for c in 'TCAG']


def codon_id_to_tcga(i):
    """our ACGT-ordered codon index -> index into the TCGA-ordered genetic code string"""
    a, b, c = i // 16, (i // 4) % 4, i % 4
    inv = {0: 2, 1: 1, 2: 3, 3: 0}          # A->2 C->1 G->3 T->0 in TCGA order
    return inv[a] * 16 + inv[b] * 4 + inv[c]


def kmer_spectrum(seq, k=6):
    v = seq[seq <= 3].astype(np.int64)
    n = len(v) - k + 1
    code = np.zeros(n, np.int64)
    for j in range(k):
        code = code * 4 + v[j:j + n]
    c = np.bincount(code, minlength=4 ** k).astype(np.float64)
    return c / c.sum()


def js(p, q):
    m = 0.5 * (p + q)
    def kl(a, b):
        s = a > 0
        return float((a[s] * np.log2(a[s] / b[s])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def bench_gen(models, n=1500000, seed=1):
    ref = np.load(HS / '20.npy')
    ref = ref[(len(ref) // 3):(len(ref) // 3) + 4000000]
    ref = ref[ref <= 3]
    sp_ref = kmer_spectrum(ref)
    rng = np.random.default_rng(seed)
    shuf = ref.copy(); rng.shuffle(shuf)
    zref = np.load(T / 'anno_train.npz')
    cu_ref = zref['codon'].astype(np.float64)
    stops = [3 * 16 + 0 * 4 + 0, 3 * 16 + 2 * 4 + 0, 3 * 16 + 0 * 4 + 2]
    cu_ref[stops] = 0; cu_ref /= cu_ref.sum()
    out = {'shuffled_control': dict(kmer6_js=round(js(sp_ref, kmer_spectrum(shuf)), 5),
                                    gc=round(float(((shuf == 1) | (shuf == 2)).mean()), 4))}
    out['reference_second_half'] = dict(
        kmer6_js=round(js(kmer_spectrum(ref[:len(ref)//2]), kmer_spectrum(ref[len(ref)//2:])), 5))
    for m in models:
        t0 = time.time()
        seq, path = I.sample(n, seed, 1, *m.args(), S.G.idx['IG'], 1.0, 1024)
        # prompted: start at a transcription start site so gene structure is exercised
        gs, gp = [], []
        for r in range(240):
            a, b = I.sample(20000, 1000 + r, 1, *m.args(), S.G.idx['UTR5'], 1.0, 1024)
            gs.append(a); gp.append(b)
        gseq, gpath = np.concatenate(gs), np.concatenate(gp)
        sp = kmer_spectrum(seq)
        cs = three_class(path)
        gc = float(((seq == 1) | (seq == 2)).mean())
        c = seq[seq <= 3]
        cg = float(np.mean((c[:-1] == 1) & (c[1:] == 2)))
        pc = float((c == 1).mean()); pg = float((c == 2).mean())
        # codon usage of generated coding stretches, read in the model's own frame
        cds0 = np.flatnonzero(np.isin(gpath, [S.G.idx['CDS0']]))
        cds0 = cds0[(cds0 + 2) < len(gseq)]
        res = dict(kmer6_js=round(js(sp_ref, sp), 5), gc=round(gc, 4),
                   cpg_obs_exp=round(cg / (pc * pg), 3),
                   coding_frac=round(float((cs == 1).mean()), 5),
                   intron_frac=round(float((cs == 2).mean()), 5),
                   sec=round(time.time() - t0, 1), kb_per_sec=round(n / 1000 / (time.time() - t0), 1),
                   prompted_coding_frac=round(float((three_class(gpath) == 1).mean()), 4),
                   prompted_intron_frac=round(float((three_class(gpath) == 2).mean()), 4),
                   n_pas=int((gpath == S.G.idx['PAS0']).sum()))
        if len(cds0) > 200:
            cod = gseq[cds0[:, None] + np.arange(3)]
            ci = cod[:, 0] * 16 + cod[:, 1] * 4 + cod[:, 2]
            cu = np.bincount(ci, minlength=64).astype(np.float64)
            res['n_codons'] = int(len(ci))
            res['internal_stop_rate'] = round(float(sum((ci == s).sum() for s in stops) / len(ci)), 5)
            cu[stops] = 0; cu /= max(cu.sum(), 1)
            res['codon_usage_js'] = round(js(cu_ref, cu), 5)
        # splice-site fidelity of generated introns
        d1 = np.flatnonzero(np.isin(gpath, [S.G.idx[f'DON{e}_3'] for e in range(3)] +
                                    [S.G.idx['DONnc_3']]))
        d1 = d1[(d1 + 1) < len(gseq)]
        if len(d1) > 20:
            res['n_introns'] = int(len(d1))
            res['donor_GT_frac'] = round(float(np.mean((gseq[d1] == 2) & (gseq[d1 + 1] == 3))), 4)
        a2 = np.flatnonzero(np.isin(gpath, [S.G.idx[f'ACC{e}_18'] for e in range(3)] +
                                    [S.G.idx['ACCnc_18']]))
        a2 = a2[(a2 + 1) < len(gseq)]
        if len(a2) > 20:
            res['acceptor_AG_frac'] = round(float(np.mean((gseq[a2] == 0) & (gseq[a2 + 1] == 2))), 4)
        st = np.flatnonzero(gpath == S.G.idx['ATG0'])
        st = st[(st + 2) < len(gseq)]
        if len(st) > 5:
            res['start_ATG_frac'] = round(float(np.mean((gseq[st] == 0) & (gseq[st + 1] == 3) &
                                                        (gseq[st + 2] == 2))), 4)
        sp_ = np.flatnonzero(gpath == S.G.idx['STOP0'])
        sp_ = sp_[(sp_ + 2) < len(gseq)]
        if len(sp_) > 5:
            si = gseq[sp_[:, None] + np.arange(3)]
            si = si[:, 0] * 16 + si[:, 1] * 4 + si[:, 2]
            res['stop_codon_frac'] = round(float(np.mean(np.isin(si, stops))), 4)
        out[m.name] = res
        print(f'  gen {m.name}: 6-mer JS {res["kmer6_js"]:.5f} (shuffled '
              f'{out["shuffled_control"]["kmer6_js"]:.5f}), GT {res.get("donor_GT_frac")}, '
              f'AG {res.get("acceptor_AG_frac")}, codonJS {res.get("codon_usage_js")}', flush=True)
        np.save(RES / f'sample_{m.name}.npy', seq[:200000])
        np.save(RES / f'samplepath_{m.name}.npy', path[:200000])
        np.save(RES / f'gene_{m.name}.npy', gseq[:120000])
        np.save(RES / f'genepath_{m.name}.npy', gpath[:120000])
    return out





# ------------------------------------------------------------------ 4. BRCA1 variant effects
STATE_GROUPS = {
    'IG': ['IG'], 'CGI': ['CGI'], 'TANDEM': ['TANDEM'], 'UTR5': ['UTR5'], 'UTR3': ['UTR3'],
    'NCRNA': ['NCRNA'],
    'CDS': [f'CDS{i}' for i in range(3)] + [f'ATG{i}' for i in range(3)] +
           [f'STOP{i}' for i in range(3)],
    'INTRON': [f'INTRON{e}_{d}' for e in range(3) for d in 'SL'] +
              [f'INTRONnc_{d}' for d in 'SL'],
    'SPLICE': [f'DON{e}_{j}' for e in range(3) for j in range(9)] +
              [f'ACC{e}_{j}' for e in range(3) for j in range(23)] +
              [f'DONnc_{j}' for j in range(9)] + [f'ACCnc_{j}' for j in range(23)],
    'PAS': [f'PAS{i}' for i in range(6)]}
GROUP_IDX = {g: np.array([S.G.idx[n] for n in v]) for g, v in STATE_GROUPS.items()}


def summary(m, seq):
    gcb = A.gc_bins(seq, A.GC_WIN, *A.GC_EDGES)
    return I.forward_summary(seq, gcb, *m.args())


def logp_sym_fast(m, seq):
    a, n, _ = summary(m, seq)
    b, _, _ = summary(m, A.revcomp(seq))
    mx = max(a, b)
    return mx + np.log2(2.0 ** (a - mx) + 2.0 ** (b - mx)) - 1.0


def splice_matrix_score(m, chr17, pos, ref, alt, radius=35, thresh=3.0):
    """Zero-shot splice disruption from Quill's 176 measured matrix parameters alone."""
    pw = DQ[m.pwm]; pw = pw / pw.sum(1, keepdims=True)
    do, dl = m.pwm_meta['donor']; ao, al_ = m.pwm_meta['acceptor']
    bg = np.log2(np.array([.295, .205, .205, .295]))
    mats = [np.log2(pw[do:do + dl]) - bg, np.log2(pw[ao:ao + al_]) - bg]

    def scan(seq, M):
        L = len(M)
        w = np.lib.stride_tricks.sliding_window_view(seq, L)
        v = M[np.arange(L)[None, :], np.where(w <= 3, w, 0)].sum(1)
        return np.where((w <= 3).all(1), v, -99.0)

    out = np.zeros(len(pos))
    for i in range(len(pos)):
        p = int(pos[i]); a, b = p - radius, p + radius
        best = 0.0
        for flip in (False, True):
            w0 = chr17[a:b].copy(); w1 = w0.copy(); w1[p - a] = alt[i]
            if flip:
                w0, w1 = A.revcomp(w0), A.revcomp(w1)
            for M in mats:
                s0, s1 = scan(w0, M), scan(w1, M)
                k = int(np.argmax(s0))
                if s0[k] > thresh:
                    best = max(best, float(s0[k] - s1[k]))
        out[i] = best
    return out


def bench_brca1(models, pad=20000):
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    d = Path('/home/claude/data/repos/evo2/notebooks/brca1')
    df = pd.read_excel(d / '41586_2018_461_MOESM3_ESM.xlsx', header=2)
    lut = np.full(256, 4, np.uint8)
    for i, c in enumerate('ACGT'):
        lut[ord(c)] = i; lut[ord(c.lower())] = i
    with gzip.open(d / 'GRCh37.p13_chr17.fna.gz', 'rt') as fh:
        fh.readline()
        chr17 = lut[np.frombuffer(''.join(l.strip() for l in fh).encode(), np.uint8)]
    pos = (df['position (hg19)'].values - 1).astype(np.int64)
    ref = np.array(['ACGT'.index(x) for x in df['reference']], np.int64)
    alt = np.array(['ACGT'.index(x) for x in df['alt']], np.int64)
    assert (chr17[pos] == ref).all(), 'reference base mismatch'
    y = (df['func.class'] == 'LOF').values
    csq = df['consequence'].values
    coding = np.isin(csq, ['Missense', 'Nonsense', 'Synonymous'])
    spl = np.isin(csq, ['Canonical splice', 'Splice region', 'Intronic'])
    lo, hi = pos.min() - pad, pos.max() + pad
    rp = (pos - lo).astype(np.int64)

    def report(sc):
        r = dict(all=round(roc_auc_score(y, sc), 4),
                 coding=round(roc_auc_score(y[coding], sc[coding]), 4),
                 noncoding=round(roc_auc_score(y[~coding], sc[~coding]), 4),
                 splice_and_intronic=round(roc_auc_score(y[spl], sc[spl]), 4))
        sp = np.isin(csq, ['Canonical splice', 'Intronic'])
        r['locate_canonical_splice'] = round(
            roc_auc_score(csq[sp] == 'Canonical splice', sc[sp]), 4)
        for c in np.unique(csq):
            s = csq == c
            if y[s].sum() > 3 and (~y[s]).sum() > 3:
                r[f'auroc_{c}'] = round(roc_auc_score(y[s], sc[s]), 4)
        return r

    out = {'n': len(df), 'n_lof': int(y.sum()), 'n_coding': int(coding.sum()),
           'n_noncoding': int((~coding).sum()), 'n_splice_intronic': int(spl.sum()),
           'consequence_counts': {c: int((csq == c).sum()) for c in np.unique(csq)},
           'lof_rate_by_consequence': {c: round(float(y[csq == c].mean()), 4)
                                       for c in np.unique(csq)}}
    for nm, col in (('CADD', 'CADD.score'), ('phyloP_mammalian', 'phyloP (mammalian)')):
        v = pd.to_numeric(df[col], errors='coerce').values
        out[nm] = report(np.where(np.isnan(v), np.nanmedian(v), v))
        print(f'  baseline {nm}: all {out[nm]["all"]:.4f} coding {out[nm]["coding"]:.4f} '
              f'noncoding {out[nm]["noncoding"]:.4f} '
              f'splice/intronic {out[nm]["splice_and_intronic"]:.4f}', flush=True)
    for m in models:
        t0 = time.time()
        sm = splice_matrix_score(m, chr17, pos, ref, alt)
        out[f'{m.name}_splice_matrix'] = report(sm)
        np.save(RES / f'brca1_splicemat_{m.name}.npy', sm)
        smo = np.zeros(len(df))
        for orient in ('+', '-'):
            sq = chr17[lo:hi].copy()
            if orient == '-':
                sq = A.revcomp(sq)
            n = len(sq); gcb = A.gc_bins(sq, A.GC_WIN, *A.GC_EDGES)
            P = (rp if orient == '+' else n - 1 - rp).astype(np.int64)
            R = (ref if orient == '+' else 3 - ref).astype(np.int64)
            AL = (alt if orient == '+' else 3 - alt).astype(np.int64)
            o, _ = I.smoothed_lr(sq, gcb, *m.args(), P, R, AL, P[:1])
            smo += o
        out[f'{m.name}_smoothed_lr'] = report(smo)
        np.save(RES / f'brca1_smoothed_{m.name}.npy', smo)
        cache = {}; dl = np.zeros(len(df))
        for i in range(len(df)):
            p = int(pos[i]); a, b = p - 4096, p + 4096
            if (a, b) not in cache:
                cache[(a, b)] = logp_sym_fast(m, chr17[a:b].copy())
            w = chr17[a:b].copy(); w[p - a] = alt[i]
            dl[i] = logp_sym_fast(m, w) - cache[(a, b)]
        out[f'{m.name}_window_delta'] = report(-dl)
        np.save(RES / f'brca1_window_{m.name}.npy', dl)
        out[f'{m.name}_window_delta']['sec'] = round(time.time() - t0, 1)
        print(f'  {m.name}: splice-matrix all {out[f"{m.name}_splice_matrix"]["all"]:.4f} '
              f'(splice/intronic {out[f"{m.name}_splice_matrix"]["splice_and_intronic"]:.4f}, '
              f'locate {out[f"{m.name}_splice_matrix"]["locate_canonical_splice"]:.4f}) | '
              f'window-delta {out[f"{m.name}_window_delta"]["all"]:.4f} | '
              f'smoothed-LR {out[f"{m.name}_smoothed_lr"]["all"]:.4f}', flush=True)
    df[['chromosome', 'position (hg19)', 'reference', 'alt', 'func.class',
        'consequence']].to_csv(RES / 'brca1_variants.csv', index=False)
    return out


# ------------------------------------------------------------------ 5. Genomic Benchmarks
GB_TASKS = {
    'human_nontata_promoters': ('negative', 'positive'),
    'human_enhancers_cohn': ('negative', 'positive'),
    'human_enhancers_ensembl': ('negative', 'positive'),
    'human_ocr_ensembl': ('negative', 'positive'),
    'human_ensembl_regulatory': ('ocr', 'promoter', 'enhancer'),
    'demo_coding_vs_intergenomic_seqs': ('intergenomic_seqs', 'coding_seqs'),
}
ZERO_SHOT = {'demo_coding_vs_intergenomic_seqs': ['CDS'],
             'human_nontata_promoters': ['CGI', 'UTR5'],
             'human_enhancers_cohn': ['CGI', 'UTR5'],
             'human_enhancers_ensembl': ['CGI', 'UTR5'],
             'human_ocr_ensembl': ['CGI', 'UTR5']}
HELD_CHR = {'chr17', 'chr19', 'chr20', 'chr21', 'chr22'}
_CHROMS, _TX = {}, None


def chrom(name):
    if name not in _CHROMS:
        p = HS / f'{name.replace("chr", "")}.npy'
        if not p.exists():
            return None
        _CHROMS[name] = np.load(p, mmap_mode='r')
    return _CHROMS[name]


def tx_index():
    """ENST -> spliced exon sequence, for the cDNA-based coding class."""
    global _TX
    if _TX is None:
        _TX = {}
        for c in [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']:
            meta = pickle.load(open(HS / f'{c}.pkl', 'rb'))
            for tid, (kind, strand, segs, gene) in meta['tx'].items():
                _TX[tid] = (c, strand, segs)
    return _TX


def get_seq(region, start, end, strand):
    if region.startswith('chr'):
        c = chrom(region)
        if c is None:
            return None
        s = np.asarray(c[start:end])
    elif region.startswith('ENST'):
        rec = tx_index().get(region.split('.')[0])
        if rec is None:
            return None
        cname, tstrand, segs = rec
        c = chrom('chr' + cname)
        if c is None:
            return None
        full = A.oriented(np.asarray(c), segs, tstrand)
        if end > len(full):
            return None
        s = full[start:end]
    else:
        return None
    if len(s) == 0:
        return None
    return A.revcomp(s) if strand == '-' else s.copy()


def features(m, seq):
    lp, n, acc = summary(m, seq)
    tot = max(acc.sum(), 1e-9)
    f = [-lp / max(n, 1)]
    for g in STATE_GROUPS:
        f.append(np.log(acc[GROUP_IDX[g]].sum() / tot + 1e-6))
    v = seq[seq <= 3]
    if len(v) < 4:
        return None
    gc = float(((v == 1) | (v == 2)).mean())
    pc = max(float((v == 1).mean()), 1e-6); pg = max(float((v == 2).mean()), 1e-6)
    cg = float(np.mean((v[:-1] == 1) & (v[1:] == 2)))
    f += [gc, cg / (pc * pg)]
    return f if np.isfinite(f).all() else None


def load_rows(task, split, cap=None, seed=0):
    import csv
    rows = []
    for ci, cname in enumerate(GB_TASKS[task]):
        p = GB / task / split / f'{cname}.csv.gz'
        with gzip.open(p, 'rt') as fh:
            rd = list(csv.DictReader(fh))
        if cap and len(rd) > cap:
            rd = list(np.random.default_rng(seed).choice(np.array(rd, dtype=object), cap, replace=False))
        for r in rd:
            rows.append((r['region'], int(r['start']), int(r['end']), r['strand'], ci))
    return rows


def bench_gb(models, train_cap=6000, test_cap=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    out = {}
    for task in GB_TASKS:
        tr = load_rows(task, 'train', train_cap)
        te = load_rows(task, 'test', test_cap)
        res = {'n_train': len(tr), 'n_test': len(te), 'n_class': len(GB_TASKS[task])}
        for m in models:
            t0 = time.time(); bad = 0
            data = {}
            for tag, rows in (('tr', tr), ('te', te)):
                X, y, held = [], [], []
                for region, a, b, st, ci in rows:
                    s = get_seq(region, a, b, st)
                    if s is None or len(s) < 20:
                        bad += 1; continue
                    fv = features(m, s)
                    if fv is None:
                        bad += 1; continue
                    X.append(fv); y.append(ci); held.append(region in HELD_CHR)
                data[tag] = (np.array(X), np.array(y), np.array(held))
            Xtr, ytr, _ = data['tr']; Xte, yte, hte = data['te']
            if len(Xtr) < 50 or len(Xte) < 50:
                res[m.name] = {'skipped': f'too few usable sequences (missing {bad})'}
                continue
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(Xtr), ytr)
            pred = clf.predict(sc.transform(Xte))
            r = dict(probe_accuracy=round(float((pred == yte).mean()), 4),
                     probe_accuracy_heldout_chrom=round(float((pred[hte] == yte[hte]).mean()), 4)
                     if hte.sum() > 50 else None,
                     n_heldout=int(hte.sum()), missing=bad,
                     sec=round(time.time() - t0, 1), n_feat=Xte.shape[1])
            if task in ZERO_SHOT and len(GB_TASKS[task]) == 2:
                cols = [1 + list(STATE_GROUPS).index(g) for g in ZERO_SHOT[task]]
                z = np.logaddexp.reduce(Xte[:, cols], axis=1)
                r['zero_shot_groups'] = ZERO_SHOT[task]
                r['zero_shot_auroc'] = round(float(roc_auc_score(yte, z)), 4)
                r['zero_shot_accuracy'] = round(float(((z > np.median(z)).astype(int) == yte).mean()), 4)
            res[m.name] = r
            print(f'  {task} {m.name}: probe {r["probe_accuracy"]:.4f} '
                  f'(held-out chrom {r["probe_accuracy_heldout_chrom"]}) '
                  f'zero-shot AUROC {r.get("zero_shot_auroc")}  [{bad} missing]', flush=True)
        out[task] = res
    return out


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    import os
    names = os.environ.get('MODELS', 'quill-k,quill-s,quill-l').split(',')
    models = [Model(n) for n in names]
    print(json.dumps({m.name: m.nparam for m in models}, indent=1))
    res = {}
    if which in ('all', 'lm'):
        print('== language modelling'); res['lm'] = bench_lm(models)
    if which in ('all', 'gen'):
        print('== generation'); res['gen'] = bench_gen(models)
    if which in ('all', 'anno'):
        print('== annotation'); res['anno'] = bench_anno(models)
    if which in ('all', 'brca1'):
        print('== BRCA1 variant effects'); res['brca1'] = bench_brca1(models)
    if which in ('all', 'gb'):
        print('== genomic benchmarks'); res['gb'] = bench_gb(models)
    old = {}
    f = RES / 'bench.json'
    if f.exists():
        old = json.loads(f.read_text())
    old.update(res)
    f.write_text(json.dumps(old, indent=1))
    print('saved', f)
