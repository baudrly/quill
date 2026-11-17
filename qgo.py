"""Function-conditioned coding model: P(codon | functional class, position).

Build turns the written amino-acid knowledge in function.py into quantised codon weight
tables, one set per class, with no genome read (`--written`), or counts the same tables
from the training chromosomes (`--counted`). Scoring reads a coding DNA sequence and
returns one log2 likelihood per class, which makes zero-shot functional classification
an argmax and generation a sampling from the chosen class's tables.

    python3 qgo.py build            # both arms
    python3 qgo.py data             # assemble the evaluation set
"""
import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import function as F
import knowledge as K
import ontology as O
from qbuild import quant, dequant_table

HS = Path('/home/claude/data/hs')
GO = Path('/home/claude/data/go')
MODELS = HERE / 'models'
DATA = Path('/home/claude/data/goset')
TRAIN = [str(i) for i in range(1, 17)] + ['18', 'X', 'Y', 'MT']
HELD = ['17', '19', '20', '21', '22']
B = 'ACGT'
BI = {c: i for i, c in enumerate(B)}
STOP = {'TAA', 'TAG', 'TGA'}
MAXP = max(s['period'] for s in F.PERIOD_PROFILES.values())
NTERM_MAX = max(v['length'] for v in F.NTERM.values())

CODONS = [a + b + c for a in B for b in B for c in B]
CODON_AA = {}
for cod in CODONS:
    aa = K.GENETIC_CODE[cod]
    CODON_AA[cod] = aa
AA_CODONS = {}
for cod, aa in CODON_AA.items():
    if aa != '*':
        AA_CODONS.setdefault(aa, []).append(cod)


# ---------------------------------------------------------------- written tables
def synonymous_usage(gc3=None):
    """P(codon | amino acid) from the written codon usage table, optionally GC3-tilted."""
    out = np.zeros(64)
    for aa, cods in AA_CODONS.items():
        w = np.array([K.CODON_USAGE[c] for c in cods], np.float64)
        if gc3 is not None:
            tilt = np.array([1.0 + (gc3 - 0.5) * 2.0 if c[2] in 'GC' else
                             1.0 - (gc3 - 0.5) * 2.0 for c in cods])
            w = w * np.maximum(tilt, 1e-3)
        w = w / w.sum()
        for c, v in zip(cods, w):
            out[CODONS.index(c)] = v
    return out


def codon_from_aa(aa_vec, gc3):
    """P(codon) implied by an amino-acid composition and the synonymous usage."""
    syn = synonymous_usage(gc3)
    p = np.zeros(64)
    for aa, cods in AA_CODONS.items():
        share = aa_vec[F.AA_IDX[aa]]
        idx = [CODONS.index(c) for c in cods]
        sub = syn[idx]
        p[idx] = share * (sub / sub.sum())
    for c in STOP:
        p[CODONS.index(c)] = F.CDS_STOP_LEAK
    return p / p.sum()


def written_prior(specs):
    """The written per-class share of coding genes, normalised over the classes used."""
    src = F.CLASS_PRIOR if set(specs) & set(F.CLASS_PRIOR) else F.FAMILY_PRIOR
    v = np.array([src.get(k, 1.0 / len(specs)) for k in specs], np.float64)
    return (v / v.sum()).astype(np.float32)


def build_written(specs):
    """One codon table per class, plus N-terminal, periodic, length and TM parameters."""
    n = len(specs)
    base = np.empty((n, 64))
    nterm = np.empty((n, NTERM_MAX, 64))
    nterm_len = np.zeros(n, np.int64)
    period = np.empty((n, MAXP, 64))
    period_n = np.zeros(n, np.int64)
    period_w = np.zeros(n, np.float64)
    len_mean = np.zeros(n)
    tm = np.zeros(n)

    for i, (name, spec) in enumerate(specs.items()):
        gc3 = spec.get('gc3', 0.55)
        aa = F.composition(spec['aa'])
        base[i] = codon_from_aa(aa, gc3)
        len_mean[i] = spec['mean_aa']
        tm[i] = spec['tm_expect']

        nt = spec.get('nterm')
        if nt:
            spec_nt = F.NTERM[nt]
            nterm_len[i] = spec_nt['length']
            row = codon_from_aa(F.composition(spec_nt['aa']), gc3)
            nterm[i, :] = row[None, :]
        else:
            nterm[i, :] = base[i][None, :]

        pr = spec.get('period')
        if pr:
            prof = F.period_profile(pr, aa)
            period_n[i] = len(prof)
            period_w[i] = spec.get('period_weight', 0.5)
            for j, row in enumerate(prof):
                period[i, j] = codon_from_aa(row, gc3)
            for j in range(len(prof), MAXP):
                period[i, j] = base[i]
        else:
            period[i, :] = base[i][None, :]

    return dict(base=quant(base), nterm=quant(nterm), nterm_len=nterm_len,
                period=quant(period), period_n=period_n,
                period_w=period_w.astype(np.float32),
                len_mean=len_mean.astype(np.float32), tm=tm.astype(np.float32),
                tm_table=quant(codon_from_aa(F.composition(F.TM_SEGMENT), 0.55)),
                prior=written_prior(specs), names=json.dumps(list(specs)),
                labels=json.dumps([s['label'] for s in specs.values()]))


# ---------------------------------------------------------------- counted tables
def build_counted(specs, records):
    """The same tables, tallied from training-chromosome coding sequences instead."""
    n = len(specs)
    names = list(specs)
    idx = {k: i for i, k in enumerate(names)}
    cb = np.ones((n, 64)) * 0.5
    cn = np.ones((n, NTERM_MAX, 64)) * 0.5
    cp = np.ones((n, MAXP, 64)) * 0.5
    lens = [[] for _ in range(n)]
    tms = [[] for _ in range(n)]
    nterm_len = np.zeros(n, np.int64)
    period_n = np.zeros(n, np.int64)
    period_w = np.zeros(n, np.float64)
    for i, (name, spec) in enumerate(specs.items()):
        nt = spec.get('nterm')
        nterm_len[i] = F.NTERM[nt]['length'] if nt else 0
        pr = spec.get('period')
        if pr:
            period_n[i] = F.PERIOD_PROFILES[pr]['period']
            period_w[i] = spec.get('period_weight', 0.5)

    for rec in records:
        c = rec['cls']
        if c not in idx:
            continue
        i = idx[c]
        cods = rec['codons']
        lens[i].append(len(cods))
        tms[i].append(tm_count(rec['prot']))
        nl, pn = nterm_len[i], period_n[i]
        for j, cd in enumerate(cods):
            if cd < 0:
                continue
            cb[i, cd] += 1.0
            if j < nl:
                cn[i, j, cd] += 1.0
            if pn:
                cp[i, j % pn, cd] += 1.0

    base = cb / cb.sum(1, keepdims=True)
    nterm = cn / cn.sum(2, keepdims=True)
    period = cp / cp.sum(2, keepdims=True)
    for i in range(n):
        if nterm_len[i] == 0:
            nterm[i, :] = base[i][None, :]
        if period_n[i] == 0:
            period[i, :] = base[i][None, :]
        else:
            for j in range(period_n[i], MAXP):
                period[i, j] = base[i]
    len_mean = np.array([np.mean(v) if v else specs[names[i]]['mean_aa']
                         for i, v in enumerate(lens)])
    tmv = np.array([np.mean(v) if v else specs[names[i]]['tm_expect']
                    for i, v in enumerate(tms)])
    cnt = np.array([max(len(v), 1) for v in lens], np.float64)
    return dict(tm_table=quant(codon_from_aa(F.composition(F.TM_SEGMENT), 0.55)),
                base=quant(base), nterm=quant(nterm), nterm_len=nterm_len,
                period=quant(period), period_n=period_n,
                period_w=period_w.astype(np.float32),
                len_mean=len_mean.astype(np.float32), tm=tmv.astype(np.float32),
                prior=(cnt / cnt.sum()).astype(np.float32), names=json.dumps(names),
                labels=json.dumps([s['label'] for s in specs.values()]),
                n_train=json.dumps({names[i]: len(lens[i]) for i in range(n)}))


def tm_count(prot):
    """Windows of TM_WINDOW residues holding at least TM_MIN_HYDROPHOBIC hydrophobics."""
    w, need = F.TM_WINDOW, F.TM_MIN_HYDROPHOBIC
    if len(prot) < w:
        return 0
    h = np.frombuffer(prot.encode(), np.uint8)
    mask = np.isin(h, np.frombuffer(F.HYDROPHOBIC.encode(), np.uint8)).astype(np.int32)
    run = np.convolve(mask, np.ones(w, np.int32), 'valid')
    hits, i = 0, 0
    while i < len(run):
        if run[i] >= need:
            hits += 1
            i += w
        else:
            i += 1
    return hits


def _lse2(a, b):
    """log2(2**a + 2**b), stable."""
    if a < b:
        a, b = b, a
    d = b - a
    return a if d < -40.0 else a + np.log2(1.0 + 2.0 ** d)


# ---------------------------------------------------------------- the model
class GoModel:
    def __init__(self, path):
        z = np.load(path, allow_pickle=True)
        dq = dequant_table()
        self.base = np.log2(np.maximum(dq[z['base']], 1e-12))
        self.nterm = np.log2(np.maximum(dq[z['nterm']], 1e-12))
        self.period = np.log2(np.maximum(dq[z['period']], 1e-12))
        self.nterm_len = z['nterm_len']
        self.period_n = z['period_n']
        self.period_w = z['period_w'].astype(np.float64)
        self.len_mean = z['len_mean'].astype(np.float64)
        self.tm = np.maximum(z['tm'].astype(np.float64), 1e-3)
        self.prior = (np.log2(np.maximum(z['prior'].astype(np.float64), 1e-9))
                      if 'prior' in z.files else None)
        self.tm_table = (np.log2(np.maximum(dq[z['tm_table']], 1e-12))
                         if 'tm_table' in z.files else None)
        self.names = json.loads(str(z['names']))
        self.labels = json.loads(str(z['labels']))
        self.n = len(self.names)
        self.nparam = int(z['base'].size + z['nterm'].size + z['period'].size +
                          self.nterm_len.size + self.period_n.size +
                          self.period_w.size + self.len_mean.size + self.tm.size +
                          (z['tm_table'].size if 'tm_table' in z.files else 0) +
                          (z['prior'].size if 'prior' in z.files else 0))

    def score(self, codons, prot=None, per_base=False):
        """log2 likelihood of one coding sequence under every class.

        `prot` is accepted for callers that have it but is not read: everything the
        score depends on is in the codons.
        """
        w = F.SCORE_WEIGHTS
        cods = np.asarray(codons, np.int64)
        ok = cods >= 0
        cods = cods[ok]
        m = len(cods)
        if m == 0:
            return np.zeros(self.n)
        pos = np.arange(m)
        out = np.zeros(self.n)
        for i in range(self.n):
            nl = int(self.nterm_len[i])
            pn = int(self.period_n[i])
            flat = self.base[i][cods]
            if pn:
                # the repeat's register is arbitrary, so take its best phase
                pw = self.period_w[i]
                best = None
                for off in range(pn):
                    cand = (1.0 - pw) * flat + pw * self.period[i][(pos + off) % pn, cods]
                    tot = cand.sum()
                    if best is None or tot > best[0]:
                        best = (tot, cand)
                lp = best[1].copy()
            else:
                lp = flat.copy()
            if nl:
                k = min(nl, m)
                lp[:k] = self.nterm[i][pos[:k], cods[:k]]
            if self.tm_table is not None and w.get('tm', 0.0) > 0.0:
                # loop / transmembrane chain over the protein, so that hydrophobic
                # residues are scored as helices rather than as a composition
                s = w['composition'] * self._tm_forward(lp, self.tm_table[cods], i, m)
            else:
                s = w['composition'] * lp.sum()
            # written lognormal prior on coding length, in amino acids
            z = (np.log(max(m, 1)) - np.log(self.len_mean[i])) / F.LEN_LOGSIGMA
            s += w['length'] * (-0.5 * z * z - np.log(F.LEN_LOGSIGMA)) / np.log(2.0)
            if self.prior is not None:
                s += w.get('prior', 1.0) * self.prior[i]
            out[i] = s
            if per_base:
                out[i] /= max(3 * m, 1)
        return out

    def _tm_rates(self, i, m):
        """Per-residue enter/stay rates implied by the written segment length and count."""
        stay = 1.0 - 1.0 / F.TM_SEGMENT_LEN
        expect = max(float(self.tm[i]), 1e-3)
        loop_total = max(m - expect * F.TM_SEGMENT_LEN, 1.0)
        enter = min(max(expect / loop_total, 1e-9), 0.5)
        return enter, stay

    def _tm_forward(self, loop_lp, tm_lp, i, m):
        """log2 P(sequence) under a two-state loop/transmembrane chain."""
        enter, stay = self._tm_rates(i, m)
        a_loop, a_tm = 0.0, -60.0           # log2, start in a loop
        l_stay, l_enter = np.log2(1.0 - enter), np.log2(enter)
        t_stay, t_exit = np.log2(stay), np.log2(1.0 - stay)
        for j in range(m):
            n_loop = _lse2(a_loop + l_stay, a_tm + t_exit) + loop_lp[j]
            n_tm = _lse2(a_loop + l_enter, a_tm + t_stay) + tm_lp[j]
            a_loop, a_tm = n_loop, n_tm
        return _lse2(a_loop, a_tm)

    def classify(self, codons, prot):
        s = self.score(codons, prot)
        return self.names[int(np.argmax(s))], s


# ---------------------------------------------------------------- dataset
def gene_class(terms, specs):
    """First class in priority order whose written GO terms the gene carries."""
    t = set(terms)
    for name, spec in specs.items():
        if t & set(spec['go']):
            return name
    return None


def family_of(symbol):
    for name, spec in F.GENE_FAMILIES.items():
        if re.search(spec['symbol'], symbol):
            return name
    return None


def codons_of(seq, segs, strand):
    """Concatenate CDS segments in transcription order and return codon indices."""
    parts = []
    for a, b in (segs if strand == '+' else list(reversed(segs))):
        s = np.asarray(seq[a:b], np.int8)
        if strand == '-':
            s = s[::-1]
            s = np.where(s <= 3, 3 - s, s)      # complement, leaving N (>3) alone
        parts.append(s)
    v = np.concatenate(parts) if parts else np.zeros(0, np.int8)
    n = (len(v) // 3) * 3
    v = v[:n]
    if n == 0:
        return np.zeros(0, np.int64)
    tri = v.reshape(-1, 3).astype(np.int64)
    bad = (tri > 3).any(1)
    idx = tri[:, 0] * 16 + tri[:, 1] * 4 + tri[:, 2]
    idx[bad] = -1
    return idx


def assemble(chroms, out):
    """One record per gene: longest coding transcript, its codons, protein, GO class."""
    recs = []
    for c in chroms:
        gp = GO / f'{c}.pkl'
        hp = HS / f'{c}.pkl'
        if not (gp.exists() and hp.exists()):
            continue
        g = pickle.load(open(gp, 'rb'))
        h = pickle.load(open(hp, 'rb'))
        seq = np.load(HS / f'{c}.npy')
        best = {}
        for tid, strand, segs, gid, prot in h['cds']:
            if gid not in best or len(prot) > len(best[gid][3]):
                best[gid] = (tid, strand, segs, prot)
        for gid, (tid, strand, segs, prot) in best.items():
            terms = g['gene_go'].get(gid)
            sym = h['genes'].get(gid, ('', '', 0, 0))[0]
            cls = gene_class(terms, F.GO_CLASSES) if terms else None
            fam = family_of(sym) if sym else None
            if cls is None and fam is None:
                continue
            cod = codons_of(seq, segs, strand)
            if len(cod) < 40:
                continue
            recs.append(dict(gene=gid, symbol=sym, chrom=c, cls=cls, family=fam,
                             prot=prot, codons=cod.astype(np.int16).tolist(),
                             n_terms=len(terms) if terms else 0))
        del seq
        print(f'  chr{c}: {len(recs)} records so far', flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'wb') as fh:
        pickle.dump(recs, fh, protocol=4)
    return recs


def load(split):
    with open(DATA / f'{split}.pkl', 'rb') as fh:
        recs = pickle.load(fh)
    for r in recs:
        r['codons'] = np.asarray(r['codons'], np.int64)
    return recs


# ---------------------------------------------------------------- entry points
def cmd_data(_):
    DATA.mkdir(parents=True, exist_ok=True)
    for split, chroms in (('train', TRAIN), ('held', HELD)):
        print(f'{split}:')
        recs = assemble(chroms, DATA / f'{split}.pkl')
        import collections
        cc = collections.Counter(r['cls'] for r in recs if r['cls'])
        fc = collections.Counter(r['family'] for r in recs if r['family'])
        print(f'  {split}: {len(recs)} genes')
        print('  classes:', dict(cc))
        print('  families:', dict(fc))


def cmd_build(_):
    MODELS.mkdir(exist_ok=True)
    for tag, specs in (('go', F.GO_CLASSES), ('fam', F.GENE_FAMILIES)):
        d = build_written(specs)
        p = MODELS / f'quill-k-{tag}.npz'
        np.savez_compressed(p, **d)
        m = GoModel(p)
        print(f'{p.name}: {m.nparam:,} parameters, {p.stat().st_size/1024:.1f} KB, '
              f'{m.n} classes (written)')
    if (DATA / 'train.pkl').exists():
        train = load('train')
        for tag, specs, key in (('go', F.GO_CLASSES, 'cls'),
                                ('fam', F.GENE_FAMILIES, 'family')):
            recs = [dict(cls=r[key], codons=r['codons'], prot=r['prot'])
                    for r in train if r[key]]
            d = build_counted(specs, recs)
            p = MODELS / f'quill-s-{tag}.npz'
            np.savez_compressed(p, **d)
            m = GoModel(p)
            print(f'{p.name}: {m.nparam:,} parameters, {p.stat().st_size/1024:.1f} KB, '
                  f'{m.n} classes (counted from {len(recs)} genes)')
    else:
        print('no training records yet: run `python3 qgo.py data` first')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['data', 'build'])
    a = ap.parse_args()
    {'data': cmd_data, 'build': cmd_build}[a.cmd](a)
