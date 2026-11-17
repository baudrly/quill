"""quill-v: a written variant-effect scorer. Nothing is fitted to any variant data.

A single-nucleotide variant in an annotated gene is scored by rules written before
looking at the benchmark:
  tier         nonsense and start-lost 10; canonical splice dinucleotide 9;
               missense 2 + m; everything else 0
  missense m   substitution severity (4 - BLOSUM62) / 4, in 0..2
               + 2 x conservation of the reference residue among the remembered
                 homologues that align over it (identity 1, conservative 0.5), the
                 alignment memory of qalign.py with tracebacks
               + residue rules: reference Cys +1; reference Trp, Gly or Pro +0.5;
                 proline introduced +0.5; hydrophobic to charged or amide +0.5
  splicing     added to every variant: the loss, in bits (capped at 8), of the best
               donor or acceptor match of the written quill-k matrices within 35 nt

    python3 qvariant.py brca1     # Findlay et al. 2018 saturation editing -> results/variant.json
"""
import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import parasail
from scipy.stats import rankdata

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import qalign
import qgox

RES = HERE / 'results'
BRCA1 = Path('/home/claude/data/repos/evo2/notebooks/brca1')
HS = Path('/home/claude/data/hs')
B62 = parasail.blosum62
TIER = dict(nonsense=10.0, start_lost=10.0, canonical_splice=9.0, missense=2.0)
CONS_MIN_SCORE, SPLICE_CAP = 50.0, 8.0
HYDROPHOBIC, POLAR = set('LIVMFWC'), set('DEKRNQ')
CODE = 'ACGT'
AA = {a + b + c: aa for (a, b, c), aa in zip(
    ((a, b, c) for a in 'TCAG' for b in 'TCAG' for c in 'TCAG'),
    'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG')}


def blosum(a, b):
    return int(B62.matrix[B62.mapper[ord(a)], B62.mapper[ord(b)]])


def conservation(prot, aligner):
    """Per residue: how conserved it is among aligned remembered homologues, 0..1."""
    q = qalign.clean(prot)
    num, den, cover = (np.zeros(len(q)) for _ in range(3))
    for j, s in aligner.hits([prot])[0][1]:
        if s < CONS_MIN_SCORE:
            continue
        r = parasail.sw_trace_striped_16(q, aligner.prots[j], qalign.GAP_OPEN,
                                         qalign.GAP_EXTEND, B62)
        qa, ta = r.traceback.query, r.traceback.ref
        i = r.end_query - sum(c != '-' for c in qa) + 1
        for a, b in zip(qa, ta):
            if a == '-':
                continue
            if b != '-':
                num[i] += s * (1.0 if a == b else 0.5 if blosum(a, b) > 0 else 0.0)
                den[i] += s
                cover[i] += 1
            i += 1
    return num / np.maximum(den, 1e-9) * np.minimum(1.0, cover / 3.0)


def missense(ref, alt, cons):
    return ((4 - blosum(ref, alt)) / 4.0 + 2.0 * cons + (ref == 'C') + 0.5 * (ref in 'WGP')
            + 0.5 * (alt == 'P') + 0.5 * (ref in HYDROPHOBIC and alt in POLAR))


def classify(pos, alt, segs, strand, chrom):
    """Consequence of each variant from the coding segments alone.

    Returns per variant (class, reference amino acid, alternate amino acid, residue index)."""
    comp = lambda b: 3 - b
    order = sorted(segs, reverse=strand == '-')
    cds_pos = np.concatenate([np.arange(a, b)[::-1 if strand == '-' else 1] for a, b in order])
    where = {int(p): k for k, p in enumerate(cds_pos)}
    cds = chrom[cds_pos] if strand == '+' else comp(chrom[cds_pos])
    edges = set()
    for k, (a, b) in enumerate(sorted(segs)):
        if k:
            edges.update((a - 1, a - 2))
        if k < len(segs) - 1:
            edges.update((b, b + 1))
    out = []
    for p, x in zip(pos, alt):
        p = int(p)
        if p in edges:
            out.append(('canonical_splice', '', '', -1))
            continue
        k = where.get(p)
        if k is None:
            out.append(('noncoding', '', '', -1))
            continue
        c = cds[k - k % 3:k - k % 3 + 3].copy()
        ra = AA[''.join(CODE[b] for b in c)]
        c[k % 3] = x if strand == '+' else comp(x)
        aa = AA[''.join(CODE[b] for b in c)]
        cls = 'synonymous' if aa == ra else 'nonsense' if aa == '*' else \
            'start_lost' if k < 3 else 'missense'
        out.append((cls, ra, aa, k // 3))
    return out


def score(pos, alt, segs, strand, chrom, prot, aligner, splice, parts=('sub', 'cons', 'rules')):
    cons = conservation(prot, aligner)
    s = np.minimum(np.asarray(splice, float), SPLICE_CAP)
    cls = classify(pos, alt, segs, strand, chrom)
    for i, (c, ra, aa, k) in enumerate(cls):
        s[i] += TIER.get(c, 0.0)
        if c == 'missense':
            full = missense(ra, aa, cons[k])
            sub = (4 - blosum(ra, aa)) / 4.0
            s[i] += ('sub' in parts) * sub + ('cons' in parts) * 2.0 * cons[k] + \
                ('rules' in parts) * (full - sub - 2.0 * cons[k])
    return s, cls, cons


def auroc(y, s):
    y = np.asarray(y, bool)
    r = rankdata(s)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def lift_segments(segs, seq38, chrom37, lo, hi):
    """Coding segments in the benchmark's assembly, by exact sequence match."""
    hay = bytes(chrom37[lo:hi].astype(np.uint8))
    out = []
    for a, b in segs:
        k = hay.find(bytes(np.asarray(seq38[a:b], np.uint8)))
        if k < 0:
            raise SystemExit(f'coding segment {a}-{b} not found in the benchmark assembly')
        out.append((lo + k, lo + k + b - a))
    return out


def cmd_brca1():
    import pandas as pd
    import qbench
    df = pd.read_excel(BRCA1 / '41586_2018_461_MOESM3_ESM.xlsx', header=2)
    lut = np.full(256, 4, np.uint8)
    for i, c in enumerate(CODE):
        lut[[ord(c), ord(c.lower())]] = i
    with gzip.open(BRCA1 / 'GRCh37.p13_chr17.fna.gz', 'rt') as fh:
        fh.readline()
        chrom = lut[np.frombuffer(''.join(l.strip() for l in fh).encode(), np.uint8)]
    pos = (df['position (hg19)'].values - 1).astype(np.int64)
    ref = np.array([CODE.index(x) for x in df['reference']], np.int64)
    alt = np.array([CODE.index(x) for x in df['alt']], np.int64)
    y = (df['func.class'] == 'LOF').values
    csq = df['consequence'].values
    sets = qgox.load_sets()
    rec = next(r for r in sets['held'] if r['symbol'] == 'BRCA1')
    h = pickle.load(open(HS / '17.pkl', 'rb'))
    tid, strand, segs, gid, prot = next(c for c in h['cds'] if c[3] == rec['gene'] and
                                        c[4] == rec['prot'])
    segs = lift_segments(segs, np.load(HS / '17.npy', mmap_mode='r'), chrom,
                         pos.min() - 200000, pos.max() + 200000)
    with open(qalign.CACHE.with_name('align_self.pkl'), 'rb') as fh:
        aligner = qalign.Aligner([r['prot'] for r in sets['train']], pickle.load(fh))
    splice = qbench.splice_matrix_score(qbench.Model('quill-k'), chrom, pos, ref, alt)
    groups = {'all': np.ones(len(y), bool),
              'coding': np.isin(csq, ['Missense', 'Nonsense', 'Synonymous']),
              'missense': csq == 'Missense'}
    groups['noncoding'] = ~groups['coding']
    out = dict(gene='BRCA1', transcript=tid, protein_length=len(prot), n=len(y),
               n_lof=int(y.sum()), n_missense=int(groups['missense'].sum()), variants={})
    for tag, parts in (('quill-v', ('sub', 'cons', 'rules')), ('BLOSUM62 only', ('sub',)),
                       ('BLOSUM62 + residue rules', ('sub', 'rules')),
                       ('BLOSUM62 + conservation', ('sub', 'cons')), ('tiers only', ())):
        s, cls, cons = score(pos, alt, segs, strand, chrom, prot, aligner, splice, parts)
        out['variants'][tag] = {g: round(auroc(y[m], s[m]), 4) for g, m in groups.items()}
        print(f'{tag:26s}', out['variants'][tag], flush=True)
    mine = np.array([c[0] for c in cls])
    truth = {'Missense': 'missense', 'Nonsense': 'nonsense', 'Synonymous': 'synonymous',
             'Canonical splice': 'canonical_splice'}
    named = np.isin(csq, list(truth))
    out['consequence_agreement'] = round(float(np.mean(
        [truth[c] == m or (c == 'Missense' and m == 'start_lost')
         for c, m in zip(csq[named], mine[named])])), 4)
    res = np.array([c[3] for c in cls])
    mis = mine == 'missense'
    out['residues_with_homologue_cover'] = round(float((cons[np.unique(res[mis])] > 0).mean()), 4)
    out['homologues_aligned'] = int(sum(1 for _, s in aligner.hits([prot])[0][1]
                                        if s >= CONS_MIN_SCORE))
    for nm, col in (('CADD', 'CADD.score'), ('phyloP (mammalian)', 'phyloP (mammalian)')):
        v = pd.to_numeric(df[col], errors='coerce').values
        v = np.where(np.isnan(v), np.nanmedian(v), v)
        out['variants'][nm] = {g: round(auroc(y[m], v[m]), 4) for g, m in groups.items()}
        print(f'{nm:26s}', out['variants'][nm], flush=True)
    print({k: v for k, v in out.items() if k != 'variants'})
    (RES / 'variant.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    cmd_brca1()
