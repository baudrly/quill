"""Describe a messenger RNA with the written knowledge of transcriptome.py.

For each annotated gene, the longest coding transcript is spliced into its mature mRNA
and split into 5'UTR, CDS and 3'UTR. describe() reads RNA-binding-protein sites,
microRNA seed matches, AU-rich elements, Pumilio and CPE elements, poly(A) signals, the
histone stem-loop, upstream ORFs, the 5'TOP tract, G-quadruplexes, m6A consensus sites,
the start-codon context, codon adaptation to the written usage table, exon junctions
past the stop codon (the NMD rule), and folding of the 5' end with ViennaRNA, whose
energy parameters were measured in the lab rather than fitted here.

    python3 qrna.py cache      # describe every annotated gene's mRNA -> data/gox/rna.pkl
    python3 qrna.py validate   # zero-shot tests of the written RNA knowledge
"""
import json
import math
import pickle
import random
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import knowledge as K
import regulatory as Rg
import transcriptome as T

HS = Path('/home/claude/data/hs')
CACHE = Path('/home/claude/data/gox/rna.pkl')
RES = HERE / 'results'
COMP = str.maketrans('ACGTN', 'TGCAN')


def iupac_rx(site):
    return re.compile(''.join('[%s]' % Rg.IUPAC[c] if len(Rg.IUPAC[c]) > 1 else c
                              for c in site))


RBP_RX = {k: iupac_rx(v) for k, v in T.RBP_SITES.items()}
# a 7mer-m8 site is the reverse complement of seed nucleotides 2-8
MIR_SITES = {k: v.translate(COMP)[::-1] for k, v in T.MIRNA_SEEDS.items()}
G4_RX = re.compile(T.G_QUADRUPLEX)
DRACH_RX = re.compile(T.DRACH)
PRE_RX = iupac_rx(T.PRE)

# relative adaptiveness of each codon under the written usage table
_W = {}
for aa in set(K.GENETIC_CODE.values()) - {'*'}:
    cods = [c for c, a in K.GENETIC_CODE.items() if a == aa]
    mx = max(K.CODON_USAGE[c] for c in cods)
    for c in cods:
        _W[c] = K.CODON_USAGE[c] / mx


def cai(cds):
    ws = [_W[cds[i:i + 3]] for i in range(0, len(cds) - 2, 3)
          if cds[i:i + 3] in _W and K.GENETIC_CODE[cds[i:i + 3]] not in 'MW*']
    if not ws:
        return 0.0
    return float(math.exp(sum(math.log(max(w, 1e-4)) for w in ws) / len(ws)))


def count(s, sub):
    n, i = 0, s.find(sub)
    while i >= 0:
        n += 1
        i = s.find(sub, i + 1)
    return n


def fold(seq):
    import RNA
    s, mfe = RNA.fold(seq.replace('T', 'U'))
    return float(mfe)


def fold_z(seq, n_shuffles=T.FOLD_SHUFFLES, seed=0):
    """MFE and its z-score against mononucleotide shuffles of the same sequence."""
    if len(seq) < 20:
        return 0.0, 0.0
    m = fold(seq)
    rnd = random.Random(seed)
    sh = []
    for _ in range(n_shuffles):
        l = list(seq)
        rnd.shuffle(l)
        sh.append(fold(''.join(l)))
    sd = float(np.std(sh)) or 1.0
    return m, (m - float(np.mean(sh))) / sd


def assemble(seq, exons, cds_segs, strand):
    """(5'UTR, CDS, 3'UTR, junction positions in transcript coordinates) or None."""
    exons = sorted(exons)
    order = exons if strand == '+' else list(reversed(exons))
    parts, junc, pos = [], [], 0
    gpos = []
    for a, b in order:
        s = seq[a:b]
        if strand == '-':
            s = s[::-1]
            s = np.where(s <= 3, 3 - s, s)
        parts.append(s)
        gpos.append(np.arange(a, b) if strand == '+' else np.arange(b - 1, a - 1, -1))
        pos += b - a
        junc.append(pos)
    junc = junc[:-1]
    tx = ''.join('ACGTN'[min(int(v), 4)] for v in np.concatenate(parts))
    g = np.concatenate(gpos)
    lo = min(a for a, b in cds_segs)
    hi = max(b for a, b in cds_segs) - 1
    first, last = (lo, hi) if strand == '+' else (hi, lo)
    i0 = np.flatnonzero(g == first)
    i1 = np.flatnonzero(g == last)
    if not len(i0) or not len(i1) or i1[0] < i0[0]:
        return None
    i0, i1 = int(i0[0]), int(i1[0])
    return tx[:i0], tx[i0:i1 + 1], tx[i1 + 1:], junc, i1


def describe(utr5, cds, utr3, junc, stop_end, do_fold=True):
    tail = utr3[-60:] if utr3 else ''
    # 5'TOP: C at the cap followed by a pyrimidine run
    top = 0
    if utr5[:1] == 'C':
        while top + 1 < len(utr5) and utr5[top + 1] in 'CT':
            top += 1
    uorfs = sum(1 for m in re.finditer('ATG', utr5))
    kozak = cds[:4] if len(cds) >= 4 else ''
    ctx = (utr5[-3:] if len(utr5) >= 3 else '') + kozak
    strong = bool(re.match(T.KOZAK_STRONG, ctx)) if len(ctx) == 7 else False
    gc3 = (sum(1 for i in range(2, len(cds), 3) if cds[i] in 'GC') /
           max(len(cds) // 3, 1))
    nmd = any(j - stop_end > T.NMD_JUNCTION_DISTANCE for j in junc)
    d = dict(
        utr5_len=len(utr5), utr3_len=len(utr3), cds_len=len(cds), exons=len(junc) + 1,
        utr5_gc=round((utr5.count('G') + utr5.count('C')) / max(len(utr5), 1), 3),
        utr3_gc=round((utr3.count('G') + utr3.count('C')) / max(len(utr3), 1), 3),
        top_tract=top, uorfs=uorfs, kozak_strong=strong, gc3=round(gc3, 3),
        cai_written=round(cai(cds), 4),
        g4_5utr=bool(G4_RX.search(utr5)), g4_3utr=bool(G4_RX.search(utr3)),
        ire_5utr=T.IRE_LOOP in utr5,
        are_count=count(utr3, T.ARE_PENTAMER), are_nonamer=T.ARE_NONAMER in utr3,
        pre_count=len(PRE_RX.findall(utr3)), cpe_count=count(utr3, 'TTTTAT'),
        pas=next((p for p in T.POLYA_SIGNALS if p in tail), None),
        histone_stem_loop=T.HISTONE_SL in (utr3[:200] if utr3 else '') or
        T.HISTONE_SL in tail,
        m6a_per_kb=round(1000 * len(DRACH_RX.findall(utr3)) / max(len(utr3), 1), 2),
        ese_cds=sum(count(cds, h) for h in T.ESE_HEXAMERS),
        ess_cds=sum(count(cds, h) for h in T.ESS_HEXAMERS),
        nmd_junction_in_3utr=nmd,
        rbp_3utr={k: len(rx.findall(utr3)) for k, rx in RBP_RX.items()},
        mirna_sites={k: count(utr3, s) for k, s in MIR_SITES.items()})
    if do_fold and len(utr5) >= 20:
        m, z = fold_z(utr5[:T.FOLD_5UTR_MAX])
        d['utr5_mfe_per_nt'] = round(m / min(len(utr5), T.FOLD_5UTR_MAX), 4)
        d['utr5_fold_z'] = round(z, 3)
    else:
        d['utr5_mfe_per_nt'], d['utr5_fold_z'] = 0.0, 0.0
    return d


BINS = dict(utr5_len=[1, 50, 100, 200, 400], utr3_len=[1, 200, 500, 1000, 2000, 4000],
            exons=[2, 4, 8, 15, 30], utr5_gc=[0.5, 0.6, 0.7], utr3_gc=[0.35, 0.45, 0.55],
            gc3=[0.45, 0.55, 0.65, 0.75], cai_written=[0.7, 0.75, 0.8, 0.85],
            uorfs=[1, 2, 4], are_count=[1, 3, 6, 12], m6a_per_kb=[5, 10, 20],
            utr5_mfe_per_nt=[-0.5, -0.4, -0.3, -0.2], utr5_fold_z=[-2.0, -1.0, 0.0])


def features(d):
    f = [f'rna:{k}:{int(np.searchsorted(e, d[k], side="right"))}' for k, e in BINS.items()]
    for k in ('kozak_strong', 'g4_5utr', 'g4_3utr', 'ire_5utr', 'are_nonamer',
              'histone_stem_loop', 'nmd_junction_in_3utr'):
        if d[k]:
            f.append(f'rna:{k}')
    if d['top_tract'] >= T.TOP_MIN:
        f.append('rna:top_tract')
    f.append(f'rna:pas:{d["pas"]}')
    for k in ('pre_count', 'cpe_count'):
        if d[k]:
            f.append(f'rna:{k}:' + ('1' if d[k] == 1 else '2+'))
    for k, c in d['rbp_3utr'].items():
        if c:
            f.append(f'rna:rbp:{k}:' + ('1' if c == 1 else ('2-4' if c <= 4 else '5+')))
    for k, c in d['mirna_sites'].items():
        if c:
            f.append(f'rna:mir:{k}:' + ('1' if c == 1 else '2+'))
    return f


def _one(job):
    gene, args = job
    return gene, describe(*args)


def cmd_cache():
    import qgox
    from multiprocessing import Pool
    sets = qgox.load_sets()
    want = {}
    for s in ('train', 'calib', 'held'):
        for r in sets[s]:
            want.setdefault(r['chrom'], set()).add(r['gene'])
    out = {}
    for c, genes in sorted(want.items()):
        h = pickle.load(open(HS / f'{c}.pkl', 'rb'))
        seq = np.load(HS / f'{c}.npy')
        cand = {}
        for tid, strand, segs, gid, prot in h['cds']:
            if gid in genes and tid in h['tx']:
                cand.setdefault(gid, []).append((len(prot), tid, strand, segs))
        jobs = []
        for gid, cs in cand.items():
            # prefer a complete mRNA: stop codon at the end of the CDS and a 3'UTR,
            # then the longest protein; a fragment is the fallback
            fallback = None
            for _, tid, strand, segs in sorted(cs, reverse=True):
                a = assemble(seq, h['tx'][tid][2], segs, strand)
                if not a:
                    continue
                if a[1][-3:] in ('TAA', 'TAG', 'TGA') and len(a[1]) % 3 == 0 and a[2]:
                    jobs.append((gid, a))
                    fallback = None
                    break
                fallback = fallback or a
            else:
                if fallback:
                    jobs.append((gid, fallback))
        del seq
        with Pool(2) as pool:
            for g, d in pool.imap_unordered(_one, jobs, 32):
                out[g] = d
        print(f'  chr{c}: {len(out)} mRNAs described', flush=True)
    with open(CACHE, 'wb') as fh:
        pickle.dump(out, fh, protocol=4)
    print(f'{CACHE}: {len(out)} mRNAs')


def load_cache():
    with open(CACHE, 'rb') as fh:
        return pickle.load(fh)


def cmd_validate():
    """Written RNA predictors against GO truth, and folding against SO classes."""
    import qgox
    import qso
    from qgobench import auroc
    sets, meta = qgox.load_sets(), qgox.load_terms()
    by_name = {v['name']: t for t, v in meta.items()}
    rna = load_cache()
    held = [r for r in sets['held'] if r['gene'] in rna]
    score_of = {
        'top_tract': lambda d: d['top_tract'],
        'are_count': lambda d: d['are_count'] * 1000.0 / max(d['utr3_len'], 200),
        'histone_stem_loop': lambda d: float(d['histone_stem_loop']),
        'cai_written': lambda d: d['cai_written'],
        'long_3utr': lambda d: d['utr3_len'],
        'g4_5utr': lambda d: float(d['g4_5utr']),
    }
    out = {}
    for pred, names in T.EXPECTED_TERMS.items():
        s = np.array([score_of[pred](rna[r['gene']]) for r in held], float)
        for nm in names:
            tid = by_name.get(nm)
            if not tid:
                continue
            y = np.array([tid in set(r['terms']) for r in held])
            out[f'{pred} -> {nm}'] = dict(auroc=round(auroc(y, s), 4),
                                          positives=int(y.sum()), n=len(held))
            print(f'  {pred:18s} -> {nm:40s} AUROC {out[f"{pred} -> {nm}"]["auroc"]:.4f} '
                  f'({int(y.sum())} positives)', flush=True)
    # folding: structured small RNAs against long non-coding and pseudogene transcripts
    srecs = qso.load('held')
    structured = {'snoRNA_CD', 'snoRNA_HACA', 'snRNA', 'pre_miRNA', 'tRNA', 'Y_RNA',
                  'srpRNA_7SL'}
    loose = {'lincRNA', 'antisense_lncRNA', 'processed_pseudogene'}
    rnd = random.Random(1)
    rows = []
    for r in srecs:
        if r['so'] not in structured | loose:
            continue
        s = ''.join('ACGT'[int(x)] if x <= 3 else 'N' for x in r['arr'])
        if 'N' in s:
            continue
        if len(s) > 150:            # a 150-nt window, so long and short RNAs compare fairly
            st = rnd.randrange(0, len(s) - 150 + 1)
            s = s[st:st + 150]
        m, z = fold_z(s, 6, seed=len(rows))
        rows.append((r['so'] in structured, z, r['so']))
    y = np.array([r[0] for r in rows])
    z = np.array([r[1] for r in rows])
    per = {}
    for cls in sorted({r[2] for r in rows}):
        v = [r[1] for r in rows if r[2] == cls]
        per[cls] = dict(n=len(v), mean_fold_z=round(float(np.mean(v)), 3))
    out['folding'] = dict(auroc_structured_vs_loose=round(auroc(y, -z), 4),
                          n=len(rows), n_structured=int(y.sum()), per_class=per)
    print(f'  folding z, structured small RNAs vs lncRNA/pseudogene: AUROC '
          f'{out["folding"]["auroc_structured_vs_loose"]:.4f} ({len(rows)} transcripts)')
    for k, v in per.items():
        print(f'     {k:22s} n={v["n"]:4d}  mean z {v["mean_fold_z"]:+.2f}')
    cov = {k: round(float(np.mean([bool(rna[r['gene']][k]) for r in held])), 4)
           for k in ('g4_5utr', 'kozak_strong', 'are_nonamer', 'nmd_junction_in_3utr',
                     'histone_stem_loop', 'ire_5utr')}
    out['prevalence'] = cov
    out['n_mrnas'] = len(rna)
    cur = json.loads((RES / 'rna.json').read_text()) if (RES / 'rna.json').exists() else {}
    cur.update(dict(validate=out))
    (RES / 'rna.json').write_text(json.dumps(cur, indent=1))


if __name__ == '__main__':
    {'cache': cmd_cache, 'validate': cmd_validate}[sys.argv[1]]()
