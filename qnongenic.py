"""quill-xl, nongenic side: transcription-factor sites, regulatory elements, repeats.

  tf        each written site in regulatory.py, scored by the curated JASPAR 2024 matrix
            for the same factor: how much of the matrix's best score the written site
            reaches, and whether, among all 879 vertebrate matrices, the written site
            picks out its own factor
  reg       regulatory-element classification on Genomic Benchmarks with a linear probe
            over (a) sequence composition, (b) the quill-s grammar, (c) the written TF
            sites, (d) JASPAR matrices, and their combinations
  repeats   genome copies of the written Alu and L1 heads on held-out chromosomes,
            against shuffles of the same heads

Writes results/nongenic.json.
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import ontology as O
import regulatory as R

RES = HERE / 'results'
OUT = RES / 'nongenic.json'
HS = Path('/home/claude/data/hs')
MODELS = HERE / 'models'
BI = np.full(256, 4, np.int8)
for _i, _c in enumerate('ACGT'):
    BI[ord(_c)] = _i
    BI[ord(_c.lower())] = _i


def save(d):
    cur = json.loads(OUT.read_text()) if OUT.exists() else {}
    cur.update(d)
    OUT.write_text(json.dumps(cur, indent=1))


def jaspar():
    """Vertebrate CORE matrices from JASPAR 2024 as (name, family, log-odds L x 4)."""
    from pyjaspar import jaspardb
    out = []
    for m in jaspardb(release='JASPAR2024').fetch_motifs(collection='CORE',
                                                        tax_group=['vertebrates']):
        c = np.array([m.counts[b] for b in 'ACGT'], np.float64).T
        p = (c + 0.25) / (c.sum(1, keepdims=True) + 1.0)
        out.append((m.name.upper(), (m.tf_class or ['?'])[0], np.log2(p / 0.25)))
    return out


def best_alignment(lo_mat, site_mat):
    """Best normalised score of a written site under a JASPAR matrix.

    Slides the written matrix along the JASPAR one, on both strands, requiring overlap of
    at least 80% of the shorter. At each offset the written site's expected log-odds is
    compared with the JASPAR maximum and minimum over the same columns, giving a number
    in [0, 1] where 1 means the written site is the matrix's own consensus.
    """
    S = np.asarray(site_mat, np.float64)
    best = 0.0
    for strand in (S, S[::-1, ::-1]):
        Lj, Ls = len(lo_mat), len(strand)
        need = int(np.ceil(0.8 * min(Lj, Ls)))
        for off in range(-(Ls - need), Lj - need + 1):
            a, b = max(0, off), min(Lj, off + Ls)
            if b - a < need:
                continue
            J = lo_mat[a:b]
            W = strand[a - off:b - off]
            got = float((J * W).sum())
            hi, lo = float(J.max(1).sum()), float(J.min(1).sum())
            v = (got - lo) / max(hi - lo, 1e-9)
            best = max(best, v)
    return best


def cmd_tf(_):
    J = jaspar()
    names = sorted({n for n, _, _ in J})
    rows, ranks = [], []
    for tf, site in R.TF_SITES.items():
        sm = R.matrix(site)
        own = [lo for n, _, lo in J if n == tf]
        if not own:
            rows.append(dict(tf=tf, site=site, jaspar=None))
            continue
        self_score = max(best_alignment(lo, sm) for lo in own)
        # retrieval: of all factors, which does the written site score highest under?
        per_tf = {}
        for n, _, lo in J:
            per_tf[n] = max(per_tf.get(n, 0.0), best_alignment(lo, sm))
        order = sorted(per_tf, key=lambda n: -per_tf[n])
        rank = order.index(tf) + 1
        ranks.append(rank)
        rows.append(dict(tf=tf, site=site, fraction_of_jaspar_max=round(self_score, 4),
                         rank_of_own_factor=rank, of_factors=len(order),
                         top3=order[:3]))
        print(f'  {tf:9s} {site:22s} {self_score:.3f}  rank {rank:4d}/{len(order)}  '
              f'top {order[:3]}', flush=True)
    ranks = np.array(ranks)
    sc = np.array([r['fraction_of_jaspar_max'] for r in rows if r.get('jaspar', 1)
                   is not None and 'fraction_of_jaspar_max' in r])
    summary = dict(written=len(R.TF_SITES), found_in_jaspar=int(len(ranks)),
                   jaspar_matrices=len(J), jaspar_factors=len(names),
                   median_fraction_of_jaspar_max=round(float(np.median(sc)), 4),
                   at_least_0_9=int((sc >= 0.9).sum()),
                   own_factor_rank_1=int((ranks == 1).sum()),
                   own_factor_top_10=int((ranks <= 10).sum()),
                   median_rank=float(np.median(ranks)))
    print(json.dumps(summary, indent=1))
    save({'tf': dict(summary=summary, sites=rows)})


# ---------------------------------------------------------------- motif features
def onehot(seq):
    a = np.zeros((len(seq), 4), np.float32)
    ok = seq <= 3
    a[np.flatnonzero(ok), seq[ok]] = 1.0
    return a


def motif_bank(mats):
    """Group log-odds matrices by length for vectorised scanning: {L: (M, L, 4)}."""
    bank = {}
    for i, m in enumerate(mats):
        bank.setdefault(len(m), ([], []))
        bank[len(m)][0].append(i)
        bank[len(m)][1].append(np.asarray(m, np.float32))
    return {L: (np.array(ix), np.stack(ms)) for L, (ix, ms) in bank.items()}


def motif_max(seq, bank, n):
    """Best score of every matrix anywhere in the sequence, either strand."""
    X = onehot(seq)
    Xr = X[::-1, ::-1].copy()
    out = np.full(n, -20.0, np.float32)
    for L, (ix, M) in bank.items():
        if len(X) < L:
            continue
        for Z in (X, Xr):
            W = swv(Z, (L, 4))[:, 0]                       # positions x L x 4
            s = np.einsum('plk,mlk->pm', W, M, optimize=True).max(0)
            out[ix] = np.maximum(out[ix], s)
    return out


def kmer3(seq):
    s = seq[seq <= 3].astype(np.int64)
    if len(s) < 3:
        return np.zeros(64, np.float32)
    v = np.bincount(s[:-2] * 16 + s[1:-1] * 4 + s[2:], minlength=64).astype(np.float32)
    return v / v.sum()


def cmd_reg(a):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import qbench as QB
    J = jaspar()
    # one matrix per factor (the most informative), to keep the probe honest in size
    best = {}
    for n, fam, lo in J:
        ic = float(lo.max(1).sum())
        if n not in best or ic > best[n][1]:
            best[n] = (lo, ic)
    jnames = sorted(best)
    jbank = motif_bank([best[n][0] for n in jnames])
    wmats = [np.log2(np.asarray(R.matrix(s)) / 0.25) for s in R.TF_SITES.values()]
    wbank = motif_bank(wmats)
    model = QB.Model('quill-s')
    tasks = a.tasks.split(',')
    out = {}
    for task in tasks:
        tr = QB.load_rows(task, 'train', a.cap)
        te = QB.load_rows(task, 'test', a.cap)
        feats = {}
        for tag, rows in (('tr', tr), ('te', te)):
            F = {k: [] for k in ('kmer', 'grammar', 'written', 'jaspar')}
            y, held = [], []
            for region, s0, s1, st, ci in rows:
                s = QB.get_seq(region, s0, s1, st)
                if s is None or len(s) < 30:
                    continue
                g = QB.features(model, s)
                if g is None:
                    continue
                F['grammar'].append(g)
                F['kmer'].append(kmer3(s))
                F['written'].append(motif_max(s, wbank, len(wmats)))
                F['jaspar'].append(motif_max(s, jbank, len(jnames)))
                y.append(ci)
                held.append(region in QB.HELD_CHR)
            feats[tag] = ({k: np.array(v) for k, v in F.items()}, np.array(y),
                          np.array(held))
            print(f'  {task} {tag}: {len(y)} sequences featurised', flush=True)
        (Ftr, ytr, _), (Fte, yte, hte) = feats['tr'], feats['te']
        res = dict(n_train=int(len(ytr)), n_test=int(len(yte)),
                   n_class=int(len(set(ytr.tolist()))), majority=round(float(
                       max(np.mean(yte == c) for c in set(yte.tolist()))), 4))
        combos = {'composition (3-mers)': ['kmer'],
                  'composition + grammar': ['kmer', 'grammar'],
                  'composition + written TF sites': ['kmer', 'written'],
                  'composition + JASPAR': ['kmer', 'jaspar'],
                  'composition + grammar + written': ['kmer', 'grammar', 'written'],
                  'all four': ['kmer', 'grammar', 'written', 'jaspar']}
        for label, parts in combos.items():
            Xtr = np.hstack([Ftr[p] for p in parts])
            Xte = np.hstack([Fte[p] for p in parts])
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=4000, C=0.5).fit(sc.transform(Xtr), ytr)
            pred = clf.predict(sc.transform(Xte))
            res[label] = dict(accuracy=round(float((pred == yte).mean()), 4),
                              n_features=int(Xtr.shape[1]))
            print(f'  {task:28s} {label:34s} {res[label]["accuracy"]:.4f} '
                  f'({Xtr.shape[1]} features)', flush=True)
        out[task] = res
    save({'regulatory': out, 'jaspar_factors_used': len(jnames),
          'written_sites_used': len(wmats)})


# ---------------------------------------------------------------- repeats
def kmer_hits(seq, probes, k):
    """Positions where any k-mer of the probes (either strand) occurs exactly.

    The k-mer code is built one column at a time so that a whole chromosome costs a few
    arrays of its own length rather than k times that.
    """
    s = seq.astype(np.int64)
    bad = s > 3
    s[bad] = 0
    n = len(s) - k + 1
    code = np.zeros(n, np.int64)
    for j in range(k):
        code <<= 2
        code |= s[j:j + n]
    cb = np.concatenate([[0], np.cumsum(bad)])
    good = (cb[k:k + n] - cb[:n]) == 0
    want = set()
    for p in probes:
        a = BI[np.frombuffer(p.encode(), np.uint8)].astype(np.int64)
        for strand in (a, (3 - a)[::-1]):
            for i in range(len(strand) - k + 1):
                v = 0
                for x in strand[i:i + k]:
                    v = (v << 2) | int(x)
                want.add(v)
    hit = np.isin(code, np.fromiter(want, np.int64)) & good
    return np.flatnonzero(hit)


def cmd_repeats(a):
    rng = np.random.default_rng(3)
    out, k = {}, a.k
    for name, head in (('Alu', O.ALU_HEAD), ('L1', O.L1_ORF2_HEAD)):
        res = {}
        for c in ('21', '22'):
            seq = np.load(HS / f'{c}.npy')
            mb = float((seq <= 3).sum()) / 1e6
            pos = kmer_hits(seq, [head], k)
            loci = int((np.diff(pos) > 50).sum() + (1 if len(pos) else 0))
            sh = []
            for _ in range(5):
                l = list(head)
                rng.shuffle(l)
                p2 = kmer_hits(seq, [''.join(l)], k)
                sh.append(int((np.diff(p2) > 50).sum() + (1 if len(p2) else 0)))
            res[f'chr{c}'] = dict(loci=loci, per_mb=round(loci / mb, 1),
                                  shuffled_loci_mean=round(float(np.mean(sh)), 1),
                                  enrichment=round(loci / max(np.mean(sh), 1.0), 1),
                                  mb=round(mb, 1))
            print(f'  {name} chr{c}: {loci} loci ({loci/mb:.1f}/Mb) vs shuffled '
                  f'{np.mean(sh):.1f}', flush=True)
            del seq
        out[name] = res
    out['k'] = k
    save({'repeats': out})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('task', choices=['tf', 'reg', 'repeats'])
    ap.add_argument('--tasks', default='human_ensembl_regulatory,human_enhancers_cohn,'
                                       'human_ocr_ensembl,human_nontata_promoters')
    ap.add_argument('--cap', type=int, default=3000)
    ap.add_argument('--k', type=int, default=16)
    a = ap.parse_args()
    {'tf': cmd_tf, 'reg': cmd_reg, 'repeats': cmd_repeats}[a.task](a)
