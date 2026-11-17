"""Head-to-head against Evo 2 on the BRCA1 saturation-mutagenesis set.

Not run in the environment that produced the preprint: Evo 2's weights are not
reachable there and it has no GPU. Run this on a CUDA machine to replace the
published Evo 2 figures with ones measured on the same variants, the same
reference sequence and the same AUROC code that scored Quill.

    pip install evo2 pandas openpyxl scikit-learn
    python3 qevo2.py --model evo2_7b --window 8192 --out results/evo2.json

Scoring follows Evo 2's own zero-shot protocol: score a window centred on the
variant with the reference base and with the alternate base, and take the
difference of total log-likelihoods. Quill's numbers for the same variants are
read from results/bench.json so the two land in one table.
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
BRCA1 = Path('/home/claude/data/repos/evo2/notebooks/brca1')
RES = HERE / 'results'
GROUPS = (('all', None), ('coding', 'coding'), ('noncoding', 'noncoding'),
          ('splice_and_intronic', 'splice'), ('missense', 'missense'))


def load_variants(brca1_dir):
    """The Findlay et al. table plus the GRCh37 chr17 reference Evo 2 ships with it."""
    import pandas as pd
    df = pd.read_excel(brca1_dir / '41586_2018_461_MOESM3_ESM.xlsx', header=2)
    lut = np.full(256, 4, np.uint8)
    for i, c in enumerate('ACGT'):
        lut[ord(c)] = i
        lut[ord(c.lower())] = i
    with gzip.open(brca1_dir / 'GRCh37.p13_chr17.fna.gz', 'rt') as fh:
        fh.readline()
        chrom = lut[np.frombuffer(''.join(l.strip() for l in fh).encode(), np.uint8)]
    pos = (df['position (hg19)'].values - 1).astype(np.int64)
    ref = np.array(['ACGT'.index(x) for x in df['reference']], np.int64)
    if not (chrom[pos] == ref).all():
        raise SystemExit('reference base mismatch: wrong assembly for this table')
    return df, chrom, pos, ref


def masks(df):
    csq = df['consequence'].values
    coding = np.isin(csq, ['Missense', 'Nonsense', 'Synonymous'])
    return {'all': np.ones(len(df), bool), 'coding': coding, 'noncoding': ~coding,
            'splice': np.isin(csq, ['Canonical splice', 'Splice region', 'Intronic']),
            'missense': csq == 'Missense'}, csq


def score_evo2(model_name, chrom, pos, ref, alt, window, batch):
    """Delta of total log-likelihood, reference window minus alternate window."""
    from evo2 import Evo2
    model = Evo2(model_name)
    code = 'ACGT'
    half = window // 2
    lo = np.maximum(pos - half, 0)
    hi = np.minimum(lo + window, len(chrom))
    lo = np.maximum(hi - window, 0)

    seqs, index = [], []
    for i in range(len(pos)):
        w = chrom[lo[i]:hi[i]].copy()
        off = pos[i] - lo[i]
        base = ''.join(code[b] if b < 4 else 'N' for b in w)
        index.append(len(seqs))
        seqs.append(base)
        mut = list(base)
        mut[off] = code[alt[i]]
        seqs.append(''.join(mut))

    scores = np.empty(len(seqs))
    for s in range(0, len(seqs), batch):
        chunk = seqs[s:s + batch]
        _, out = model.score_sequences(chunk)
        scores[s:s + len(chunk)] = np.asarray(out, np.float64)
        print(f'  scored {min(s + batch, len(seqs)):,}/{len(seqs):,}', flush=True)
    j = np.array(index)
    return scores[j + 1] - scores[j]          # negative where the variant is less likely


def auroc_table(y, score, mask_map):
    from sklearn.metrics import roc_auc_score
    return {name: round(float(roc_auc_score(y[m], score[m])), 4)
            for name, key in GROUPS
            for m in [mask_map['all'] if key is None else mask_map[key]]
            if y[m].sum() > 3 and (~y[m]).sum() > 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='evo2_7b')
    ap.add_argument('--window', type=int, default=8192)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--brca1', type=Path, default=BRCA1)
    ap.add_argument('--out', type=Path, default=RES / 'evo2.json')
    a = ap.parse_args()

    df, chrom, pos, ref = load_variants(a.brca1)
    alt = np.array(['ACGT'.index(x) for x in df['alt']], np.int64)
    y = (df['func.class'] == 'LOF').values
    mask_map, _ = masks(df)

    print(f'{a.model}: scoring {len(df):,} variants in {a.window:,} bp windows', flush=True)
    delta = score_evo2(a.model, chrom, pos, ref, alt, a.window, a.batch)
    # loss-of-function variants should lower the likelihood, so the score is -delta
    table = auroc_table(y, -delta, mask_map)

    out = {'model': a.model, 'window': a.window, 'n': int(len(df)),
           'n_lof': int(y.sum()), 'auroc': table}
    bench = RES / 'bench.json'
    if bench.exists():
        b = json.loads(bench.read_text())['brca1']
        out['quill'] = {k: {g: b[k].get({'all': 'all', 'coding': 'coding',
                                         'noncoding': 'noncoding',
                                         'splice_and_intronic': 'splice_and_intronic',
                                         'missense': 'auroc_Missense'}[g])
                            for g, _ in GROUPS}
                        for k in b if k.startswith('quill-')}
    a.out.parent.mkdir(exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(table, indent=1))
    print('saved', a.out)


if __name__ == '__main__':
    main()
