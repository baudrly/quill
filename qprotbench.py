"""Zero-shot test of the written protein knowledge on held-out genes.

Each predictor in proteome.EXPECTED_TERMS is a written rule with no counted parameter.
Its score on every held-out protein is compared with whether the gene carries the GO
term the rule says it should mark (propagated annotation). Writes results/protein.json.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import proteome as P
import qgox
import qprot
from qgobench import auroc

RES = HERE / 'results'

SCORE = {
    'signal_peptide': lambda d: float(d['signal_peptide']),
    'tm_any': lambda d: len(d['tm_helices']),
    'tm_multi': lambda d: len(d['tm_helices']),
    'mito_presequence': lambda d: d['mito_charge'] + 10 * d['mito_moment'] +
    5 * d['mito_presequence'],
    'nls': lambda d: d['slims']['NLS_monopartite'] + 2 * d['slims']['NLS_bipartite'],
    'gpi_anchor': lambda d: float(d['gpi_anchor']),
    'coiled_coil': lambda d: d['coiled_coil_fraction'],
    'high_pI': lambda d: d['pI'],
    'disordered': lambda d: d['disorder_fraction'],
    'KDEL': lambda d: float(d['slims']['KDEL']),
    'N_glycosylation_density': lambda d: d['slims']['N_glycosylation'] / d['length'],
    'cysteine_rich': lambda d: d['cysteine'],
}


def main():
    sets, meta = qgox.load_sets(), qgox.load_terms()
    by_name = {v['name']: t for t, v in meta.items()}
    with open(qprot.CACHE, 'rb') as fh:
        cache = pickle.load(fh)
    held = [r for r in sets['held'] if cache.get(r['gene'], (None,))[0]]
    out = {}
    for pred, names in P.EXPECTED_TERMS.items():
        s = np.array([SCORE[pred](cache[r['gene']][0]) for r in held], float)
        for nm in names:
            tid = by_name.get(nm)
            if not tid:
                continue
            y = np.array([tid in set(r['terms']) for r in held])
            a = auroc(y, s)
            fired = s > 0 if pred in ('signal_peptide', 'gpi_anchor', 'KDEL') else None
            row = dict(auroc=round(a, 4), positives=int(y.sum()), n=len(held))
            if fired is not None:
                row['fires_on'] = int(fired.sum())
                row['precision_when_fires'] = round(float(y[fired].mean()), 4) \
                    if fired.any() else None
                row['base_rate'] = round(float(y.mean()), 4)
            out[f'{pred} -> {nm}'] = row
            extra = (f'  fires {row["fires_on"]}, precision {row["precision_when_fires"]} '
                     f'vs base {row["base_rate"]}') if fired is not None else ''
            print(f'  {pred:24s} -> {nm:32s} AUROC {a:.4f}{extra}', flush=True)
    d = [cache[r['gene']][0] for r in held]
    out['prevalence'] = dict(
        signal_peptide=round(float(np.mean([x['signal_peptide'] for x in d])), 4),
        any_tm=round(float(np.mean([len(x['tm_helices']) > 0 for x in d])), 4),
        mito_presequence=round(float(np.mean([x['mito_presequence'] for x in d])), 4),
        gpi_anchor=round(float(np.mean([x['gpi_anchor'] for x in d])), 4),
        any_prosite=round(float(np.mean([len(cache[r['gene']][1]) > 0 for r in held])), 4))
    out['n'] = len(held)
    (RES / 'protein.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out['prevalence']))


if __name__ == '__main__':
    main()
