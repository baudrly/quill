"""Held-out evaluation of quill-xl on the whole Gene Ontology.

CAFA-style protein-centric Fmax per namespace on chromosomes 17, 19, 20, 21, 22, for:
  naive        term frequency among training genes, the same for every protein
  rules        the written signatures of domains.py, at their written confidence
  memory       recall of the most similar training genes (a homology baseline)
  dictionary   significant peptide 5-mer -> term associations
  quill-xl     dictionary + memory + prior
  quill-xl+rules
and the same on the 'remote' subset: held-out genes sharing almost no 5-mers with any
training gene, where homology transfer has nothing to work with.

Writes results/goxbench.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import qdomains
import qgox

RES = HERE / 'results'
REMOTE = 0.10        # max 5-mer containment to any training gene below which a gene is remote


def rules_matrix(prots, names, meta):
    resolved, _ = qdomains.resolve(meta)
    index = {t: i for i, t in enumerate(names)}
    P = np.zeros((len(prots), len(names)), np.float32)
    hits = 0
    for i, p in enumerate(prots):
        for rule, tid, conf in qdomains.scan(p, resolved):
            j = index.get(tid)
            if j is not None:
                P[i, j] = max(P[i, j], conf)
                hits += 1
    return P, hits


def main():
    sets, meta = qgox.load_sets(), qgox.load_terms()
    held = sets['held']
    prots = [r['prot'] for r in held]
    m = qgox.GoxModel(qgox.MODELS / 'quill-xl-go.npz')
    names = m.names
    naive = np.tile(m.prior[None, :] * 0.999, (len(held), 1))

    X5 = qgox.matrix(prots, qgox.exact5, qgox.N5).tocsr()
    _, sim = m.recall(X5, k=2)
    top = sim[:, 0]
    remote = top < REMOTE

    R, nhits = rules_matrix(prots, names, meta)
    arms = {
        'naive': m.propagate(naive.copy()),
        'rules': m.propagate(np.maximum(R, naive)),
        'memory': m.predict(prots, parts=('memory', 'prior')),
        'dictionary': m.predict(prots, parts=('dictionary', 'prior')),
        'quill-xl': m.predict(prots),
    }
    arms['quill-xl+rules'] = m.propagate(np.maximum(arms['quill-xl'], R))

    th = np.linspace(0.005, 0.99, 80)
    out = dict(n_held=len(held), n_remote=int(remote.sum()), remote_threshold=REMOTE,
               similarity_quantiles=dict(zip(['p10', 'p25', 'p50', 'p75', 'p90'],
                   [round(float(v), 4) for v in np.quantile(top, [.1, .25, .5, .75, .9])])),
               rule_hits_on_model_terms=nhits, n_terms=len(names),
               parameters=m.nparam, parts=m.parts, results={})
    sub_remote = [r for r, k in zip(held, remote) if k]
    for arm, P in arms.items():
        res = {}
        for ns in ('BP', 'MF', 'CC'):
            res[ns] = qgox.fmax(P.copy(), held, names, meta, ns, th)
            res[f'{ns}_remote'] = qgox.fmax(P[remote].copy(), sub_remote, names, meta,
                                            ns, th)
        out['results'][arm] = res
        print(f'{arm:15s} ' + '  '.join(
            f'{ns} {res[ns]["fmax"]:.3f}/{res[ns + "_remote"]["fmax"]:.3f}'
            for ns in ('BP', 'MF', 'CC')), flush=True)

    # three worked examples from the remote subset, as the model would report them
    full = arms['quill-xl+rules']
    ex = []
    order = np.argsort(top)
    resolved, _ = qdomains.resolve(meta)
    for i in order:
        rules = qdomains.scan(prots[i], resolved)
        if not rules or held[i]['symbol'] == '':
            continue
        mf = [j for j in np.argsort(-full[i]) if meta[names[j]]['ns'] == 'MF'][:4]
        ex.append(dict(symbol=held[i]['symbol'], chrom=held[i]['chrom'],
                       nearest_training_similarity=round(float(top[i]), 4),
                       rules=[r[0] for r in rules],
                       top_mf=[(meta[names[j]]['name'], round(float(full[i, j]), 3),
                                names[j] in set(held[i]['terms'])) for j in mf]))
        if len(ex) >= 6:
            break
    out['remote_examples'] = ex
    (RES / 'goxbench.json').write_text(json.dumps(out, indent=1))
    print(f'remote genes: {out["n_remote"]} of {out["n_held"]}; similarity '
          f'quantiles {out["similarity_quantiles"]}')
    for e in ex:
        print(' ', e['symbol'], e['nearest_training_similarity'], e['rules'],
              [(n[:34], p, ok) for n, p, ok in e['top_mf']])


if __name__ == '__main__':
    main()
