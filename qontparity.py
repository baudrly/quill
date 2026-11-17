"""Write the browser-parity reference for the ontology models.

Takes real held-out genes and transcripts, scores them with the Python implementations,
and stores the per-class scores. web/ontparity.js recomputes them and must match.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import qgo
import qso
from qgo import GoModel
from qso import SoModel

WEB = HERE / 'web'
N_PER = 6


def main():
    out = {}
    held = qgo.load('held')
    for arm, key in (('go', 'cls'), ('fam', 'family')):
        m = GoModel(HERE / 'models' / f'quill-k-{arm}.npz')
        out[f'{arm}_names'] = m.names
        picks, seen = [], set()
        for r in held:
            if not r[key] or r[key] in seen or len(r['codons']) < 60:
                continue
            seen.add(r[key])
            picks.append(r)
            if len(picks) >= N_PER:
                break
        rows = []
        for r in picks:
            c = np.asarray(r['codons'])
            dna = ''.join(qgo.CODONS[int(v)] if v >= 0 else 'NNN' for v in c)
            rows.append(dict(symbol=r['symbol'], truth=r[key], dna=dna,
                             scores=[float(v) for v in m.score(r['codons'], r['prot'])]))
        out[arm] = rows

    sm = SoModel(HERE / 'models' / 'quill-k-so.npz')
    out['so_names'] = sm.names
    srecs = qso.load('held')
    picks, seen = [], set()
    for r in srecs:
        if r['so'] in seen:
            continue
        seen.add(r['so'])
        picks.append(r)
        if len(picks) >= N_PER:
            break
    rows = []
    for r in picks:
        a = r['arr']
        f = qso.features(a, r['exons'])
        rows.append(dict(symbol=r['symbol'], truth=r['so'], exons=int(r['exons']),
                         dna=''.join('ACGT'[int(x)] if x <= 3 else 'N' for x in a),
                         hairpin=float(f['hairpin']), gc=float(f['gc']),
                         scores=[float(v) for v in sm.score(a, r['exons'])]))
    out['so'] = rows

    p = WEB / '_ref-ontology.json'
    p.write_text(json.dumps(out))
    print(f'{p.name}: {len(out["go"])} functional, {len(out["fam"])} family, '
          f'{len(out["so"])} SO reference sequences')


if __name__ == '__main__':
    main()
