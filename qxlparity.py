"""Browser-parity reference for quill-xl2-nano: real held-out mRNAs scored by Python."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import knowledge as K
import qgox
import qprot
import qrna
import qxl

HS = Path('/home/claude/data/hs')


def main(n_genes=16):
    m = qxl.XL(qxl.MODELS / 'quill-xl2-nano.npz')
    held = qgox.load_sets()['held']
    want = {}
    for r in held[::37]:
        want.setdefault(r['chrom'], set()).add(r['gene'])
    rows = []
    for c, genes in want.items():
        h = pickle.load(open(HS / f'{c}.pkl', 'rb'))
        seq = np.load(HS / f'{c}.npy')
        for tid, strand, segs, gid, _ in h['cds']:
            if gid not in genes or tid not in h['tx']:
                continue
            a = qrna.assemble(seq, h['tx'][tid][2], segs, strand)
            if not a or a[1][-3:] not in ('TAA', 'TAG', 'TGA') or not a[2] or not a[0]:
                continue
            genes.discard(gid)
            u5, cds, u3 = a[0], a[1], a[2]
            prot = ''.join(K.GENETIC_CODE.get(cds[i:i + 3], 'X')
                           for i in range(0, len(cds) - 3, 3))
            if '*' in prot or 'X' in prot or 'N' in u5 + cds + u3:
                continue
            d = qprot.describe(prot)
            hits = qprot.prosite(prot)
            rd = qrna.describe(u5, cds, u3, [], len(cds), do_fold=False)
            pf, rf = qprot.features(d, ()), qrna.features(rd)
            P = m.predict([prot], {'protein': [pf], 'prosite': [[f'prosite:{x}' for x in hits]],
                                   'rna': [rf]})[0]
            rows.append(dict(symbol=h['genes'][gid][0], u5=u5, cds=cds, u3=u3, prot=prot,
                             protein_features=sorted(pf), prosite=sorted(hits),
                             rna_features=sorted(f for f in rf if f in m.vocab['rna']),
                             top=[[int(j), float(P[j])] for j in np.argsort(-P)[:40]],
                             P=[round(float(v), 5) for v in P]))
        del seq
        if len(rows) >= n_genes:
            break
    (HERE / 'web' / '_ref-xl.json').write_text(json.dumps(rows))
    print(f'_ref-xl.json: {len(rows)} held-out mRNAs')


if __name__ == '__main__':
    main()
