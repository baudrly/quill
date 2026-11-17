"""Export quill-xl2-nano and all the written knowledge it reads to one browser file.

The page recomputes everything from this file: the protein description (proteome.py),
the PROSITE hits, the mRNA description (transcriptome.py), the three feature experts,
the written signatures, the per-namespace combination and the GO graph. web/xlparity.js
checks it against Python on held-out genes.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import knowledge as K
import proteome as P
import qgox
import qprot
import qrna
import qxl
import transcriptome as T

WEB = HERE / 'web'


def csr(m):
    return dict(indptr=m.indptr.astype(int).tolist(), indices=m.indices.astype(int).tolist(),
                data=[round(float(v), 4) for v in m.data], shape=list(m.shape))


def main(size='nano'):
    m = qxl.XL(qxl.MODELS / f'quill-xl2-{size}.npz')
    meta = qgox.load_terms()
    ecm = qxl.ec_map(None)
    children = {i: [] for i in range(m.T)}
    for i in range(m.T):
        for k in range(m.parent_ptr[i], m.parent_ptr[i + 1]):
            children[int(m.parent_idx[k])].append(i)
    model = dict(
        size=size, names=m.names, term_names=[meta[t]['name'] for t in m.names],
        ns=m.ns, prior=[float(v) for v in m.prior], tau=m.tau, weights=m.weights,
        experts=list(m.cfg['experts']),
        vocab={k: sorted(v, key=v.get) for k, v in m.vocab.items()},
        D={k: csr(v) for k, v in m.D.items()},
        order=m.order.astype(int).tolist(), parent_ptr=m.parent_ptr.astype(int).tolist(),
        parent_idx=m.parent_idx.astype(int).tolist(),
        has_child=[bool(children[i]) for i in range(m.T)],
        ec={str(i): sorted(e for e in ecm.get(t, ()) if '-' not in e)
            for i, t in enumerate(m.names) if ecm.get(t)},
        nparam=m.nparam)
    know = dict(
        AA=P.AA, KD=P.KYTE_DOOLITTLE, EIS=P.EISENBERG, DIS=P.TOP_IDP, HEL=P.HELIX,
        SHE=P.SHEET, MASS=P.MASS, WATER=P.WATER, PKA=P.PKA,
        rules={k: getattr(P, k) for k in dir(P) if k.isupper() and isinstance(
            getattr(P, k), (int, float, str)) and k not in ('AA', 'WATER')},
        slims=[[n, r] for n, r, _ in P.SLIMS],
        prot_bins=qprot.BINS, tm_classes=qprot.TM_CLASSES,
        prosite=[[a, n, d, rx] for a, n, d, rx, _, skip in qprot.prosite_patterns()
                 if not skip],
        rbp={k: rx.pattern for k, rx in qrna.RBP_RX.items()}, mir=qrna.MIR_SITES,
        g4=T.G_QUADRUPLEX, drach=T.DRACH, pre=qrna.PRE_RX.pattern,
        are=T.ARE_PENTAMER, are9=T.ARE_NONAMER, pas=list(T.POLYA_SIGNALS),
        histone_sl=T.HISTONE_SL, kozak=T.KOZAK_STRONG, top_min=T.TOP_MIN, ire=T.IRE_LOOP,
        cai_w=qrna._W, code=K.GENETIC_CODE, rna_bins=qrna.BINS)
    out = WEB / f'quill-xl-{size}.js'
    out.write_text('window.QUILL_XL = ' + json.dumps(dict(model=model, know=know),
                                                       separators=(',', ':')) + ';\n')
    print(f'{out.name}: {out.stat().st_size / 1e6:.2f} MB, {m.T} terms, '
          f'{m.nparam:,} parameters, {len(know["prosite"])} PROSITE patterns')


if __name__ == '__main__':
    main(*(sys.argv[1:] or ['nano']))
