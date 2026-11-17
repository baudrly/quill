"""Export the ontology models to one JavaScript file for the browser app.

The three written models are a few kilobytes each, so they ship as base64 of the same
quantised bytes the Python side uses, plus the small float vectors as plain numbers. The
browser recomputes scores from these and must agree with Python (web/ontparity.js).
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import function as F
import ontology as O
import qgo
import qso

WEB = HERE / 'web'
MODELS = HERE / 'models'


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def pack_go(name):
    z = np.load(MODELS / f'{name}.npz', allow_pickle=True)
    return dict(
        names=json.loads(str(z['names'])), labels=json.loads(str(z['labels'])),
        base=b64(z['base']), base_shape=list(z['base'].shape),
        nterm=b64(z['nterm']), nterm_shape=list(z['nterm'].shape),
        period=b64(z['period']), period_shape=list(z['period'].shape),
        tm_table=b64(z['tm_table']),
        nterm_len=[int(v) for v in z['nterm_len']],
        period_n=[int(v) for v in z['period_n']],
        period_w=[float(v) for v in z['period_w']],
        len_mean=[float(v) for v in z['len_mean']],
        tm=[float(v) for v in z['tm']],
        prior=[float(v) for v in z['prior']],
        nparam=qgo.GoModel(MODELS / f'{name}.npz').nparam)


def pack_so(name):
    z = np.load(MODELS / f'{name}.npz', allow_pickle=True)
    idx = json.loads(str(z['mat_index']))
    return dict(
        names=json.loads(str(z['names'])), boxes=json.loads(str(z['boxes'])),
        mats={k: dict(data=b64(z[f'mat_{k}']), rows=int(z[f'mat_{k}'].shape[0]))
              for k in idx},
        len_mean=[float(v) for v in z['len_mean']],
        gc=[float(v) for v in z['gc']],
        exons=[float(v) for v in z['exons']],
        hairpin=[float(v) for v in z['hairpin']],
        prior=[float(v) for v in z['prior']],
        nparam=qso.SoModel(MODELS / f'{name}.npz').nparam)


def main():
    payload = dict(
        go=pack_go('quill-k-go'), fam=pack_go('quill-k-fam'), so=pack_so('quill-k-so'),
        go_counted=pack_go('quill-s-go'), fam_counted=pack_go('quill-s-fam'),
        so_counted=pack_so('quill-s-so'),
        aa=F.AA, codons=qgo.CODONS,
        genetic_code=''.join(qgo.CODON_AA[c] for c in qgo.CODONS),
        weights=dict(F.SCORE_WEIGHTS), so_weights=dict(qso.SO_WEIGHTS),
        len_logsigma=F.LEN_LOGSIGMA, so_len_logsigma=qso.SO_LEN_LOGSIGMA,
        tm_segment_len=F.TM_SEGMENT_LEN,
        so_terms={k: v.get('so', '') for k, v in O.SO_RNA_CLASSES.items()},
        labels_go={k: v['label'] for k, v in F.GO_CLASSES.items()},
        labels_fam={k: v['label'] for k, v in F.GENE_FAMILIES.items()})
    WEB.mkdir(exist_ok=True)
    out = WEB / 'quill-ontology.js'
    out.write_text('window.QUILL_ONT = ' + json.dumps(payload, separators=(',', ':'))
                   + ';\n')
    print(f'{out.name}: {out.stat().st_size/1024:.1f} KB  '
          f'({payload["go"]["nparam"]:,} + {payload["fam"]["nparam"]:,} + '
          f'{payload["so"]["nparam"]:,} written parameters)')


if __name__ == '__main__':
    main()
