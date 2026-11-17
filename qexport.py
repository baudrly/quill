"""Export a Quill model as a single JS file for the browser build.

Writes web/quill-model.js:  window.QUILL = {<header>, blob:"<base64>"}
All numeric arrays live in one base64 blob; the header gives each one's offset,
length and dtype so the page can slice typed array views with no parsing.
"""
import base64, json, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import qstates as S
from qbench import Model, STATE_GROUPS, GROUP_IDX, CODING_STATES, INTRON_STATES

OUT = Path('/home/claude/quill/web'); OUT.mkdir(exist_ok=True)
T = Path('/home/claude/data/tables')


def main(name='quill-s'):
    m = Model(name)
    parts, header = [], {}
    off = 0

    def put(key, arr, dtype):
        nonlocal off
        if off % 8:                      # keep every array 8-byte aligned for typed views
            pad = 8 - off % 8
            parts.append(b'\0' * pad); off += pad
        a = np.ascontiguousarray(arr, dtype=dtype)
        header[key] = dict(off=off, n=int(a.size), dt=np.dtype(dtype).name)
        parts.append(a.tobytes())
        off += a.nbytes

    grp3 = np.zeros(S.G.nstate, np.int32)
    grp3[INTRON_STATES] = 2
    grp3[CODING_STATES] = 1
    # finer grouping for the display ribbon
    ribbon = np.zeros(S.G.nstate, np.int32)
    RIB = ['intergenic', 'cpg_island', 'tandem', 'utr5', 'cds', 'intron', 'splice',
           'utr3', 'ncrna', 'polya', 'start', 'stop']
    for i, nm in enumerate(S.G.name):
        if nm == 'IG':
            ribbon[i] = 0
        elif nm == 'CGI':
            ribbon[i] = 1
        elif nm == 'TANDEM':
            ribbon[i] = 2
        elif nm == 'UTR5':
            ribbon[i] = 3
        elif nm.startswith('CDS'):
            ribbon[i] = 4
        elif nm.startswith('INTRON'):
            ribbon[i] = 5
        elif nm.startswith(('DON', 'ACC')):
            j = int(S.G.pwmpos[i])
            exon = (S.G.pwm[i] == 1 and j < 3) or (S.G.pwm[i] == 2 and j >= 20)
            ribbon[i] = 6
        elif nm == 'UTR3':
            ribbon[i] = 7
        elif nm == 'NCRNA':
            ribbon[i] = 8
        elif nm.startswith('PAS'):
            ribbon[i] = 9
        elif nm.startswith('ATG'):
            ribbon[i] = 10
        elif nm.startswith('STOP'):
            ribbon[i] = 11

    put('bb_tab', m.bb_tab, 'uint8')
    put('bb_off', m.bb_off, 'int32')
    put('cls', m.cls.reshape(-1), 'uint8')
    put('cls0', m.cls0.reshape(-1), 'uint8')
    put('pwm', m.pwm.reshape(-1), 'uint8')
    put('cod1', m.cod1.reshape(-1), 'uint8')
    put('cod2', m.cod2.reshape(-1), 'uint8')
    put('cod3', m.cod3.reshape(-1), 'uint8')
    put('mix', m.mix.reshape(-1), 'float32')
    put('efrom', S.G.efrom, 'int32')
    put('eto', S.G.eto, 'int32')
    put('elog', m.elog, 'float32')
    put('init', m.init, 'float32')
    put('state_slot', S.G.slot, 'int32')
    put('state_phase', S.G.phase, 'int32')
    put('grp3', grp3, 'int32')
    put('ribbon', ribbon, 'int32')
    put('slot_kind', m.slot_kind, 'int32')
    put('slot_cls', m.slot_cls, 'int32')
    put('slot_row', m.slot_row, 'int32')
    if name == 'quill-k':
        # the zero-data model must not ship counted display statistics either
        import knowledge as K
        import qwritten as W
        cf = W.codon_freq()
        put('codon_freq', cf, 'float32')
        cls0 = np.array([[W.dequant_cls0(m, c, g) for g in range(3)] for c in range(S.NCLS)])
        put('class_base_freq', cls0.reshape(-1), 'float32')
        il = np.arange(0, 60000, 20, dtype=np.float64)
        fs = K.INTRON_SHORT_FRAC
        put('intron_len_hist', fs * np.exp(-il / K.INTRON_SHORT) / K.INTRON_SHORT +
            (1 - fs) * np.exp(-il / K.INTRON_LONG) / K.INTRON_LONG, 'float32')
        el = np.arange(0, 1200, 4, dtype=np.float64)
        put('exon_len_hist', np.exp(-el / K.RATE['exon_len']) / K.RATE['exon_len'], 'float32')
    else:
        z = np.load(T / 'anno_train.npz')
        cf = z['codon'].astype(np.float64); cf = cf / cf.sum()
        put('codon_freq', cf, 'float32')
        aa = z['cls0'].astype(np.float64)
        put('class_base_freq', (aa / np.maximum(aa.sum(-1, keepdims=True), 1)).reshape(-1), 'float32')
        il = z['inlen'].astype(np.float64)
        put('intron_len_hist', il[:60000:20], 'float32')
        el = z['exlen'].astype(np.float64)
        put('exon_len_hist', el[:1200:4], 'float32')

    blob = b''.join(parts)
    meta = dict(
        name=name, order=int(m.order), nstate=int(S.G.nstate), nslot=int(S.NSLOT),
        nedge=int(len(S.G.efrom)), nparam=m.nparam, arrays=header,
        state_names=list(S.G.name), ribbon_names=RIB,
        class_names=S.CLSNAME, pwm_meta=m.pwm_meta,
        gc_edges=list(__import__('qanno').GC_EDGES), gc_win=int(__import__('qanno').GC_WIN),
        idx={k: int(S.G.idx[k]) for k in
             ('IG', 'CGI', 'TANDEM', 'UTR5', 'UTR3', 'NCRNA', 'CDS0', 'ATG0', 'STOP0', 'PAS0')},
        blob_bytes=len(blob))
    js = 'window.QUILL=' + json.dumps(meta) + ';\nwindow.QUILL.blob="' + \
         base64.b64encode(blob).decode() + '";\n'
    p = OUT / f'{name}-model.js'
    p.write_text(js)
    print(f'{p}  blob {len(blob)/1e6:.2f} MB  file {p.stat().st_size/1e6:.2f} MB')
    print(f'params {m.nparam["total"]:,}')
    return p


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'quill-s')
