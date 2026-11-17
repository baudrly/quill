"""Write the browser-parity reference for a model.

Takes a window of held-out human sequence, scores it with the Python kernels and
stores the sequence, total log2 likelihood, bits/base and Viterbi path. web/parity.js
recomputes all of it under jsdom and must reproduce the numbers exactly.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import qanno as A
import qinfer as I
from qbench import Model

HS = Path('/home/claude/data/hs')
WEB = Path('/home/claude/quill/web')
CHROM, START, N = '22', 20_000_000, 4000


def main(name):
    seq = np.load(HS / f'{CHROM}.npy')
    off = START
    while off + N <= len(seq) and (seq[off:off + N] > 3).any():
        off += N
    sq = np.ascontiguousarray(seq[off:off + N])
    del seq
    m = Model(name)
    gcb = A.gc_bins(sq, A.GC_WIN, *A.GC_EDGES)
    logp, n, _ = I.forward(sq, gcb, *m.args(), False)
    path = I.viterbi(sq, gcb, *m.args())
    path = np.asarray(path[0] if isinstance(path, tuple) else path, np.int64)
    ref = dict(model=name, chrom='chr' + CHROM, start=int(off), n=int(n),
               seq=''.join('ACGT'[b] for b in sq),
               logp=float(logp), bits_per_base=float(-logp / n),
               gcb=[int(v) for v in gcb], path=[int(v) for v in path])
    p = WEB / f'_ref-{name}.json'
    p.write_text(json.dumps(ref))
    print(f'{p.name}: chr{CHROM}:{off}-{off+N} {n} bases  '
          f'{-logp/n:.10f} bits/base  {len(set(ref["path"]))} distinct states')
    return p


if __name__ == '__main__':
    for nm in (sys.argv[1:] or ['quill-k', 'quill-s']):
        main(nm)
