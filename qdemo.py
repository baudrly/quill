"""Pick real loci from held-out chromosomes for the webapp demos, with true annotation."""
import json, pickle, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, '.')
import qanno as A, qstates as S

HS = Path('/home/claude/data/hs')
out = []
WANT = [('21', 12000), ('20', 12000), ('19', 9000)]
for name, span in WANT:
    seq = np.load(HS / f'{name}.npy')
    meta = pickle.load(open(HS / f'{name}.pkl', 'rb'))
    best = None
    for tid, kind, strand, exons, cds, prot in A.transcripts(meta):
        if cds is None or len(cds) < 4:
            continue
        lo, hi = exons[0][0], exons[-1][1]
        L = hi - lo
        if not (span * 0.45 < L < span * 0.92):
            continue
        w0 = max(0, lo - (span - L) // 2); w1 = w0 + span
        if w1 > len(seq) or (seq[w0:w1] > 3).any():
            continue
        score = len(cds) + sum(b - a for a, b in cds) / 1000
        if best is None or score > best[0]:
            best = (score, tid, strand, exons, cds, w0, w1, meta['genes'])
    if best is None:
        print('no locus found on', name); continue
    score, tid, strand, exons, cds, w0, w1, genes = best
    sq = seq[w0:w1]
    gname = ''
    for gid, (locus, st, gs, ge) in genes.items():
        if gs <= exons[0][0] and ge >= exons[-1][1] and st == strand:
            gname = locus or gid
            break
    out.append(dict(chrom='chr' + name, start=int(w0), end=int(w1), strand=strand,
                    transcript=tid, gene=gname, assembly='GRCh38 (Ensembl r91)',
                    seq=''.join('ACGT'[b] for b in sq),
                    cds=[[int(a - w0), int(b - w0)] for a, b in cds],
                    exons=[[int(a - w0), int(b - w0)] for a, b in exons]))
    print(f'chr{name}:{w0}-{w1} {strand} {gname} {tid} {len(cds)} CDS segments, '
          f'{sum(b-a for a,b in cds)} coding bp')
    del seq
Path('web/demo-loci.js').write_text('window.QUILL_LOCI=' + json.dumps(out) + ';\n')
print('wrote web/demo-loci.js', Path('web/demo-loci.js').stat().st_size, 'bytes')
