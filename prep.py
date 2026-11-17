"""Parse Ensembl r91 GenBank dumps -> uint8 chromosomes (A0 C1 G2 T3 N4) + feature pickles."""
import gzip, re, pickle, sys, numpy as np
from pathlib import Path
SRC = Path('/home/claude/data/repos/Homo_sapiens.GRCh38.91.chromosome'); OUT = Path('/home/claude/data/hs')
LUT = np.full(256, 4, np.uint8)
for i, c in enumerate('ACGT'):
    LUT[ord(c)] = i; LUT[ord(c.lower())] = i
LOC = re.compile(r"<?(\d+)(?:\.\.>?(\d+))?")

def features(text):
    key, parts, quals, last = None, [], {}, None
    def flush():
        loc = ''.join(parts)
        if ':' in loc:  # segments on other clones
            return None
        return key, '-' if 'complement' in loc else '+', sorted((int(a) - 1, int(b or a)) for a, b in LOC.findall(loc)), quals
    for line in text.split('\n'):
        if len(line) < 22: continue
        if line[5] != ' ':
            if key:
                f = flush()
                if f: yield f
            key, parts, quals, last = line[5:21].strip(), [line[21:].strip()], {}, None
        elif key:
            t = line[21:]
            if t.startswith('/'):
                n, _, v = t[1:].partition('=')
                last = quals.setdefault(n, []); last.append(v.strip('"'))
            elif last is None: parts.append(t.strip())
            else: last[-1] += t.strip().strip('"') if n == 'translation' else ' ' + t.strip().strip('"')
    if key:
        f = flush()
        if f: yield f

for name in sys.argv[1:]:
    if (OUT / f'{name}.pkl').exists(): continue
    text = gzip.open(SRC / f'Homo_sapiens.GRCh38.91.chromosome.{name}.dat.gz', 'rb').read()
    i = text.index(b'\nORIGIN'); f0 = text.index(b'\nFEATURES')
    seq = np.frombuffer(text[i + 7:], np.uint8)
    seq = LUT[seq[(seq > 64)]]          # letters only (drops digits, spaces, newlines, //)
    np.save(OUT / f'{name}.npy', seq)
    genes, tx, cds = {}, {}, []
    for key, strand, segs, q in features(text[f0:i].decode('latin-1')):
        if key == 'gene':
            genes[q['gene'][0]] = (q.get('locus_tag', [''])[0], strand, segs[0][0], segs[-1][1])
        elif key in ('mRNA', 'misc_RNA'):
            tx[q['standard_name'][0].split('.')[0]] = (key, strand, segs, q['gene'][0])
        elif key == 'CDS':
            tid = [n for n in q.get('note', []) if n.startswith('transcript_id=')][0][14:].split('.')[0]
            cds.append((tid, strand, segs, q['gene'][0], q.get('translation', [''])[0].replace(' ', '')))
    pickle.dump(dict(genes=genes, tx=tx, cds=cds), open(OUT / f'{name}.pkl', 'wb'))
    print(name, len(seq), len(genes), len(tx), len(cds), flush=True)
