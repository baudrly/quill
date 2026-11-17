"""Ship a handful of real held-out genes and transcripts with the browser page."""
import json, sys
from pathlib import Path
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE))
import qgo, qso

WANT_FAM = ['collagen', 'zinc_finger', 'olfactory_receptor', 'histone',
            'ribosomal_protein', 'keratin', 'immunoglobulin_V']
WANT_SO = ['snoRNA_CD', 'snoRNA_HACA', 'snRNA', 'pre_miRNA', 'srpRNA_7SL',
           'lincRNA', 'processed_pseudogene']


def main():
    held = qgo.load('held')
    coding, seen = [], set()
    for fam in WANT_FAM:
        for r in held:
            if r['family'] != fam or r['symbol'] in seen:
                continue
            n = len(r['codons'])
            if not (90 <= n <= 1500):
                continue
            seen.add(r['symbol'])
            dna = ''.join(qgo.CODONS[int(v)] if v >= 0 else 'NNN' for v in r['codons'])
            coding.append(dict(label=f"{r['symbol']} (chr{r['chrom']})", dna=dna,
                               truth=fam))
            break
    for cls in ('integral_membrane', 'secreted', 'mitochondrion'):
        for r in held:
            if r['cls'] != cls or r['family'] or r['symbol'] in seen:
                continue
            if not (150 <= len(r['codons']) <= 900):
                continue
            seen.add(r['symbol'])
            dna = ''.join(qgo.CODONS[int(v)] if v >= 0 else 'NNN' for v in r['codons'])
            coding.append(dict(label=f"{r['symbol']} (chr{r['chrom']})", dna=dna,
                               truth=cls))
            break

    srecs = qso.load('held')
    nc, seen2 = [], set()
    for cls in WANT_SO:
        for r in srecs:
            if r['so'] != cls or r['symbol'] in seen2 or r['n'] > 1200:
                continue
            seen2.add(r['symbol'])
            nc.append(dict(label=f"{r['symbol']} (chr{r['chrom']})",
                           dna=''.join('ACGT'[int(x)] if x <= 3 else 'N' for x in r['arr']),
                           exons=int(r['exons']), truth=cls))
            break

    out = HERE / 'web' / 'quill-ont-examples.js'
    out.write_text('window.QUILL_ONT_EXAMPLES = ' +
                   json.dumps(dict(coding=coding, noncoding=nc),
                              separators=(',', ':')) + ';\n')
    print(f'{out.name}: {len(coding)} coding, {len(nc)} non-coding, '
          f'{out.stat().st_size/1024:.1f} KB')


if __name__ == '__main__':
    main()
