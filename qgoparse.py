"""Pull GO term cross-references and protein translations out of the Ensembl EMBL files.

The per-chromosome pickles written by prep.py already carry gene symbols, strands and
exon/CDS structures. The only things missing for function-level modelling are the GO
terms attached to each gene and the protein sequence of each coding transcript, both of
which sit in the EMBL feature table. Writes /home/claude/data/go/<chrom>.pkl.
"""
import gzip
import pickle
import re
import sys
from pathlib import Path

SRC = Path('/home/claude/data/repos/Homo_sapiens.GRCh38.91.chromosome')
OUT = Path('/home/claude/data/go')
CHROMS = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']

GENE_RE = re.compile(r'/gene="?(ENSG[0-9.]+)"?')
GO_RE = re.compile(r'/db_xref="GO:(\d{7})"')
TX_RE = re.compile(r'/note="transcript_id=(ENST[0-9.]+)"')
PROT_START_RE = re.compile(r'/translation="([^"]*)"?')


def parse(chrom):
    """One pass over the file: GO terms per gene, translation per coding transcript."""
    path = SRC / f'Homo_sapiens.GRCh38.91.chromosome.{chrom}.dat.gz'
    gene_go, tx_prot = {}, {}
    cur_gene = cur_tx = None
    prot_parts, in_prot = [], False

    with gzip.open(path, 'rt') as fh:
        for line in fh:
            if not line.startswith('     '):
                if line.startswith('ORIGIN') or line.startswith('SQ '):
                    break
                continue
            body = line[21:].rstrip()

            if in_prot:
                # a translation runs over several continuation lines until its closing quote
                end = body.find('"')
                prot_parts.append(body if end < 0 else body[:end])
                if end >= 0:
                    in_prot = False
                    if cur_tx:
                        tx_prot[cur_tx] = (cur_gene, ''.join(prot_parts))
                    prot_parts = []
                continue

            key = line[5:21].strip()
            if key in ('gene', 'mRNA', 'CDS', 'misc_RNA', 'exon', 'source', 'STS',
                       'misc_feature'):
                if key in ('gene', 'CDS'):
                    cur_tx = None
                if key not in ('gene', 'CDS'):
                    cur_gene = cur_gene if key != 'source' else None

            m = GENE_RE.search(body)
            if m:
                cur_gene = m.group(1)
            m = TX_RE.search(body)
            if m:
                cur_tx = m.group(1)
            for g in GO_RE.findall(body):
                if cur_gene:
                    gene_go.setdefault(cur_gene, set()).add(g)
            if '/translation=' in body:
                seg = body.split('/translation=', 1)[1].lstrip('"')
                end = seg.find('"')
                if end >= 0:
                    if cur_tx:
                        tx_prot[cur_tx] = (cur_gene, seg[:end])
                else:
                    prot_parts = [seg]
                    in_prot = True

    return {'gene_go': {k: sorted(v) for k, v in gene_go.items()}, 'tx_prot': tx_prot}


def main(chroms):
    OUT.mkdir(parents=True, exist_ok=True)
    for c in chroms:
        p = OUT / f'{c}.pkl'
        if p.exists():
            print(f'chr{c}: already done', flush=True)
            continue
        d = parse(c)
        with open(p, 'wb') as fh:
            pickle.dump(d, fh, protocol=4)
        ngo = sum(len(v) for v in d['gene_go'].values())
        print(f'chr{c}: {len(d["gene_go"])} genes with GO, {ngo} xrefs, '
              f'{len(d["tx_prot"])} proteins', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:] or CHROMS)
