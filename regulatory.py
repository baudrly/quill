"""Transcription-factor binding sites, written by hand.

One consensus per factor, recalled from the literature, with IUPAC codes where the site
is degenerate. qnongenic.py checks each against the curated JASPAR 2024 matrix for the
same factor, which was not consulted when this list was written, and then uses the
sites as features for regulatory-element classification.
"""

TF_SITES = {
    'CTCF':     'CCACCAGGGGGCGC',
    'SP1':      'GGGGCGGGG',
    'NRF1':     'GCGCATGCGC',
    'YY1':      'AAGATGGCG',
    'CREB1':    'TGACGTCA',
    'FOS::JUN': 'TGAGTCA',
    'RELA':     'GGGAATTTCC',
    'TP53':     'RRRCATGYYYRRRCATGYYY',
    'MAX':      'CACGTG',
    'USF1':     'GTCACGTGAC',
    'ELK1':     'CCGGAAGT',
    'SPI1':     'AAAGAGGAAGTG',
    'GATA1':    'AGATAAG',
    'FOXA1':    'TGTTTACTTA',
    'HNF1A':    'GTTAATNATTAAC',
    'HNF4A':    'GGTCAAAGGTCA',
    'CEBPB':    'TTGCGCAA',
    'SOX2':     'CCATTGTT',
    'POU5F1':   'ATGCAAAT',
    'NFYA':     'RRCCAATSRG',
    'SRF':      'CCWWWWWWGG',
    'STAT3':    'TTCCNGGAA',
    'IRF1':     'GAAASTGAAAST',
    'REST':     'TTCAGCACCATGGACAGCGCC',
    'TEAD1':    'ACATTCCA',
    'RUNX1':    'TGTGGTTT',
    'E2F1':     'TTTSSCGC',
    'MEF2A':    'CTAWWWWTAG',
    'KLF4':     'GCCACACCC',
    'NR3C1':    'AGAACANNNTGTTCT',
    'ESR1':     'AGGTCANNNTGACCT',
    'EGR1':     'GCGTGGGCG',
    'TBP':      'TATAAAAG',
}

IUPAC = {'A': 'A', 'C': 'C', 'G': 'G', 'T': 'T', 'R': 'AG', 'Y': 'CT', 'S': 'CG',
         'W': 'AT', 'K': 'GT', 'M': 'AC', 'B': 'CGT', 'D': 'AGT', 'H': 'ACT',
         'V': 'ACG', 'N': 'ACGT'}


def matrix(site):
    """A written IUPAC site as a probability matrix, rows of (A, C, G, T)."""
    rows = []
    for c in site:
        allowed = IUPAC[c]
        rows.append(tuple((0.97 / len(allowed) if b in allowed else 0.01)
                          for b in 'ACGT'))
    return [tuple(v / sum(r) for v in r) for r in rows]
