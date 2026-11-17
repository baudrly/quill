"""RNA-level knowledge, written by hand.

What happens to a messenger RNA after it is made: which proteins bind where, which
microRNAs target it, what its untranslated regions say about translation and decay,
and what folds. Sequences are written in DNA letters (T for U) so they scan genomic
transcripts directly. Nothing here reads a database; qrna.py turns it into features.
"""

# --- RNA-binding protein sites (IUPAC, DNA letters) --------------------------------------
RBP_SITES = {
    'RBFOX': 'TGCATG',           # alternative-exon enhancer when downstream
    'PUM': 'TGTAHATA',           # Pumilio response element, decay/repression
    'QKI': 'ACTAAY',             # quaking response element
    'ELAVL1_HuR': 'TTTWTTT',     # U-rich stabilising element
    'ZFP36_TTP': 'TATTTAT',      # ARE nonamer core, decay
    'SRSF1': 'GGAGA',            # purine-rich exonic splicing enhancer
    'HNRNPA1': 'TAGGGW',         # exonic splicing silencer
    'PTBP1': 'TCTCTC',           # polypyrimidine, exon repression
    'CELF': 'TGTGTGT',           # GU-rich element
    'MBNL': 'YGCTTGC',           # YGCY repeats
    'NOVA': 'YCAYYCAY',          # YCAY clusters
    'CPEB': 'TTTTAT',            # cytoplasmic polyadenylation element
    'HNRNPC': 'TTTTTT',          # poly-U
    'LIN28': 'GGAG',             # pre-let-7 terminal loop
}

# --- microRNA seeds: nucleotides 2-8 of the mature strand (RNA written in DNA letters) ---
MIRNA_SEEDS = {
    'let-7': 'GAGGTAG', 'miR-1/206': 'GGAATGT', 'miR-21': 'AGCTTAT', 'miR-122': 'GGAGTGT',
    'miR-155': 'TAATGCT', 'miR-15/16': 'AGCAGCA', 'miR-17/20': 'AAAGTGC',
    'miR-19': 'GTGCAAA', 'miR-92': 'ATTGCAC', 'miR-200a': 'AACACTG',
    'miR-200bc': 'AATACTG', 'miR-34': 'GGCAGTG', 'miR-29': 'AGCACCA',
    'miR-124': 'AAGGCAC', 'miR-125': 'CCCTGAG', 'miR-126': 'CGTACCG',
    'miR-143': 'GAGATGA', 'miR-145': 'TCCAGTT', 'miR-133': 'TTGGTCC',
    'miR-9': 'CTTTGGT', 'miR-24': 'GGCTCAG', 'miR-26': 'TCAAGTA', 'miR-27': 'TCACAGT',
    'miR-30': 'GTAAACA', 'miR-181': 'ACATTCA', 'miR-223': 'GTCAGTT', 'miR-375': 'TTGTTCG',
    'miR-142-3p': 'GTAGTGT', 'miR-150': 'CTCCCAA', 'miR-196': 'AGGTAGT',
    'miR-10': 'ACCCTGT', 'miR-103/107': 'GCAGCAT', 'miR-128': 'CACAGTG',
    'miR-208': 'TAAGACG', 'miR-302': 'AAGTGCT', 'miR-451': 'AACCGTT',
}

# --- untranslated-region elements --------------------------------------------------------
ARE_PENTAMER = 'ATTTA'
ARE_NONAMER = 'TTATTTATT'
G_QUADRUPLEX = r'G{3,}.{1,7}G{3,}.{1,7}G{3,}.{1,7}G{3,}'
IRE_LOOP = 'CAGTG'               # apical loop of the iron-responsive element
TOP_MIN = 4                      # a C at +1 followed by at least this many pyrimidines
PRE = 'TGTAHATA'
POLYA_SIGNALS = ('AATAAA', 'ATTAAA')
HISTONE_SL = 'GGCTCTTTTCAGAGCC'  # 3'-end stem-loop of replication-dependent histone mRNAs
DRACH = r'[AGT][AG]AC[ACT]'      # m6A consensus
KOZAK_STRONG = r'[AG]..ATGG'
NMD_JUNCTION_DISTANCE = 55       # a stop this far upstream of the last junction triggers NMD
# exonic splicing enhancer and silencer hexamers, a small recalled sample
ESE_HEXAMERS = ('GAAGAA', 'GAAGAT', 'AAGAAG', 'GATGAA', 'GAAGAG', 'TGAAGA', 'CGAAGA')
ESS_HEXAMERS = ('TAGGTA', 'TTAGGG', 'GGTAGG', 'TAGTTA', 'GGGTAG', 'CTAGTA')

# --- folding --------------------------------------------------------------------------
FOLD_5UTR_MAX = 200              # the 5'UTR is folded up to this length from the cap
FOLD_SHUFFLES = 4                # shuffles behind a folding z-score

# Which written RNA predictor should mark which GO term, for the zero-shot test.
EXPECTED_TERMS = {
    'top_tract': ['cytosolic ribosome', 'structural constituent of ribosome'],
    'are_count': ['cytokine activity', 'inflammatory response'],
    'histone_stem_loop': ['nucleosome', 'structural constituent of chromatin'],
    'cai_written': ['structural constituent of ribosome', 'cytosolic ribosome'],
    'long_3utr': ['nervous system development', 'neuron projection'],
    'g4_5utr': ['DNA-binding transcription factor activity'],
}
