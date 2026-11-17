"""Sequence Ontology knowledge, written by hand.

Everything here is recalled molecular biology, not measured: consensus matrices for cis
elements, the defining boxes of the small-RNA classes, length and exon-count priors per
SO term. qso.py turns it into weight tables; nothing in this file reads a genome.

Matrices are rows of (A, C, G, T) probabilities, 5'->3' on the feature's own strand.
SO accessions are given so the vocabulary is anchored to the ontology rather than to
names invented here.
"""

B = 'ACGT'

# --- 1. Core promoter elements --------------------------------------------------------------
# TATA box, SO:0000174. TATAWAWR, centred about 31 bp upstream of the TSS in the
# minority of mammalian promoters that have one at all.
TATA = [(.03, .02, .02, .93), (.94, .02, .02, .02), (.03, .02, .02, .93),
        (.92, .03, .02, .03), (.55, .05, .05, .35), (.90, .03, .03, .04),
        (.50, .08, .08, .34), (.38, .10, .40, .12)]
TATA_OFFSET = (-34, -28)        # window the box is looked for in, relative to the TSS

# CCAAT box, SO:0000172, usually -60 to -100 and often on the reverse strand.
CCAAT = [(.10, .15, .60, .15), (.08, .10, .72, .10), (.10, .72, .10, .08),
         (.10, .72, .10, .08), (.76, .08, .08, .08), (.80, .06, .06, .08),
         (.06, .06, .06, .82), (.12, .60, .14, .14), (.14, .18, .14, .54)]

# GC box, SO:0000173, the SP1 site; the commonest element in CpG-island promoters.
GC_BOX = [(.05, .05, .85, .05), (.05, .05, .85, .05), (.05, .05, .85, .05),
          (.05, .10, .75, .10), (.05, .80, .10, .05), (.05, .05, .85, .05),
          (.05, .05, .85, .05), (.05, .05, .85, .05), (.10, .10, .70, .10),
          (.10, .55, .20, .15)]

# Initiator, SO:0000015. YYANWYY with the A at +1.
INR = [(.15, .35, .10, .40), (.15, .35, .10, .40), (.75, .08, .09, .08),
       (.25, .25, .25, .25), (.40, .10, .10, .40), (.15, .35, .10, .40),
       (.15, .35, .10, .40)]

# TCT element, SO-adjacent: the polypyrimidine initiator of ribosomal protein genes,
# which replaces the Inr A at +1 with a C and starts the 5'TOP tract.
TCT = [(.10, .40, .05, .45), (.05, .45, .05, .45), (.05, .50, .05, .40),
       (.05, .45, .05, .45), (.05, .45, .05, .45), (.08, .42, .08, .42)]

PROMOTER_CLASSES = {
    # fraction of mammalian promoters, and which elements each class carries
    'CpG_island_promoter': dict(frac=0.62, elements=('GC_BOX', 'INR'), cpg_oe=0.75,
                                gc=0.62, so='SO:0000170'),
    'TATA_promoter': dict(frac=0.24, elements=('TATA', 'CCAAT', 'INR'), cpg_oe=0.22,
                          gc=0.42, so='SO:0000170'),
    'TCT_promoter': dict(frac=0.03, elements=('TCT', 'GC_BOX'), cpg_oe=0.70,
                         gc=0.58, so='SO:0000170'),
    'weak_promoter': dict(frac=0.11, elements=('INR',), cpg_oe=0.25, gc=0.44,
                          so='SO:0000170'),
}

# --- 2. Splicing elements -------------------------------------------------------------------
# Major (U2) spliceosome. The donor and acceptor consensus matrices live in knowledge.py;
# what is added here is the branch point and the polypyrimidine tract, which the v1
# grammar had no states for.

# Branch point, SO:0000611. yUnAy, the adenosine being the nucleophile. 18-40 nt from
# the 3' splice site.
BRANCH_POINT = [(.14, .36, .12, .38), (.08, .14, .06, .72), (.26, .24, .22, .28),
                (.78, .06, .10, .06), (.16, .34, .14, .36)]
BRANCH_POINT_A = 3                       # index of the branch adenosine in the matrix
BRANCH_POINT_DIST = (18, 40)             # nt upstream of the acceptor

# Polypyrimidine tract, SO:0000713: 10-20 nt of C and U between branch point and 3' SS.
PPT = (.07, .34, .05, .54)
PPT_LEN = (10, 20)

# Minor (U12) spliceosome, SO:0000588. Two subtypes; the AT-AC subtype is the
# recognisable one. The U12 branch point is far more constrained than the U2 one and
# there is no polypyrimidine tract.
U12_DONOR = [(.95, .02, .02, .01), (.02, .02, .02, .94), (.88, .04, .04, .04),
             (.04, .04, .04, .88), (.04, .90, .03, .03), (.04, .88, .04, .04),
             (.10, .10, .10, .70), (.10, .14, .08, .68)]
U12_ACCEPTOR = [(.20, .30, .20, .30), (.80, .06, .08, .06), (.10, .70, .10, .10)]
U12_BRANCH = [(.04, .04, .04, .88), (.04, .86, .05, .05), (.04, .86, .05, .05),
              (.04, .04, .04, .88), (.04, .04, .04, .88), (.40, .06, .48, .06),
              (.90, .03, .04, .03), (.14, .34, .12, .40)]
U12_FRACTION = 0.0035                    # share of mammalian introns spliced by U12

# Exon classes, SO:0000200 / 0000004 / 0000202 / 0005845
EXON_CLASSES = {
    'initial_exon': dict(so='SO:0000200', mean_len=170.0, frac=0.17),
    'internal_exon': dict(so='SO:0000004', mean_len=145.0, frac=0.62),
    'terminal_exon': dict(so='SO:0000202', mean_len=1100.0, frac=0.17),
    'single_exon': dict(so='SO:0005845', mean_len=1600.0, frac=0.04),
}

# --- 3. Translation-level elements ----------------------------------------------------------
# Kozak context, SO:0001647, gccRccAUGG; the -3 purine and the +4 G carry most of it.
KOZAK = [(.22, .38, .18, .22), (.18, .42, .18, .22), (.22, .36, .20, .22),
         (.40, .12, .36, .12), (.18, .44, .16, .22), (.22, .38, .18, .22),
         (.97, .01, .01, .01), (.01, .01, .01, .97), (.01, .01, .97, .01),
         (.18, .14, .50, .18), (.22, .22, .30, .26), (.24, .24, .26, .26)]
KOZAK_ATG = 6                            # index of the A of the AUG

# Upstream ORF, SO:0000204 in the 5' UTR: an AUG with a stop before the main start.
UORF_RATE = 0.035                        # per 5'-UTR base, chance of starting one
UORF_MEAN_LEN = 30.0                     # codons

# Selenocysteine recoding, SO:0000383: in-frame UGA plus a 3'-UTR SECIS stem-loop.
SECIS_CORE = [(.80, .06, .08, .06), (.06, .06, .06, .82), (.06, .06, .82, .06),
              (.84, .05, .06, .05), (.20, .20, .40, .20), (.18, .14, .50, .18)]

# --- 4. 3'-end elements ---------------------------------------------------------------------
# Poly(A) signal, SO:0000551, and the hexamers that stand in for AAUAAA.
POLYA_HEXAMERS = {'AATAAA': .585, 'ATTAAA': .165, 'TATAAA': .040, 'AGTAAA': .034,
                  'AATACA': .028, 'CATAAA': .022, 'AATATA': .021, 'GATAAA': .019,
                  'AATGAA': .017, 'AAGAAA': .016, 'ACTAAA': .014, 'AATAGA': .013,
                  'ATTACA': .008, 'AACAAA': .007, 'ATTATA': .006, 'AACAAG': .005,
                  'AATAAT': .004}
POLYA_SITE_DIST = (10, 30)               # cleavage site, nt downstream of the hexamer

# Downstream sequence element, SO:0000205 region: GU-rich/U-rich after the cleavage site.
DSE = (.12, .10, .30, .48)
DSE_LEN = (20, 40)
# Upstream sequence element: UGUA repeats bound by CFIm.
USE = [(.06, .06, .06, .82), (.06, .06, .82, .06), (.06, .06, .06, .82),
       (.82, .06, .06, .06)]

# Histone mRNA 3' stem-loop, SO:0000313-like: replaces the poly(A) signal in replication
# dependent histone genes. 6 bp stem, 4 nt loop, then the purine-rich HDE.
HISTONE_STEM_LOOP = [(.05, .05, .85, .05), (.05, .05, .85, .05), (.05, .80, .10, .05),
                     (.05, .05, .10, .80), (.05, .80, .10, .05), (.05, .05, .10, .80),
                     (.05, .05, .10, .80), (.05, .05, .10, .80), (.05, .05, .10, .80),
                     (.05, .80, .10, .05), (.80, .05, .10, .05), (.05, .05, .80, .10),
                     (.80, .05, .10, .05), (.05, .05, .80, .10), (.05, .80, .10, .05),
                     (.05, .80, .10, .05)]
HISTONE_HDE = (.46, .14, .28, .12)

# AU-rich element, SO:0000175-adjacent: AUUUA in a U-rich context, 3'-UTR instability.
ARE_CORE = 'ATTTA'
ARE_NONAMER = 'TTATTTATT'
# m6A consensus DRACH, commonest as GGACU; dense in long 3' UTRs and near stop codons.
M6A = [(.30, .10, .40, .20), (.45, .05, .45, .05), (.95, .02, .02, .01),
       (.02, .94, .02, .02), (.32, .30, .06, .32)]
# 5' terminal oligopyrimidine tract, the translational switch of ribosomal protein mRNAs.
TOP_TRACT = (.06, .40, .06, .48)
TOP_LEN = (5, 15)

# --- 5. Small-RNA classes, with the boxes that define them ----------------------------------
# Box C/D snoRNA, SO:0000593. Box C near the 5' end, box D within a few nt of the 3' end,
# degenerate C'/D' internally, and a short terminal stem bringing the two together.
BOX_C = [(.45, .05, .45, .05), (.05, .05, .05, .85), (.05, .05, .85, .05),
         (.88, .04, .04, .04), (.05, .05, .05, .85), (.05, .05, .85, .05),
         (.88, .04, .04, .04)]
BOX_D = [(.04, .88, .04, .04), (.04, .04, .04, .88), (.04, .04, .88, .04),
         (.88, .04, .04, .04)]
# Box H/ACA snoRNA, SO:0000594. ANANNA hinge, ACA exactly 3 nt from the 3' end.
BOX_H = [(.88, .04, .04, .04), (.25, .25, .25, .25), (.88, .04, .04, .04),
         (.25, .25, .25, .25), (.25, .25, .25, .25), (.88, .04, .04, .04)]
BOX_ACA = [(.90, .04, .03, .03), (.04, .88, .04, .04), (.90, .04, .03, .03)]
BOX_ACA_FROM_END = 3
# Sm site of the spliceosomal snRNAs, SO:0000275 class: RAU(4-6)GR.
SM_SITE = [(.40, .08, .44, .08), (.86, .05, .05, .04), (.04, .05, .05, .86),
           (.05, .06, .05, .84), (.05, .06, .05, .84), (.06, .07, .06, .81),
           (.10, .10, .10, .70), (.06, .06, .82, .06), (.38, .10, .42, .10)]
# U1 snRNA 5' end, which pairs with the donor consensus; diagnostic of U1 genes.
U1_FIVE_PRIME = 'ATACTTACCTGG'
# tRNA A and B boxes, SO:0000253; the B box is the Pol III TFIIIC anchor.
TRNA_A_BOX = [(.08, .06, .06, .80), (.06, .06, .82, .06), (.06, .06, .82, .06),
              (.06, .82, .06, .06), (.25, .25, .25, .25), (.25, .25, .25, .25),
              (.80, .06, .08, .06), (.06, .06, .82, .06), (.08, .06, .06, .80),
              (.06, .06, .82, .06), (.06, .06, .82, .06)]
TRNA_B_BOX = [(.05, .05, .85, .05), (.05, .05, .05, .85), (.05, .05, .05, .85),
              (.05, .85, .05, .05), (.05, .05, .85, .05), (.85, .05, .05, .05),
              (.85, .05, .05, .05), (.25, .25, .25, .25), (.25, .25, .25, .25),
              (.10, .70, .10, .10)]

SO_RNA_CLASSES = {
    'snoRNA_CD': dict(so='SO:0000593', len=(65, 100), mean_len=80.0, exons=1,
                      boxes=(('BOX_C', 'start', (2, 20)), ('BOX_D', 'end', (0, 8))),
                      gc=0.42, symbol=r'^SNORD\d'),
    'snoRNA_HACA': dict(so='SO:0000594', len=(110, 165), mean_len=133.0, exons=1,
                        boxes=(('BOX_H', 'middle', (50, 85)),
                               ('BOX_ACA', 'end', (2, 5))),
                        gc=0.44, symbol=r'^SNORA\d'),
    'snRNA': dict(so='SO:0000274', len=(90, 190), mean_len=120.0, exons=1,
                  boxes=(('SM_SITE', 'end', (8, 60)),), gc=0.49, symbol=r'^RNU\d'),
    'pre_miRNA': dict(so='SO:0001244', len=(60, 110), mean_len=82.0, exons=1,
                      hairpin=(22, 40, 2.0), gc=0.48, symbol=r'^MIR\d'),
    'tRNA': dict(so='SO:0000253', len=(70, 95), mean_len=75.0, exons=1,
                 boxes=(('TRNA_A_BOX', 'start', (6, 14)),
                        ('TRNA_B_BOX', 'end', (10, 25))),
                 gc=0.56, symbol=r'^TR[A-Z]-'),
    'Y_RNA': dict(so='SO:0000405', len=(80, 115), mean_len=100.0, exons=1,
                  hairpin=(14, 30, 1.2), gc=0.48, symbol=r'^RNY\d|^VTRNA'),
    'srpRNA_7SL': dict(so='SO:0000590', len=(260, 320), mean_len=295.0, exons=1,
                       gc=0.57, symbol=r'^RN7S'),
    'lincRNA': dict(so='SO:0001463', len=(300, 4000), mean_len=750.0, exons=3,
                    gc=0.44, symbol=r'^LINC\d'),
    'antisense_lncRNA': dict(so='SO:0001904', len=(300, 4000), mean_len=755.0, exons=3,
                             gc=0.47, symbol=r'-AS\d$'),
    'processed_pseudogene': dict(so='SO:0000043', len=(300, 3000), mean_len=715.0,
                                 exons=1, gc=0.45, coding_like=True, symbol=r'P\d+$'),
}

# Pyknons of the two transposon families that dominate mammalian intergenic DNA. Only the
# diagnostic head of each is written; neither is used generatively.
ALU_HEAD = 'GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCA'
L1_ORF2_HEAD = 'ATGACAAAAACCACAAAGACAAAAACCACAAAGAC'

# --- 6. What the grammar gains ---------------------------------------------------------------
# New emission slots the v2 grammar carries over v1, in the order qstates2 lays them out.
NEW_SLOTS = ('TATA', 'CCAAT', 'GC_BOX', 'INR', 'TCT', 'BRANCH_POINT', 'PPT',
             'U12_DONOR', 'U12_BRANCH', 'U12_ACCEPTOR', 'KOZAK', 'UORF',
             'HISTONE_STEM_LOOP', 'DSE', 'USE', 'TOP_TRACT', 'M6A',
             'BOX_C', 'BOX_D', 'BOX_H', 'BOX_ACA', 'SM_SITE',
             'TRNA_A_BOX', 'TRNA_B_BOX')


def matrix(name):
    """A written matrix by name, as a list of (A, C, G, T) rows."""
    v = globals()[name]
    if isinstance(v, tuple) and len(v) == 4 and not isinstance(v[0], tuple):
        return [v]
    return [tuple(r) for r in v]


def pwm_names():
    """Every written position-weight matrix in this file, longest first."""
    out = []
    for k, v in sorted(globals().items()):
        if k.startswith('_') or not k.isupper():
            continue
        if isinstance(v, list) and v and isinstance(v[0], tuple) and len(v[0]) == 4:
            out.append(k)
    return out
