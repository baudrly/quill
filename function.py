"""Gene Ontology knowledge, written by hand.

What a gene does leaves marks on its sequence: the amino acid composition of its product,
a targeting peptide at the N terminus, a repeat period, a length, whether it has introns,
which kind of promoter it uses. This file writes those marks down per functional class.
qgo.py turns them into codon weight tables; nothing here reads a genome.

Compositions are percentages and need not sum to exactly 100 - they are normalised on
load. Classes are listed with the GO terms used to assign genes to them for evaluation,
in PRIORITY order: a gene is given the first class whose terms it carries, so the classes
come out disjoint. The order runs from the most sequence-distinctive to the least.
"""

AA = 'ACDEFGHIKLMNPQRSTVWY'
AA_IDX = {a: i for i, a in enumerate(AA)}

# Human proteome average, for reference and as the fallback composition.
PROTEOME_AVG = dict(A=7.0, C=2.3, D=4.7, E=7.1, F=3.6, G=6.6, H=2.6, I=4.3, K=5.7,
                    L=10.0, M=2.1, N=3.6, P=6.3, Q=4.8, R=5.6, S=8.3, T=5.4, V=6.0,
                    W=1.2, Y=2.7)

# --- N-terminal models ----------------------------------------------------------------------
# A signal peptide is a hydrophobic core of 7-15 residues inside the first ~25, with a
# small residue at the -1 cleavage position. A mitochondrial targeting peptide is
# amphipathic: Arg, Leu and Ser rich, and almost free of acidic residues.
NTERM = {
    'signal_peptide': dict(
        length=22,
        aa=dict(A=14.0, C=4.0, D=0.3, E=0.3, F=8.0, G=6.0, H=0.4, I=6.0, K=1.0,
                L=24.0, M=3.0, N=0.8, P=5.0, Q=1.5, R=2.0, S=8.0, T=4.0, V=10.0,
                W=2.0, Y=0.7)),
    'mito_targeting': dict(
        length=25,
        aa=dict(A=12.0, C=0.5, D=0.5, E=0.5, F=5.0, G=6.0, H=1.5, I=3.0, K=3.0,
                L=16.0, M=4.0, N=1.0, P=6.0, Q=4.0, R=12.0, S=12.0, T=6.0, V=5.0,
                W=1.0, Y=1.5)),
}

# --- Periodic profiles ----------------------------------------------------------------------
# A periodic class is written as a repeat length plus anchors: positions within the repeat
# where particular residues dominate. Unanchored positions take the class composition.
PERIOD_PROFILES = {
    # Collagen: the Gly-X-Y triplet, with proline heavily favoured at X and Y.
    'collagen_GXY': dict(period=3, anchors={
        0: dict(G=92.0, A=4.0, S=2.0),
        1: dict(P=34.0, E=8.0, A=9.0, G=6.0, K=5.0, S=5.0, Q=5.0, D=4.0, R=4.0, T=3.0),
        2: dict(P=30.0, A=10.0, G=6.0, K=7.0, R=6.0, E=6.0, Q=5.0, S=5.0, T=4.0, D=4.0),
    }),
    # Coiled coil: the heptad, hydrophobic at a and d.
    'coiled_coil_heptad': dict(period=7, anchors={
        0: dict(L=28.0, I=10.0, V=10.0, M=5.0, F=5.0, A=10.0, N=6.0),
        3: dict(L=30.0, I=9.0, V=9.0, M=5.0, F=5.0, A=11.0),
        1: dict(E=16.0, K=12.0, Q=9.0, S=9.0, R=8.0),
        4: dict(E=17.0, K=13.0, Q=8.0, R=8.0, D=7.0),
        5: dict(K=14.0, E=13.0, R=8.0, Q=7.0, S=8.0),
    }),
    # C2H2 zinc finger: a 28-residue repeat, two cysteines then two histidines, closing
    # with the TGEKP linker.
    'zinc_finger_C2H2': dict(period=28, anchors={
        2: dict(C=86.0), 5: dict(C=86.0),
        9: dict(F=40.0, Y=14.0, L=10.0), 12: dict(L=24.0, F=12.0, I=10.0),
        18: dict(H=88.0), 22: dict(H=86.0),
        23: dict(T=52.0, S=16.0), 24: dict(G=58.0, A=10.0),
        25: dict(E=54.0, D=10.0), 26: dict(K=56.0, R=14.0), 27: dict(P=58.0, S=10.0),
    }),
}

# --- GO slim classes, in assignment priority order ------------------------------------------
GO_CLASSES = {
    'ribosome_translation': dict(
        go=('0003735', '0005840', '0006412', '0022625', '0022627'),
        aa=dict(A=7.8, C=0.9, D=3.4, E=5.4, F=3.0, G=7.4, H=2.2, I=4.4, K=10.6, L=8.4,
                M=1.8, N=3.0, P=5.2, Q=3.6, R=8.6, S=6.6, T=5.0, V=6.6, W=0.6, Y=2.6),
        mean_aa=175.0, promoter='TCT_promoter', gc3=0.60, intronless=0.02,
        utr5='TOP_TRACT', tm_expect=0.1, label='ribosome and translation'),
    'secreted': dict(
        go=('0005576', '0005615', '0070062'),
        aa=dict(A=7.0, C=4.4, D=4.6, E=5.4, F=3.6, G=7.6, H=2.4, I=3.8, K=4.2, L=10.2,
                M=1.8, N=4.2, P=7.6, Q=4.2, R=5.2, S=8.6, T=5.8, V=6.2, W=1.4, Y=2.8),
        mean_aa=420.0, nterm='signal_peptide', promoter='TATA_promoter', gc3=0.56,
        intronless=0.03, tm_expect=1.0, label='secreted and extracellular'),
    'integral_membrane': dict(
        go=('0016021', '0005887'),
        aa=dict(A=8.0, C=2.2, D=3.2, E=4.2, F=5.2, G=7.0, H=2.2, I=6.0, K=4.2, L=14.5,
                M=2.6, N=3.2, P=5.2, Q=3.6, R=4.4, S=8.0, T=5.6, V=8.2, W=1.7, Y=3.3),
        mean_aa=480.0, promoter='CpG_island_promoter', gc3=0.58, intronless=0.05,
        hydrophobic_run=(19, 3.0), tm_expect=4.0, label='integral membrane'),
    'mitochondrion': dict(
        go=('0005739', '0005743', '0005759'),
        aa=dict(A=8.2, C=1.8, D=4.4, E=6.4, F=4.0, G=6.8, H=2.4, I=4.8, K=6.2, L=11.0,
                M=2.2, N=3.4, P=5.4, Q=4.2, R=6.2, S=7.6, T=5.4, V=6.4, W=1.0, Y=2.6),
        mean_aa=330.0, nterm='mito_targeting', promoter='CpG_island_promoter',
        gc3=0.55, intronless=0.04, tm_expect=0.6, label='mitochondrial'),
    'cytoskeleton': dict(
        go=('0005856', '0015629', '0005882', '0005874'),
        aa=dict(A=7.4, C=1.4, D=4.6, E=9.6, F=3.0, G=5.4, H=2.2, I=4.4, K=7.4, L=11.2,
                M=2.0, N=3.4, P=5.0, Q=6.0, R=6.2, S=8.0, T=5.2, V=5.6, W=0.8, Y=2.2),
        mean_aa=620.0, period='coiled_coil_heptad', period_weight=0.45,
        promoter='CpG_island_promoter', gc3=0.54, intronless=0.02,
        tm_expect=0.15, label='cytoskeletal'),
    'nucleus_dna_binding': dict(
        go=('0003677', '0006355', '0006351', '0000981', '0003700'),
        aa=dict(A=6.4, C=1.7, D=4.8, E=8.2, F=2.9, G=6.0, H=2.6, I=3.5, K=7.6, L=8.6,
                M=1.9, N=3.4, P=7.2, Q=5.4, R=6.6, S=9.6, T=5.2, V=5.0, W=1.0, Y=2.4),
        mean_aa=520.0, promoter='CpG_island_promoter', gc3=0.57, intronless=0.04,
        tm_expect=0.1, label='nuclear, DNA binding'),
    'rna_binding': dict(
        go=('0003723', '0000166', '0003676'),
        aa=dict(A=6.6, C=1.6, D=4.6, E=7.4, F=3.4, G=8.6, H=2.4, I=3.8, K=6.6, L=8.4,
                M=2.0, N=3.6, P=6.2, Q=4.6, R=7.0, S=9.4, T=5.0, V=5.4, W=0.9, Y=2.5),
        mean_aa=440.0, promoter='CpG_island_promoter', gc3=0.56, intronless=0.03,
        tm_expect=0.1, label='RNA binding'),
    'atp_binding': dict(
        go=('0005524', '0004672', '0016740', '0016787'),
        aa=dict(A=6.8, C=1.9, D=5.4, E=7.2, F=3.8, G=7.4, H=2.4, I=5.0, K=6.2, L=9.6,
                M=2.2, N=3.8, P=5.0, Q=4.2, R=5.4, S=8.0, T=5.6, V=6.6, W=1.1, Y=3.0),
        mean_aa=560.0, promoter='CpG_island_promoter', gc3=0.55, intronless=0.03,
        motif='GxxxxGK[ST]', tm_expect=0.35, label='ATP binding, catalytic'),
    'metal_ion_binding': dict(
        go=('0046872', '0008270', '0005509'),
        aa=dict(A=6.6, C=3.6, D=5.2, E=6.6, F=3.4, G=6.4, H=3.6, I=4.0, K=6.2, L=9.0,
                M=1.9, N=3.6, P=6.0, Q=4.6, R=5.8, S=8.6, T=5.4, V=5.6, W=1.0, Y=2.8),
        mean_aa=500.0, promoter='CpG_island_promoter', gc3=0.57, intronless=0.04,
        tm_expect=0.3, label='metal ion binding'),
    'cytosol_soluble': dict(
        go=('0005829', '0005737', '0005515'),
        aa=dict(A=7.0, C=2.0, D=4.9, E=7.4, F=3.4, G=6.4, H=2.5, I=4.2, K=6.0, L=9.6,
                M=2.1, N=3.6, P=6.0, Q=4.9, R=5.4, S=8.4, T=5.4, V=6.0, W=1.1, Y=2.7),
        mean_aa=450.0, promoter='CpG_island_promoter', gc3=0.55, intronless=0.04,
        tm_expect=0.2, label='soluble cytoplasmic'),
}

# --- Gene families with extreme written signatures ------------------------------------------
# These are the showcase cases: families whose sequence is so characteristic that a few
# written numbers should be enough to pick them out. Assigned by HGNC symbol pattern,
# which is a proxy for the family and is stated as such.
GENE_FAMILIES = {
    'collagen': dict(
        symbol=r'^COL\d+A\d+$',
        aa=dict(A=11.0, C=0.7, D=2.8, E=4.5, F=1.2, G=32.0, H=0.6, I=1.2, K=3.0, L=2.2,
                M=0.5, N=1.2, P=21.0, Q=3.0, R=4.0, S=3.2, T=1.8, V=1.8, W=0.2, Y=0.6),
        mean_aa=1300.0, period='collagen_GXY', period_weight=0.85,
        exon_multiple=54, tm_expect=0.05, label='collagen'),
    'keratin': dict(
        symbol=r'^KRT\d+$',
        aa=dict(A=4.6, C=4.8, D=3.0, E=8.0, F=2.6, G=13.0, H=1.2, I=3.0, K=3.2, L=8.6,
                M=1.0, N=3.0, P=3.0, Q=5.6, R=6.0, S=14.5, T=5.0, V=6.4, W=0.4, Y=2.4),
        mean_aa=520.0, period='coiled_coil_heptad', period_weight=0.55,
        tm_expect=0.1, label='keratin'),
    'zinc_finger': dict(
        symbol=r'^ZNF\d+$',
        aa=dict(A=4.6, C=5.6, D=4.0, E=7.0, F=4.2, G=5.2, H=4.2, I=4.0, K=7.2, L=6.8,
                M=2.0, N=3.4, P=4.6, Q=4.4, R=5.6, S=8.4, T=6.0, V=4.2, W=0.6, Y=3.0),
        mean_aa=520.0, period='zinc_finger_C2H2', period_weight=0.60,
        tm_expect=0.05, label='C2H2 zinc finger'),
    'olfactory_receptor': dict(
        symbol=r'^OR\d+[A-Z]\d+$',
        aa=dict(A=6.8, C=3.0, D=2.8, E=3.4, F=6.0, G=6.0, H=2.2, I=6.6, K=3.6, L=12.4,
                M=3.0, N=3.4, P=4.0, Q=3.2, R=4.0, S=8.2, T=6.0, V=8.0, W=1.4, Y=3.6),
        mean_aa=312.0, single_coding_exon=True, hydrophobic_run=(19, 3.5),
        tm_expect=7.0, label='olfactory receptor'),
    'histone': dict(
        symbol=r'^(HIST\d|H1-\d|H2A|H2B|H3|H4)',
        aa=dict(A=12.0, C=0.4, D=3.0, E=5.0, F=2.0, G=8.2, H=1.6, I=5.0, K=11.0, L=7.0,
                M=1.0, N=2.8, P=4.2, Q=3.4, R=9.0, S=5.0, T=6.2, V=6.2, W=0.2, Y=2.2),
        mean_aa=135.0, intronless=0.95, three_prime='HISTONE_STEM_LOOP',
        tm_expect=0.05, label='histone'),
    'ribosomal_protein': dict(
        symbol=r'^RP[LS]\d+[A-Z]?$',
        aa=dict(A=7.8, C=0.9, D=3.4, E=5.4, F=3.0, G=7.4, H=2.2, I=4.4, K=10.6, L=8.4,
                M=1.8, N=3.0, P=5.2, Q=3.6, R=8.6, S=6.6, T=5.0, V=6.6, W=0.6, Y=2.6),
        mean_aa=175.0, promoter='TCT_promoter', utr5='TOP_TRACT',
        tm_expect=0.1, label='ribosomal protein'),
    'immunoglobulin_V': dict(
        symbol=r'^IG[HKL]V',
        aa=dict(A=6.0, C=3.4, D=4.0, E=4.0, F=3.4, G=8.0, H=1.4, I=3.0, K=5.4, L=7.0,
                M=1.2, N=3.0, P=5.0, Q=4.6, R=4.4, S=10.0, T=7.0, V=8.0, W=2.6, Y=4.6),
        mean_aa=120.0, tm_expect=0.4, label='immunoglobulin V region'),
}

# Recombination signal sequence flanking immunoglobulin and TCR V, D and J segments.
RSS_HEPTAMER = 'CACAGTG'
RSS_NONAMER = 'ACAAAAACC'
RSS_SPACERS = (12, 23)

# How much of the total score each written component is allowed to contribute when a
# coding sequence is scored against a class. Written, not fitted.
SCORE_WEIGHTS = dict(composition=1.0, nterm=1.0, period=1.0, length=0.35,
                     tm=0.6, prior=1.0)
# Written share of human protein-coding genes per class, used as the Bayes prior when a
# sequence is assigned to one. Recalled proportions, not counted.
CLASS_PRIOR = dict(integral_membrane=0.23, secreted=0.16, nucleus_dna_binding=0.13,
                   cytosol_soluble=0.13, atp_binding=0.09, rna_binding=0.07,
                   mitochondrion=0.06, metal_ion_binding=0.06, cytoskeleton=0.05,
                   ribosome_translation=0.02)
FAMILY_PRIOR = dict(olfactory_receptor=0.40, zinc_finger=0.30, immunoglobulin_V=0.09,
                    histone=0.08, keratin=0.06, ribosomal_protein=0.05, collagen=0.02)

# Internal stop codons essentially do not occur inside a real coding sequence; this is
# the written mass left on each of the three, so that one does not cost infinity.
CDS_STOP_LEAK = 1e-4
# Spread of the written length prior, as a lognormal in amino acids.
LEN_LOGSIGMA = 0.85
# A window of this many residues with at least this many hydrophobics counts as one
# transmembrane-helix-like segment; used only to report what a sequence contains.
TM_WINDOW, TM_MIN_HYDROPHOBIC = 19, 16
HYDROPHOBIC = 'AILMFVWC'

# A membrane protein does not merely contain hydrophobic residues, it gathers them into
# helices of about 21 crossing the bilayer. That is a two-state chain over the protein,
# so the composition of a crossing segment and its length are written here and the
# per-class tm_expect becomes the entry rate.
TM_SEGMENT = dict(A=11.0, C=2.0, D=0.3, E=0.3, F=8.0, G=7.0, H=0.6, I=10.0, K=0.5,
                  L=25.0, M=3.0, N=1.0, P=2.0, Q=0.8, R=0.5, S=6.0, T=5.0, V=12.0,
                  W=2.0, Y=2.0)
TM_SEGMENT_LEN = 21.0


def composition(vec):
    """A written percentage dict as a normalised 20-vector in AA order."""
    out = [float(vec.get(a, 0.0)) for a in AA]
    t = sum(out)
    return [v / t for v in out]


def period_profile(name, base):
    """The per-position compositions of a periodic profile, anchors over a base vector."""
    spec = PERIOD_PROFILES[name]
    rows = []
    for i in range(spec['period']):
        anc = spec['anchors'].get(i)
        if not anc:
            rows.append(list(base))
            continue
        # anchored residues take their written share; the rest of the mass follows base
        row = [0.0] * len(AA)
        share = sum(anc.values()) / 100.0
        share = min(share, 0.97)
        for a, v in anc.items():
            row[AA_IDX[a]] = (v / 100.0) / max(sum(anc.values()) / 100.0, 1e-9) * share
        rest = 1.0 - share
        bt = sum(base[j] for j in range(len(AA)) if row[j] == 0.0)
        for j in range(len(AA)):
            if row[j] == 0.0:
                row[j] = rest * (base[j] / bt if bt > 0 else 1.0 / len(AA))
        rows.append(row)
    return rows


def class_names():
    return list(GO_CLASSES)


def family_names():
    return list(GENE_FAMILIES)
