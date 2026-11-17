"""Protein-level knowledge, written by hand.

Physicochemical scales, sorting-signal rules, transmembrane topology, short linear
motifs and modification sites, all recalled from the literature. qprot.py turns them
into a description of a protein and into features for the function experts. Nothing in
this file reads a sequence database.
"""

AA = 'ACDEFGHIKLMNPQRSTVWY'

# --- scales -----------------------------------------------------------------------------
KYTE_DOOLITTLE = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4,
                      H=-3.2, I=4.5, L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8,
                      T=-0.7, W=-0.9, Y=-1.3, V=4.2)
# Eisenberg consensus hydrophobicity, for the hydrophobic moment of amphipathic helices
EISENBERG = dict(A=0.62, R=-2.53, N=-0.78, D=-0.90, C=0.29, Q=-0.85, E=-0.74, G=0.48,
                 H=-0.40, I=1.38, L=1.06, K=-1.50, M=0.64, F=1.19, P=0.12, S=-0.18,
                 T=-0.05, W=0.81, Y=0.26, V=1.08)
# pKa values used for the isoelectric point (EMBOSS set)
PKA = dict(Nterm=8.6, Cterm=3.6, K=10.8, R=12.5, H=6.5, D=3.9, E=4.1, C=8.5, Y=10.1)
# average residue masses in a chain, Da
MASS = dict(A=71.0788, R=156.1875, N=114.1038, D=115.0886, C=103.1388, E=129.1155,
            Q=128.1307, G=57.0519, H=137.1411, I=113.1594, L=113.1594, K=128.1741,
            M=131.1926, F=147.1766, P=97.1167, S=87.0782, T=101.1051, W=186.2132,
            Y=163.1760, V=99.1326)
WATER = 18.015
# TOP-IDP disorder propensity: positive promotes disorder
TOP_IDP = dict(W=-0.884, F=-0.697, Y=-0.510, I=-0.486, M=-0.397, L=-0.326, V=-0.121,
               N=0.007, C=0.02, T=0.059, A=0.06, G=0.166, R=0.180, D=0.192, H=0.303,
               Q=0.318, K=0.586, S=0.341, E=0.736, P=0.987)
# Chou-Fasman conformational propensities
HELIX = dict(E=1.51, M=1.45, A=1.42, L=1.21, K=1.16, F=1.13, Q=1.11, W=1.08, I=1.08,
             V=1.06, D=1.01, H=1.00, R=0.98, T=0.83, S=0.77, C=0.70, Y=0.69, N=0.67,
             P=0.57, G=0.57)
SHEET = dict(V=1.70, I=1.60, Y=1.47, F=1.38, W=1.37, L=1.30, C=1.19, T=1.19, Q=1.10,
             M=1.05, R=0.93, N=0.89, H=0.87, A=0.83, S=0.75, G=0.75, K=0.74, P=0.55,
             D=0.54, E=0.37)

# --- segment rules ----------------------------------------------------------------------
TM_WINDOW, TM_THRESHOLD = 19, 1.6          # Kyte-Doolittle helix detector
TM_MIN_GAP = 4                             # helices closer than this are one segment
# von Heijne signal peptide: n-region, a hydrophobic h-region of 7-15, then a c-region
# whose -1 and -3 positions hold small residues (the AxA rule)
SP_SEARCH = 35
SP_H_WINDOW, SP_H_THRESHOLD = 8, 1.9
SP_SMALL = 'AGSCT'
SP_C_MIN, SP_C_MAX = 3, 8                  # cleavage this far after the h-region ends
SP_MAX_H = 17                              # a longer hydrophobic run is a signal anchor
# mitochondrial presequence: amphipathic, basic, almost no acidic residues
MITO_SEARCH = 35
MITO_MIN_BASIC, MITO_MAX_ACIDIC = 3, 1
MITO_MOMENT_ANGLE, MITO_MOMENT_WINDOW, MITO_MOMENT_MIN = 100.0, 18, 0.35
# GPI anchor: hydrophobic C-terminal tail after a small-residue omega site, plus a signal
GPI_TAIL = 30
GPI_H_WINDOW, GPI_H_THRESHOLD = 11, 1.6
GPI_OMEGA = 'SNAGDC'
# coiled coil: heptads with hydrophobic a and d, charged e and g
CC_WINDOW, CC_MIN_SCORE = 28, 0.72
CC_A_D = 'LIVMA'
CC_E_G = 'EKRQD'
# low complexity: Shannon entropy over a short window
LC_WINDOW, LC_MAX_BITS = 12, 2.2
# disorder: smoothed TOP-IDP over a window; long regions are what matter
DIS_WINDOW, DIS_THRESHOLD, DIS_LONG = 21, 0.10, 30

# --- short linear motifs and modification sites -----------------------------------------
# (name, regex, what it does). Regexes run on the one-letter protein sequence.
SLIMS = [
    ('NLS_monopartite', r'(?<![KR])K[KR].[KR]', 'importin-alpha nuclear import'),
    ('NLS_bipartite', r'[KR]{2}.{10,12}[KR]{3}', 'importin-alpha nuclear import'),
    ('NES_leucine_rich', r'[LIVFM].{2,3}[LIVFM].{2,3}[LIVFM].[LIVFM]', 'CRM1 nuclear export'),
    ('SH3_class_I', r'[RK]..P..P', 'SH3 domain binding'),
    ('SH3_class_II', r'P..P.[RK]', 'SH3 domain binding'),
    ('WW_PPxY', r'PP.Y', 'WW domain binding'),
    ('14-3-3', r'R..[ST].P', '14-3-3 binding when phosphorylated'),
    ('Rb_LxCxE', r'L.C.E', 'retinoblastoma protein binding'),
    ('PIP_box', r'Q..[ILM]..[FY][FY]', 'PCNA binding'),
    ('D_box', r'R..L..[LIVMF]', 'APC/C degron'),
    ('KEN_box', r'KEN', 'APC/C degron'),
    ('SUMO', r'[VILMFA]K.E', 'SUMOylation'),
    ('caspase_cleavage', r'[DE]..D[GSA]', 'caspase cleavage'),
    ('PDZ_class_I', r'[ST].[VIL]$', 'PDZ domain binding'),
    ('PDZ_class_II', r'[FYVIL].[FYVIL]$', 'PDZ domain binding'),
    ('YxxPhi_endocytic', r'Y..[LIMFV]', 'AP2 endocytic sorting'),
    ('dileucine', r'[DE]...L[LI]', 'endosomal-lysosomal sorting'),
    ('TRAF2_binding', r'[PSAT].[QE]E', 'TRAF2 binding'),
    ('PP1_RVxF', r'[KR].?[VI].[FW]', 'protein phosphatase 1 docking'),
    ('clathrin_box', r'L[LI].[LI][DE]', 'clathrin binding'),
    ('calmodulin_IQ', r'[FILV]Q...[RK]G...[RK]', 'calmodulin binding'),
    ('ER_export_DxE', r'D.E', 'COPII ER export'),
    ('KDEL', r'[KHR]DEL$', 'ER retention'),
    ('PTS1', r'[SAC][KRH][LM]$', 'peroxisomal import'),
    ('PTS2', r'^.{0,30}R[LVIQ]..[LVIHQ][LSGAQ].[HQ][LAF]', 'peroxisomal import'),
    ('CaaX', r'C[AVILMCST][AVILMCST][SMQALCFT]$', 'prenylation'),
    ('N_myristoylation', r'^MG[^EDRKHPFYW]..[STAGCN][^P]', 'N-myristoylation'),
    ('N_glycosylation', r'N[^P][ST]', 'N-glycosylation sequon'),
    ('phospho_PKA', r'[RK][RK].[ST][^P]', 'PKA phosphorylation'),
    ('phospho_CK2', r'[ST]..[DE]', 'CK2 phosphorylation'),
    ('phospho_CDK', r'[ST]P.[KR]', 'CDK phosphorylation'),
    ('phospho_MAPK', r'P.[ST]P', 'MAP kinase phosphorylation'),
    ('phospho_ATM', r'[ST]Q', 'ATM/ATR phosphorylation'),
    ('phospho_GSK3', r'[ST]...[ST]', 'GSK3 phosphorylation'),
    ('arginine_methyl_RGG', r'RGG', 'arginine methylation, RNA binding'),
    ('polyQ', r'Q{8,}', 'polyglutamine tract'),
    ('polyE_D', r'[DE]{8,}', 'acidic tract'),
    ('polyP', r'P{6,}', 'polyproline tract'),
    ('polyA_S', r'[AS]{8,}', 'low-complexity tract'),
]

# Which written predictor should mark which GO term, for the zero-shot test. Terms are
# by name and resolved against the ontology at run time.
EXPECTED_TERMS = {
    'signal_peptide': ['extracellular region'],
    'tm_any': ['membrane'],
    'tm_multi': ['plasma membrane'],
    'mito_presequence': ['mitochondrion'],
    'nls': ['nucleus'],
    'gpi_anchor': ['plasma membrane', 'cell surface'],
    'coiled_coil': ['cytoskeleton'],
    'high_pI': ['RNA binding', 'nucleus'],
    'disordered': ['nucleus'],
    'KDEL': ['endoplasmic reticulum lumen', 'endoplasmic reticulum'],
    'N_glycosylation_density': ['extracellular region'],
    'cysteine_rich': ['extracellular region'],
}
