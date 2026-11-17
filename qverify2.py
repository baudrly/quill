"""Check every ontology-layer number quoted in the preprint against the result files.

Same contract as qverify.py: each entry pairs the figure as written in the preprint with
the figure recomputed from results/gobench.json and results/sobench.json. A mismatch is a
reporting error. Run both scripts before publishing anything.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import function as F
import ontology as O
import qgo
import qso

RES = HERE / 'results'
G = json.loads((RES / 'gobench.json').read_text())
S = json.loads((RES / 'sobench.json').read_text())
checks = []


def eq(name, claimed, actual, tol=5e-4):
    ok = abs(float(claimed) - float(actual)) <= tol
    checks.append((ok, name, claimed, round(float(actual), 6)))


def exact(name, claimed, actual):
    checks.append((claimed == actual, name, claimed, actual))


# ---- written knowledge: sizes of the vocabularies
exact('GO slim classes', 10, len(F.GO_CLASSES))
exact('gene families', 7, len(F.GENE_FAMILIES))
exact('SO RNA classes', 10, len(O.SO_RNA_CLASSES))
exact('written position-weight matrices in ontology.py', 21, len(O.pwm_names()))
exact('new emission slots named', 24, len(O.NEW_SLOTS))
exact('periodic profiles', 3, len(F.PERIOD_PROFILES))
exact('N-terminal models', 2, len(F.NTERM))
exact('collagen repeat period', 3, F.PERIOD_PROFILES['collagen_GXY']['period'])
exact('heptad period', 7, F.PERIOD_PROFILES['coiled_coil_heptad']['period'])
exact('zinc finger repeat period', 28, F.PERIOD_PROFILES['zinc_finger_C2H2']['period'])
exact('written transmembrane helix length', 21.0, F.TM_SEGMENT_LEN)

# every written matrix is a normalised probability matrix
bad = [(n, i) for n in O.pwm_names() for i, r in enumerate(O.matrix(n))
       if abs(sum(r) - 1.0) > 0.02]
checks.append((not bad, 'every written matrix row sums to 1', 'none', bad[:3]))
# and reads as the consensus the literature gives
cons = {n: ''.join('ACGT'[max(range(4), key=lambda j: r[j])] for r in O.matrix(n))
        for n in O.pwm_names()}
for n, want in (('BOX_C', 'ATGATGA'), ('BOX_D', 'CTGA'), ('BOX_ACA', 'ACA'),
                ('SM_SITE', 'GATTTTTGG'), ('TATA', 'TATAAAAG'),
                ('CCAAT', 'GGCCAATCT'), ('GC_BOX', 'GGGGCGGGGC'),
                ('U12_DONOR', 'ATATCCTT'), ('U12_BRANCH', 'TCCTTGAT'),
                ('HISTONE_STEM_LOOP', 'GGCTCTTTTCAGAGCC')):
    exact(f'{n} consensus', want, cons[n])

# ---- parameter counts of the three ontology models
for tag, npar in (('go', 34684), ('fam', 24298), ('so', 260)):
    mod = (qgo.GoModel if tag != 'so' else qso.SoModel)(
        HERE / 'models' / f'quill-k-{tag}.npz')
    exact(f'quill-k-{tag} parameters', npar, mod.nparam)

# ---- dataset sizes
exact('held-out genes with a functional label', 3578, G['classify']['go/quill-k']['n'])
exact('held-out genes in a family', 387, G['classify']['fam/quill-k']['n'])
exact('held-out non-coding transcripts', 2080, S['classify']['quill-k']['n'])
exact('held-out introns examined', 38692, S['minor_intron']['n_introns'])
exact('AT-AC introns found', 39, S['minor_intron']['n_atac'])
exact('transcription starts examined', 19018, S['cis_elements']['n_tss'])

# ---- functional classification
eq('family macro AUROC, written', 0.986, G['classify']['fam/quill-k']['macro_auroc'])
eq('family macro AUROC, counted', 0.945, G['classify']['fam/quill-s']['macro_auroc'])
eq('family accuracy, written', 0.848, G['classify']['fam/quill-k']['accuracy'])
eq('family accuracy, counted', 0.904, G['classify']['fam/quill-s']['accuracy'])
eq('family majority baseline', 0.669, G['classify']['fam/quill-k']['majority_baseline'])
eq('GO slim macro AUROC, written', 0.662, G['classify']['go/quill-k']['macro_auroc'])
eq('GO slim macro AUROC, counted', 0.680, G['classify']['go/quill-s']['macro_auroc'])
eq('GO slim majority baseline', 0.249, G['classify']['go/quill-k']['majority_baseline'])
fam_k = G['classify']['fam/quill-k']['per_class_auroc']
fam_s = G['classify']['fam/quill-s']['per_class_auroc']
for k, v in (('olfactory_receptor', 1.000), ('immunoglobulin_V', 0.998),
             ('histone', 0.996), ('collagen', 0.990), ('zinc_finger', 0.974),
             ('keratin', 0.974), ('ribosomal_protein', 0.971)):
    eq(f'family AUROC written {k}', v, fam_k[k], tol=1e-3)
for k, v in (('collagen', 0.863), ('keratin', 0.921), ('ribosomal_protein', 0.867)):
    eq(f'family AUROC counted {k}', v, fam_s[k], tol=1e-3)
checks.append((min(fam_k.values()) >= 0.97, 'every family AUROC written >= 0.97',
               '>=0.97', round(min(fam_k.values()), 4)))
rarer = ['collagen', 'keratin', 'ribosomal_protein', 'histone']
checks.append((all(fam_k[k] > fam_s[k] for k in rarer),
               'written beats counted on the four rarest families', 'all four',
               [k for k in rarer if fam_k[k] > fam_s[k]]))
go_k = G['classify']['go/quill-k']['per_class_auroc']
eq('integral membrane AUROC, written', 0.787, go_k['integral_membrane'], tol=1e-3)
eq('cytoskeleton AUROC, written', 0.561, go_k['cytoskeleton'], tol=1e-3)

# ---- conditioning on function as a coding model
eq('family conditioning gain, written', 0.0804, G['bits']['fam/quill-k']['gain'])
eq('family conditioning gain, counted', 0.0694, G['bits']['fam/quill-s']['gain'])
eq('GO slim conditioning gain, written', -0.0060, G['bits']['go/quill-k']['gain'])
eq('GO slim conditioning gain, counted', 0.0055, G['bits']['go/quill-s']['gain'])
eq('family conditioned bits/base, written', 1.8801,
   G['bits']['fam/quill-k']['bits_per_base_class_conditioned'])

# ---- generation reproduces the written signatures
gen = G['generate']
checks.append((max(v['composition_js_to_written'] for v in gen.values()) < 0.02,
               'every class generates within 0.02 JS of its written composition',
               '<0.02', round(max(v['composition_js_to_written']
                                  for v in gen.values()), 5)))
col = gen['fam/collagen']['glycine_by_triplet_phase']
eq('generated collagen glycine at phase 0', 0.826, col[0], tol=0.02)
checks.append((max(col[1:]) < 0.15, 'generated collagen glycine off-phase', '<0.15',
               round(max(col[1:]), 3)))
eq('generated collagen proline at phase 1', 0.334,
   gen['fam/collagen']['proline_by_triplet_phase'][1], tol=0.03)
znf = gen['fam/zinc_finger']
eq('generated zinc finger cysteine at anchored positions', 0.557,
   znf['cysteine_at_anchored_positions'], tol=0.03)
eq('generated zinc finger cysteine elsewhere', 0.052, znf['cysteine_elsewhere'],
   tol=0.02)
eq('generated zinc finger histidine at anchored positions', 0.530,
   znf['histidine_at_anchored_positions'], tol=0.03)
exact('generated zinc finger exact TGEKP linkers', 0, znf['TGEKP_linkers_found'])
ker = gen['fam/keratin']
eq('generated keratin hydrophobic at heptad a and d', 0.423,
   ker['hydrophobic_at_a_and_d'], tol=0.03)
eq('generated keratin hydrophobic elsewhere', 0.193, ker['hydrophobic_elsewhere'],
   tol=0.03)
eq('generated olfactory receptor membrane segments', 2.5,
   gen['fam/olfactory_receptor']['tm_segments_mean'], tol=0.6)
eq('generated integral membrane segments', 1.4,
   gen['go/integral_membrane']['tm_segments_mean'], tol=0.6)


# ---- sequence ontology
eq('SO macro AUROC, written', 0.876, S['classify']['quill-k']['macro_auroc'], tol=1e-3)
eq('SO macro AUROC, counted', 0.868, S['classify']['quill-s']['macro_auroc'], tol=1e-3)
eq('SO accuracy, written', 0.470, S['classify']['quill-k']['accuracy'], tol=1e-3)
eq('SO majority baseline', 0.401, S['classify']['quill-k']['majority_baseline'],
   tol=1e-3)
eq('SO macro AUROC without the length prior', 0.711,
   S['classify']['quill-k-without-length']['macro_auroc'], tol=1e-3)
eq('SO macro AUROC without the boxes', 0.868,
   S['classify']['quill-k-without-boxes']['macro_auroc'], tol=1e-3)
checks.append((S['classify']['quill-k-without-boxes']['accuracy'] >
               S['classify']['quill-k']['accuracy'],
               'dropping the boxes does not hurt SO accuracy', 'higher without',
               S['classify']['quill-k-without-boxes']['accuracy']))

# ---- the boxes are really there
bx = S['boxes']
eq('box C in C/D snoRNAs, bits', 7.45, bx['snoRNA_CD/BOX_C']['mean_in_members'],
   tol=5e-3)
eq('box C in a shuffle of the same bases', 0.89,
   bx['snoRNA_CD/BOX_C']['mean_in_shuffled'], tol=5e-3)
eq('box C in other RNA classes', 0.04,
   bx['snoRNA_CD/BOX_C']['mean_in_other_classes'], tol=5e-3)
eq('box C AUROC vs shuffled', 0.892, bx['snoRNA_CD/BOX_C']['auroc_vs_shuffled'],
   tol=1e-3)
eq('box ACA AUROC vs shuffled', 0.841, bx['snoRNA_HACA/BOX_ACA']['auroc_vs_shuffled'],
   tol=1e-3)
eq('Sm site AUROC vs shuffled', 0.720, bx['snRNA/SM_SITE']['auroc_vs_shuffled'],
   tol=1e-3)
checks.append((all(v['auroc_vs_shuffled'] > 0.70 for v in bx.values()),
               'every written box beats its own shuffle at AUROC > 0.70', '>0.70',
               round(min(v['auroc_vs_shuffled'] for v in bx.values()), 4)))

# ---- minor introns and promoter elements
mi = S['minor_intron']
eq('AT-AC share of held-out introns', 0.00101, mi['atac_fraction'], tol=1e-5)
eq('U12 AUROC, all three written matrices', 0.9996, mi['auroc_u12_vs_u2_score'],
   tol=1e-3)
eq('U12 AUROC, donor matrix alone', 0.9831, mi['auroc_donor_only'], tol=1e-3)
eq('U12 AUROC, branch-point matrix alone', 0.8221, mi['auroc_branch_only'], tol=1e-3)
ce = S['cis_elements']
for k, v in (('TATA', 0.4664), ('CCAAT', 0.4932), ('GC_BOX', 0.5689),
             ('INR', 0.5286), ('TCT', 0.5003)):
    eq(f'{k} AUROC vs composition-matched control', v,
       ce[k]['auroc_vs_shuffled_same_window'], tol=1e-3)
checks.append((max(ce[k]['auroc_vs_shuffled_same_window']
                   for k in ('TATA', 'CCAAT', 'GC_BOX', 'INR', 'TCT')) < 0.60,
               'no core-promoter element reaches 0.60 against its own shuffle', '<0.60',
               round(max(ce[k]['auroc_vs_shuffled_same_window']
                         for k in ('TATA', 'CCAAT', 'GC_BOX', 'INR', 'TCT')), 4)))

bad = [c for c in checks if not c[0]]
for ok, name, claimed, actual in checks:
    if not ok:
        print(f'FAIL  {name}: preprint says {claimed}, results say {actual}')
print(f'{len(checks) - len(bad)}/{len(checks)} ontology numbers in the preprint match '
      f'the result files')
sys.exit(1 if bad else 0)
