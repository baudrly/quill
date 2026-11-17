"""Check every quill-xl2 number quoted in the preprint against the result files."""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import proteome
import transcriptome

RES = HERE / 'results'
X = json.loads((RES / 'xl.json').read_text())
R = json.loads((RES / 'rna.json').read_text())['validate']
P = json.loads((RES / 'protein.json').read_text())
checks = []


def eq(name, claimed, actual, tol=6e-4):
    checks.append((abs(float(claimed) - float(actual)) <= tol, name, claimed,
                   round(float(actual), 6)))


def exact(name, claimed, actual):
    checks.append((claimed == actual, name, claimed, actual))


M = X['models']
exact('held-out genes', 3719, X['n_held'])
exact('remote genes', 2793, X['n_remote'])
exact('held-out genes with an EC number', 1006, X['n_ec_genes'])
exact('generic GO slim terms', 140, X['n_slim_terms'])
table = {
    'nano': (97269, 0.28, 2387, .412, .366, .539, .506, .628, .610, .429, .401, .483),
    'small': (307244, 1.1, 4175, .425, .375, .580, .542, .631, .610, .473, .428, .484),
    'medium': (13100443, 44.93, 4175, .476, .400, .627, .557, .639, .615, .557, .466, .514),
    'large': (15918893, 56.11, 6230, .478, .402, .613, .550, .640, .615, .566, .465, .518)}
keys = ('BP', 'BP_remote', 'MF', 'MF_remote', 'CC', 'CC_remote', 'EC', 'EC_remote', 'slim')
for s, v in table.items():
    exact(f'{s} parameters', v[0], M[s]['parameters'])
    eq(f'{s} MB', v[1], M[s]['mb'], 0.01)
    exact(f'{s} terms', v[2], M[s]['terms'])
    eq(f'{s} file on disk MB', v[1], os.path.getsize(HERE / f'models/quill-xl2-{s}.npz') / 1e6, 0.01)
    for k, c in zip(keys, v[3:]):
        eq(f'{s} {k}', c, M[s]['held_out'][k])
naive = (.364, .332, .410, .393, .585, .563, .267, .257, .380)
for s in table:
    for k, c in zip(keys, naive):
        eq(f'naive at {s} {k}', c, M[s]['naive'][k])
S = X['standalone']
for e, mf, cc, ec, bp in (('kmer', .567, .600, .459, .414), ('memory', .594, .614, .508, .458),
                          ('protein', .476, .622, .328, .387), ('prosite', .543, .610, .430, .394),
                          ('rna', .412, .591, .289, .361), ('signatures', .483, .589, .358, .364)):
    eq(f'{e} alone MF', mf, S[e]['MF'])
    eq(f'{e} alone CC', cc, S[e]['CC'])
    eq(f'{e} alone EC', ec, S[e]['EC'])
    eq(f'{e} alone BP', bp, S[e]['BP'])
eq('rna alone MF remote', .408, S['rna']['MF_remote'])
checks.append((S['protein']['CC'] == max(v['CC'] for v in S.values()),
               'protein features are the best single CC predictor', 'best',
               max(S, key=lambda e: S[e]['CC'])))
A = X['ablation']
for e, k, c in (('memory', 'BP', .413), ('memory', 'MF', .591), ('memory', 'MF_remote', .550),
                ('kmer', 'MF', .612), ('kmer', 'MF_remote', .533), ('prosite', 'MF', .605),
                ('prosite', 'EC', .545), ('protein', 'CC', .628), ('protein', 'CC_remote', .600),
                ('protein', 'slim', .497), ('protein', 'BP', .480), ('signatures', 'MF_remote', .553),
                ('rna', 'MF', .627), ('rna', 'BP', .476)):
    eq(f'medium without {e} {k}', c, A[e]['without'][k])
for ns in ('BP', 'MF', 'CC'):
    exact(f'medium RNA weight {ns}', 0.0, M['medium']['weights'][ns]['rna'])
exact('medium protein weight MF', 0.0, M['medium']['weights']['MF']['protein'])

# written protein knowledge, zero-shot
for k, c in (('signal_peptide -> extracellular region', .6681), ('tm_any -> membrane', .7217),
             ('tm_multi -> plasma membrane', .6926), ('mito_presequence -> mitochondrion', .6527),
             ('nls -> nucleus', .6133), ('disordered -> nucleus', .6856),
             ('coiled_coil -> cytoskeleton', .6063), ('gpi_anchor -> plasma membrane', .5116),
             ('cysteine_rich -> extracellular region', .5070)):
    eq(f'zero-shot {k}', c, P[k]['auroc'])
eq('signal peptide precision when it fires', .556, P['signal_peptide -> extracellular region']['precision_when_fires'])
eq('extracellular base rate', .236, P['signal_peptide -> extracellular region']['base_rate'])
eq('signal peptide prevalence', .190, P['prevalence']['signal_peptide'])
exact('KDEL fires', 6, P['KDEL -> endoplasmic reticulum lumen']['fires_on'])
eq('KDEL precision', 1.0, P['KDEL -> endoplasmic reticulum lumen']['precision_when_fires'])
checks.append((all(v['auroc'] > 0.5 for k, v in P.items() if isinstance(v, dict) and 'auroc' in v),
               'every written protein predictor scores above 0.5', '>0.5',
               min(v['auroc'] for k, v in P.items() if isinstance(v, dict) and 'auroc' in v)))
exact('written short linear motifs', 39, len(proteome.SLIMS))

# written RNA knowledge, zero-shot
for k, c in (('are_count -> cytokine activity', .6671), ('cai_written -> cytosolic ribosome', .6452),
             ('top_tract -> cytosolic ribosome', .5264), ('long_3utr -> nervous system development', .5888),
             ('g4_5utr -> DNA-binding transcription factor activity', .5098)):
    eq(f'zero-shot {k}', c, R[k]['auroc'])
exact('histone stem-loop positives on held-out chromosomes', 3, R['histone_stem_loop -> nucleosome']['positives'])
eq('folding AUROC structured vs loose', .7156, R['folding']['auroc_structured_vs_loose'])
exact('transcripts folded', 2080, R['folding']['n'])
eq('pre-miRNA mean fold z', -6.06, R['folding']['per_class']['pre_miRNA']['mean_fold_z'], 0.005)
eq('lincRNA mean fold z', -0.94, R['folding']['per_class']['lincRNA']['mean_fold_z'], 0.005)
eq('snoRNA H/ACA mean fold z', -2.69, R['folding']['per_class']['snoRNA_HACA']['mean_fold_z'], 0.005)
exact('mRNAs described', 19016, R['n_mrnas'])
exact('written RBP sites', 14, len(transcriptome.RBP_SITES))
exact('written miRNA seeds', 36, len(transcriptome.MIRNA_SEEDS))

eq('7SL mean fold z', -2.49, R['folding']['per_class']['srpRNA_7SL']['mean_fold_z'], 0.005)
eq('processed pseudogene mean fold z (stored to 3 dp)', -0.95, R['folding']['per_class']['processed_pseudogene']['mean_fold_z'], 0.0051)
eq('BP lost without the gene memory', 0.063, M['medium']['held_out']['BP'] - A['memory']['without']['BP'])
eq('quill-xl MF it improves on', 0.615, json.loads((RES / 'goxbench.json').read_text())['results']['quill-xl+rules']['MF']['fmax'], 6e-4)
exact('fitted per-namespace weights in medium', 18, sum(len(v) for v in M['medium']['weights'].values()))
import proteome as _P
exact('coiled-coil threshold', 0.72, _P.CC_MIN_SCORE)

bad = [c for c in checks if not c[0]]
for ok, name, claimed, actual in checks:
    if not ok:
        print(f'FAIL  {name}: preprint says {claimed}, results say {actual}')
print(f'{len(checks) - len(bad)}/{len(checks)} quill-xl2 numbers in the preprint match the '
      f'result files')
sys.exit(1 if bad else 0)
