"""Check every number quoted in the preprint against results/bench.json and the weight files.

Each entry is (claim as written in the preprint, value recomputed from the result files).
Any mismatch is a reporting error, not a modelling one, and is printed as FAIL.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RES = HERE / 'results'
B = json.loads((RES / 'bench.json').read_text())
S = json.loads((RES / 'scaling.json').read_text())
LM, GEN, BR, AN, GB = B['lm'], B['gen'], B['brca1'], B['anno'], B['gb']

checks = []


def eq(name, claimed, actual, tol=5e-4):
    ok = abs(float(claimed) - float(actual)) <= tol
    checks.append((ok, name, claimed, round(float(actual), 6)))


def exact(name, claimed, actual):
    ok = claimed == actual
    checks.append((ok, name, claimed, actual))


# ---- sizes and parameter counts
for nm, npar, kb in (('quill-k', 10252, 5.2), ('quill-s', 1406988, 960), ('quill-l', 67467276, 49920)):
    z = np.load(HERE / 'models' / f'{nm}.npz', allow_pickle=True)
    exact(f'{nm} parameters', npar, json.loads(str(z['nparam']))['total'])
    eq(f'{nm} file KB', kb, (HERE / 'models' / f'{nm}.npz').stat().st_size / 1024, tol=kb * 0.03)

# ---- language modelling table
lm_claims = {('quill-k', 'human_heldout_chr19_22'): 1.9252,
             ('quill-k', 'human_chr17_heldout'): 1.9266,
             ('quill-k', 'mouse_chr19_heldout_species'): 1.9223,
             ('quill-s', 'human_heldout_chr19_22'): 1.7560,
             ('quill-s', 'human_chr17_heldout'): 1.7339,
             ('quill-s', 'mouse_chr19_heldout_species'): 1.9043,
             ('quill-l', 'human_heldout_chr19_22'): 1.6675,
             ('quill-l', 'human_chr17_heldout'): 1.6429,
             ('quill-l', 'mouse_chr19_heldout_species'): 1.8980}
for (m, src), v in lm_claims.items():
    eq(f'bits/base {m} {src}', v, LM[src][m]['bits_per_base'])

h = {m: LM['human_heldout_chr19_22'][m]['bits_per_base'] for m in ('quill-k', 'quill-s', 'quill-l')}
mo = {m: LM['mouse_chr19_heldout_species'][m]['bits_per_base'] for m in h}
eq('human->mouse cost quill-k = -0.003', -0.003, mo['quill-k'] - h['quill-k'], tol=1e-3)
eq('human->mouse cost quill-s = +0.148', 0.148, mo['quill-s'] - h['quill-s'], tol=1e-3)
eq('human->mouse cost quill-l = +0.231', 0.231, mo['quill-l'] - h['quill-l'], tol=1e-3)
eq('counting buys 0.17 at 1.4M', 0.17, h['quill-k'] - h['quill-s'], tol=1e-3)
eq('counting buys 0.26 at 67.5M (2 dp)', 0.26, h['quill-k'] - h['quill-l'], tol=5e-3)
eq('mouse gap k vs l = 0.024', 0.024, mo['quill-k'] - mo['quill-l'], tol=1e-3)
eq('human gap k vs l = 0.258', 0.258, h['quill-k'] - h['quill-l'], tol=1e-3)

# ---- scaling curve
eq('order 2->12 gain human chr20 = 0.236', 0.236, S['human_chr20'][0] - S['human_chr20'][-1], tol=1e-3)
eq('order 2->12 gain mouse chr19 = 0.004', 0.004, S['mouse_chr19'][0] - S['mouse_chr19'][-1], tol=1e-3)
eq('order-2 Markov human chr20 = 1.932', 1.932, S['human_chr20'][0], tol=1e-3)

# ---- BRCA1
br_claims = {
    ('quill-k_splice_matrix', 'all'): 0.548, ('quill-k_splice_matrix', 'coding'): 0.461,
    ('quill-k_splice_matrix', 'noncoding'): 0.790,
    ('quill-k_splice_matrix', 'splice_and_intronic'): 0.783,
    ('quill-k_splice_matrix', 'auroc_Missense'): 0.445,
    ('quill-k_splice_matrix', 'locate_canonical_splice'): 0.836,
    ('quill-s_splice_matrix', 'all'): 0.580, ('quill-s_splice_matrix', 'coding'): 0.479,
    ('quill-s_splice_matrix', 'noncoding'): 0.873,
    ('quill-s_splice_matrix', 'splice_and_intronic'): 0.867,
    ('quill-s_splice_matrix', 'auroc_Missense'): 0.471,
    ('quill-s_splice_matrix', 'locate_canonical_splice'): 0.926,
    ('quill-s_window_delta', 'all'): 0.490, ('quill-s_window_delta', 'coding'): 0.516,
    ('quill-s_window_delta', 'noncoding'): 0.426,
    ('quill-s_window_delta', 'splice_and_intronic'): 0.423,
    ('quill-s_window_delta', 'auroc_Missense'): 0.514,
    ('quill-l_window_delta', 'all'): 0.526, ('quill-l_window_delta', 'coding'): 0.527,
    ('quill-l_window_delta', 'noncoding'): 0.523,
    ('quill-l_window_delta', 'splice_and_intronic'): 0.531,
    ('quill-l_window_delta', 'auroc_Missense'): 0.499,
    ('quill-l_smoothed_lr', 'all'): 0.513, ('quill-l_smoothed_lr', 'coding'): 0.484,
    ('quill-l_smoothed_lr', 'noncoding'): 0.587,
    ('quill-l_smoothed_lr', 'splice_and_intronic'): 0.589,
    ('quill-l_smoothed_lr', 'auroc_Missense'): 0.475,
    ('phyloP_mammalian', 'all'): 0.781, ('phyloP_mammalian', 'coding'): 0.736,
    ('phyloP_mammalian', 'noncoding'): 0.887,
    ('phyloP_mammalian', 'splice_and_intronic'): 0.886,
    ('phyloP_mammalian', 'auroc_Missense'): 0.724,
    ('CADD', 'all'): 0.804, ('CADD', 'coding'): 0.841, ('CADD', 'noncoding'): 0.875,
    ('CADD', 'splice_and_intronic'): 0.887, ('CADD', 'auroc_Missense'): 0.738,
}
for (k, g), v in br_claims.items():
    eq(f'BRCA1 {k} {g}', v, BR[k][g], tol=1e-3)
exact('BRCA1 n variants', 3893, BR['n'])
exact('BRCA1 n LOF', 823, BR['n_lof'])
exact('BRCA1 n noncoding', 1125, BR['n_noncoding'])
exact('BRCA1 n splice+intronic', 1078, BR['n_splice_intronic'])
exact('BRCA1 n missense', 2086, BR['consequence_counts']['Missense'])

# every likelihood head is at chance on every subset, as claimed
lh = [BR[f'{m}_{hd}'][g] for m in ('quill-k', 'quill-s', 'quill-l')
      for hd in ('smoothed_lr', 'window_delta')
      for g in ('all', 'coding', 'noncoding', 'splice_and_intronic')]
checks.append((min(lh) >= 0.42 and max(lh) <= 0.59,
               'likelihood heads all within 0.42-0.59', '0.42-0.59',
               f'{min(lh):.3f}-{max(lh):.3f}'))
best_lh = max(lh)
eq('best likelihood head = 0.589', 0.589, best_lh, tol=1e-3)
eq('second likelihood head = 0.587', 0.587, sorted(lh)[-2], tol=1e-3)
others = sorted(lh)[:-2]
checks.append((len(others) == 22 and max(abs(v - 0.5) for v in others) <= 0.08,
               'other 22 likelihood heads within 0.08 of chance', '22, <=0.08',
               f'{len(others)}, {max(abs(v - 0.5) for v in others):.4f}'))
exact('24 likelihood-head combinations', 24,
      len([1 for m in ('quill-k', 'quill-s', 'quill-l')
           for hd in ('smoothed_lr', 'window_delta')
           for g in ('all', 'coding', 'noncoding', 'splice_and_intronic')]))

# ---- generation
gen_claims = {('quill-k', 'kmer6_js'): 0.0209, ('quill-s', 'kmer6_js'): 0.0083,
              ('quill-l', 'kmer6_js'): 0.0081,
              ('quill-k', 'gc'): 0.401, ('quill-s', 'gc'): 0.406, ('quill-l', 'gc'): 0.405,
              ('quill-k', 'cpg_obs_exp'): 0.255, ('quill-s', 'cpg_obs_exp'): 0.265,
              ('quill-l', 'cpg_obs_exp'): 0.250,
              ('quill-k', 'codon_usage_js'): 0.0036, ('quill-s', 'codon_usage_js'): 0.0028,
              ('quill-l', 'codon_usage_js'): 0.0053,
              ('quill-k', 'donor_GT_frac'): 0.871, ('quill-s', 'donor_GT_frac'): 0.972,
              ('quill-l', 'donor_GT_frac'): 0.967,
              ('quill-k', 'acceptor_AG_frac'): 0.889, ('quill-s', 'acceptor_AG_frac'): 0.974,
              ('quill-l', 'acceptor_AG_frac'): 0.981,
              ('quill-k', 'internal_stop_rate'): 0.0125,
              ('quill-s', 'internal_stop_rate'): 0.0076,
              ('quill-l', 'internal_stop_rate'): 0.0107}
for (m, k), v in gen_claims.items():
    eq(f'gen {m} {k}', v, GEN[m][k], tol=1e-3)
eq('shuffled control JS 0.0905', 0.0905, GEN['shuffled_control']['kmer6_js'], tol=1e-4)
eq('reference self JS 0.0052', 0.0052, GEN['reference_second_half']['kmer6_js'], tol=1e-4)
eq('counted within 1.6x of reference self', 1.6,
   GEN['quill-s']['kmer6_js'] / GEN['reference_second_half']['kmer6_js'], tol=0.08)
eq('written 4x further than reference self', 4.0,
   GEN['quill-k']['kmer6_js'] / GEN['reference_second_half']['kmer6_js'], tol=0.1)
eq('written 4x closer than shuffled', 4.3,
   GEN['shuffled_control']['kmer6_js'] / GEN['quill-k']['kmer6_js'], tol=0.1)

# ---- annotation
eq('chr20 best coding F1 = 0.142', 0.142, AN['human_chr20_heldout']['quill-s']['posterior']['coding']['f1'])
eq('chr20 best coding MCC = 0.206', 0.206, AN['human_chr20_heldout']['quill-s']['posterior']['coding']['mcc'])
eq('chr21 coding MCC = 0.05', 0.051, AN['human_chr21_heldout']['quill-s']['posterior']['coding']['mcc'])
eq('mouse coding MCC quill-k = 0.330', 0.330, AN['mouse_chr19_heldout_species']['quill-k']['posterior']['coding']['mcc'])
eq('mouse coding MCC quill-s = 0.339', 0.339, AN['mouse_chr19_heldout_species']['quill-s']['posterior']['coding']['mcc'])
eq('mouse coding MCC quill-l = 0.262', 0.262, AN['mouse_chr19_heldout_species']['quill-l']['posterior']['coding']['mcc'])
best20 = max(AN['human_chr20_heldout'][m]['posterior']['coding']['f1'] for m in ('quill-k', 'quill-s', 'quill-l'))
checks.append((abs(best20 - 0.142) < 1e-3, 'chr20 coding F1 really is the maximum', 0.142, round(best20, 4)))

# ---- genomic benchmarks
gb_claims = {('demo_coding_vs_intergenomic_seqs', 'quill-k', 'probe_accuracy'): 0.849,
             ('demo_coding_vs_intergenomic_seqs', 'quill-s', 'probe_accuracy'): 0.908,
             ('demo_coding_vs_intergenomic_seqs', 'quill-l', 'probe_accuracy'): 0.915,
             ('demo_coding_vs_intergenomic_seqs', 'quill-s', 'zero_shot_auroc'): 0.958,
             ('human_nontata_promoters', 'quill-k', 'probe_accuracy'): 0.798,
             ('human_nontata_promoters', 'quill-s', 'probe_accuracy'): 0.822,
             ('human_nontata_promoters', 'quill-l', 'probe_accuracy'): 0.821,
             ('human_nontata_promoters', 'quill-s', 'zero_shot_auroc'): 0.861,
             ('human_enhancers_cohn', 'quill-k', 'probe_accuracy'): 0.673,
             ('human_enhancers_cohn', 'quill-s', 'probe_accuracy'): 0.701,
             ('human_enhancers_cohn', 'quill-l', 'probe_accuracy'): 0.718,
             ('human_enhancers_cohn', 'quill-s', 'zero_shot_auroc'): 0.662,
             ('human_enhancers_ensembl', 'quill-k', 'probe_accuracy'): 0.635,
             ('human_enhancers_ensembl', 'quill-s', 'probe_accuracy'): 0.686,
             ('human_enhancers_ensembl', 'quill-l', 'probe_accuracy'): 0.693,
             ('human_enhancers_ensembl', 'quill-s', 'zero_shot_auroc'): 0.615,
             ('human_ocr_ensembl', 'quill-k', 'probe_accuracy'): 0.595,
             ('human_ocr_ensembl', 'quill-s', 'probe_accuracy'): 0.618,
             ('human_ocr_ensembl', 'quill-l', 'probe_accuracy'): 0.609,
             ('human_ocr_ensembl', 'quill-s', 'zero_shot_auroc'): 0.513,
             ('human_ensembl_regulatory', 'quill-k', 'probe_accuracy'): 0.544,
             ('human_ensembl_regulatory', 'quill-s', 'probe_accuracy'): 0.541,
             ('human_ensembl_regulatory', 'quill-l', 'probe_accuracy'): 0.541}
for (t, m, k), v in gb_claims.items():
    eq(f'gb {t} {m} {k}', v, GB[t][m][k], tol=1e-3)

bad = [c for c in checks if not c[0]]
for ok, name, claimed, actual in checks:
    if not ok:
        print(f'FAIL  {name}: preprint says {claimed}, results say {actual}')
print(f'{len(checks) - len(bad)}/{len(checks)} numbers in the preprint match the result files')
sys.exit(1 if bad else 0)
