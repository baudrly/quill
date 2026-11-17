"""Check every quill-xl number quoted in the preprint against the result files."""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import domains
import regulatory

RES = HERE / 'results'
G = json.loads((RES / 'goxbench.json').read_text())
N = json.loads((RES / 'nongenic.json').read_text())
D = json.loads((RES / 'domains.json').read_text())
checks = []


def eq(name, claimed, actual, tol=5e-4):
    checks.append((abs(float(claimed) - float(actual)) <= tol, name, claimed,
                   round(float(actual), 6)))


def exact(name, claimed, actual):
    checks.append((claimed == actual, name, claimed, actual))


R = G['results']
# ---- the gene model
exact('GO terms modelled', 4175, G['n_terms'])
exact('quill-xl parameters', 12989491, G['parameters'])
exact('dictionary entries', 4151898, G['parts']['dictionary'])
exact('memory k-mer entries', 7845744, G['parts']['memory_kmers'])
exact('held-out genes', 3719, G['n_held'])
exact('remote held-out genes', 2793, G['n_remote'])
eq('quill-xl-go.npz MB', 43.6, os.path.getsize(HERE / 'models/quill-xl-go.npz') / 1e6, 0.05)
eq('quill-xl-go-full.npz MB', 63.8, os.path.getsize(HERE / 'models/quill-xl-go-full.npz') / 1e6, 0.05)
claims = {('naive', 'MF'): .410, ('naive', 'BP'): .364, ('naive', 'CC'): .585,
          ('naive', 'MF_remote'): .393, ('naive', 'BP_remote'): .332,
          ('naive', 'CC_remote'): .563,
          ('rules', 'MF'): .484, ('rules', 'MF_remote'): .461,
          ('memory', 'MF'): .594, ('memory', 'BP'): .458, ('memory', 'CC'): .614,
          ('memory', 'MF_remote'): .515, ('memory', 'BP_remote'): .385,
          ('dictionary', 'MF'): .567, ('dictionary', 'MF_remote'): .510,
          ('quill-xl', 'MF'): .606, ('quill-xl', 'BP'): .462, ('quill-xl', 'CC'): .612,
          ('quill-xl', 'MF_remote'): .535, ('quill-xl', 'BP_remote'): .390,
          ('quill-xl', 'CC_remote'): .589,
          ('quill-xl+rules', 'MF'): .615, ('quill-xl+rules', 'MF_remote'): .548,
          ('quill-xl+rules', 'BP'): .462, ('quill-xl+rules', 'CC'): .613}
for (arm, k), v in claims.items():
    eq(f'Fmax {arm} {k}', v, R[arm][k]['fmax'], 6e-4)
eq('rules MF precision', 0.756, R['rules']['MF']['precision'], 6e-4)

eq('MF gain over naive', 0.205, R['quill-xl+rules']['MF']['fmax'] - R['naive']['MF']['fmax'], 6e-4)
eq('MF remote gain over naive', 0.154, R['quill-xl+rules']['MF_remote']['fmax'] - R['naive']['MF_remote']['fmax'], 6e-4)
eq('BP gain over naive', 0.098, R['quill-xl+rules']['BP']['fmax'] - R['naive']['BP']['fmax'], 6e-4)
eq('CC gain over naive', 0.028, R['quill-xl+rules']['CC']['fmax'] - R['naive']['CC']['fmax'], 6e-4)
eq('written signatures add on remote MF', 0.013,
   R['quill-xl+rules']['MF_remote']['fmax'] - R['quill-xl']['MF_remote']['fmax'], 6e-4)
eq('dense-to-sparse shrink factor', 74, 307012800 / G['parts']['dictionary'], 0.6)
import qgox
exact('live GO terms', 37944, len(qgox.load_terms()))

# ---- written signatures
s = D['summary']
exact('signatures written', 67, len(domains.RULES))
exact('signatures resolved by name', 67, s['rules_resolved'])
exact('testable signatures', 42, s['rules_with_5_or_more_matches'])
exact('Bonferroni-significant signatures', 36, s['rules_significant_bonferroni'])
eq('median precision', 0.937, s['median_precision'], 6e-4)
eq('median lift', 46.0, s['median_lift'], 0.05)
exact('signatures with precision >= 0.8', 27, s['rules_with_precision_at_least_0_8'])
eq('mean written confidence', 0.785, s['mean_written_confidence'], 6e-4)
eq('mean measured precision', 0.770, s['mean_measured_precision'], 6e-4)
eq('Spearman confidence vs precision', 0.50, s['spearman_confidence_vs_precision'], 6e-3)
eq('fraction of held-out genes matching a signature', 0.351,
   s['fraction_of_genes_matching_any_rule'], 6e-4)
rules = {r['rule']: r for r in D['rules']}
eq('C2H2 pattern precision for zinc ion binding', 0.06,
   rules['C2H2 zinc finger']['precision'], 6e-3)
eq('leucine zipper lift', 0.54, rules['leucine zipper heptad']['lift'], 6e-3)
exact('CxxCH heme matches with heme binding', 0,
      rules['cytochrome c heme attachment']['with_term'])
eq('protein kinase loop precision', 0.97, rules['protein kinase catalytic loop']['precision'], 6e-3)
exact('protein kinase loop matches', 64, rules['protein kinase catalytic loop']['matches'])

# ---- TF sites
t = N['tf']['summary']
exact('written TF sites', 33, len(regulatory.TF_SITES))
exact('JASPAR factors', 816, t['jaspar_factors'])
eq('median fraction of JASPAR max', 0.973, t['median_fraction_of_jaspar_max'], 6e-4)
exact('sites >= 0.9 of JASPAR max', 31, t['at_least_0_9'])
exact('own factor ranked first', 7, t['own_factor_rank_1'])
exact('own factor in top 10', 23, t['own_factor_top_10'])
exact('median rank of own factor', 4.0, t['median_rank'])

# ---- regulatory probe
reg = N['regulatory']
for task, comp, best, jas, allf in (
        ('human_ensembl_regulatory', .553, .599, .589, .594),
        ('human_enhancers_cohn', .696, .722, .680, .687),
        ('human_ocr_ensembl', .619, .653, .638, .633),
        ('human_nontata_promoters', .816, .839, .800, .814)):
    eq(f'{task} composition', comp, reg[task]['composition (3-mers)']['accuracy'], 6e-4)
    eq(f'{task} grammar+written', best, reg[task]['composition + grammar + written']['accuracy'], 6e-4)
    eq(f'{task} +JASPAR', jas, reg[task]['composition + JASPAR']['accuracy'], 6e-4)
    eq(f'{task} all four', allf, reg[task]['all four']['accuracy'], 6e-4)
    acc = {k: v['accuracy'] for k, v in reg[task].items() if isinstance(v, dict)}
    checks.append((max(acc, key=acc.get) == 'composition + grammar + written',
                   f'{task}: grammar + written is the best feature set', 'best',
                   max(acc, key=acc.get)))
    checks.append((acc['all four'] < acc['composition + grammar + written'],
                   f'{task}: adding JASPAR to grammar + written lowers accuracy', 'lower',
                   round(acc['all four'] - acc['composition + grammar + written'], 4)))

# ---- repeats
rp = N['repeats']
exact('Alu loci chr21', 4771, rp['Alu']['chr21']['loci'])
exact('Alu loci chr22', 9360, rp['Alu']['chr22']['loci'])
eq('Alu per Mb chr21', 119.0, rp['Alu']['chr21']['per_mb'], 0.05)
eq('Alu per Mb chr22', 239.0, rp['Alu']['chr22']['per_mb'], 0.05)
checks.append((max(rp['Alu']['chr21']['shuffled_loci_mean'],
                   rp['Alu']['chr22']['shuffled_loci_mean']) < 1.0,
               'shuffled Alu heads find under one locus', '<1',
               rp['Alu']['chr22']['shuffled_loci_mean']))
exact('L1 loci chr21', 62, rp['L1']['chr21']['loci'])
eq('L1 shuffled chr21', 4.6, rp['L1']['chr21']['shuffled_loci_mean'], 0.05)
eq('L1 per Mb chr22', 1.2, rp['L1']['chr22']['per_mb'], 0.05)

bad = [c for c in checks if not c[0]]
for ok, name, claimed, actual in checks:
    if not ok:
        print(f'FAIL  {name}: preprint says {claimed}, results say {actual}')
print(f'{len(checks) - len(bad)}/{len(checks)} quill-xl numbers in the preprint match the '
      f'result files')
sys.exit(1 if bad else 0)
