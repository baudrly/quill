"""Check every quill-xl3 and quill-v number quoted in the preprint against the result files."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import qalign
import qvariant

RES = HERE / 'results'
X = json.loads((RES / 'xl3.json').read_text())
V = json.loads((RES / 'variant.json').read_text())
B = json.loads((RES / 'bench.json').read_text())['brca1']
checks = []


def eq(name, claimed, actual, tol=6e-4):
    checks.append((abs(float(claimed) - float(actual)) <= tol, name, claimed,
                   round(float(actual), 6)))


def exact(name, claimed, actual):
    checks.append((claimed == actual, name, claimed, actual))


KEYS = ('BP', 'BP_remote', 'MF', 'MF_remote', 'CC', 'CC_remote', 'EC', 'EC_remote', 'slim')
TABLE = {
    'nano': (97273, 0.29, 2387, .413, .366, .541, .510, .632, .612, .433, .404, .483),
    'small': (360636, 1.25, 4175, .425, .375, .580, .542, .636, .613, .474, .429, .499),
    'medium': (21201398, 50.99, 4175, .551, .465, .690, .645, .679, .648, .672, .598, .620),
    'large': (24022344, 62.19, 6230, .554, .470, .704, .657, .679, .648, .694, .622, .618)}
PREVIOUS = {'nano': (.412, .539, .628, .429), 'small': (.425, .580, .631, .473),
            'medium': (.476, .627, .639, .557), 'large': (.478, .613, .640, .566)}
M = X['models']
exact('held-out genes', 3719, X['n_held'])
exact('remote genes', 2793, X['n_remote'])
for s, v in TABLE.items():
    exact(f'{s} parameters', v[0], M[s]['parameters'])
    eq(f'{s} MB', v[1], M[s]['mb'], 0.006)
    exact(f'{s} terms', v[2], M[s]['terms'])
    for k, c in zip(KEYS, v[3:]):
        eq(f'{s} {k}', c, M[s]['held_out'][k])
        eq(f'{s} {k} second pass changes nothing', M[s]['first_pass'][k], M[s]['held_out'][k], 1e-9)
    for k, c in zip(('BP', 'MF', 'CC', 'EC'), PREVIOUS[s]):
        eq(f'{s} quill-xl2 {k}', c, X['previous'][s]['held_out'][k])
    exact(f'{s} crosstalk rejected', False, any(M[s]['crosstalk'].values()))
for k, c in zip(('BP', 'MF', 'CC', 'EC'), (.364, .410, .585, .267)):
    eq(f'naive {k}', c, M['medium']['naive'][k])
eq('medium MF gain', 0.063, M['medium']['held_out']['MF'] - X['previous']['medium']['held_out']['MF'], 1e-3)
eq('medium BP gain', 0.075, M['medium']['held_out']['BP'] - X['previous']['medium']['held_out']['BP'], 1e-3)
eq('medium EC gain', 0.115, M['medium']['held_out']['EC'] - X['previous']['medium']['held_out']['EC'], 1e-3)
eq('medium MF remote gain', 0.088, M['medium']['held_out']['MF_remote'] - X['previous']['medium']['held_out']['MF_remote'], 1e-3)
exact('alignment share of medium parameters', 8047559, M['medium']['parts']['align'])
eq('medium seconds per 1000 genes', 3.9, M['medium']['seconds_per_1000_genes'], 0.06)
ALONE = {'align': (.532, .704, .669, .673), 'memory': (.458, .594, .614, .508),
         'kmer': (.414, .567, .600, .459), 'protein': (.387, .476, .622, .328),
         'prosite': (.394, .543, .610, .430), 'rna': (.361, .412, .591, .289),
         'dna': (.364, .411, .586, .267), 'signatures': (.364, .483, .589, .358)}
for e, v in ALONE.items():
    for k, c in zip(('BP', 'MF', 'CC', 'EC'), v):
        eq(f'{e} alone {k}', c, X['standalone'][e][k])
for k, c in zip(('BP', 'MF', 'CC', 'EC'), (.430, .588, .634, .522)):
    eq(f'without align {k}', c, X['without']['align'][k])
eq('without dna CC', .676, X['without']['dna']['CC'])
eq('without dna CC remote', .640, X['without']['dna']['CC_remote'])
eq('without protein CC', .664, X['without']['protein']['CC'])
for e in ('medium', 'large'):
    for ns in ('BP', 'MF', 'CC'):
        exact(f'{e} {ns} align weight', 1.5, M[e]['weights'][ns]['align'])
exact('medium CC dna weight', 1.0, M['medium']['weights']['CC']['dna'])
exact('medium BP dna weight', 0.0, M['medium']['weights']['BP']['dna'])
exact('medium MF dna weight', 0.0, M['medium']['weights']['MF']['dna'])
R = X['reliability']
eq('ECE', 0.024, R['ece'])
eq('top bin predicted', 0.981, R['bins'][9]['predicted'])
eq('top bin observed', 0.702, R['bins'][9]['observed'])
eq('0.5 bin observed', 0.196, R['bins'][5]['observed'])
exact('alignment rule constants', (150, 100.0, 30, 2, 11, 1, 8), (
    qalign.CANDIDATES, qalign.MIN_SCORE, qalign.TOP_HITS, qalign.POWER, qalign.GAP_OPEN,
    qalign.GAP_EXTEND, qalign.SEED_K))
exact('variant tiers', (10.0, 9.0, 2.0), (qvariant.TIER['nonsense'],
                                           qvariant.TIER['canonical_splice'],
                                           qvariant.TIER['missense']))
exact('BRCA1 variants', 3893, V['n'])
exact('BRCA1 missense variants', 2086, V['n_missense'])
exact('BRCA1 homologues aligned', 139, V['homologues_aligned'])
eq('consequence agreement', 0.998, V['consequence_agreement'])
eq('residues with homologue cover', 0.452, V['residues_with_homologue_cover'])
VT = {'quill-v': (.772, .729, .587, .852), 'BLOSUM62 only': (.771, .724, .581, .852),
      'BLOSUM62 + residue rules': (.780, .741, .609, .852),
      'BLOSUM62 + conservation': (.763, .711, .556, .852), 'tiers only': (.729, .649, .459, .852),
      'CADD': (.804, .841, .738, .875), 'phyloP (mammalian)': (.781, .736, .724, .887)}
for t, v in VT.items():
    for k, c in zip(('all', 'coding', 'missense', 'noncoding'), v):
        eq(f'{t} {k}', c, V['variants'][t][k])
eq('earlier quill-k splice-matrix all', .548, B['quill-k_splice_matrix']['all'])
eq('earlier quill-k splice-matrix missense', .445, B['quill-k_splice_matrix']['auroc_Missense'])
bad = [c for c in checks if not c[0]]
for ok, name, claimed, actual in bad:
    print(f'MISMATCH {name}: claimed {claimed}, actual {actual}')
print(f'{len(checks) - len(bad)}/{len(checks)} quill-xl3 and quill-v figures agree with the result files')
sys.exit(bool(bad))
