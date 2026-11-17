"""Describe a protein with the written knowledge of proteome.py, plus PROSITE patterns.

describe(prot)      physicochemistry, transmembrane helices and topology, signal
                    peptide and cleavage site, mitochondrial presequence, GPI anchor,
                    coiled coils, disorder, low complexity, short linear motifs and
                    modification sites, all from written rules
prosite(prot)       PROSITE pattern hits (release 2019_10, 1,311 patterns)
features(d, hits)   the binary feature names a description implies, for the experts

    python3 qprot.py cache     # describe every annotated gene once -> data/gox/prot.pkl
"""
import math
import pickle
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import proteome as P

PROSITE_DAT = Path('/home/claude/data/ext/ps/prosite.dat')
CACHE = Path('/home/claude/data/gox/prot.pkl')
AIDX = {a: i for i, a in enumerate(P.AA)}


def _vec(scale):
    return np.array([scale[a] for a in P.AA], np.float64)


KD, EIS, DIS, HEL, SHE, MASS = (_vec(P.KYTE_DOOLITTLE), _vec(P.EISENBERG),
                               _vec(P.TOP_IDP), _vec(P.HELIX), _vec(P.SHEET),
                               _vec(P.MASS))


def encode(prot):
    """Residue indices; anything outside the 20 standard residues is dropped."""
    return np.array([AIDX[c] for c in prot if c in AIDX], np.int64)


def smooth(x, w):
    if len(x) < w:
        return np.zeros(0)
    c = np.concatenate([[0.0], np.cumsum(x)])
    return (c[w:] - c[:-w]) / w


def runs(mask, min_gap=0):
    """(start, end) of runs of True, merging gaps shorter than min_gap."""
    out = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if out and i - out[-1][1] < min_gap:
                out[-1] = (out[-1][0], j)
            else:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def isoelectric_point(prot):
    cnt = {k: prot.count(k) for k in 'KRHDECY'}

    def charge(ph):
        pos = (1 / (1 + 10 ** (ph - P.PKA['Nterm'])) +
               sum(cnt[k] / (1 + 10 ** (ph - P.PKA[k])) for k in 'KRH'))
        neg = (1 / (1 + 10 ** (P.PKA['Cterm'] - ph)) +
               sum(cnt[k] / (1 + 10 ** (P.PKA[k] - ph)) for k in 'DECY'))
        return pos - neg

    lo, hi = 0.0, 14.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if charge(mid) > 0:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 3), round(charge(7.0), 3)


def tm_helices(x):
    """Kyte-Doolittle helices: segments whose 19-residue mean clears the threshold."""
    h = smooth(KD[x], P.TM_WINDOW)
    segs = runs(h >= P.TM_THRESHOLD, P.TM_MIN_GAP)
    return [(a, b + P.TM_WINDOW - 1) for a, b in segs]


def topology(prot, helices):
    """Positive-inside rule: the end with more K+R in its first loop faces the cytosol."""
    if not helices:
        return None
    a, b = helices[0]
    n_loop = prot[max(0, a - 15):a]
    c_loop = prot[b:b + 15] if len(helices) == 1 else prot[b:helices[1][0]][:15]
    kn = sum(n_loop.count(k) for k in 'KR')
    kc = sum(c_loop.count(k) for k in 'KR')
    return 'N_in' if kn >= kc else 'N_out'


def signal_peptide(prot, x):
    """von Heijne rule: (has_signal, cleavage position or None, h-region)."""
    head = x[:P.SP_SEARCH]
    if len(head) < 20:
        return False, None, None
    h = smooth(KD[head], P.SP_H_WINDOW)
    best = None
    for s in range(1, min(len(h), 20)):
        if h[s] >= P.SP_H_THRESHOLD:
            e = s + P.SP_H_WINDOW
            while e < len(head) and KD[head[e]] > 0:
                e += 1
            if e - s > P.SP_MAX_H:
                return False, None, (s, e)       # too long: a signal anchor, not a signal
            for c in range(e + P.SP_C_MIN, min(e + P.SP_C_MAX, len(prot) - 1) + 1):
                if prot[c - 1] in P.SP_SMALL and prot[c - 3] in P.SP_SMALL \
                        and prot[c] != 'P' and prot[c - 1] != 'P':
                    best = (c, (s, e))
                    break
            if best:
                break
    if not best:
        return False, None, None
    return True, best[0], best[1]


def hydrophobic_moment(x, angle_deg, window):
    if len(x) < window:
        return 0.0
    th = math.radians(angle_deg)
    k = np.arange(window)
    cos, sin = np.cos(th * k), np.sin(th * k)
    best = 0.0
    for s in range(len(x) - window + 1):
        h = EIS[x[s:s + window]]
        m = math.hypot(float((h * cos).sum()), float((h * sin).sum())) / window
        best = max(best, m)
    return best


def mito_presequence(prot, x):
    head = prot[:P.MITO_SEARCH]
    basic = sum(head.count(k) for k in 'RK')
    acidic = sum(head.count(k) for k in 'DE')
    mom = hydrophobic_moment(x[:P.MITO_SEARCH], P.MITO_MOMENT_ANGLE, P.MITO_MOMENT_WINDOW)
    ok = basic >= P.MITO_MIN_BASIC and acidic <= P.MITO_MAX_ACIDIC and \
        mom >= P.MITO_MOMENT_MIN and prot[:1] == 'M'
    return ok, round(mom, 3), basic - 2 * acidic


def gpi_anchor(prot, x, has_signal):
    tail = x[-P.GPI_TAIL:]
    if not has_signal or len(tail) < P.GPI_TAIL:
        return False
    h = smooth(KD[tail], P.GPI_H_WINDOW)
    hyd = len(h) and h[-12:].max() >= P.GPI_H_THRESHOLD
    omega = any(c in P.GPI_OMEGA for c in prot[-P.GPI_TAIL:-P.GPI_TAIL + 12])
    return bool(hyd and omega)


def coiled_coil_fraction(prot):
    n = len(prot)
    if n < P.CC_WINDOW:
        return 0.0
    covered = np.zeros(n, bool)
    ad = np.array([c in P.CC_A_D for c in prot])
    eg = np.array([c in P.CC_E_G for c in prot])
    for s in range(0, n - P.CC_WINDOW + 1, 7):
        best = 0.0
        for reg in range(7):
            idx = np.arange(s, s + P.CC_WINDOW)
            ph = (idx - s + reg) % 7
            sa = ad[idx[(ph == 0) | (ph == 3)]].mean()
            se = eg[idx[(ph == 4) | (ph == 6)]].mean()
            best = max(best, 0.6 * sa + 0.4 * se)
        if best >= P.CC_MIN_SCORE:
            covered[s:s + P.CC_WINDOW] = True
    return float(covered.mean())


def low_complexity_fraction(x):
    n, w = len(x), P.LC_WINDOW
    if n < w:
        return 0.0
    covered = np.zeros(n, bool)
    for s in range(n - w + 1):
        c = np.bincount(x[s:s + w], minlength=20)
        p = c[c > 0] / w
        if -(p * np.log2(p)).sum() <= P.LC_MAX_BITS:
            covered[s:s + w] = True
    return float(covered.mean())


SLIM_RX = [(n, re.compile(r), d) for n, r, d in P.SLIMS]


def describe(prot):
    prot = ''.join(c for c in prot.upper() if c in AIDX)
    x = encode(prot)
    n = len(x)
    if n == 0:
        return None
    comp = np.bincount(x, minlength=20) / n
    pi, charge = isoelectric_point(prot)
    helices = tm_helices(x)
    sp, cleave, hreg = signal_peptide(prot, x)
    if sp:   # the signal's own h-region is not a transmembrane helix
        helices = [h for h in helices if h[0] > cleave - 5]
    mito, moment, mito_charge = mito_presequence(prot, x)
    dis = smooth(DIS[x], P.DIS_WINDOW)
    dmask = dis >= P.DIS_THRESHOLD
    long_dis = [r for r in runs(dmask) if r[1] - r[0] >= P.DIS_LONG]
    slims = {}
    for name, rx, _ in SLIM_RX:
        slims[name] = len(rx.findall(prot))
    return dict(
        length=n, mass_kda=round((MASS[x].sum() + P.WATER) / 1000.0, 2), pI=pi,
        charge_pH7=charge, gravy=round(float(KD[x].mean()), 3),
        aromatic=round(float(comp[[AIDX[a] for a in 'FWY']].sum()), 4),
        cysteine=round(float(comp[AIDX['C']]), 4),
        aliphatic_index=round(float(100 * (comp[AIDX['A']] + 2.9 * comp[AIDX['V']] +
                                           3.9 * (comp[AIDX['I']] + comp[AIDX['L']]))), 1),
        helix_propensity=round(float(HEL[x].mean()), 3),
        sheet_propensity=round(float(SHE[x].mean()), 3),
        disorder_fraction=round(float(dmask.mean()) if len(dmask) else 0.0, 3),
        long_disordered_regions=[(int(a), int(b + P.DIS_WINDOW - 1)) for a, b in long_dis],
        coiled_coil_fraction=round(coiled_coil_fraction(prot), 3),
        low_complexity_fraction=round(low_complexity_fraction(x), 3),
        tm_helices=[(int(a), int(b)) for a, b in helices], topology=topology(prot, helices),
        signal_peptide=bool(sp), cleavage_site=cleave,
        mito_presequence=bool(mito), mito_moment=moment, mito_charge=mito_charge,
        gpi_anchor=gpi_anchor(prot, x, sp), slims=slims)


# ---------------------------------------------------------------- PROSITE
def _prosite_to_regex(pat):
    pat = pat.rstrip('.')
    out = []
    for el in pat.split('-'):
        rep = ''
        m = re.match(r'^(.*?)\((\d+)(?:,(\d+))?\)$', el)
        if m:
            el = m.group(1)
            rep = '{%s}' % m.group(2) if m.group(3) is None else \
                '{%s,%s}' % (m.group(2), m.group(3))
        start = el.startswith('<')
        end = el.endswith('>') and not el.endswith('>]')
        el = el.lstrip('<')
        if end:
            el = el[:-1]
        if el == 'x':
            core = '.'
        elif el.startswith('['):
            inner = el[1:-1]
            if '>' in inner:
                core = '(?:[%s]|$)' % inner.replace('>', '') if inner.replace('>', '') \
                    else '$'
            else:
                core = '[%s]' % inner
        elif el.startswith('{'):
            core = '[^%s]' % el[1:-1]
        else:
            core = el
        out.append(('^' if start else '') + core + rep + ('$' if end else ''))
    return ''.join(out)


@lru_cache(maxsize=1)
def prosite_patterns():
    """[(accession, name, description, regex string, compiled, skip_flag)]"""
    from Bio.ExPASy import Prosite
    out = []
    with open(PROSITE_DAT) as fh:
        for r in Prosite.parse(fh):
            if r.type != 'PATTERN' or not r.pattern:
                continue
            try:
                rx = _prosite_to_regex(r.pattern)
                comp = re.compile(rx)
            except re.error:
                continue
            out.append((r.accession, r.name, r.description, rx, comp,
                        bool(getattr(r, 'cc_skip_flag', False))))
    return out


def prosite(prot):
    return [acc for acc, _, _, _, rx, skip in prosite_patterns()
            if not skip and rx.search(prot)]


# ---------------------------------------------------------------- features
BINS = dict(length=[100, 200, 300, 450, 650, 900, 1300, 2000],
            pI=[5, 6, 7, 8, 9, 10, 11], gravy=[-1.0, -0.6, -0.3, 0.0, 0.3],
            disorder_fraction=[0.1, 0.2, 0.35, 0.5, 0.7],
            coiled_coil_fraction=[0.02, 0.1, 0.25, 0.5],
            low_complexity_fraction=[0.02, 0.05, 0.1, 0.2],
            cysteine=[0.01, 0.02, 0.035, 0.06, 0.1], aromatic=[0.06, 0.08, 0.1, 0.12, 0.15],
            helix_propensity=[0.98, 1.02, 1.06, 1.1], sheet_propensity=[0.95, 1.0, 1.05, 1.1])
TM_CLASSES = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 6), (7, 7), (8, 11), (12, 999)]


def _bin(v, edges):
    return int(np.searchsorted(edges, v, side='right'))


def features(d, hits=()):
    """Binary feature names implied by a description and its PROSITE hits."""
    f = [f'{k}:{_bin(d[k], e)}' for k, e in BINS.items()]
    ntm = len(d['tm_helices'])
    f.append('tm:' + next(f'{a}-{b}' for a, b in TM_CLASSES if a <= ntm <= b))
    if ntm:
        f.append(f'topology:{d["topology"]}')
    if d['signal_peptide']:
        f.append('signal_peptide')
        f.append('secreted_like' if ntm == 0 else ('type_I_membrane' if ntm == 1
                                                   else 'signal_multi_tm'))
    elif ntm and d['tm_helices'][0][0] < 40:
        f.append('signal_anchor')
    if d['gpi_anchor']:
        f.append('gpi_anchor')
    if d['mito_presequence']:
        f.append('mito_presequence')
    if d['long_disordered_regions']:
        f.append('long_disorder:%d' % min(len(d['long_disordered_regions']), 3))
    per100 = 100.0 / max(d['length'], 1)
    for name, c in d['slims'].items():
        if c:
            dens = c * per100
            f.append(f'slim:{name}:' + ('1' if c == 1 else ('2-3' if c <= 3 else '4+')))
            if name.startswith(('phospho_', 'N_glyc', 'SUMO', 'D_box', 'ER_export')):
                f.append(f'slimdens:{name}:{_bin(dens, [0.5, 1, 2, 4])}')
    if d['signal_peptide'] and d['slims'].get('N_glycosylation', 0) >= 3:
        f.append('signal_and_glycosylated')
    f.extend(f'prosite:{a}' for a in hits)
    return f


def _describe_one(args):
    gene, prot = args
    d = describe(prot)
    return gene, d, prosite(prot) if d else []


def cmd_cache():
    """Describe every annotated gene (train, calibration, held-out) once."""
    import qgox
    from multiprocessing import Pool
    sets = qgox.load_sets()
    jobs = [(r['gene'], r['prot']) for s in ('train', 'calib', 'held') for r in sets[s]]
    prosite_patterns()
    out = {}
    with Pool(2) as pool:
        for i, (g, d, h) in enumerate(pool.imap_unordered(_describe_one, jobs, 64)):
            out[g] = (d, h)
            if (i + 1) % 2000 == 0:
                print(f'  {i + 1}/{len(jobs)} proteins described', flush=True)
    with open(CACHE, 'wb') as fh:
        pickle.dump(out, fh, protocol=4)
    print(f'{CACHE}: {len(out)} proteins, {len(prosite_patterns())} PROSITE patterns '
          f'({sum(1 for p in prosite_patterns() if not p[5])} used)')


if __name__ == '__main__':
    if sys.argv[1:] == ['cache']:
        cmd_cache()
