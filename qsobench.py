"""Benchmark the Sequence Ontology layer on held-out chromosomes.

  classify  - SO class of a non-coding transcript from written motifs, written vs counted
  minor     - find the U12 (AT-AC) introns among held-out introns with the written matrices
  cis       - do the written core-promoter matrices fire where promoters actually are

Writes results/sobench.json.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import knowledge as K
import ontology as O
import qso
from qgobench import auroc, metrics

HS = Path('/home/claude/data/hs')
RES = HERE / 'results'
RES.mkdir(exist_ok=True)
OUT = RES / 'sobench.json'
HELD = ['17', '19', '20', '21', '22']


def save(d):
    cur = json.loads(OUT.read_text()) if OUT.exists() else {}
    cur.update(d)
    OUT.write_text(json.dumps(cur, indent=1))


def cmd_classify(_):
    held = qso.load('held')
    truth = [r['so'] for r in held]
    present = sorted(set(truth))
    out = {}
    for arm in ('quill-k', 'quill-s'):
        m = qso.SoModel(HERE / 'models' / f'{arm}-so.npz')
        S = np.array([m.score(r['arr'], r['exons']) for r in held])
        keep = [i for i, nm in enumerate(m.names) if nm in present]
        res = metrics(S[:, keep], truth, [m.names[i] for i in keep])
        res['parameters'] = m.nparam
        out[arm] = res
        print(f'{arm}: acc {res["accuracy"]:.3f}  balanced {res["balanced_accuracy"]:.3f}'
              f'  macro AUROC {res["macro_auroc"]:.3f}  (majority '
              f'{res["majority_baseline"]:.3f}, {res["n"]} transcripts)', flush=True)
    # which written component carries it
    saved = dict(qso.SO_WEIGHTS)
    for drop in ('boxes', 'length', 'gc', 'hairpin', 'exons'):
        qso.SO_WEIGHTS.update(saved)
        qso.SO_WEIGHTS[drop] = 0.0
        m = qso.SoModel(HERE / 'models' / 'quill-k-so.npz')
        S = np.array([m.score(r['arr'], r['exons']) for r in held])
        keep = [i for i, nm in enumerate(m.names) if nm in present]
        res = metrics(S[:, keep], truth, [m.names[i] for i in keep])
        out[f'quill-k-without-{drop}'] = dict(
            accuracy=res['accuracy'], macro_auroc=res['macro_auroc'],
            balanced_accuracy=res['balanced_accuracy'])
        print(f'  written without {drop:9s}: acc {res["accuracy"]:.3f}  '
              f'macro AUROC {res["macro_auroc"]:.3f}', flush=True)
    qso.SO_WEIGHTS.update(saved)
    save({'classify': out})


def _logodds(mat):
    return np.log2(np.maximum(np.asarray(mat, np.float64), 1e-4) / 0.25)


def _score_at(seq, logmat, pos):
    L = logmat.shape[0]
    if pos < 0 or pos + L > len(seq):
        return None
    w = seq[pos:pos + L]
    if (w > 3).any():
        return None
    return float(logmat[np.arange(L), w].sum())


def cmd_minor(_):
    """U12 introns are ~0.35% of introns; the written matrices have to find them."""
    u12d, u12a, u12b = (_logodds(O.U12_DONOR), _logodds(O.U12_ACCEPTOR),
                        _logodds(O.U12_BRANCH))
    u2d, u2a = _logodds(K.DONOR), _logodds(K.ACCEPTOR)
    rows = []
    for c in HELD:
        h = pickle.load(open(HS / f'{c}.pkl', 'rb'))
        seq = np.load(HS / f'{c}.npy')
        seen = set()
        for tid, strand, segs, gid, prot in h['cds']:
            ex = segs if strand == '+' else list(reversed(segs))
            for k in range(len(ex) - 1):
                if strand == '+':
                    a, b = ex[k][1], ex[k + 1][0]
                else:
                    a, b = ex[k + 1][1], ex[k][0]
                if b - a < 60 or (a, b, strand) in seen:
                    continue
                seen.add((a, b, strand))
                s = np.asarray(seq[a:b], np.int8)
                if strand == '-':
                    s = s[::-1]
                    s = np.where(s <= 3, 3 - s, s)
                if (s[:2] > 3).any() or (s[-2:] > 3).any():
                    continue
                first2 = ''.join('ACGT'[int(x)] for x in s[:2])
                last2 = ''.join('ACGT'[int(x)] for x in s[-2:])
                is_atac = (first2 == 'AT' and last2 == 'AC')
                d12 = _score_at(s, u12d, 0)
                a12 = _score_at(s, u12a, len(s) - u12a.shape[0])
                d2 = _score_at(s, u2d, 0)
                a2 = _score_at(s, u2a, len(s) - u2a.shape[0])
                if None in (d12, a12, d2, a2):
                    continue
                # best written U12 branch point in its window upstream of the 3' site
                bl = u12b.shape[0]
                lo = max(0, len(s) - 45)
                best_b = -30.0
                for p in range(lo, len(s) - bl - 8):
                    v = _score_at(s, u12b, p)
                    if v is not None and v > best_b:
                        best_b = v
                rows.append((is_atac, d12 + a12 + best_b - (d2 + a2), d12, best_b))
        del seq
        print(f'  chr{c}: {len(rows)} introns', flush=True)
    y = np.array([r[0] for r in rows])
    res = dict(n_introns=len(rows), n_atac=int(y.sum()),
               atac_fraction=round(float(y.mean()), 5),
               written_u12_fraction=O.U12_FRACTION,
               auroc_u12_vs_u2_score=round(auroc(y, np.array([r[1] for r in rows])), 4),
               auroc_donor_only=round(auroc(y, np.array([r[2] for r in rows])), 4),
               auroc_branch_only=round(auroc(y, np.array([r[3] for r in rows])), 4))
    print(f'U12 detection: {res["n_atac"]} AT-AC of {res["n_introns"]} introns '
          f'({res["atac_fraction"]*100:.3f}%); AUROC {res["auroc_u12_vs_u2_score"]:.4f} '
          f'(donor alone {res["auroc_donor_only"]:.4f}, '
          f'branch alone {res["auroc_branch_only"]:.4f})')
    save({'minor_intron': res})


def cmd_cis(_):
    """Core-promoter matrices at real transcription starts versus shifted controls."""
    tata, ccaat, gcb, inr, tct = (np.array(O.matrix(n), np.float64) for n in
                                  ('TATA', 'CCAAT', 'GC_BOX', 'INR', 'TCT'))
    rng = np.random.default_rng(5)
    KEYS = ('TATA', 'CCAAT', 'GC_BOX', 'INR', 'TCT')
    real = {k: [] for k in KEYS}
    ctrl = {k: [] for k in KEYS}      # same window, bases shuffled: composition-matched
    dist = {k: [] for k in KEYS}      # a distant window, for reference only
    n = 0
    for c in HELD:
        h = pickle.load(open(HS / f'{c}.pkl', 'rb'))
        seq = np.load(HS / f'{c}.npy')
        starts = []
        for tid, (kind, strand, exons, gid) in h['tx'].items():
            if not exons or kind != 'mRNA':
                continue
            tss = exons[0][0] if strand == '+' else exons[-1][1] - 1
            starts.append((tss, strand))
        for tss, strand in starts:
            for bag, off in ((real, 0), (dist, int(rng.integers(20000, 120000)) *
                                         (1 if rng.random() < .5 else -1))):
                q = tss + off
                # 120 bp upstream and 60 downstream IN TRANSCRIPTION ORIENTATION, so the
                # window reaches the promoter on both strands and the TSS lands at 120
                lo, hi = (q - 120, q + 60) if strand == '+' else (q - 59, q + 121)
                if lo < 0 or hi >= len(seq):
                    continue
                w = np.asarray(seq[lo:hi], np.int8)
                if strand == '-':
                    w = w[::-1]
                    w = np.where(w <= 3, 3 - w, w)
                if (w > 3).mean() > 0.1:
                    continue
                centre = 120
                bag['TATA'].append(_best(w, tata, centre - 40, centre - 22))
                bag['CCAAT'].append(_best(w, ccaat, centre - 110, centre - 45))
                bag['GC_BOX'].append(_best(w, gcb, centre - 110, centre + 10))
                bag['INR'].append(_best(w, inr, centre - 4, centre + 6))
                bag['TCT'].append(_best(w, tct, centre - 4, centre + 10))
                if bag is real:
                    # the composition-matched control: identical bases, order destroyed
                    sh = w.copy()
                    good = sh <= 3
                    v = sh[good]
                    rng.shuffle(v)
                    sh[good] = v
                    ctrl['TATA'].append(_best(sh, tata, centre - 40, centre - 22))
                    ctrl['CCAAT'].append(_best(sh, ccaat, centre - 110, centre - 45))
                    ctrl['GC_BOX'].append(_best(sh, gcb, centre - 110, centre + 10))
                    ctrl['INR'].append(_best(sh, inr, centre - 4, centre + 6))
                    ctrl['TCT'].append(_best(sh, tct, centre - 4, centre + 10))
            n += 1
        del seq
        print(f'  chr{c}: {n} transcription starts', flush=True)
    out = {'n_tss': n}
    for k in KEYS:
        a = np.array([v for v in real[k] if v is not None])
        b = np.array([v for v in ctrl[k] if v is not None])
        c = np.array([v for v in dist[k] if v is not None])
        m = min(len(a), len(b))
        y = np.r_[np.ones(m, bool), np.zeros(m, bool)]
        md = min(len(a), len(c))
        yd = np.r_[np.ones(md, bool), np.zeros(md, bool)]
        out[k] = dict(auroc_vs_shuffled_same_window=round(
                          auroc(y, np.r_[a[:m], b[:m]]), 4),
                      auroc_vs_distal_window=round(auroc(yd, np.r_[a[:md], c[:md]]), 4),
                      mean_at_tss=round(float(a[:m].mean()), 3),
                      mean_shuffled=round(float(b[:m].mean()), 3),
                      mean_distal=round(float(c[:md].mean()), 3), n=int(m))
        o = out[k]
        print(f'  {k:7s} AUROC vs shuffled {o["auroc_vs_shuffled_same_window"]:.4f}'
              f'  (vs distal {o["auroc_vs_distal_window"]:.4f})'
              f'  mean {o["mean_at_tss"]:+.2f} shuffled {o["mean_shuffled"]:+.2f}')
    save({'cis_elements': out})


def _best(w, mat, a, b, bg=None):
    """Best log2 odds of a written matrix in a window, against a background.

    The background must be the local base composition, not a uniform 0.25: an AT-rich
    matrix such as the TATA box scores high in any AT-rich stretch, so a uniform
    background measures GC content rather than the presence of the element.
    """
    L = mat.shape[0]
    a, b = max(0, a), min(len(w) - L, b)
    if b < a:
        return None
    ok = w[w <= 3]
    if bg is None:
        if len(ok) < 40:
            return None
        bg = np.array([(ok == k).mean() for k in range(4)])
        bg = np.maximum(bg, 0.02)
        bg = bg / bg.sum()
    lm = np.log2(np.maximum(mat, 1e-4) / bg[None, :])
    idx = np.arange(L)
    best = None
    for s in range(a, b + 1):
        x = w[s:s + L]
        if (x > 3).any():
            continue
        v = float(lm[idx, x].sum())
        if best is None or v > best:
            best = v
    return best


def cmd_boxes(_):
    """Is the defining box actually there? Score it at its written window in true members,
    in members of the other classes, and in the same transcripts with their bases shuffled."""
    held = qso.load('held')
    rng = np.random.default_rng(7)
    out = {}
    for cls, spec in O.SO_RNA_CLASSES.items():
        for mat_name, where, window in spec.get('boxes', ()):
            lm = _logodds(O.matrix(mat_name))
            mine, other, shuf = [], [], []
            for r in held:
                v = qso._box(r['arr'], lm, qso.WHERE[where], window)
                if v <= -8.0:
                    continue
                (mine if r['so'] == cls else other).append(v)
                if r['so'] == cls:
                    a = r['arr'].copy()
                    rng.shuffle(a)
                    w = qso._box(a, lm, qso.WHERE[where], window)
                    if w > -8.0:
                        shuf.append(w)
            if len(mine) < 10 or len(shuf) < 10:
                continue
            mine, other, shuf = map(np.array, (mine, other, shuf))
            m = min(len(mine), len(shuf))
            y = np.r_[np.ones(m, bool), np.zeros(m, bool)]
            k = min(len(mine), len(other))
            yo = np.r_[np.ones(k, bool), np.zeros(k, bool)]
            out[f'{cls}/{mat_name}'] = dict(
                n_members=int(len(mine)),
                mean_in_members=round(float(mine.mean()), 3),
                mean_in_shuffled=round(float(shuf.mean()), 3),
                mean_in_other_classes=round(float(other.mean()), 3),
                auroc_vs_shuffled=round(auroc(y, np.r_[mine[:m], shuf[:m]]), 4),
                auroc_vs_other_classes=round(auroc(yo, np.r_[mine[:k], other[:k]]), 4))
            o = out[f'{cls}/{mat_name}']
            print(f'  {cls:22s} {mat_name:12s} members {o["mean_in_members"]:+.2f} '
                  f'shuffled {o["mean_in_shuffled"]:+.2f} other {o["mean_in_other_classes"]:+.2f}'
                  f'  AUROC vs shuffled {o["auroc_vs_shuffled"]:.3f}'
                  f'  vs others {o["auroc_vs_other_classes"]:.3f}', flush=True)
    save({'boxes': out})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('task', nargs='?', default='all',
                    choices=['all', 'classify', 'minor', 'cis', 'boxes'])
    a = ap.parse_args()
    for t in (['classify', 'boxes', 'minor', 'cis'] if a.task == 'all' else [a.task]):
        print(f'== {t}', flush=True)
        globals()[f'cmd_{t}'](a)
