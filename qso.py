"""Sequence-Ontology layer: classify a transcript into an SO class from written motifs.

The small-RNA classes are defined by boxes at written positions (box C and D for C/D
snoRNAs, box H and ACA for H/ACA snoRNAs, the Sm site for spliceosomal snRNAs, the A and
B boxes for tRNAs), by a length, by a GC fraction, and for the hairpin classes by how
well the transcript pairs with itself. All of those numbers are written in ontology.py.

    python3 qso.py data     # assemble non-coding transcripts with their SO truth
    python3 qso.py build    # written and counted weight files
"""
import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import ontology as O
from qbuild import dequant_table, quant

HS = Path('/home/claude/data/hs')
MODELS = HERE / 'models'
DATA = Path('/home/claude/data/soset')
TRAIN = [str(i) for i in range(1, 17)] + ['18', 'X', 'Y', 'MT']
HELD = ['17', '19', '20', '21', '22']
CLASSES = list(O.SO_RNA_CLASSES)
WHERE = {'start': 0, 'middle': 1, 'end': 2}
# Written weights on the parts of an SO score. Not fitted.
SO_WEIGHTS = dict(boxes=1.0, length=1.0, gc=1.0, exons=0.6, hairpin=1.0)
SO_LEN_LOGSIGMA = 0.45
PAIR = {0: 3, 3: 0, 1: 2, 2: 1}


# ---------------------------------------------------------------- features
def hairpin_score(seq):
    """Longest self-pairing stem the transcript can form, as a fraction of its length."""
    n = len(seq)
    if n < 30:
        return 0.0
    best = 0
    for centre in range(8, n - 8):
        for gap in (3, 5, 7, 9, 11):
            i, j = centre - gap // 2 - 1, centre + gap // 2 + 1
            stem = mism = 0
            while i >= 0 and j < n:
                a, b = int(seq[i]), int(seq[j])
                if a <= 3 and b <= 3 and PAIR.get(a) == b:
                    stem += 1
                else:
                    mism += 1
                    if mism > max(2, stem // 5):
                        break
                i -= 1
                j += 1
            if stem > best:
                best = stem
    return best / n


def features(seq, n_exons):
    """Everything the written SO models read off one transcript."""
    ok = seq[seq <= 3]
    gc = float(np.mean((ok == 1) | (ok == 2))) if len(ok) else 0.5
    return dict(n=len(seq), gc=gc, exons=n_exons, hairpin=hairpin_score(seq))


# ---------------------------------------------------------------- the model
class SoModel:
    """Written or counted parameters for every SO class, and the score of a transcript."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=True)
        dq = dequant_table()
        self.names = json.loads(str(z['names']))
        self.boxes = json.loads(str(z['boxes']))
        self.mats = {k: np.log2(np.maximum(dq[z[f'mat_{k}']], 1e-4) / 0.25)
                     for k in json.loads(str(z['mat_index']))}
        self.len_mean = z['len_mean'].astype(np.float64)
        self.gc = z['gc'].astype(np.float64)
        self.exons = z['exons'].astype(np.float64)
        self.hairpin = z['hairpin'].astype(np.float64)
        self.prior = np.log2(np.maximum(z['prior'].astype(np.float64), 1e-9))
        self.nparam = int(sum(z[f'mat_{k}'].size for k in self.mats) +
                          self.len_mean.size + self.gc.size + self.exons.size +
                          self.hairpin.size * 2 + self.prior.size)

    def score(self, seq, n_exons):
        f = features(seq, n_exons)
        out = np.zeros(len(self.names))
        for i, nm in enumerate(self.names):
            s = self.prior[i]
            for mat_name, where, window in self.boxes[nm]:
                m = self.mats[mat_name]
                s += SO_WEIGHTS['boxes'] * _box(seq, m, WHERE[where], window)
            z = (np.log(max(f['n'], 1)) - np.log(self.len_mean[i])) / SO_LEN_LOGSIGMA
            s += SO_WEIGHTS['length'] * (-0.5 * z * z) / np.log(2.0)
            s += SO_WEIGHTS['gc'] * (-40.0 * (f['gc'] - self.gc[i]) ** 2)
            s += SO_WEIGHTS['exons'] * (-0.8 * abs(f['exons'] - self.exons[i]))
            s += SO_WEIGHTS['hairpin'] * (-18.0 * (f['hairpin'] - self.hairpin[i]) ** 2)
            out[i] = s
        return out

    def classify(self, seq, n_exons):
        s = self.score(seq, n_exons)
        return self.names[int(np.argmax(s))], s


def _box(seq, logmat, where, window):
    n, L = len(seq), logmat.shape[0]
    if n < L:
        return -8.0
    lo, hi = window
    if where == 0:
        a, b = lo, min(n - L, hi)
    elif where == 2:
        a, b = max(0, n - L - hi), max(0, n - L - lo)
    else:
        a, b = max(0, lo), min(n - L, hi)
    if b < a:
        a, b = 0, n - L
    best = -8.0
    idx = np.arange(L)
    for s in range(a, b + 1):
        w = seq[s:s + L]
        if len(w) < L or (w > 3).any():
            continue
        v = float(logmat[idx, w].sum())
        if v > best:
            best = v
    return best


# ---------------------------------------------------------------- build
def _mat_names():
    used = set()
    for spec in O.SO_RNA_CLASSES.values():
        for mat, _, _ in spec.get('boxes', ()):
            used.add(mat)
    return sorted(used)


def build_written():
    """Every number from ontology.py; no transcript is read."""
    mats = _mat_names()
    boxes = {k: [list(b) for b in O.SO_RNA_CLASSES[k].get('boxes', ())] for k in CLASSES}
    d = {f'mat_{m}': quant(np.array(O.matrix(m), np.float64)) for m in mats}
    d['mat_index'] = json.dumps({m: f'mat_{m}' for m in mats})
    d['names'] = json.dumps(CLASSES)
    d['boxes'] = json.dumps(boxes)
    d['len_mean'] = np.array([O.SO_RNA_CLASSES[k]['mean_len'] for k in CLASSES], np.float32)
    d['gc'] = np.array([O.SO_RNA_CLASSES[k]['gc'] for k in CLASSES], np.float32)
    d['exons'] = np.array([O.SO_RNA_CLASSES[k]['exons'] for k in CLASSES], np.float32)
    d['hairpin'] = np.array([
        (O.SO_RNA_CLASSES[k]['hairpin'][0] / O.SO_RNA_CLASSES[k]['mean_len'])
        if 'hairpin' in O.SO_RNA_CLASSES[k] else 0.12 for k in CLASSES], np.float32)
    # written share of the non-coding gene set each class takes
    share = dict(snoRNA_CD=.038, snoRNA_HACA=.043, snRNA=.18, pre_miRNA=.19,
                 tRNA=.004, Y_RNA=.006, srpRNA_7SL=.07, lincRNA=.19,
                 antisense_lncRNA=.13, processed_pseudogene=.15)
    v = np.array([share[k] for k in CLASSES], np.float64)
    d['prior'] = (v / v.sum()).astype(np.float32)
    return d


def build_counted(records):
    """The same shape, with lengths, GC, exon counts and hairpin means tallied instead."""
    d = build_written()
    by = {}
    for r in records:
        by.setdefault(r['so'], []).append(r)
    ln, gc, ex, hp, pri = [], [], [], [], []
    for k in CLASSES:
        rs = by.get(k, [])
        if rs:
            ln.append(float(np.mean([r['n'] for r in rs])))
            gc.append(float(np.mean([r['gc'] for r in rs])))
            ex.append(float(np.mean([r['exons'] for r in rs])))
            hp.append(float(np.mean([r['hairpin'] for r in rs])))
        else:
            i = CLASSES.index(k)
            ln.append(float(d['len_mean'][i])); gc.append(float(d['gc'][i]))
            ex.append(float(d['exons'][i])); hp.append(float(d['hairpin'][i]))
        pri.append(max(len(rs), 1))
    d['len_mean'] = np.array(ln, np.float32)
    d['gc'] = np.array(gc, np.float32)
    d['exons'] = np.array(ex, np.float32)
    d['hairpin'] = np.array(hp, np.float32)
    p = np.array(pri, np.float64)
    d['prior'] = (p / p.sum()).astype(np.float32)
    d['n_train'] = json.dumps({k: len(by.get(k, [])) for k in CLASSES})
    return d


# ---------------------------------------------------------------- dataset
def so_of(symbol):
    for name, spec in O.SO_RNA_CLASSES.items():
        if re.search(spec['symbol'], symbol):
            return name
    return None


def spliced(seq, exons, strand):
    parts = []
    for a, b in (exons if strand == '+' else list(reversed(exons))):
        s = np.asarray(seq[a:b], np.int8)
        if strand == '-':
            s = s[::-1]
            s = np.where(s <= 3, 3 - s, s)
        parts.append(s)
    return np.concatenate(parts) if parts else np.zeros(0, np.int8)


def assemble(chroms, out):
    recs = []
    for c in chroms:
        hp = HS / f'{c}.pkl'
        if not hp.exists():
            continue
        h = pickle.load(open(hp, 'rb'))
        seq = np.load(HS / f'{c}.npy')
        coding = {gid for _, _, _, gid, _ in h['cds']}
        best = {}
        for tid, (kind, strand, exons, gid) in h['tx'].items():
            L = sum(b - a for a, b in exons)
            if gid not in best or L > best[gid][2]:
                best[gid] = (strand, exons, L)
        for gid, (strand, exons, L) in best.items():
            if gid in coding:
                continue
            sym = h['genes'].get(gid, ('', '', 0, 0))[0]
            so = so_of(sym) if sym else None
            if so is None or L < 40 or L > 6000:
                continue
            s = spliced(seq, exons, strand)
            f = features(s, len(exons))
            recs.append(dict(gene=gid, symbol=sym, chrom=c, so=so,
                             seq=s.astype(np.int8).tobytes(), **f))
        del seq
        print(f'  chr{c}: {len(recs)} records so far', flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'wb') as fh:
        pickle.dump(recs, fh, protocol=4)
    return recs


def load(split):
    with open(DATA / f'{split}.pkl', 'rb') as fh:
        recs = pickle.load(fh)
    for r in recs:
        r['arr'] = np.frombuffer(r['seq'], np.int8)
    return recs


def cmd_data(_):
    import collections
    DATA.mkdir(parents=True, exist_ok=True)
    for split, chroms in (('train', TRAIN), ('held', HELD)):
        print(f'{split}:')
        recs = assemble(chroms, DATA / f'{split}.pkl')
        print(f'  {split}: {len(recs)} transcripts')
        print('  ', dict(collections.Counter(r['so'] for r in recs)))


def cmd_build(_):
    MODELS.mkdir(exist_ok=True)
    p = MODELS / 'quill-k-so.npz'
    np.savez_compressed(p, **build_written())
    m = SoModel(p)
    print(f'{p.name}: {m.nparam:,} parameters, {p.stat().st_size/1024:.1f} KB, '
          f'{len(m.names)} SO classes (written)')
    if (DATA / 'train.pkl').exists():
        recs = load('train')
        p = MODELS / 'quill-s-so.npz'
        np.savez_compressed(p, **build_counted(recs))
        m = SoModel(p)
        print(f'{p.name}: {m.nparam:,} parameters, {p.stat().st_size/1024:.1f} KB, '
              f'counted from {len(recs)} transcripts')
    else:
        print('no training transcripts yet: run `python3 qso.py data` first')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['data', 'build'])
    a = ap.parse_args()
    {'data': cmd_data, 'build': cmd_build}[a.cmd](a)
