"""Alignment memory: transfer function from remembered proteins by real sequence alignment.

No weight here is trained. The expert is three written rules around a published
substitution matrix:
  1. candidates   the 150 remembered proteins sharing the most 8-mers with the query in a
                  ten-letter reduced amino-acid alphabet (Murphy et al. 2000), so that
                  conservative substitutions do not break a seed
  2. alignment    Smith-Waterman local alignment of the query against each candidate with
                  BLOSUM62 and gap penalties 11/1 (parasail)
  3. transfer     a term's probability is the share of squared alignment score carried by
                  the hits annotated with it (top 30 hits scoring at least 100), times a
                  confidence 1 - exp(-r / sigma) in the best hit, r being its score relative
                  to the smaller of the two self-alignment scores

    python3 qalign.py cache      # align calibration and held-out proteins -> data/gox/align.pkl
"""
import hashlib
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import parasail

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import qgox

CACHE = Path('/home/claude/data/gox/align.pkl')
GROUPS = ('LVIM', 'C', 'A', 'G', 'ST', 'P', 'FYW', 'EDNQ', 'KR', 'H')
SEED_K, N_SEED = 8, 10 ** 8
CANDIDATES, KEEP_SCORE = 150, 40.0
MIN_SCORE, TOP_HITS, POWER = 100.0, 30, 2
GAP_OPEN, GAP_EXTEND = 11, 1
_RED = np.full(256, -1, np.int64)
for _i, _g in enumerate(GROUPS):
    _RED[np.frombuffer(_g.encode(), np.uint8)] = _i


def reduced8(prot):
    """Distinct 8-mers of the protein in the reduced alphabet."""
    a = _RED[np.frombuffer(prot.encode('ascii', 'replace'), np.uint8)]
    n = len(a) - SEED_K + 1
    if n <= 0:
        return np.zeros(0, np.int64)
    ok, code = np.ones(n, bool), np.zeros(n, np.int64)
    for j in range(SEED_K):
        ok &= a[j:n + j] >= 0
        code = code * 10 + np.maximum(a[j:n + j], 0)
    return np.unique(code[ok])


def clean(prot):
    return prot.replace('*', '').replace('U', 'C') or 'A'


def self_score(prot):
    p = clean(prot)
    return float(parasail.sw_striped_16(p, p, GAP_OPEN, GAP_EXTEND, parasail.blosum62).score)


def key(prot):
    return hashlib.md5(prot.encode()).hexdigest()


class Aligner:
    """The remembered proteins, their seed index and a persistent cache of alignments."""

    def __init__(self, prots, self_scores=None, cache=CACHE):
        self.prots = [clean(p) for p in prots]
        self.seeds = qgox.matrix(prots, reduced8, N_SEED).tocsr()
        self.self = np.array([self_score(p) for p in prots]) if self_scores is None \
            else np.asarray(self_scores, np.float64)
        self.path = cache
        self.memo = pickle.load(open(cache, 'rb')) if cache and Path(cache).exists() else {}
        self.fresh = 0

    def hits(self, prots):
        """Per query: its self score and [(remembered index, score)] with score >= 40."""
        todo = [i for i, p in enumerate(prots) if key(p) not in self.memo]
        for a in range(0, len(todo), 256):
            ch = [prots[i] for i in todo[a:a + 256]]
            S = (qgox.matrix(ch, reduced8, N_SEED).tocsr() @ self.seeds.T).toarray()
            c = min(CANDIDATES, S.shape[1] - 1)
            idx = np.argpartition(-S, c, axis=1)[:, :c]
            for i, p in enumerate(ch):
                q = clean(p)
                prof = parasail.profile_create_16(q, parasail.blosum62)
                row = [(int(j), float(s)) for j in idx[i] if S[i, j] > 0 for s in (
                    parasail.sw_striped_profile_16(prof, self.prots[j], GAP_OPEN,
                                                   GAP_EXTEND).score,) if s >= KEEP_SCORE]
                self.memo[key(p)] = (self_score(p), row)
            self.fresh += len(ch)
        return [self.memo[key(p)] for p in prots]

    def save(self):
        if self.path and self.fresh:
            with open(self.path, 'wb') as fh:
                pickle.dump(self.memo, fh, protocol=4)
            self.fresh = 0

    def ranked(self, prot):
        """One query's usable hits, best first: (index, score, relative score)."""
        sq, row = self.hits([prot])[0]
        row = sorted((h for h in row if h[1] >= MIN_SCORE), key=lambda h: -h[1])[:TOP_HITS]
        return [(j, s, s / min(sq, self.self[j])) for j, s in row]

    def transfer(self, prots, Y, sigma, exclude=None):
        """Queries x terms probabilities from the labels Y (CSR) of the remembered proteins.

        exclude: per query, a remembered index to ignore (leave-one-out on the memory)."""
        P = np.zeros((len(prots), Y.shape[1]), np.float32)
        for i, (sq, row) in enumerate(self.hits(prots)):
            if exclude is not None:
                row = [h for h in row if h[0] != exclude[i]]
            row = sorted((h for h in row if h[1] >= MIN_SCORE), key=lambda h: -h[1])[:TOP_HITS]
            if not row:
                continue
            w = np.array([s ** POWER for _, s in row])
            for (j, _), wi in zip(row, w):
                P[i, Y.indices[Y.indptr[j]:Y.indptr[j + 1]]] += wi
            best = max(s / min(sq, self.self[j]) for j, s in row)
            P[i] *= (1.0 - np.exp(-best / sigma)) / w.sum()
        return P


def main():
    sets = qgox.load_sets()
    t0 = time.time()
    al = Aligner([r['prot'] for r in sets['train']])
    print(f'seed index over {len(al.prots)} proteins in {time.time() - t0:.0f} s', flush=True)
    for s in ('calib', 'held'):
        t0 = time.time()
        h = al.hits([r['prot'] for r in sets[s]])
        al.save()
        n = [sum(1 for _, sc in row if sc >= MIN_SCORE) for _, row in h]
        print(f'{s}: {len(h)} queries in {time.time() - t0:.0f} s; '
              f'{np.mean(np.array(n) > 0):.3f} have a hit >= {MIN_SCORE:.0f}, '
              f'median {int(np.median(n))} hits', flush=True)
    with open(CACHE.with_name('align_self.pkl'), 'wb') as fh:
        pickle.dump(al.self, fh)


if __name__ == '__main__':
    main()
