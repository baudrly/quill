"""Backbone-only bits/base on held-out sequence: the context-model scaling curve."""
import sys, numpy as np
from numba import njit
from pathlib import Path
T = Path('/home/claude/data/tables')
DQ = (2.0 ** (-np.arange(256) / 16.0)).astype(np.float64)

@njit(cache=True)
def bits(seq, tabs, offs, orders, dq, nmax):
    """-log2 p per base, using the highest order whose context is fully known."""
    tot = np.zeros(len(orders)); cnt = 0
    ctx = np.uint64(0); run = 0
    for i in range(len(seq)):
        b = seq[i]
        if b > 3:
            run = 0; ctx = np.uint64(0); continue
        if run >= nmax:
            cnt += 1
            for j in range(len(orders)):
                k = orders[j]
                use = k if run >= k else run
                while use > 0 and offs[use] < 0:
                    use -= 1
                m = (np.uint64(1) << np.uint64(2 * use)) - np.uint64(1)
                base = offs[use] + ((ctx & m) * np.uint64(4))
                s = 0.0
                for x in range(4):
                    s += dq[tabs[base + np.uint64(x)]]
                p = dq[tabs[base + np.uint64(b)]] / s
                tot[j] += -np.log2(p)
        ctx = (ctx << np.uint64(2)) | np.uint64(b); run += 1
    return tot, cnt

def load(orders):
    offs = np.full(13, -1, np.int64); parts = []; cur = 0
    for k in range(13):
        f = T / 'baked' / f'bb{k}.npy'
        if f.exists():
            a = np.load(f).reshape(-1)
            offs[k] = cur; parts.append(a); cur += len(a)
    return np.concatenate(parts), offs

if __name__ == '__main__':
    orders = np.array([2, 4, 6, 8, 9, 10, 11, 12], np.int64)
    tabs, offs = load(orders)
    print(f'table bytes total {len(tabs)/1e6:.1f} MB')
    for name, path in (('human chr20 (held out)', '/home/claude/data/hs/20.npy'),
                       ('human chr21 (held out)', '/home/claude/data/hs/21.npy'),
                       ('mouse chr19 (held out species)', '/home/claude/data/mm/19.npy')):
        seq = np.load(path)
        t, n = bits(seq, tabs, offs, orders, DQ, 12)
        print(f'{name}: {n/1e6:.1f} Mb scored')
        for k, v in zip(orders, t):
            print(f'   order {k:2d}  params {4**k*4/1e6:9.3f}M   {v/n:.4f} bits/base')
