"""Turn counts into parameters.

Backbone: interpolated absolute discounting over orders 0..K, with the discount
per order taken from that order's count-of-counts (Ney et al.). The whole chain
is baked into one table per order, then log-quantised to a byte per base.
No gradients anywhere; the only iterative step is EM over the ~1k mixture
weights, in qmix.py.
"""
import json, sys, time
import numpy as np
from pathlib import Path

T = Path('/home/claude/data/tables')
QSTEP = 16.0          # bytes hold round(-QSTEP*log2 p), so 1/16 bit resolution
QMAX = 255


def quant(p):
    """Probabilities -> log-quantised bytes."""
    p = np.asarray(p, np.float64)
    p = np.clip(p, 2.0 ** (-QMAX / QSTEP), 1.0)
    return np.round(-QSTEP * np.log2(p)).clip(0, QMAX).astype(np.uint8)


def dequant_table():
    return (2.0 ** (-np.arange(256) / QSTEP)).astype(np.float32)


def discount(cnt):
    """D = n1/(n1+2 n2) from the count-of-counts of a single order."""
    n1 = int((cnt == 1).sum()); n2 = int((cnt == 2).sum())
    return n1 / (n1 + 2 * n2) if n1 + n2 else 0.5


def bake_backbone(kmax, want, block=1 << 21):
    """Bake the interpolated distribution for every context at each order in `want`."""
    counts = np.load(T / 'exact_counts.npy', mmap_mode='r')
    offsets = np.load(T / 'exact_offsets.npy')
    out = {}
    uni = np.asarray(counts[offsets[1]:offsets[1] + 16], np.float64).reshape(4, 4).sum(0)
    prev = (uni / uni.sum())[None, :]                     # order 0
    out[0] = prev.copy()
    D = {}
    for k in range(1, kmax + 1):
        n = 4 ** k
        cnt_all = counts[offsets[k]:offsets[k] + n * 4]
        D[k] = discount(np.asarray(cnt_all[:min(len(cnt_all), 1 << 26)]))
        cur = np.empty((n, 4), np.float32)
        lowmask = 4 ** (k - 1)
        for s in range(0, n, block):
            e = min(n, s + block)
            c = np.asarray(cnt_all[s * 4:e * 4], np.float64).reshape(-1, 4)
            N = c.sum(1); u = (c > 0).sum(1)
            low = prev[(np.arange(s, e) % lowmask) if k > 1 else 0]
            if k == 1:
                low = np.repeat(prev, e - s, axis=0)
            num = np.maximum(c - D[k], 0.0) + (D[k] * u)[:, None] * low
            with np.errstate(invalid='ignore', divide='ignore'):
                p = np.where(N[:, None] > 0, num / np.maximum(N, 1)[:, None], low)
            cur[s:e] = p / p.sum(1, keepdims=True)
        prev = cur
        if k in want:
            out[k] = cur
        print(f'  order {k}: D={D[k]:.4f} baked {n} contexts', flush=True)
        if k == max(want):
            break
    return out, D


def fold_hash(hc, bits):
    """Collapse 2^22 hash buckets down to 2^bits by summing (the low bits are themselves a hash)."""
    n = 1 << bits
    return hc.reshape(n, -1, 4).sum(1)


def repeat_dict(hc, keep_frac=None, min_count=4096):
    """Prune the hashed long-context table to buckets dominated by a frequent context."""
    N = hc.sum(1)
    sel = np.flatnonzero(N >= min_count)
    p = hc[sel].astype(np.float64)
    p = (p + 0.2) / (p.sum(1, keepdims=True) + 0.8)
    conf = np.clip(np.log2(N[sel] / min_count) / 2, 0, 3).astype(np.uint8)
    return sel.astype(np.int64), quant(p), conf, N[sel]


def main():
    want = set(range(13))
    t0 = time.time()
    baked, D = bake_backbone(12, want)
    T.joinpath('baked').mkdir(exist_ok=True)
    for k, p in baked.items():
        np.save(T / 'baked' / f'bb{k}.npy', quant(p))
    json.dump({str(k): v for k, v in D.items()}, open(T / 'baked' / 'discounts.json', 'w'), indent=1)
    print(f'backbone baked in {time.time()-t0:.0f}s', flush=True)

    hc = np.load(T / 'hash_counts.npy')
    meta = json.load(open(T / 'count_meta.json'))
    for j, order in enumerate(meta['hash_orders']):
        N = hc[j].sum(1)
        qs = np.percentile(N, [50, 90, 99, 99.9, 99.99])
        print(f'order {order} hashed: mean N={N.mean():.0f} pct50/90/99/99.9/99.99={qs.astype(int)}',
              flush=True)
        for mc in (1024, 4096, 16384, 65536):
            k, q, c, n = repeat_dict(hc[j], min_count=mc)
            print(f'   min_count={mc:6d}: {len(k):8d} buckets, {n.sum()/N.sum():.3%} of mass, '
                  f'{len(k)*6/1e6:.1f} MB', flush=True)


if __name__ == '__main__':
    main()
