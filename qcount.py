"""Pass 1: context counts over the training chromosomes, both strands.

Produces exact n-gram count tables for orders 1..KMAX and hashed tables for
long contexts. Nothing here touches a held-out chromosome.
"""
import sys, time
import numpy as np
from numba import njit
from pathlib import Path

HS = Path('/home/claude/data/hs')
OUT = Path('/home/claude/data/tables'); OUT.mkdir(exist_ok=True)

TRAIN = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
         '13', '14', '15', '16', '18', 'X', 'Y', 'MT']
HELD = ['17', '19', '20', '21', '22']          # 17 carries BRCA1; 19-22 are the LM test
KMAX = 12
HASH_ORDERS = (16, 24)
HASH_BITS = 22
MIX = np.uint64(0x9E3779B97F4A7C15)


@njit(cache=True)
def count_pass(seq, tables, offsets, kmax, hcounts, horders, hbits):
    """Accumulate counts for every exact order 1..kmax and each hashed order."""
    ctx = np.uint64(0)
    run = 0
    nh = len(horders)
    hmask = (np.uint64(1) << np.uint64(hbits)) - np.uint64(1)
    mix = np.uint64(0x9E3779B97F4A7C15)
    for i in range(len(seq)):
        b = seq[i]
        if b > 3:
            run = 0
            ctx = np.uint64(0)
            continue
        # counts use the context *before* this base
        for k in range(1, kmax + 1):
            if run >= k:
                m = (np.uint64(1) << np.uint64(2 * k)) - np.uint64(1)
                c = ctx & m
                tables[offsets[k] + c * np.uint64(4) + np.uint64(b)] += 1
        for j in range(nh):
            k = horders[j]
            if run >= k:
                m = (np.uint64(1) << np.uint64(2 * k)) - np.uint64(1)
                h = ((ctx & m) + np.uint64(1)) * mix
                h = (h ^ (h >> np.uint64(29))) * mix
                bucket = (h >> np.uint64(33)) & hmask
                hcounts[j, bucket, b] += 1
        ctx = (ctx << np.uint64(2)) | np.uint64(b)
        run += 1


@njit(cache=True)
def revcomp(seq):
    n = len(seq)
    out = np.empty(n, np.uint8)
    for i in range(n):
        b = seq[n - 1 - i]
        out[i] = b if b > 3 else 3 - b
    return out


def main():
    offsets = np.zeros(KMAX + 2, np.uint64)
    tot = 0
    for k in range(1, KMAX + 1):
        offsets[k] = tot
        tot += 4 ** k * 4
    print(f'exact table cells: {tot/1e6:.1f}M ({tot*4/2**30:.2f} GiB)', flush=True)
    tables = np.zeros(tot, np.uint32)
    horders = np.array(HASH_ORDERS, np.int64)
    hcounts = np.zeros((len(HASH_ORDERS), 2 ** HASH_BITS, 4), np.uint32)

    nbase = 0
    for name in TRAIN:
        seq = np.load(HS / f'{name}.npy')
        t0 = time.time()
        count_pass(seq, tables, offsets, KMAX, hcounts, horders, HASH_BITS)
        rc = revcomp(seq)
        count_pass(rc, tables, offsets, KMAX, hcounts, horders, HASH_BITS)
        v = int((seq < 4).sum())
        nbase += v
        print(f'chr{name}: {len(seq)/1e6:.1f} Mb, {v/1e6:.1f} Mb valid, {time.time()-t0:.0f}s', flush=True)
        del seq, rc

    np.save(OUT / 'exact_counts.npy', tables)
    np.save(OUT / 'exact_offsets.npy', offsets)
    np.save(OUT / 'hash_counts.npy', hcounts)
    meta = dict(train=TRAIN, held=HELD, kmax=KMAX, hash_orders=list(HASH_ORDERS),
                hash_bits=HASH_BITS, nbase=nbase)
    import json
    (OUT / 'count_meta.json').write_text(json.dumps(meta, indent=1))
    print('valid training bases (one strand):', nbase, flush=True)


if __name__ == '__main__':
    main()
