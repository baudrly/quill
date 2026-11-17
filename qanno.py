"""Pass 2: everything the grammar needs, measured from annotation + sequence.

Per chromosome it builds an oriented label track (a forward view and a
reverse-complement view) and accumulates:
  * class-conditional order-0/1/2 emission counts, conditioned on local GC bin
  * codon usage and codon-pair counts
  * splice donor / acceptor / Kozak / stop matrices and a 3'-end hexamer scan
  * region length histograms, which become the HMM durations
Label ids (also the emission classes):
  0 IG  1 CGI  2 UTR5  3 CDS0  4 CDS1  5 CDS2  6 UTR3  7 INTRON  8 NCRNA  9 TANDEM
255 = masked (a feature of the opposite strand, or ambiguous)
"""
import json, pickle, sys, time
import numpy as np
from numba import njit
from pathlib import Path

HS = Path('/home/claude/data/hs')
OUT = Path('/home/claude/data/tables'); OUT.mkdir(exist_ok=True)
NCLS, NGC = 10, 3
IG, CGI, UTR5, CDS0, UTR3, INTRON, NCRNA, TANDEM, MASK = 0, 1, 2, 3, 6, 7, 8, 9, 255
GC_EDGES = (0.42, 0.52)
GC_WIN = 1024
DON = (3, 8)        # last exon bases | first intron bases
ACC = (20, 3)       # last intron bases | first exon bases
KOZ = (8, 4)        # before ATG | ATG + 1
STP = (0, 9)        # stop codon + 6 downstream
PAS_W = 300
KEEP_LABELS = {'17', '19', '20', '21', '22', '1', '2'}
TRAIN = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
         '13', '14', '15', '16', '18', 'X', 'Y', 'MT']
HELD = ['17', '19', '20', '21', '22']


@njit(cache=True)
def revcomp(seq):
    n = len(seq)
    out = np.empty(n, np.uint8)
    for i in range(n):
        b = seq[n - 1 - i]
        out[i] = b if b > 3 else 3 - b
    return out


@njit(cache=True)
def gc_bins(seq, win, e0, e1):
    """Per-base GC bin of a centred window, two-pointer running counts."""
    n = len(seq)
    out = np.zeros(n, np.uint8)
    gc = 0; tot = 0; lo = 0; hi = 0
    for i in range(n):
        wl = i - win
        if wl < 0:
            wl = 0
        wh = i                       # trailing window only: keeps the model causal
        if wh > n:
            wh = n
        while hi < wh:
            b = seq[hi]
            if b < 4:
                tot += 1
                if b == 1 or b == 2:
                    gc += 1
            hi += 1
        while lo < wl:
            b = seq[lo]
            if b < 4:
                tot -= 1
                if b == 1 or b == 2:
                    gc -= 1
            lo += 1
        f = gc / tot if tot > 50 else 0.42
        out[i] = 0 if f < e0 else (1 if f < e1 else 2)
    return out


@njit(cache=True)
def find_cgi(seq, step=25, w=200, minlen=200, gc_min=0.5, oe_min=0.6):
    """Gardiner-Garden CpG islands: windows passing GC% and obs/exp CpG, merged, length filtered."""
    n = len(seq)
    mark = np.zeros(n, np.uint8)
    i = 0
    while i + w <= n:
        c = 0; g = 0; cg = 0; valid = 0
        for j in range(i, i + w):
            b = seq[j]
            if b > 3:
                continue
            valid += 1
            if b == 1:
                c += 1
                if j + 1 < n and seq[j + 1] == 2:
                    cg += 1
            elif b == 2:
                g += 1
        if valid > 150:
            exp = c * g / valid
            if (c + g) / valid >= gc_min and exp > 0 and cg / exp >= oe_min:
                for j in range(i, i + w):
                    mark[j] = 1
        i += step
    i = 0
    while i < n:
        if mark[i]:
            j = i
            while j < n and mark[j]:
                j += 1
            if j - i < minlen:
                for k in range(i, j):
                    mark[k] = 0
            i = j
        else:
            i += 1
    return mark


@njit(cache=True)
def find_tandem(seq, span=24, need=22):
    """Period 1..6 tandem repeats, O(6n) with a running match count per period."""
    n = len(seq)
    mark = np.zeros(n, np.uint8)
    for p in range(1, 7):
        lim = n - p
        if lim <= span:
            continue
        cnt = 0
        for j in range(span):
            a = seq[j]; b = seq[j + p]
            if a < 4 and a == b:
                cnt += 1
        for i in range(lim - span):
            if cnt >= need:
                for j in range(span + p):
                    mark[i + j] = 1
            a = seq[i]; b = seq[i + p]
            if a < 4 and a == b:
                cnt -= 1
            a = seq[i + span]; b = seq[i + span + p]
            if a < 4 and a == b:
                cnt += 1
    return mark


@njit(cache=True)
def count_classes(seq, lab, gcb, cls2, cls1, cls0, runs):
    n = len(seq)
    prev = 255; start = 0
    for i in range(n):
        c = lab[i]
        if c != prev:
            if prev < NCLS:
                L = i - start
                runs[prev, L if L < 4095 else 4095] += 1
            prev = c; start = i
        if c >= NCLS:
            continue
        b = seq[i]
        if b > 3:
            continue
        g = gcb[i]
        cls0[c, g, b] += 1
        if i >= 1 and seq[i - 1] < 4:
            cls1[c, g, seq[i - 1], b] += 1
            if i >= 2 and seq[i - 2] < 4:
                cls2[c, g, seq[i - 2] * 4 + seq[i - 1], b] += 1


def oriented(seq, segs, strand):
    parts = [seq[a:b] for a, b in segs]
    if not parts:
        return np.empty(0, np.uint8)
    if strand == '-':
        return np.concatenate([revcomp(p) for p in reversed(parts)])
    return np.concatenate(parts)


def win(seq, cut, before, after, strand):
    """Window spanning `before` bases up to the cut and `after` past it, transcript-oriented."""
    n = len(seq)
    if strand == '+':
        a, b = cut - before, cut + after
        return seq[a:b] if 0 <= a and b <= n else None
    a, b = cut - after, cut + before
    return revcomp(seq[a:b]) if 0 <= a and b <= n else None


def add_pwm(acc, w):
    if w is None or len(w) != len(acc) or (w > 3).any():
        return
    acc[np.arange(len(w)), w] += 1


def transcripts(meta):
    bycds = {}
    for tid, strand, segs, gene, prot in meta['cds']:
        bycds.setdefault(tid, (segs, prot))
    for tid, (kind, strand, segs, gene) in meta['tx'].items():
        cds = bycds.get(tid)
        yield tid, kind, strand, segs, (cds[0] if cds else None), (cds[1] if cds else '')


def build_labels(seq, meta, cgi, tand):
    """Returns (labels for + strand features, labels for - strand features in revcomp coords)."""
    n = len(seq)
    tracks = {'+': np.full(n, IG, np.uint8), '-': np.full(n, IG, np.uint8)}
    txs = list(transcripts(meta))
    # 1. gene bodies -> INTRON
    for tid, kind, strand, exons, cds, prot in txs:
        tracks[strand][exons[0][0]:exons[-1][1]] = INTRON
    # 2. sequence features win over IG and INTRON
    for t in tracks.values():
        t[cgi == 1] = CGI
        t[tand == 1] = TANDEM
    # 3. exons: NCRNA, then UTR, then CDS with spliced phase
    for tid, kind, strand, exons, cds, prot in txs:
        t = tracks[strand]
        if cds is None:
            for a, b in exons:
                t[a:b] = NCRNA
            continue
        cl, ch = cds[0][0], cds[-1][1]
        for a, b in exons:
            pos = np.arange(a, b)
            t[a:b] = np.where(pos < cl, UTR5, UTR3) if strand == '+' else \
                     np.where(pos >= ch, UTR5, UTR3)
        pos = 0
        for a, b in (cds if strand == '+' else list(reversed(cds))):
            L = b - a
            ph = np.arange(pos, pos + L) % 3
            t[a:b] = (CDS0 + (ph[::-1] if strand == '-' else ph)).astype(np.uint8)
            pos += L
    # 4. mask the opposite strand's gene bodies
    for gid, (locus, strand, gs, ge) in meta['genes'].items():
        tracks['-' if strand == '+' else '+'][gs:ge] = MASK
    return tracks['+'], tracks['-'][::-1].copy()


def main(which):
    names = {'train': TRAIN, 'held': HELD}[which]
    cls2 = np.zeros((NCLS, NGC, 16, 4), np.int64)
    cls1 = np.zeros((NCLS, NGC, 4, 4), np.int64)
    cls0 = np.zeros((NCLS, NGC, 4), np.int64)
    runs = np.zeros((NCLS, 4096), np.int64)
    codon = np.zeros(64, np.int64)
    codon_pair = np.zeros((65, 64), np.int64)
    pwm = {k: np.zeros((a + b, 4), np.int64) for k, (a, b) in
           (('donor', DON), ('acceptor', ACC), ('kozak', KOZ), ('stop', STP))}
    exlen = np.zeros(4096, np.int64); inlen = np.zeros(1 << 19, np.int64)
    utr5len = np.zeros(4096, np.int64); utr3len = np.zeros(16384, np.int64)
    nexon = np.zeros(128, np.int64); hexam = np.zeros((4096, PAS_W), np.int64)
    stats = dict(nbase=0, ncgi=0, ntand=0, ngene=0, ntx=0, ncds=0, nintron=0, nexon_tot=0)

    for name in names:
        t0 = time.time()
        seq = np.load(HS / f'{name}.npy')
        meta = pickle.load(open(HS / f'{name}.pkl', 'rb'))
        cgi = find_cgi(seq); tand = find_tandem(seq); gcb = gc_bins(seq, GC_WIN, *GC_EDGES)
        fwd, rev = build_labels(seq, meta, cgi, tand)
        rseq = revcomp(seq); rgcb = gcb[::-1].copy()
        count_classes(seq, fwd, gcb, cls2, cls1, cls0, runs)
        count_classes(rseq, rev, rgcb, cls2, cls1, cls0, runs)
        if name in KEEP_LABELS:
            np.save(OUT / f'lab_{name}.npy', fwd); np.save(OUT / f'labr_{name}.npy', rev)
        stats['nbase'] += int((seq < 4).sum()); stats['ncgi'] += int(cgi.sum())
        stats['ntand'] += int(tand.sum()); stats['ngene'] += len(meta['genes'])

        for tid, kind, strand, exons, cds, prot in transcripts(meta):
            stats['ntx'] += 1; stats['nexon_tot'] += len(exons)
            nexon[min(len(exons), 127)] += 1
            for (a, b), (c, d) in zip(exons[:-1], exons[1:]):
                stats['nintron'] += 1
                inlen[min(c - b, (1 << 19) - 1)] += 1
                if strand == '+':
                    add_pwm(pwm['donor'], win(seq, b, *DON, '+'))
                    add_pwm(pwm['acceptor'], win(seq, c, *ACC, '+'))
                else:
                    add_pwm(pwm['donor'], win(seq, c, *DON, '-'))
                    add_pwm(pwm['acceptor'], win(seq, b, *ACC, '-'))
            if cds is None:
                continue
            stats['ncds'] += 1
            cseq = oriented(seq, cds, strand)
            if len(cseq) >= 6 and not (cseq > 3).any():
                cod = cseq[:len(cseq) // 3 * 3].reshape(-1, 3)
                idx = cod[:, 0] * 16 + cod[:, 1] * 4 + cod[:, 2]
                np.add.at(codon, idx, 1)
                np.add.at(codon_pair, (np.concatenate(([64], idx[:-1])), idx), 1)
            for a, b in cds:
                exlen[min(b - a, 4095)] += 1
            cl, ch = cds[0][0], cds[-1][1]
            starts_atg = len(cseq) >= 3 and cseq[0] == 0 and cseq[1] == 3 and cseq[2] == 2
            lastcod = -1
            if len(cseq) >= 3 and not (cseq[-3:] > 3).any():
                lastcod = int(cseq[-3]) * 16 + int(cseq[-2]) * 4 + int(cseq[-1])
            if starts_atg:
                add_pwm(pwm['kozak'], win(seq, cl if strand == '+' else ch, *KOZ, strand))
            if lastcod in (48, 56, 50):
                add_pwm(pwm['stop'], win(seq, ch - 3 if strand == '+' else cl + 3, *STP, strand))
            stats['atg_ok'] = stats.get('atg_ok', 0) + int(starts_atg)
            stats['stop_ok'] = stats.get('stop_ok', 0) + int(lastcod in (48, 56, 50))
            if strand == '+':
                n5 = sum(max(0, min(b, cl) - a) for a, b in exons)
                n3 = sum(max(0, b - max(a, ch)) for a, b in exons)
            else:
                n5 = sum(max(0, b - max(a, ch)) for a, b in exons)
                n3 = sum(max(0, min(b, cl) - a) for a, b in exons)
            utr5len[min(n5, 4095)] += 1; utr3len[min(n3, 16383)] += 1
            if n3 >= 12:
                ex = oriented(seq, exons, strand)
                tail = ex[-min(n3, PAS_W):]
                if len(tail) >= 6 and not (tail > 3).any():
                    t32 = tail.astype(np.int32)
                    k = (t32[:-5] * 1024 + t32[1:-4] * 256 + t32[2:-3] * 64 +
                         t32[3:-2] * 16 + t32[4:-1] * 4 + t32[5:])
                    np.add.at(hexam, (k, np.arange(len(k)) + PAS_W - len(tail)), 1)
        print(f'chr{name} {time.time()-t0:.0f}s', flush=True)
        del seq, rseq, fwd, rev, cgi, tand, gcb, rgcb

    np.savez_compressed(OUT / f'anno_{which}.npz', cls2=cls2, cls1=cls1, cls0=cls0, runs=runs,
                        codon=codon, codon_pair=codon_pair, exlen=exlen, inlen=inlen,
                        utr5len=utr5len, utr3len=utr3len, nexon=nexon, hexam=hexam,
                        **{f'pwm_{k}': v for k, v in pwm.items()})
    (OUT / f'anno_{which}.json').write_text(json.dumps(stats, indent=1))
    print(stats)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'train')
