"""Describe a gene at the DNA level: architecture, promoter, genomic neighbourhood.

The third layer of the central dogma for the function experts. For every annotated gene:
  architecture   gene span, intron total, longest and first intron, number of annotated
                 transcripts, coding exons, intron GC
  promoter       300 bp upstream to 100 bp downstream of the transcription start: GC,
                 CpG observed/expected, the written TATA box in its window, the 33 written
                 TF sites of regulatory.py, and JASPAR 2024 matrices (one per factor)
                 scoring at least 90% of their maximum
  neighbourhood  local gene density, distance to the nearest gene, a head-to-head
                 (bidirectional) promoter, 100 kb isochore GC, and whether a neighbouring
                 gene encodes a similar protein (a tandem duplicate) - all label-free: no
                 neighbour's annotation is read

    python3 qdna.py cache      # -> data/gox/dna.pkl
"""
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import ontology as O
import qgox
import qnongenic as NG
import regulatory as R

HS = Path('/home/claude/data/hs')
CACHE = Path('/home/claude/data/gox/dna.pkl')
UP, DOWN = 300, 100
SITE_FRACTION = 0.90
BINS = dict(span=[5e3, 2e4, 5e4, 1.5e5, 5e5], intron_total=[2e3, 1e4, 4e4, 1.5e5],
            longest_intron=[1e3, 5e3, 2e4, 1e5], first_intron=[500, 2e3, 1e4, 5e4],
            transcripts=[2, 4, 8, 15], promoter_gc=[0.45, 0.55, 0.65, 0.72],
            promoter_cpg=[0.3, 0.5, 0.75, 0.95], isochore_gc=[0.38, 0.42, 0.46, 0.52],
            density=[4, 8, 16, 30], nearest=[2e3, 1e4, 5e4, 2e5], intron_gc=[0.38, 0.44, 0.5])


def _bin(v, e):
    return int(np.searchsorted(e, v, side='right'))


def rc(a):
    a = a[::-1]
    return np.where(a <= 3, 3 - a, a)


def cpg_oe(w):
    ok = w[w <= 3]
    if len(ok) < 50:
        return 0.0, 0.5
    c, g = (ok == 1).mean(), (ok == 2).mean()
    cg = ((w[:-1] == 1) & (w[1:] == 2)).mean()
    return float(cg / max(c * g, 1e-6)), float(c + g)


def main():
    sets = qgox.load_sets()
    want, prot_of = {}, {}
    for s in ('train', 'calib', 'held'):
        for r in sets[s]:
            want.setdefault(r['chrom'], set()).add(r['gene'])
            prot_of[r['gene']] = r['prot']
    J = NG.jaspar()
    best = {}
    for n, fam, lo in J:
        ic = float(lo.max(1).sum())
        if n not in best or ic > best[n][1]:
            best[n] = (lo, ic)
    jnames = sorted(best)
    jbank = NG.motif_bank([best[n][0] for n in jnames])
    jmax = np.array([best[n][1] for n in jnames])
    wnames = list(R.TF_SITES)
    wm = [np.log2(np.asarray(R.matrix(s)) / 0.25) for s in R.TF_SITES.values()]
    wbank = NG.motif_bank(wm)
    wmax = np.array([float(m.max(1).sum()) for m in wm])
    tata = np.log2(np.asarray(O.matrix('TATA')) / 0.25)
    out = {}
    for c, genes in sorted(want.items()):
        h = pickle.load(open(HS / f'{c}.pkl', 'rb'))
        seq = np.load(HS / f'{c}.npy')
        ntx, tx_of = {}, {}
        for tid, (kind, strand, exons, gid) in h['tx'].items():
            ntx[gid] = ntx.get(gid, 0) + 1
        for tid, strand, segs, gid, prot in h['cds']:
            if gid in genes and tid in h['tx'] and (gid not in tx_of or
                                                     len(prot) > tx_of[gid][2]):
                tx_of[gid] = (tid, strand, len(prot))
        # every gene on the chromosome, annotated or not, as landmarks
        marks = sorted((a, b, st, gid) for gid, (nm, st, a, b) in h['genes'].items())
        starts = np.array([m[0] for m in marks])
        order = {m[3]: i for i, m in enumerate(marks)}
        f5 = {g: set(qgox.exact5(p).tolist()) for g, p in prot_of.items() if g in genes}
        for gid in genes:
            if gid not in tx_of or gid not in h['genes']:
                continue
            tid, strand, _ = tx_of[gid]
            exons = sorted(h['tx'][tid][2])
            nm, st, ga, gb = h['genes'][gid]
            introns = [(exons[i][1], exons[i + 1][0]) for i in range(len(exons) - 1)]
            il = [b - a for a, b in introns]
            first = (il[0] if strand == '+' else il[-1]) if il else 0
            tss = exons[0][0] if strand == '+' else exons[-1][1] - 1
            lo, hi = (tss - UP, tss + DOWN) if strand == '+' else (tss - DOWN + 1,
                                                                   tss + UP + 1)
            if lo < 0 or hi > len(seq):
                continue
            w = np.asarray(seq[lo:hi], np.int8)
            if strand == '-':
                w = rc(w)
            oe, gc = cpg_oe(w)
            ig = np.concatenate([seq[a:b][:2000] for a, b in introns[:6]]) if introns \
                else np.zeros(0, np.int8)
            igc = float(((ig == 1) | (ig == 2)).mean()) if len(ig) else 0.45
            iso = seq[max(0, tss - 50000):tss + 50000]
            iso = iso[iso <= 3]
            isogc = float(((iso == 1) | (iso == 2)).mean()) if len(iso) else 0.42
            i = order[gid]
            dens = int(np.searchsorted(starts, tss + 500000) -
                       np.searchsorted(starts, tss - 500000))
            near = min([abs(marks[k][0] - gb) if k > i else abs(ga - marks[k][1])
                        for k in (i - 1, i + 1) if 0 <= k < len(marks)] or [1e6])
            hh = False
            for k in (i - 1, i + 1):
                if 0 <= k < len(marks) and marks[k][2] != strand:
                    otss = marks[k][0] if marks[k][2] == '+' else marks[k][1]
                    if abs(otss - tss) < 1000:
                        hh = True
            tandem = False
            mine = f5.get(gid, set())
            for k in range(max(0, i - 5), min(len(marks), i + 6)):
                og = marks[k][3]
                if og != gid and og in f5 and mine:
                    sh = len(mine & f5[og]) / max(min(len(mine), len(f5[og])), 1)
                    if sh >= 0.15:
                        tandem = True
            f = [f'dna:{k}:{_bin(v, BINS[k])}' for k, v in (
                ('span', gb - ga), ('intron_total', sum(il)),
                ('longest_intron', max(il) if il else 0), ('first_intron', first),
                ('transcripts', ntx.get(gid, 1)), ('promoter_gc', gc),
                ('promoter_cpg', oe), ('isochore_gc', isogc), ('density', dens),
                ('nearest', near), ('intron_gc', igc))]
            if not il:
                f.append('dna:intronless')
            if hh:
                f.append('dna:bidirectional_promoter')
            if tandem:
                f.append('dna:tandem_duplicate')
            idx = np.arange(len(tata))
            tb = max(float(tata[idx, w[s:s + len(tata)]].sum())
                     for s in range(UP - 40, UP - 20)
                     if (w[s:s + len(tata)] <= 3).all()) if (w <= 3).all() else -9
            if tb >= 0.8 * float(tata.max(1).sum()):
                f.append('dna:TATA_box')
            if (w <= 3).mean() > 0.95:
                ws = NG.motif_max(w, wbank, len(wnames)) / wmax
                f.extend(f'dna:site:{wnames[k]}' for k in np.flatnonzero(ws >= SITE_FRACTION))
                js = NG.motif_max(w, jbank, len(jnames)) / jmax
                f.extend(f'dna:jaspar:{jnames[k]}' for k in np.flatnonzero(js >= SITE_FRACTION))
            out[gid] = f
        del seq
        print(f'  chr{c}: {len(out)} genes described', flush=True)
    with open(CACHE, 'wb') as fh:
        pickle.dump(out, fh, protocol=4)
    print(f'{CACHE}: {len(out)} genes, {np.mean([len(v) for v in out.values()]):.1f} '
          f'features each')


def load_cache():
    with open(CACHE, 'rb') as fh:
        return pickle.load(fh)


if __name__ == '__main__':
    main()
