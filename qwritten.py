"""Build quill-k: every weight written from knowledge.py, nothing counted.

The output .npz has exactly the layout qbench.Model expects, so the zero-data
model runs through the same forward pass, Viterbi decoder and sampler as the
counted models. No genome, no annotation and no EM is touched here.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import knowledge as K
import qstates as S
from qbuild import quant

MODELS = Path('/home/claude/quill/models'); MODELS.mkdir(exist_ok=True)
NCLS, NGC, NCONF, NEXP = S.NCLS, 3, 4, 4
BB_ORDER = 4
IDX = {b: i for i, b in enumerate(K.B)}


def _norm(a, axis=-1):
    a = np.asarray(a, np.float64)
    return a / a.sum(axis, keepdims=True)


def base_freq(gc):
    """A C G T at a given GC fraction, strand-symmetric."""
    return np.array([(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2])


def rho_matrix(cpg=None):
    """RHO as a 4x4 odds-ratio matrix, optionally overriding the CpG entry."""
    r = np.ones((4, 4))
    for xy, v in K.RHO.items():
        r[IDX[xy[0]], IDX[xy[1]]] = v
    if cpg is not None:
        r[IDX['C'], IDX['G']] = cpg
    return r


def _context_boost(ctx, x):
    """Homopolymer and alternation factors for emitting x after the bases in ctx."""
    f = 1.0
    run = 0
    for b in reversed(ctx):
        if b != x:
            break
        run += 1
    if run >= 3:
        tab = K.HOMO_GC if x in (IDX['C'], IDX['G']) else K.HOMO_AT
        f *= tab[min(run - 3, 2)]
    if len(ctx) >= 2 and ctx[-2] == x and ctx[-1] != x:
        f *= K.ALT_BOOST[0]
        if len(ctx) >= 4 and ctx[-4] == x and ctx[-3] == ctx[-1]:
            f *= K.ALT_BOOST[1] / K.ALT_BOOST[0]
    return f


def backbone(order=BB_ORDER):
    """Written backbone: base composition x dinucleotide odds ratio x run structure."""
    gc = K.BG_GC[1]
    f0, rho = base_freq(gc), rho_matrix()
    tabs, offs, cur = [], np.full(13, -1, np.int64), 0
    for k in range(order + 1):
        n = 4 ** k
        p = np.empty((n, 4))
        if k == 0:
            p[0] = f0
        else:
            ctxs = np.empty((n, k), np.int64)
            for j in range(k):
                ctxs[:, j] = (np.arange(n) >> (2 * (k - 1 - j))) & 3
            for i in range(n):
                ctx = ctxs[i]
                row = f0 * rho[ctx[-1]]
                for x in range(4):
                    row[x] *= _context_boost(ctx, x)
                p[i] = row
        offs[k] = cur
        q = quant(_norm(p))
        tabs.append(q.reshape(-1)); cur += q.size
    return np.concatenate(tabs), offs


def codon_freq():
    """P(codon) from written usage, in-frame stops kept only at STOP_LEAK."""
    p = np.zeros(64)
    for cod, v in K.CODON_USAGE.items():
        i = (IDX[cod[0]] * 16 + IDX[cod[1]] * 4 + IDX[cod[2]])
        p[i] = v if K.GENETIC_CODE[cod] != '*' else 0.0
    for cod in K.STOP_CODONS:
        p[IDX[cod[0]] * 16 + IDX[cod[1]] * 4 + IDX[cod[2]]] = K.STOP_LEAK
    return p / p.sum()


def class_tables():
    """cls[c, g, ab, x] and cls0[c, g, x] for the ten annotation classes."""
    cf = codon_freq().reshape(4, 4, 4)
    phase_f = [_norm(cf.sum((1, 2))), _norm(cf.sum((0, 2))), _norm(cf.sum((0, 1)))]
    cls = np.empty((NCLS, NGC, 16, 4)); cls0 = np.empty((NCLS, NGC, 4))
    for g in range(NGC):
        gc, cpg = K.BG_GC[g], K.BG_CPG[g]
        bg = base_freq(gc)
        for c in range(NCLS):
            if c in (S.IG, S.INTRON, S.TANDEM):
                f, pg = bg, cpg
            elif c == S.NCRNA:
                f, pg = _norm(bg * np.array([1.0, 1.05, 1.05, 1.0])), cpg * 1.2
            elif c == S.CGI:
                f, pg = np.array(K.REGION['CGI'][0]), K.REGION['CGI'][1]
            elif c == S.UTR5:
                f, pg = np.array(K.REGION['UTR5'][0]), K.REGION['UTR5'][1]
            elif c == S.UTR3:
                f, pg = np.array(K.REGION['UTR3'][0]), K.REGION['UTR3'][1]
            else:
                ph = c - S.CDS0
                f = phase_f[ph].copy()
                if ph == 2:
                    f = _norm(f * np.exp(np.array([0.0, 1.0, 1.0, 0.0]) * K.GC3_TILT[g]))
                pg = cpg * 1.4
            rho = rho_matrix(pg)
            cls0[c, g] = f
            for a in range(4):
                for b in range(4):
                    row = f * rho[b]
                    for x in range(4):
                        row[x] *= _context_boost((a, b), x)
                    cls[c, g, a * 4 + b] = _norm(row)
    return quant(cls), quant(cls0)


def pwm_tables():
    """donor(9) acceptor(23) atg(3) stop(3) pas(6), laid out as the counted build does."""
    rows, meta = [], {}

    def push(name, mat):
        meta[name] = (len(rows), len(mat))
        rows.extend(_norm(np.asarray(mat, np.float64) + 1e-4).tolist())

    push('donor', list(K.DONOR_EXON) + list(K.DONOR))
    acc = [K.ACC_PY, K.ACC_PY] + list(K.ACCEPTOR) + [K.EXON_START] * S.ACC_EXON
    push('acceptor', acc)
    push('atg', K.START)
    stop = np.zeros((3, 4))
    for cod, w in K.STOP_CODONS.items():
        for j, b in enumerate(cod):
            stop[j, IDX[b]] += w
    push('stop', stop)
    push('pas', K.PAS)
    assert meta['donor'][1] == 9 and meta['acceptor'][1] == 23, meta
    return quant(np.array(rows)), meta


def codon_tables():
    """P(base | phase, partial codon, previous codon) from written usage alone."""
    p = codon_freq().reshape(4, 4, 4)
    c1 = np.tile(_norm(p.sum((1, 2)))[None, :], (65, 1))
    c2 = np.tile(_norm(p.sum(2), axis=1)[None, :, :], (65, 1, 1))
    c3 = np.tile(_norm(p).reshape(1, 16, 4), (65, 1, 1))
    for prev in range(64):
        if prev % 4 == IDX['C']:                       # CpG across the codon boundary
            r = c1[prev].copy(); r[IDX['G']] *= K.BOUNDARY_CPG
            c1[prev] = _norm(r)
    return quant(c1), quant(c2), quant(c3)


def mixture():
    w = np.empty((S.NSLOT, NCONF, NEXP), np.float32)
    for sl in range(S.NSLOT):
        if sl >= NCLS:
            v = K.MIX_PWM
        elif S.CDS0 <= sl <= S.CDS2:
            v = K.MIX_CDS
        else:
            v = K.MIX_CLS
        w[sl] = np.array(v, np.float32)[None, :]
    return w


def transitions():
    """Per-base rates written from RATE, normalised over each state's outgoing edges."""
    R = K.RATE
    don = K.CDS_DON_RATE
    ex_s, ex_l = 1.0 / K.INTRON_SHORT, 1.0 / K.INTRON_LONG
    fs = K.INTRON_SHORT_FRAC
    rate = {
        'self_IG': 1.0 - (R['cgi_enter'] + R['tandem_enter'] + R['tss_enter'] + K.NC_ENTER),
        'ig_cgi': R['cgi_enter'], 'ig_tand': R['tandem_enter'],
        'ig_tss': R['tss_enter'], 'ig_nc': K.NC_ENTER,
        'self_CGI': 1.0 - 1.0 / R['cgi_len'],
        'cgi_ig': K.IG_FROM_CGI / R['cgi_len'],
        'cgi_tss': R['cgi_to_tss'] / R['cgi_len'],
        'cgi_nc': max(1.0 - K.IG_FROM_CGI - R['cgi_to_tss'], 1e-3) / R['cgi_len'],
        'self_TANDEM': K.TANDEM_STAY, 'tand_ig': 1.0 - K.TANDEM_STAY,
        'self_UTR5': 1.0 - 1.0 / R['utr5_len'],
        'utr5_atg': K.UTR5_TO_ATG / R['utr5_len'],
        'utr5_don': K.UTR5_TO_DON / R['utr5_len'],
        'utr5_abort': R['utr5_abort'] / R['utr5_len'],
        'cds_stop': R['stop_per_codon'],
        'self_UTR3': 1.0 - 1.0 / R['utr3_len'],
        'utr3_pas': R['utr3_pas'] / R['utr3_len'],
        'utr3_end': K.UTR3_TO_IG / R['utr3_len'],
        'utr3_don': K.UTR3_TO_DON / R['utr3_len'],
        'self_NCRNA': 1.0 - 1.0 / K.NCRNA_LEN,
        'nc_don': K.NCRNA_DON / K.NCRNA_LEN,
        'nc_end': (1.0 - K.NCRNA_DON) / K.NCRNA_LEN,
        'don_intronS': fs, 'don_intronL': 1.0 - fs,
        'self_intronS': 1.0 - ex_s, 'intronS_acc': ex_s,
        'self_intronL': 1.0 - ex_l, 'intronL_acc': ex_l,
        'accnc_utr5': 1.0, 'accnc_utr3': 1.0, 'accnc_nc': 1.0, 'fixed': 1.0,
    }
    for i in range(3):
        rate[f'cds_cont{i}'] = 1.0 - don - (R['stop_per_codon'] if i == 2 else 0.0)
        rate[f'cds_don{i}'] = don

    efam, efrom = S.G.efam, S.G.efrom
    w = np.array([rate[f] for f in efam], np.float64)
    elog = np.zeros(len(efrom), np.float32)
    for s in range(S.G.nstate):
        e = np.flatnonzero(efrom == s)
        if len(e) == 0:
            continue
        v = w[e]
        elog[e] = np.log2(v / v.sum())

    # Stationary occupancy of the three backgrounds decides the start distribution.
    init = np.full(S.G.nstate, -30.0, np.float64)
    occ = np.zeros(S.G.nstate)
    coding = R['exon_enter'] * R['exon_len']
    occ[S.G.idx['IG']] = 1.0
    occ[S.G.idx['CGI']] = R['cgi_enter'] * R['cgi_len']
    occ[S.G.idx['TANDEM']] = R['tandem_enter'] / (1.0 - K.TANDEM_STAY)
    occ[S.G.idx['UTR5']] = R['tss_enter'] * R['utr5_len']
    occ[S.G.idx['UTR3']] = R['tss_enter'] * R['utr3_len']
    occ[S.G.idx['NCRNA']] = K.NC_ENTER * K.NCRNA_LEN
    for i in range(3):
        occ[S.G.idx[f'CDS{i}']] = coding / 3.0
        for d, frac in (('S', fs), ('L', 1.0 - fs)):
            mean = K.INTRON_SHORT if d == 'S' else K.INTRON_LONG
            occ[S.G.idx[f'INTRON{i}_{d}']] = coding * don * frac * mean / 3.0
    occ = np.maximum(occ, 1e-12)
    init[:] = np.log2(occ / occ.sum())
    return elog, init.astype(np.float32)


def build():
    bb_tab, bb_off = backbone()
    cls, cls0 = class_tables()
    pwm, pwm_meta = pwm_tables()
    cod1, cod2, cod3 = codon_tables()
    mix = mixture()
    elog, init = transitions()

    kind = np.zeros(S.NSLOT, np.int64); scls = np.zeros(S.NSLOT, np.int64)
    srow = np.zeros(S.NSLOT, np.int64)
    for s in range(S.G.nstate):
        sl = S.G.slot[s]
        if S.G.pwm[s]:
            kind[sl] = 1
            srow[sl] = pwm_meta[S.PWMNAME[int(S.G.pwm[s])]][0] + int(S.G.pwmpos[s])
        else:
            kind[sl] = 0; scls[sl] = int(S.G.cls[s])

    nparam = dict(backbone=int(bb_tab.size), classes=int(cls.size + cls0.size),
                  pwm=int(pwm.size), codon=int(cod1.size + cod2.size + cod3.size),
                  mixture=int(mix.size), transitions=int(elog.size + init.size))
    nparam['total'] = sum(nparam.values())
    path = MODELS / 'quill-k.npz'
    np.savez_compressed(path, order=BB_ORDER, bb_tab=bb_tab, bb_off=bb_off, cls=cls,
                        cls0=cls0, pwm=pwm, cod1=cod1, cod2=cod2, cod3=cod3, mix=mix,
                        elog=elog, init=init, slot_kind=kind, slot_cls=scls, slot_row=srow,
                        pwm_meta=json.dumps(pwm_meta),
                        pas_top=json.dumps([['AATAAA', 1.0]]),
                        nparam=json.dumps(nparam))
    return path, nparam


if __name__ == '__main__':
    p, n = build()
    print(f'{p} ({p.stat().st_size/1024:.1f} KB)')
    print(' ', n)


def dequant_cls0(model, c, g):
    """The written base frequencies of one class and GC bin, read back from the model."""
    from qbuild import dequant_table
    p = dequant_table()[model.cls0[c, g]].astype(np.float64)
    return p / p.sum()
