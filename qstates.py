"""The Quill state grammar: a causal gene-structure HMM, and the map from
annotation to a state path (used to measure transitions and mixture weights).

Emission slots are the distinct conditional distributions a state can draw on.
Many states share one (every acceptor position is shared across phases), so the
forward pass evaluates ~54 distributions per base rather than one per state.
"""
import numpy as np

NCLS = 10
IG, CGI, UTR5, CDS0, CDS1, CDS2, UTR3, INTRON, NCRNA, TANDEM = range(10)
CLSNAME = ['IG', 'CGI', 'UTR5', 'CDS0', 'CDS1', 'CDS2', 'UTR3', 'INTRON', 'NCRNA', 'TANDEM']
# PWM id -> length. Donor spans 3 exon + 6 intron bases; acceptor 20 intron + 3 exon.
PWM_LEN = {1: 9, 2: 23, 3: 3, 4: 3, 5: 6}
PWMNAME = {0: '-', 1: 'donor', 2: 'acceptor', 3: 'atg', 4: 'stop', 5: 'pas'}
DON_EXON, ACC_EXON = 3, 3
NSLOT_CLS = NCLS
SLOT_PWM = {1: NSLOT_CLS, 2: NSLOT_CLS + 9, 3: NSLOT_CLS + 32, 4: NSLOT_CLS + 35, 5: NSLOT_CLS + 38}
NSLOT = NSLOT_CLS + 9 + 23 + 3 + 3 + 6                      # 54


class Grammar:
    """States as parallel arrays plus a sparse transition list."""

    def __init__(self):
        self.name, self.cls, self.pwm, self.pwmpos, self.phase = [], [], [], [], []
        self.edges = {}          # (from, to) -> label for the measured transition family
        self.idx = {}

    def add(self, name, cls=-1, pwm=0, pwmpos=-1, phase=-1):
        s = len(self.name)
        self.name.append(name); self.cls.append(cls); self.pwm.append(pwm)
        self.pwmpos.append(pwmpos); self.phase.append(phase); self.idx[name] = s
        return s

    def link(self, a, b, fam=None):
        self.edges[(self.idx[a] if isinstance(a, str) else a,
                    self.idx[b] if isinstance(b, str) else b)] = fam or 'auto'

    def chain(self, names):
        for a, b in zip(names[:-1], names[1:]):
            self.link(a, b, 'fixed')

    def finish(self):
        n = len(self.name)
        self.cls = np.array(self.cls, np.int16); self.pwm = np.array(self.pwm, np.int16)
        self.pwmpos = np.array(self.pwmpos, np.int16); self.phase = np.array(self.phase, np.int16)
        self.slot = np.array([slot_of(c, p, q) for c, p, q in zip(self.cls, self.pwm, self.pwmpos)], np.int16)
        e = sorted(self.edges)
        self.efrom = np.array([a for a, b in e], np.int32)
        self.eto = np.array([b for a, b in e], np.int32)
        self.efam = [self.edges[k] for k in e]
        self.nstate = n
        return self


def slot_of(cls, pwm, pwmpos):
    if pwm:
        return SLOT_PWM[pwm] + pwmpos
    return int(cls)


def build_grammar():
    """IG / CpG island / tandem background, a coding gene with phase-tracked
    introns, and one shared non-coding intron machinery for UTRs and ncRNA."""
    g = Grammar()
    g.add('IG', cls=IG); g.add('CGI', cls=CGI); g.add('TANDEM', cls=TANDEM)
    g.add('UTR5', cls=UTR5); g.add('UTR3', cls=UTR3); g.add('NCRNA', cls=NCRNA)
    for i in range(3):
        g.add(f'ATG{i}', cls=CDS0 + i, pwm=3, pwmpos=i, phase=i)
    for i in range(3):
        g.add(f'CDS{i}', cls=CDS0 + i, phase=i)
    for i in range(3):
        g.add(f'STOP{i}', pwm=4, pwmpos=i)
    for i in range(6):
        g.add(f'PAS{i}', pwm=5, pwmpos=i)
    # coding introns, one copy per resume phase
    for e in range(3):
        for j in range(9):
            cl = (CDS0 + (e + j) % 3) if j < DON_EXON else -1
            g.add(f'DON{e}_{j}', cls=cl if j < DON_EXON else -1, pwm=1, pwmpos=j,
                  phase=(e + j) % 3 if j < DON_EXON else -1)
        g.add(f'INTRON{e}_S', cls=INTRON, phase=e)
        g.add(f'INTRON{e}_L', cls=INTRON, phase=e)
        for j in range(23):
            g.add(f'ACC{e}_{j}', pwm=2, pwmpos=j,
                  phase=(e + j - 20) % 3 if j >= 20 else -1)
    # one non-coding intron machinery, shared by UTR5, UTR3 and ncRNA hosts
    for j in range(9):
        g.add(f'DONnc_{j}', pwm=1, pwmpos=j)
    g.add('INTRONnc_S', cls=INTRON)
    g.add('INTRONnc_L', cls=INTRON)
    for j in range(23):
        g.add(f'ACCnc_{j}', pwm=2, pwmpos=j)

    # ---- transitions
    for a in ('IG', 'CGI', 'TANDEM'):
        g.link(a, a, f'self_{a}')
    g.link('IG', 'CGI', 'ig_cgi'); g.link('CGI', 'IG', 'cgi_ig')
    g.link('IG', 'TANDEM', 'ig_tand'); g.link('TANDEM', 'IG', 'tand_ig')
    g.link('IG', 'UTR5', 'ig_tss'); g.link('CGI', 'UTR5', 'cgi_tss')
    g.link('IG', 'NCRNA', 'ig_nc'); g.link('CGI', 'NCRNA', 'cgi_nc')
    g.link('UTR5', 'UTR5', 'self_UTR5'); g.link('UTR5', 'ATG0', 'utr5_atg')
    g.link('UTR5', 'DONnc_0', 'utr5_don'); g.link('UTR5', 'IG', 'utr5_abort')
    g.chain(['ATG0', 'ATG1', 'ATG2']); g.link('ATG2', 'CDS0', 'fixed')
    for i in range(3):
        g.link(f'CDS{i}', f'CDS{(i+1)%3}', f'cds_cont{i}')
        g.link(f'CDS{i}', f'DON{(i+1)%3}_0', f'cds_don{i}')
    g.link('CDS2', 'STOP0', 'cds_stop')
    g.chain(['STOP0', 'STOP1', 'STOP2']); g.link('STOP2', 'UTR3', 'fixed')
    g.link('UTR3', 'UTR3', 'self_UTR3'); g.link('UTR3', 'PAS0', 'utr3_pas')
    g.link('UTR3', 'DONnc_0', 'utr3_don'); g.link('UTR3', 'IG', 'utr3_end')
    g.chain([f'PAS{i}' for i in range(6)]); g.link('PAS5', 'UTR3', 'fixed')
    g.link('NCRNA', 'NCRNA', 'self_NCRNA'); g.link('NCRNA', 'DONnc_0', 'nc_don')
    g.link('NCRNA', 'IG', 'nc_end')
    for e in range(3):
        g.chain([f'DON{e}_{j}' for j in range(9)])
        for d in 'SL':
            g.link(f'DON{e}_8', f'INTRON{e}_{d}', f'don_intron{d}')
            g.link(f'INTRON{e}_{d}', f'INTRON{e}_{d}', f'self_intron{d}')
            g.link(f'INTRON{e}_{d}', f'ACC{e}_0', f'intron{d}_acc')
        g.chain([f'ACC{e}_{j}' for j in range(23)])
        g.link(f'ACC{e}_22', f'CDS{e}', 'fixed')
    g.chain([f'DONnc_{j}' for j in range(9)])
    for d in 'SL':
        g.link('DONnc_8', f'INTRONnc_{d}', f'don_intron{d}')
        g.link(f'INTRONnc_{d}', f'INTRONnc_{d}', f'self_intron{d}')
        g.link(f'INTRONnc_{d}', 'ACCnc_0', f'intron{d}_acc')
    g.chain([f'ACCnc_{j}' for j in range(23)])
    for host, fam in (('UTR5', 'accnc_utr5'), ('UTR3', 'accnc_utr3'), ('NCRNA', 'accnc_nc')):
        g.link('ACCnc_22', host, fam)
    return g.finish()


INTRON_SPLIT = [1500]          # set from the measured length mixture before any path is built
G = build_grammar()
MASK = -1


def canonical(meta):
    """One transcript per gene: the longest CDS, else the longest span."""
    from qanno import transcripts
    best = {}
    for tid, kind, strand, exons, cds, prot in transcripts(meta):
        gene = None
        key = (strand, exons[0][0] // 1000000)
        clen = sum(b - a for a, b in cds) if cds else 0
        span = exons[-1][1] - exons[0][0]
        rec = (clen, span, tid, kind, strand, exons, cds)
        gid = (strand, exons[0][0], exons[-1][1])
        best.setdefault(gid, rec)
        if rec > best[gid]:
            best[gid] = rec
    return [(t, k, s, e, c) for _, _, t, k, s, e, c in best.values()]


def state_path(n, meta, cgi, tand, strand_want, seq=None, pas_hex=()):
    """Annotation -> HMM state path for one strand, in that strand's reading order.

    Returns an int16 array of length n (genomic order) with -1 where the path is
    unknown (gene on the opposite strand, or an ambiguous overlap).
    """
    path = np.full(n, G.idx['IG'], np.int16)
    path[cgi == 1] = G.idx['CGI']
    path[tand == 1] = G.idx['TANDEM']
    txs = [t for t in canonical(meta) if t[2] == strand_want]
    for tid, kind, strand, exons, cds in txs:
        fwd = strand == '+'
        ex = exons if fwd else [(a, b) for a, b in reversed(exons)]

        def put(a, b, val):
            if a < b:
                path[a:b] = val

        def oriented_put(start, length, states):
            """Write `states` to `length` bases beginning at transcript-order offset."""
            for k, s in enumerate(states):
                p = start + k if fwd else start - k
                if 0 <= p < n:
                    path[p] = s
        # intron interiors first; exon labels overwrite
        for (a, b), (c, d) in zip(exons[:-1], exons[1:]):
            L = c - b
            if L <= 0:
                continue
            put(b, c, G.idx['INTRONnc_S' if L <= INTRON_SPLIT[0] else 'INTRONnc_L'])
        if cds is None:
            for a, b in exons:
                put(a, b, G.idx['NCRNA'])
        else:
            cl, ch = cds[0][0], cds[-1][1]
            for a, b in exons:
                pos = np.arange(a, b)
                lab = np.where(pos < cl, G.idx['UTR5'], G.idx['UTR3']) if fwd else \
                      np.where(pos >= ch, G.idx['UTR5'], G.idx['UTR3'])
                path[a:b] = lab.astype(np.int16)
            # coding bases carry spliced phase; last 3 are the stop codon
            cs = cds if fwd else [(a, b) for a, b in reversed(cds)]
            clen = sum(b - a for a, b in cds)
            use_atg = use_stop = False
            if seq is not None and clen >= 6:
                from qanno import oriented as _or
                cseq = _or(seq, cds, strand)
                use_atg = bool(cseq[0] == 0 and cseq[1] == 3 and cseq[2] == 2)
                last = int(cseq[-3]) * 16 + int(cseq[-2]) * 4 + int(cseq[-1])
                use_stop = last in (48, 56, 50)
            off = 0
            for a, b in cs:
                L = b - a
                ph = (np.arange(off, off + L)) % 3
                k = off + np.arange(L)
                st = (G.idx['CDS0'] + ph).astype(np.int64)
                if use_atg:
                    st = np.where(k < 3, G.idx['ATG0'] + ph, st)
                if use_stop:
                    st = np.where(k >= clen - 3, G.idx['STOP0'] + np.clip(k - (clen - 3), 0, 2), st)
                st = st.astype(np.int16)
                path[a:b] = st if fwd else st[::-1]
                off += L
        # splice pipelines, written in transcript order so phases line up
        for k in range(len(ex) - 1):
            (a, b) = ex[k]
            (c, d) = ex[k + 1]
            don_cut = b if fwd else a                     # exon -> intron boundary
            acc_cut = c if fwd else d                     # intron -> exon boundary
            ilen = (c - b) if fwd else (a - d)
            if ilen < DON_EXON + 6 + 20 + 1:
                continue
            if cds is not None:
                pstate = path[don_cut - 1] if fwd else path[don_cut]
                ph = G.phase[pstate] if 0 <= pstate < G.nstate else -1
                e = (ph + 1) % 3 if ph >= 0 else -1
            else:
                e = -1
            pre = f'DON{e}_' if e >= 0 else 'DONnc_'
            acc = f'ACC{e}_' if e >= 0 else 'ACCnc_'
            dur = 'S' if ilen <= INTRON_SPLIT[0] else 'L'
            mid = (f'INTRON{e}_{dur}' if e >= 0 else f'INTRONnc_{dur}')
            start = don_cut - DON_EXON if fwd else don_cut + DON_EXON - 1
            oriented_put(start, 9, [G.idx[f'{pre}{j}'] for j in range(9)])
            istart = (don_cut + 6) if fwd else (don_cut - 7)
            iend = (acc_cut - 20) if fwd else (acc_cut + 20)
            if fwd:
                put(istart, iend, G.idx[mid])
            else:
                put(iend + 1, istart + 1, G.idx[mid])
            astart = acc_cut - 20 if fwd else acc_cut + 19
            oriented_put(astart, 23, [G.idx[f'{acc}{j}'] for j in range(23)])
        # poly(A) signal: the last match to a measured hexamer near the 3' end
        if cds is not None and seq is not None and len(pas_hex):
            utr3 = []
            for a, b in ex:
                rng = range(a, b) if fwd else range(b - 1, a - 1, -1)
                for p in rng:
                    if (p >= ch) if fwd else (p < cl):
                        utr3.append(p)
            if len(utr3) >= 20:
                tailpos = utr3[-70:-8]
                bases = np.array([seq[p] for p in tailpos], np.int16)
                if fwd is False:
                    bases = np.where(bases < 4, 3 - bases, bases)
                best = -1
                for i in range(len(bases) - 6, -1, -1):
                    w = bases[i:i + 6]
                    if (w > 3).any():
                        continue
                    code = 0
                    for v in w:
                        code = code * 4 + int(v)
                    if code in pas_hex:
                        best = i
                        break
                if best >= 0:
                    for j in range(6):
                        path[tailpos[best + j]] = G.idx[f'PAS{j}']
    # positions covered by a gene on the other strand are unknown for this strand
    for gid, (locus, strand, gs, ge) in meta['genes'].items():
        if strand != strand_want:
            path[gs:ge] = MASK
    return path
