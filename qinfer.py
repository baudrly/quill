"""Quill inference: emissions, forward likelihood, Viterbi decoding, sampling.

Every distribution is conditioned only on bases already emitted, so
P(x) = prod_t P(x_t | x_<t) is a proper density and bits/base is comparable with
any autoregressive model.
"""
import numpy as np
from numba import njit

NEXP = 4          # backbone, slot-primary, codon, uniform
NCONF = 4
EPS = 1e-12


@njit(cache=True, inline='always')
def _bb_lookup(tabs, offs, ctx, run, k, dq, out):
    """Backbone distribution from the highest baked order the history supports."""
    use = k if run >= k else run
    while use > 0 and offs[use] < 0:
        use -= 1
    m = (np.uint64(1) << np.uint64(2 * use)) - np.uint64(1)
    base = offs[use] + ((ctx & m) * np.uint64(4))
    s = 0.0
    for x in range(4):
        v = dq[tabs[base + np.uint64(x)]]
        out[x] = v
        s += v
    for x in range(4):
        out[x] /= s
    return out


@njit(cache=True)
def slot_dists(seq, t, run, ctx, gcb, bb_tab, bb_off, bb_k, dq, cls_tab, cls0_tab,
               pwm_tab, cod1, cod2, cod3, mix, slot_kind, slot_cls, slot_row,
               pbb, dist):
    """Fill dist[NSLOT, 4] with the mixed emission distribution for every slot."""
    _bb_lookup(bb_tab, bb_off, ctx, run, bb_k, dq, pbb)
    mx = 0.0
    for x in range(4):
        if pbb[x] > mx:
            mx = pbb[x]
    conf = 0 if mx < 0.35 else (1 if mx < 0.5 else (2 if mx < 0.7 else 3))
    g = gcb[t]
    c2 = -1
    if run >= 2:
        c2 = int(seq[t - 2]) * 4 + int(seq[t - 1])
    nslot = len(slot_kind)
    prim = np.empty(4); cod = np.empty(4)
    for s in range(nslot):
        have_cod = False
        if slot_kind[s] == 0:
            c = slot_cls[s]
            if c2 >= 0:
                for x in range(4):
                    prim[x] = dq[cls_tab[c, g, c2, x]]
            else:
                for x in range(4):
                    prim[x] = dq[cls0_tab[c, g, x]]
            if 3 <= c <= 5:
                ph = c - 3
                need = ph + 3
                if run >= need:
                    ok = True
                    pc = 0
                    for j in range(3):
                        bb = seq[t - ph - 3 + j]
                        if bb > 3:
                            ok = False
                        pc = pc * 4 + int(bb if bb < 4 else 0)
                    if ok:
                        have_cod = True
                        if ph == 0:
                            for x in range(4):
                                cod[x] = dq[cod1[pc, x]]
                        elif ph == 1:
                            c0 = int(seq[t - 1])
                            for x in range(4):
                                cod[x] = dq[cod2[pc, c0, x]]
                        else:
                            c01 = int(seq[t - 2]) * 4 + int(seq[t - 1])
                            for x in range(4):
                                cod[x] = dq[cod3[pc, c01, x]]
        else:
            r = slot_row[s]
            for x in range(4):
                prim[x] = dq[pwm_tab[r, x]]
        w0 = mix[s, conf, 0]; w1 = mix[s, conf, 1]
        w2 = mix[s, conf, 2] if have_cod else 0.0
        w3 = mix[s, conf, 3]
        z = w0 + w1 + w2 + w3
        sp = 0.0; sc = 0.0
        for x in range(4):
            sp += prim[x]
            if have_cod:
                sc += cod[x]
        tot = 0.0
        for x in range(4):
            v = (w0 * pbb[x] + w1 * prim[x] / sp + w3 * 0.25) / z
            if have_cod:
                v += (w2 * cod[x] / sc) / z
            dist[s, x] = v
            tot += v
        for x in range(4):
            dist[s, x] /= tot
    return conf


@njit(cache=True)
def forward(seq, gcb, bb_tab, bb_off, bb_k, dq, cls_tab, cls0_tab, pwm_tab,
            cod1, cod2, cod3, mix, slot_kind, slot_cls, slot_row,
            efrom, eto, elog, state_slot, init, want_post):
    """Total log2 P(seq) under the grammar, plus optional per-base state posteriors."""
    S = len(state_slot); n = len(seq)
    nslot = len(slot_kind)
    alpha = np.exp2(init.astype(np.float64))
    alpha = alpha / alpha.sum()
    new = np.zeros(S)
    dist = np.zeros((nslot, 4)); pbb = np.zeros(4)
    tp = np.exp2(elog.astype(np.float64))
    logp = 0.0
    nvalid = 0
    post = np.zeros((n, S) if want_post else (1, 1))
    ctx = np.uint64(0); run = 0
    for t in range(n):
        b = seq[t]
        if b > 3:
            # unobserved base: advance the chain without scoring
            new[:] = 0.0
            for e in range(len(efrom)):
                new[eto[e]] += alpha[efrom[e]] * tp[e]
            z = new.sum()
            if z > 0:
                alpha = new / z
            run = 0; ctx = np.uint64(0)
            if want_post:
                post[t] = alpha
            continue
        slot_dists(seq, t, run, ctx, gcb, bb_tab, bb_off, bb_k, dq, cls_tab, cls0_tab,
                   pwm_tab, cod1, cod2, cod3, mix, slot_kind, slot_cls, slot_row, pbb, dist)
        new[:] = 0.0
        for e in range(len(efrom)):
            new[eto[e]] += alpha[efrom[e]] * tp[e]
        z = 0.0
        for s in range(S):
            new[s] *= dist[state_slot[s], b]
            z += new[s]
        if z <= 0:
            z = 1e-300
        logp += np.log2(z)
        nvalid += 1
        for s in range(S):
            alpha[s] = new[s] / z
        if want_post:
            post[t] = alpha
        ctx = (ctx << np.uint64(2)) | np.uint64(b); run += 1
    return logp, nvalid, post


@njit(cache=True)
def forward_summary(seq, gcb, bb_tab, bb_off, bb_k, dq, cls_tab, cls0_tab, pwm_tab,
                    cod1, cod2, cod3, mix, slot_kind, slot_cls, slot_row,
                    efrom, eto, elog, state_slot, init):
    """log2 P(seq) plus the time-summed state posteriors (no per-base storage)."""
    S = len(state_slot); n = len(seq); nslot = len(slot_kind)
    alpha = np.exp2(init.astype(np.float64)); alpha /= alpha.sum()
    new = np.zeros(S); acc = np.zeros(S)
    dist = np.zeros((nslot, 4)); pbb = np.zeros(4)
    tp = np.exp2(elog.astype(np.float64))
    logp = 0.0; nvalid = 0
    ctx = np.uint64(0); run = 0
    for t in range(n):
        b = seq[t]
        if b > 3:
            new[:] = 0.0
            for e in range(len(efrom)):
                new[eto[e]] += alpha[efrom[e]] * tp[e]
            z = new.sum()
            if z > 0:
                alpha = new / z
            run = 0; ctx = np.uint64(0)
            continue
        slot_dists(seq, t, run, ctx, gcb, bb_tab, bb_off, bb_k, dq, cls_tab, cls0_tab,
                   pwm_tab, cod1, cod2, cod3, mix, slot_kind, slot_cls, slot_row, pbb, dist)
        new[:] = 0.0
        for e in range(len(efrom)):
            new[eto[e]] += alpha[efrom[e]] * tp[e]
        z = 0.0
        for s in range(S):
            new[s] *= dist[state_slot[s], b]
            z += new[s]
        if z <= 0:
            z = 1e-300
        logp += np.log2(z); nvalid += 1
        for s in range(S):
            alpha[s] = new[s] / z
            acc[s] += alpha[s]
        ctx = (ctx << np.uint64(2)) | np.uint64(b); run += 1
    return logp, nvalid, acc


@njit(cache=True)
def viterbi(seq, gcb, bb_tab, bb_off, bb_k, dq, cls_tab, cls0_tab, pwm_tab,
            cod1, cod2, cod3, mix, slot_kind, slot_cls, slot_row,
            efrom, eto, elog, state_slot, init):
    S = len(state_slot); n = len(seq); nslot = len(slot_kind)
    NEG = -1e30
    d = init.astype(np.float64).copy()
    bp = np.zeros((n, S), np.int16)
    dist = np.zeros((nslot, 4)); pbb = np.zeros(4)
    nd = np.empty(S)
    ctx = np.uint64(0); run = 0
    for t in range(n):
        b = seq[t]
        obs = b <= 3
        if obs:
            slot_dists(seq, t, run, ctx, gcb, bb_tab, bb_off, bb_k, dq,
                       cls_tab, cls0_tab, pwm_tab, cod1, cod2, cod3, mix,
                       slot_kind, slot_cls, slot_row, pbb, dist)
        for s in range(S):
            nd[s] = NEG
        for e in range(len(efrom)):
            v = d[efrom[e]] + elog[e]
            if v > nd[eto[e]]:
                nd[eto[e]] = v
                bp[t, eto[e]] = efrom[e]
        if obs:
            for s in range(S):
                nd[s] += np.log2(dist[state_slot[s], b] + 1e-300)
        mx = NEG
        for s in range(S):
            if nd[s] > mx:
                mx = nd[s]
        for s in range(S):
            d[s] = nd[s] - mx
        if obs:
            ctx = (ctx << np.uint64(2)) | np.uint64(b); run += 1
        else:
            ctx = np.uint64(0); run = 0
    path = np.zeros(n, np.int16)
    best = 0
    for s in range(S):
        if d[s] > d[best]:
            best = s
    for t in range(n - 1, -1, -1):
        path[t] = best
        best = bp[t, best]
    return path


@njit(cache=True)
def run_of(t, seq):
    r = 0
    while t - 1 - r >= 0 and seq[t - 1 - r] <= 3 and r < 32:
        r += 1
    return r


@njit(cache=True)
def ctx_of(t, seq, k):
    c = np.uint64(0)
    r = run_of(t, seq)
    m = k if r > k else r
    for j in range(m, 0, -1):
        c = (c << np.uint64(2)) | np.uint64(seq[t - j])
    return c


@njit(cache=True)
def sample(n, seed, gc_default, bb_tab, bb_off, bb_k, dq, cls_tab, cls0_tab, pwm_tab,
           cod1, cod2, cod3, mix, slot_kind, slot_cls, slot_row,
           efrom, eto, elog, state_slot, init, start_state, temp, gcwin):
    """Draw a sequence and its state path from the generative model."""
    np.random.seed(seed)
    S = len(state_slot); nslot = len(slot_kind)
    seq = np.zeros(n, np.uint8)
    path = np.zeros(n, np.int16)
    gcb = np.zeros(n + 1, np.uint8)
    dist = np.zeros((nslot, 4)); pbb = np.zeros(4)
    tp = np.exp2(elog.astype(np.float64))
    # outgoing edge slices per state
    deg = np.zeros(S + 1, np.int64)
    for e in range(len(efrom)):
        deg[efrom[e] + 1] += 1
    for s in range(S):
        deg[s + 1] += deg[s]
    st = start_state
    ctx = np.uint64(0); run = 0
    gcrun = 0; gccnt = 0
    for t in range(n):
        gcb[t] = gc_default if gcrun < 64 else (0 if gccnt / gcrun < 0.42 else
                                                (1 if gccnt / gcrun < 0.52 else 2))
        slot_dists(seq, t, run, ctx, gcb, bb_tab, bb_off, bb_k, dq, cls_tab, cls0_tab,
                   pwm_tab, cod1, cod2, cod3, mix, slot_kind, slot_cls, slot_row, pbb, dist)
        sl = state_slot[st]
        p = np.empty(4)
        s = 0.0
        for x in range(4):
            v = dist[sl, x] ** (1.0 / temp)
            p[x] = v; s += v
        u = np.random.random() * s
        acc = 0.0; b = 3
        for x in range(4):
            acc += p[x]
            if u <= acc:
                b = x
                break
        seq[t] = b; path[t] = st
        ctx = (ctx << np.uint64(2)) | np.uint64(b); run += 1
        if b == 1 or b == 2:
            gccnt += 1
        gcrun += 1
        if gcrun > gcwin:
            ob = seq[t - gcwin]
            if ob == 1 or ob == 2:
                gccnt -= 1
            gcrun = gcwin
        # move
        lo = deg[st]; hi = deg[st + 1]
        tot = 0.0
        for e in range(lo, hi):
            tot += tp[e]
        u = np.random.random() * tot
        acc = 0.0; nxt = eto[lo] if hi > lo else st
        for e in range(lo, hi):
            acc += tp[e]
            if u <= acc:
                nxt = eto[e]
                break
        st = nxt
    return seq, path


@njit(cache=True)
def predictive(seq, gcb, bb_tab, bb_off, bb_k, dq, cls_tab, cls0_tab, pwm_tab,
               cod1, cod2, cod3, mix, slot_kind, slot_cls, slot_row,
               efrom, eto, elog, state_slot, init, want):
    """P(x_t = b) for every t in `want`, mixed over the forward state posterior.

    The distribution at t conditions only on x_<t, so it is the model's honest
    predictive distribution at that site.
    """
    S = len(state_slot); n = len(seq); nslot = len(slot_kind)
    alpha = np.exp2(init.astype(np.float64)); alpha /= alpha.sum()
    new = np.zeros(S)
    dist = np.zeros((nslot, 4)); pbb = np.zeros(4)
    tp = np.exp2(elog.astype(np.float64))
    mark = np.zeros(n, np.uint8)
    for j in range(len(want)):
        if 0 <= want[j] < n:
            mark[want[j]] = 1
    outp = np.zeros((n, 4))
    ctx = np.uint64(0); run = 0
    for t in range(n):
        b = seq[t]
        new[:] = 0.0
        for e in range(len(efrom)):
            new[eto[e]] += alpha[efrom[e]] * tp[e]
        z = new.sum()
        if z <= 0:
            z = 1e-300
        if b > 3:
            for s in range(S):
                alpha[s] = new[s] / z
            run = 0; ctx = np.uint64(0)
            continue
        slot_dists(seq, t, run, ctx, gcb, bb_tab, bb_off, bb_k, dq, cls_tab, cls0_tab,
                   pwm_tab, cod1, cod2, cod3, mix, slot_kind, slot_cls, slot_row, pbb, dist)
        if mark[t]:
            for x in range(4):
                v = 0.0
                for s in range(S):
                    v += (new[s] / z) * dist[state_slot[s], x]
                outp[t, x] = v
        zz = 0.0
        for s in range(S):
            new[s] *= dist[state_slot[s], b]
            zz += new[s]
        if zz <= 0:
            zz = 1e-300
        for s in range(S):
            alpha[s] = new[s] / zz
        ctx = (ctx << np.uint64(2)) | np.uint64(b); run += 1
    return outp


@njit(cache=True)
def path_logp_range(seq, gcb, path, lo, hi, bb_tab, bb_off, bb_k, dq, cls_tab, cls0_tab,
                    pwm_tab, cod1, cod2, cod3, mix, slot_kind, slot_cls, slot_row,
                    state_slot):
    """Sum of log2 emission probabilities over [lo, hi) with the state path held fixed."""
    nslot = len(slot_kind)
    dist = np.zeros((nslot, 4)); pbb = np.zeros(4)
    run = 0
    while lo - 1 - run >= 0 and seq[lo - 1 - run] <= 3 and run < 40:
        run += 1
    ctx = np.uint64(0)
    mm = bb_k if run > bb_k else run
    for j in range(mm, 0, -1):
        ctx = (ctx << np.uint64(2)) | np.uint64(seq[lo - j])
    tot = 0.0
    for t in range(lo, hi):
        b = seq[t]
        if b > 3:
            run = 0; ctx = np.uint64(0)
            continue
        slot_dists(seq, t, run, ctx, gcb, bb_tab, bb_off, bb_k, dq, cls_tab, cls0_tab,
                   pwm_tab, cod1, cod2, cod3, mix, slot_kind, slot_cls, slot_row, pbb, dist)
        tot += np.log2(dist[state_slot[path[t]], b] + 1e-300)
        ctx = (ctx << np.uint64(2)) | np.uint64(b); run += 1
    return tot


@njit(cache=True)
def expected_lr(seq, gcb, bb_tab, bb_off, bb_k, dq, cls_tab, cls0_tab, pwm_tab,
                cod1, cod2, cod3, mix, slot_kind, slot_cls, slot_row,
                efrom, eto, elog, state_slot, init, want, refb, altb):
    """Posterior-expected pointwise log-likelihood ratio at each wanted site.

    sum_s P(state s at t | x_<t) * [log2 P(ref | s) - log2 P(alt | s)]
    Unlike the ratio of posterior-mixed probabilities, a decisive hypothesis
    carrying little posterior mass still contributes its full log-ratio.
    """
    S = len(state_slot); n = len(seq); nslot = len(slot_kind)
    alpha = np.exp2(init.astype(np.float64)); alpha /= alpha.sum()
    new = np.zeros(S)
    dist = np.zeros((nslot, 4)); pbb = np.zeros(4)
    tp = np.exp2(elog.astype(np.float64))
    nw = len(want)
    first = np.full(n, -1, np.int64)
    nxt = np.full(nw, -1, np.int64)
    for j in range(nw - 1, -1, -1):
        t = want[j]
        if 0 <= t < n:
            nxt[j] = first[t]
            first[t] = j
    out = np.zeros(nw)
    ctx = np.uint64(0); run = 0
    for t in range(n):
        b = seq[t]
        new[:] = 0.0
        for e in range(len(efrom)):
            new[eto[e]] += alpha[efrom[e]] * tp[e]
        z = new.sum()
        if z <= 0:
            z = 1e-300
        if b > 3:
            for s in range(S):
                alpha[s] = new[s] / z
            run = 0; ctx = np.uint64(0)
            continue
        slot_dists(seq, t, run, ctx, gcb, bb_tab, bb_off, bb_k, dq, cls_tab, cls0_tab,
                   pwm_tab, cod1, cod2, cod3, mix, slot_kind, slot_cls, slot_row, pbb, dist)
        j = first[t]
        while j >= 0:
            acc = 0.0
            for s in range(S):
                w = new[s] / z
                if w > 1e-9:
                    sl = state_slot[s]
                    acc += w * (np.log2(dist[sl, refb[j]] + 1e-300) -
                                np.log2(dist[sl, altb[j]] + 1e-300))
            out[j] = acc
            j = nxt[j]
        zz = 0.0
        for s in range(S):
            new[s] *= dist[state_slot[s], b]
            zz += new[s]
        if zz <= 0:
            zz = 1e-300
        for s in range(S):
            alpha[s] = new[s] / zz
        ctx = (ctx << np.uint64(2)) | np.uint64(b); run += 1
    return out


@njit(cache=True)
def smoothed_lr(seq, gcb, bb_tab, bb_off, bb_k, dq, cls_tab, cls0_tab, pwm_tab,
                cod1, cod2, cod3, mix, slot_kind, slot_cls, slot_row,
                efrom, eto, elog, state_slot, init, want, refb, altb, want_gamma):
    """Variant score under the structure the model infers from the reference.

    gamma_s(t) is the forward-backward posterior computed on the reference
    sequence; the score is sum_s gamma_s(t) [log2 P(ref|s) - log2 P(alt|s)].
    Structure comes from the reference only, so no variant label or annotation
    is used anywhere.
    """
    S = len(state_slot); n = len(seq); nslot = len(slot_kind)
    tp = np.exp2(elog.astype(np.float64))
    em = np.zeros((n, S), np.float32)          # emission of the observed base per state
    al = np.zeros((n, S), np.float32)
    dist = np.zeros((nslot, 4)); pbb = np.zeros(4)
    alpha = np.exp2(init.astype(np.float64)); alpha /= alpha.sum()
    new = np.zeros(S)
    obs = np.zeros(n, np.uint8)
    # ---- forward
    ctx = np.uint64(0); run = 0
    for t in range(n):
        b = seq[t]
        new[:] = 0.0
        for e in range(len(efrom)):
            new[eto[e]] += alpha[efrom[e]] * tp[e]
        z = new.sum()
        if z <= 0:
            z = 1e-300
        if b > 3:
            for s in range(S):
                alpha[s] = new[s] / z
                em[t, s] = 1.0
                al[t, s] = alpha[s]
            run = 0; ctx = np.uint64(0)
            continue
        obs[t] = 1
        slot_dists(seq, t, run, ctx, gcb, bb_tab, bb_off, bb_k, dq, cls_tab, cls0_tab,
                   pwm_tab, cod1, cod2, cod3, mix, slot_kind, slot_cls, slot_row, pbb, dist)
        zz = 0.0
        for s in range(S):
            e_ = dist[state_slot[s], b]
            em[t, s] = e_
            new[s] *= e_
            zz += new[s]
        if zz <= 0:
            zz = 1e-300
        for s in range(S):
            alpha[s] = new[s] / zz
            al[t, s] = alpha[s]
        ctx = (ctx << np.uint64(2)) | np.uint64(b); run += 1
    # ---- backward
    beta = np.ones(S)
    nb = np.zeros(S)
    gam = np.zeros((len(want_gamma), S), np.float64)
    gmark = np.full(n, -1, np.int64)
    for j in range(len(want_gamma)):
        if 0 <= want_gamma[j] < n:
            gmark[want_gamma[j]] = j
    gnow = np.zeros((n, 0), np.float64)
    out = np.zeros(len(want))
    first = np.full(n, -1, np.int64)
    nxt = np.full(len(want), -1, np.int64)
    for j in range(len(want) - 1, -1, -1):
        t = want[j]
        if 0 <= t < n:
            nxt[j] = first[t]
            first[t] = j
    gcur = np.zeros(S)
    for t in range(n - 1, -1, -1):
        # gamma at t from alpha_t * beta_t
        zg = 0.0
        for s in range(S):
            gcur[s] = al[t, s] * beta[s]
            zg += gcur[s]
        if zg <= 0:
            zg = 1e-300
        for s in range(S):
            gcur[s] /= zg
        j = gmark[t]
        if j >= 0:
            for s in range(S):
                gam[j, s] = gcur[s]
        if first[t] >= 0 and obs[t]:
            # recompute the slot distributions at this site
            run = 0
            while t - 1 - run >= 0 and seq[t - 1 - run] <= 3 and run < 40:
                run += 1
            ctx = np.uint64(0)
            mm = bb_k if run > bb_k else run
            for q in range(mm, 0, -1):
                ctx = (ctx << np.uint64(2)) | np.uint64(seq[t - q])
            slot_dists(seq, t, run, ctx, gcb, bb_tab, bb_off, bb_k, dq, cls_tab, cls0_tab,
                       pwm_tab, cod1, cod2, cod3, mix, slot_kind, slot_cls, slot_row, pbb, dist)
            j = first[t]
            while j >= 0:
                acc = 0.0
                for s in range(S):
                    if gcur[s] > 1e-9:
                        sl = state_slot[s]
                        acc += gcur[s] * (np.log2(dist[sl, refb[j]] + 1e-300) -
                                          np.log2(dist[sl, altb[j]] + 1e-300))
                out[j] = acc
                j = nxt[j]
        # step beta back
        nb[:] = 0.0
        for e in range(len(efrom)):
            nb[efrom[e]] += tp[e] * em[t, eto[e]] * beta[eto[e]]
        z = nb.sum()
        if z <= 0:
            z = 1e-300
        for s in range(S):
            beta[s] = nb[s] / z
    return out, gam


@njit(cache=True)
def group_posterior(seq, gcb, bb_tab, bb_off, bb_k, dq, cls_tab, cls0_tab, pwm_tab,
                    cod1, cod2, cod3, mix, slot_kind, slot_cls, slot_row,
                    efrom, eto, elog, state_slot, init, state_group, ngroup):
    """Forward-backward posterior, collapsed onto state groups. Returns (n, ngroup)."""
    S = len(state_slot); n = len(seq); nslot = len(slot_kind)
    tp = np.exp2(elog.astype(np.float64))
    em = np.zeros((n, S), np.float32); al = np.zeros((n, S), np.float32)
    dist = np.zeros((nslot, 4)); pbb = np.zeros(4)
    alpha = np.exp2(init.astype(np.float64)); alpha /= alpha.sum()
    new = np.zeros(S)
    ctx = np.uint64(0); run = 0
    for t in range(n):
        b = seq[t]
        new[:] = 0.0
        for e in range(len(efrom)):
            new[eto[e]] += alpha[efrom[e]] * tp[e]
        z = new.sum()
        if z <= 0:
            z = 1e-300
        if b > 3:
            for s in range(S):
                alpha[s] = new[s] / z; em[t, s] = 1.0; al[t, s] = alpha[s]
            run = 0; ctx = np.uint64(0)
            continue
        slot_dists(seq, t, run, ctx, gcb, bb_tab, bb_off, bb_k, dq, cls_tab, cls0_tab,
                   pwm_tab, cod1, cod2, cod3, mix, slot_kind, slot_cls, slot_row, pbb, dist)
        zz = 0.0
        for s in range(S):
            e_ = dist[state_slot[s], b]
            em[t, s] = e_; new[s] *= e_; zz += new[s]
        if zz <= 0:
            zz = 1e-300
        for s in range(S):
            alpha[s] = new[s] / zz; al[t, s] = alpha[s]
        ctx = (ctx << np.uint64(2)) | np.uint64(b); run += 1
    out = np.zeros((n, ngroup), np.float32)
    beta = np.ones(S); nb = np.zeros(S)
    for t in range(n - 1, -1, -1):
        zg = 0.0
        for s in range(S):
            zg += al[t, s] * beta[s]
        if zg <= 0:
            zg = 1e-300
        for s in range(S):
            out[t, state_group[s]] += al[t, s] * beta[s] / zg
        nb[:] = 0.0
        for e in range(len(efrom)):
            nb[efrom[e]] += tp[e] * em[t, eto[e]] * beta[eto[e]]
        z = nb.sum()
        if z <= 0:
            z = 1e-300
        for s in range(S):
            beta[s] = nb[s] / z
    return out
