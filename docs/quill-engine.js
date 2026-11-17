/* Quill browser engine: a direct port of the Python inference kernels.
   Everything here reads the exported parameter blob; no training, no network. */
window.QuillInit = function (Q) {
  const raw = atob(Q.blob);
  const bin = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bin[i] = raw.charCodeAt(i);
  function arr(key) {
    const a = Q.arrays[key], off = bin.byteOffset + a.off;
    if (a.dt === 'uint8') return new Uint8Array(bin.buffer, off, a.n);
    if (a.dt === 'int32') return new Int32Array(bin.buffer, off, a.n);
    return new Float32Array(bin.buffer, off, a.n);
  }
  const bb_tab = arr('bb_tab'), bb_off = arr('bb_off');
  const cls = arr('cls'), cls0 = arr('cls0'), pwm = arr('pwm');
  const cod1 = arr('cod1'), cod2 = arr('cod2'), cod3 = arr('cod3');
  const mix = arr('mix');
  const efrom = arr('efrom'), eto = arr('eto'), elog = arr('elog'), init = arr('init');
  const state_slot = arr('state_slot'), state_phase = arr('state_phase');
  const grp3 = arr('grp3'), ribbon = arr('ribbon');
  const slot_kind = arr('slot_kind'), slot_cls = arr('slot_cls'), slot_row = arr('slot_row');
  const ORDER = Q.order, S = Q.nstate, NSLOT = Q.nslot, NE = Q.nedge;
  const E0 = Q.gc_edges[0], E1 = Q.gc_edges[1], GW = Q.gc_win;

  const dq = new Float64Array(256);
  for (let i = 0; i < 256; i++) dq[i] = Math.fround(Math.pow(2, -i / 16));
  const tp = new Float64Array(NE);
  for (let e = 0; e < NE; e++) tp[e] = Math.pow(2, elog[e]);
  const init0 = new Float64Array(S);
  { let z = 0; for (let s = 0; s < S; s++) { init0[s] = Math.pow(2, init[s]); z += init0[s]; }
    for (let s = 0; s < S; s++) init0[s] /= z; }
  // outgoing-edge index, for sampling
  const deg = new Int32Array(S + 1);
  for (let e = 0; e < NE; e++) deg[efrom[e] + 1]++;
  for (let s = 0; s < S; s++) deg[s + 1] += deg[s];

  const CODE = 'ACGT';
  function encode(txt) {
    const up = txt.toUpperCase().replace(/[^ACGTUN]/g, '');
    const out = new Uint8Array(up.length);
    for (let i = 0; i < up.length; i++) {
      const c = up[i];
      out[i] = c === 'A' ? 0 : c === 'C' ? 1 : c === 'G' ? 2 : (c === 'T' || c === 'U') ? 3 : 4;
    }
    return out;
  }
  function decode(seq) { let s = ''; for (let i = 0; i < seq.length; i++) s += seq[i] < 4 ? CODE[seq[i]] : 'N'; return s; }
  function revcomp(seq) {
    const n = seq.length, o = new Uint8Array(n);
    for (let i = 0; i < n; i++) { const b = seq[n - 1 - i]; o[i] = b > 3 ? b : 3 - b; }
    return o;
  }
  function gcBins(seq) {
    const n = seq.length, out = new Uint8Array(n);
    let gc = 0, tot = 0, lo = 0, hi = 0;
    for (let i = 0; i < n; i++) {
      let wl = i - GW; if (wl < 0) wl = 0;
      while (hi < i) { const b = seq[hi]; if (b < 4) { tot++; if (b === 1 || b === 2) gc++; } hi++; }
      while (lo < wl) { const b = seq[lo]; if (b < 4) { tot--; if (b === 1 || b === 2) gc--; } lo++; }
      const f = tot > 50 ? gc / tot : 0.42;
      out[i] = f < E0 ? 0 : (f < E1 ? 1 : 2);
    }
    return out;
  }

  const pbb = new Float64Array(4), prim = new Float64Array(4), cod = new Float64Array(4);
  function bbLookup(ctx, run) {
    let use = run > ORDER ? ORDER : run;
    while (use > 0 && bb_off[use] < 0) use--;
    const m = (1 << (2 * use)) - 1;
    const base = bb_off[use] + (ctx & m) * 4;
    let s = 0;
    for (let x = 0; x < 4; x++) { const v = dq[bb_tab[base + x]]; pbb[x] = v; s += v; }
    for (let x = 0; x < 4; x++) pbb[x] /= s;
  }
  /* fills dist (Float64Array NSLOT*4) and returns the backbone-confidence bucket */
  function slotDists(seq, t, run, ctx, gcb, dist) {
    bbLookup(ctx, run);
    let mx = 0;
    for (let x = 0; x < 4; x++) if (pbb[x] > mx) mx = pbb[x];
    const conf = mx < 0.35 ? 0 : (mx < 0.5 ? 1 : (mx < 0.7 ? 2 : 3));
    const g = gcb[t];
    let c2 = -1;
    if (run >= 2) c2 = seq[t - 2] * 4 + seq[t - 1];
    for (let s = 0; s < NSLOT; s++) {
      let haveCod = false;
      if (slot_kind[s] === 0) {
        const c = slot_cls[s];
        if (c2 >= 0) { const o = ((c * 3 + g) * 16 + c2) * 4; for (let x = 0; x < 4; x++) prim[x] = dq[cls[o + x]]; }
        else { const o = (c * 3 + g) * 4; for (let x = 0; x < 4; x++) prim[x] = dq[cls0[o + x]]; }
        if (c >= 3 && c <= 5) {
          const ph = c - 3;
          if (run >= ph + 3) {
            let ok = true, pc = 0;
            for (let j = 0; j < 3; j++) { const bq = seq[t - ph - 3 + j]; if (bq > 3) ok = false; pc = pc * 4 + (bq < 4 ? bq : 0); }
            if (ok) {
              haveCod = true;
              if (ph === 0) { const o = pc * 4; for (let x = 0; x < 4; x++) cod[x] = dq[cod1[o + x]]; }
              else if (ph === 1) { const o = (pc * 4 + seq[t - 1]) * 4; for (let x = 0; x < 4; x++) cod[x] = dq[cod2[o + x]]; }
              else { const o = (pc * 16 + seq[t - 2] * 4 + seq[t - 1]) * 4; for (let x = 0; x < 4; x++) cod[x] = dq[cod3[o + x]]; }
            }
          }
        }
      } else {
        const o = slot_row[s] * 4;
        for (let x = 0; x < 4; x++) prim[x] = dq[pwm[o + x]];
      }
      const mo = (s * 4 + conf) * 4;
      const w0 = mix[mo], w1 = mix[mo + 1], w2 = haveCod ? mix[mo + 2] : 0, w3 = mix[mo + 3];
      const z = w0 + w1 + w2 + w3;
      let sp = 0, sc = 0;
      for (let x = 0; x < 4; x++) { sp += prim[x]; if (haveCod) sc += cod[x]; }
      let tot = 0;
      const base = s * 4;
      for (let x = 0; x < 4; x++) {
        let v = (w0 * pbb[x] + w1 * prim[x] / sp + w3 * 0.25) / z;
        if (haveCod) v += (w2 * cod[x] / sc) / z;
        dist[base + x] = v; tot += v;
      }
      for (let x = 0; x < 4; x++) dist[base + x] /= tot;
    }
    return conf;
  }

  /* forward: total log2 P(seq) and per-base bits. Optionally keeps scaled alphas. */
  function forward(seq, gcb, keepAlpha) {
    const n = seq.length;
    const dist = new Float64Array(NSLOT * 4);
    let alpha = new Float64Array(S), nxt = new Float64Array(S);
    for (let s = 0; s < S; s++) alpha[s] = init0[s];
    const bits = new Float32Array(n);
    const al = keepAlpha ? new Float32Array(n * S) : null;
    const em = keepAlpha ? new Float32Array(n * S) : null;
    let logp = 0, nv = 0, ctx = 0, run = 0;
    const mask = (1 << (2 * ORDER)) - 1;
    for (let t = 0; t < n; t++) {
      const b = seq[t];
      nxt.fill(0);
      for (let e = 0; e < NE; e++) nxt[eto[e]] += alpha[efrom[e]] * tp[e];
      let z = 0; for (let s = 0; s < S; s++) z += nxt[s];
      if (z <= 0) z = 1e-300;
      if (b > 3) {
        for (let s = 0; s < S; s++) { alpha[s] = nxt[s] / z; if (keepAlpha) { al[t * S + s] = alpha[s]; em[t * S + s] = 1; } }
        run = 0; ctx = 0; bits[t] = 2;
        continue;
      }
      slotDists(seq, t, run, ctx, gcb, dist);
      let zz = 0;
      for (let s = 0; s < S; s++) {
        const e_ = dist[state_slot[s] * 4 + b];
        if (keepAlpha) em[t * S + s] = e_;
        nxt[s] = (nxt[s] / z) * e_;
        zz += nxt[s];
      }
      if (zz <= 0) zz = 1e-300;
      bits[t] = -Math.log2(zz);
      logp += Math.log2(zz); nv++;
      for (let s = 0; s < S; s++) { alpha[s] = nxt[s] / zz; if (keepAlpha) al[t * S + s] = alpha[s]; }
      ctx = ((ctx << 2) | b) & mask; run++;
    }
    return { logp: logp, n: nv, bits: bits, alpha: al, em: em };
  }

  /* forward-backward state posteriors */
  function posterior(seq, gcb) {
    const n = seq.length;
    const f = forward(seq, gcb, true);
    const gam = new Float32Array(n * S);
    let beta = new Float64Array(S), nb = new Float64Array(S);
    beta.fill(1);
    for (let t = n - 1; t >= 0; t--) {
      let zg = 0;
      for (let s = 0; s < S; s++) { const v = f.alpha[t * S + s] * beta[s]; gam[t * S + s] = v; zg += v; }
      if (zg <= 0) zg = 1e-300;
      for (let s = 0; s < S; s++) gam[t * S + s] /= zg;
      nb.fill(0);
      for (let e = 0; e < NE; e++) nb[efrom[e]] += tp[e] * f.em[t * S + eto[e]] * beta[eto[e]];
      let z = 0; for (let s = 0; s < S; s++) z += nb[s];
      if (z <= 0) z = 1e-300;
      for (let s = 0; s < S; s++) beta[s] = nb[s] / z;
    }
    return { gamma: gam, bits: f.bits, logp: f.logp, nvalid: f.n };
  }

  function viterbi(seq, gcb) {
    const n = seq.length, NEG = -1e30;
    const dist = new Float64Array(NSLOT * 4);
    let d = new Float64Array(S), nd = new Float64Array(S);
    for (let s = 0; s < S; s++) d[s] = init[s];
    const bp = new Int16Array(n * S);
    let ctx = 0, run = 0;
    const mask = (1 << (2 * ORDER)) - 1;
    for (let t = 0; t < n; t++) {
      const b = seq[t], obs = b <= 3;
      if (obs) slotDists(seq, t, run, ctx, gcb, dist);
      nd.fill(NEG);
      for (let e = 0; e < NE; e++) {
        const v = d[efrom[e]] + elog[e];
        if (v > nd[eto[e]]) { nd[eto[e]] = v; bp[t * S + eto[e]] = efrom[e]; }
      }
      if (obs) for (let s = 0; s < S; s++) nd[s] += Math.log2(dist[state_slot[s] * 4 + b] + 1e-300);
      let mx = NEG;
      for (let s = 0; s < S; s++) if (nd[s] > mx) mx = nd[s];
      for (let s = 0; s < S; s++) d[s] = nd[s] - mx;
      if (obs) { ctx = ((ctx << 2) | b) & mask; run++; } else { ctx = 0; run = 0; }
    }
    const path = new Int16Array(n);
    let best = 0;
    for (let s = 0; s < S; s++) if (d[s] > d[best]) best = s;
    for (let t = n - 1; t >= 0; t--) { path[t] = best; best = bp[t * S + best]; }
    return path;
  }

  /* generation: sample bases and states together */
  function sample(n, startState, temp, seedState) {
    const seq = new Uint8Array(n), path = new Int16Array(n), gcb = new Uint8Array(n);
    const dist = new Float64Array(NSLOT * 4);
    let st = startState, ctx = 0, run = 0, gc = 0, win = 0;
    const mask = (1 << (2 * ORDER)) - 1;
    const forceGC = seedState === undefined ? -1 : seedState;
    for (let t = 0; t < n; t++) {
      let f;
      if (win > 50) f = gc / win; else f = 0.42;
      gcb[t] = forceGC >= 0 ? forceGC : (f < E0 ? 0 : (f < E1 ? 1 : 2));
      slotDists(seq, t, run, ctx, gcb, dist);
      const sl = state_slot[st] * 4;
      let s = 0;
      const p = [0, 0, 0, 0];
      for (let x = 0; x < 4; x++) { p[x] = Math.pow(dist[sl + x], 1 / temp); s += p[x]; }
      let u = Math.random() * s, acc = 0, b = 3;
      for (let x = 0; x < 4; x++) { acc += p[x]; if (u <= acc) { b = x; break; } }
      seq[t] = b; path[t] = st;
      ctx = ((ctx << 2) | b) & mask; run++;
      if (b === 1 || b === 2) gc++;
      win++;
      if (win > GW) { const ob = seq[t - GW]; if (ob === 1 || ob === 2) gc--; win = GW; }
      const lo = deg[st], hi = deg[st + 1];
      let tot = 0;
      for (let e = lo; e < hi; e++) tot += tp[e];
      u = Math.random() * tot; acc = 0;
      let nx = hi > lo ? eto[lo] : st;
      for (let e = lo; e < hi; e++) { acc += tp[e]; if (u <= acc) { nx = eto[e]; break; } }
      st = nx;
    }
    return { seq: seq, path: path, gcb: gcb };
  }

  /* Sample a state path alone. Transitions do not depend on the bases, so
     conditioning on a property of the path is exact conditional sampling. */
  function samplePath(start, n) {
    const path = new Int16Array(n);
    let st = start;
    for (let t = 0; t < n; t++) {
      path[t] = st;
      const lo = deg[st], hi = deg[st + 1];
      let tot = 0;
      for (let e = lo; e < hi; e++) tot += tp[e];
      let u = Math.random() * tot, acc = 0, nx = hi > lo ? eto[lo] : st;
      for (let e = lo; e < hi; e++) { acc += tp[e]; if (u <= acc) { nx = eto[e]; break; } }
      st = nx;
    }
    return path;
  }
  /* A path that starts transcription and finishes a complete coding gene inside
     maxLen bases. Returns {path, tries, complete}. */
  function samplePathGene(start, maxLen, maxTries) {
    const atg = Q.idx.ATG0, stop2 = Q.idx.STOP0 + 2, ig = Q.idx.IG;
    let bestLen = -1, best = null, tries = 0;
    for (; tries < maxTries; tries++) {
      const p = samplePath(start, maxLen);
      let sawAtg = -1, sawStop = -1;
      for (let t = 0; t < maxLen; t++) {
        if (sawAtg < 0 && p[t] === atg) sawAtg = t;
        if (sawAtg >= 0 && p[t] === stop2) { sawStop = t; break; }
      }
      if (sawStop > 0) {
        // keep a little intergenic tail after the gene
        return { path: p, tries: tries + 1, complete: true };
      }
      if (sawAtg >= 0 && sawAtg > bestLen) { bestLen = sawAtg; best = p; }
    }
    return { path: best || samplePath(start, maxLen), tries: tries, complete: false };
  }
  /* Emit bases along a fixed state path. */
  /* Optional constrained decoding: forbid the third base of a codon when it would
     create an in-frame stop. The grammar's codon expert already assigns stops zero
     probability, but it only carries ~1/3 of the mixture weight, so without this a
     long ORF picks up stops from the context model. */
  function generateAlongPath(path, temp, forceGC, noStop, gcTilt) {
    const n = path.length;
    const seq = new Uint8Array(n), gcb = new Uint8Array(n);
    const dist = new Float64Array(NSLOT * 4);
    let ctx = 0, run = 0, gc = 0, win = 0;
    const mask = (1 << (2 * ORDER)) - 1;
    for (let t = 0; t < n; t++) {
      const f = win > 50 ? gc / win : 0.42;
      gcb[t] = forceGC >= 0 ? forceGC : (f < E0 ? 0 : (f < E1 ? 1 : 2));
      slotDists(seq, t, run, ctx, gcb, dist);
      const st0 = path[t];
      const sl = state_slot[st0] * 4;
      let s = 0;
      const p = [0, 0, 0, 0];
      for (let x = 0; x < 4; x++) { p[x] = Math.pow(dist[sl + x], 1 / temp); s += p[x]; }
      if (gcTilt) {
        const f = Math.exp(gcTilt);
        p[1] *= f; p[2] *= f;
        s = p[0] + p[1] + p[2] + p[3];
      }
      if (noStop && state_phase[st0] === 2 && t >= 2 && !/^STOP/.test(Q.state_names[st0])) {
        const a = seq[t - 2], b2 = seq[t - 1];
        if (a === 3 && b2 === 0) { p[0] = 0; p[2] = 0; }      // TAA, TAG
        else if (a === 3 && b2 === 2) { p[0] = 0; }            // TGA
        s = p[0] + p[1] + p[2] + p[3];
        if (s <= 0) { for (let x = 0; x < 4; x++) p[x] = 0.25; s = 1; }
      }
      let u = Math.random() * s, acc = 0, b = 3;
      for (let x = 0; x < 4; x++) { acc += p[x]; if (u <= acc) { b = x; break; } }
      seq[t] = b;
      ctx = ((ctx << 2) | b) & mask; run++;
      if (b === 1 || b === 2) gc++;
      win++;
      if (win > GW) { const ob = seq[t - GW]; if (ob === 1 || ob === 2) gc--; win = GW; }
    }
    return { seq: seq, path: path, gcb: gcb };
  }

  /* every single-base substitution scored under the structure inferred from the input */
  function mutationScan(seq, gcb) {
    const n = seq.length;
    const post = posterior(seq, gcb);
    const gam = post.gamma;
    const dist = new Float64Array(NSLOT * 4);
    const out = new Float32Array(n * 4);
    let ctx = 0, run = 0;
    const mask = (1 << (2 * ORDER)) - 1;
    for (let t = 0; t < n; t++) {
      const b = seq[t];
      if (b > 3) { run = 0; ctx = 0; continue; }
      slotDists(seq, t, run, ctx, gcb, dist);
      const lg = [0, 0, 0, 0];
      for (let x = 0; x < 4; x++) lg[x] = 0;
      for (let s = 0; s < S; s++) {
        const w = gam[t * S + s];
        if (w > 1e-6) {
          const sl = state_slot[s] * 4;
          const lref = Math.log2(dist[sl + b] + 1e-300);
          for (let x = 0; x < 4; x++) lg[x] += w * (lref - Math.log2(dist[sl + x] + 1e-300));
        }
      }
      for (let x = 0; x < 4; x++) out[t * 4 + x] = lg[x];
      ctx = ((ctx << 2) | b) & mask; run++;
    }
    return { scores: out, bits: post.bits, gamma: gam, logp: post.logp, nvalid: post.nvalid };
  }

  /* splice-site scan straight from the measured matrices */
  function matrixLogOdds(which) {
    const mt = Q.pwm_meta[which], off = mt[0], L = mt[1];
    const bg = [Math.log2(0.295), Math.log2(0.205), Math.log2(0.205), Math.log2(0.295)];
    const M = [];
    for (let j = 0; j < L; j++) {
      const row = [];
      let z = 0;
      for (let x = 0; x < 4; x++) z += dq[pwm[(off + j) * 4 + x]];
      for (let x = 0; x < 4; x++) row.push(Math.log2(dq[pwm[(off + j) * 4 + x]] / z) - bg[x]);
      M.push(row);
    }
    return M;
  }
  function matrixProbs(which) {
    const mt = Q.pwm_meta[which], off = mt[0], L = mt[1];
    const M = [];
    for (let j = 0; j < L; j++) {
      let z = 0;
      const row = [];
      for (let x = 0; x < 4; x++) z += dq[pwm[(off + j) * 4 + x]];
      for (let x = 0; x < 4; x++) row.push(dq[pwm[(off + j) * 4 + x]] / z);
      M.push(row);
    }
    return M;
  }
  function scanMatrix(seq, M) {
    const L = M.length, n = seq.length - L + 1, out = new Float32Array(Math.max(n, 0));
    for (let i = 0; i < n; i++) {
      let v = 0, ok = true;
      for (let j = 0; j < L; j++) { const b = seq[i + j]; if (b > 3) { ok = false; break; } v += M[j][b]; }
      out[i] = ok ? v : -99;
    }
    return out;
  }

  return {
    Q: Q, S: S, NSLOT: NSLOT, ORDER: ORDER,
    encode: encode, decode: decode, revcomp: revcomp, gcBins: gcBins,
    forward: forward, posterior: posterior, viterbi: viterbi, sample: sample,
    mutationScan: mutationScan, matrixLogOdds: matrixLogOdds, matrixProbs: matrixProbs,
    samplePath: samplePath, samplePathGene: samplePathGene, generateAlongPath: generateAlongPath,
    scanMatrix: scanMatrix,
    ribbon: ribbon, grp3: grp3, state_slot: state_slot, state_phase: state_phase,
    stateNames: Q.state_names, ribbonNames: Q.ribbon_names,
    codonFreq: arr('codon_freq'), classBaseFreq: arr('class_base_freq'),
    intronLenHist: arr('intron_len_hist'), exonLenHist: arr('exon_len_hist'),
    mixWeights: mix, edges: { from: efrom, to: eto, logp: elog }, init: init,
    binGC: (function () {
      const f = arr('class_base_freq'), out = [];
      for (let b = 0; b < 3; b++) { const o = (0 * 3 + b) * 4; out.push(f[o + 1] + f[o + 2]); }
      return out;
    })()
  };
};
