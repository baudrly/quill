/* Ontology engine: the browser side of qgo.py and qso.py.

   Scores a coding sequence against the written functional classes (Gene Ontology slim
   and gene families), scores a transcript against the written Sequence Ontology classes,
   and samples a coding sequence from a chosen class. Must agree with Python: see
   web/ontparity.js. */
window.QuillOnt = (function () {
  const O = window.QUILL_ONT;
  if (!O) throw new Error('quill-ontology.js must load first');

  const dq = new Float32Array(256);
  for (let i = 0; i < 256; i++) dq[i] = Math.fround(Math.pow(2, -i / 16));

  function unb64(s) {
    const raw = atob(s), a = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) a[i] = raw.charCodeAt(i);
    return a;
  }
  function logTable(bytes) {            // quantised byte -> log2 probability
    const q = unb64(bytes), out = new Float32Array(q.length);
    for (let i = 0; i < q.length; i++) out[i] = Math.log2(Math.max(dq[q[i]], 1e-12));
    return out;
  }
  function lse2(a, b) {
    if (a < b) { const t = a; a = b; b = t; }
    const d = b - a;
    return d < -40 ? a : a + Math.log2(1 + Math.pow(2, d));
  }

  const BI = { A: 0, C: 1, G: 2, T: 3, U: 3 };
  function encode(txt) {
    const s = String(txt).toUpperCase().replace(/[^ACGTUN]/g, '');
    const a = new Int8Array(s.length);
    for (let i = 0; i < s.length; i++) a[i] = s[i] in BI ? BI[s[i]] : 4;
    return a;
  }
  function codonsOf(seq) {
    const n = Math.floor(seq.length / 3), out = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      const a = seq[3 * i], b = seq[3 * i + 1], c = seq[3 * i + 2];
      out[i] = (a > 3 || b > 3 || c > 3) ? -1 : a * 16 + b * 4 + c;
    }
    return out;
  }
  function translate(cods) {
    let s = '';
    for (const c of cods) s += c < 0 ? 'X' : O.genetic_code[c];
    return s;
  }
  const STOPS = [O.codons.indexOf('TAA'), O.codons.indexOf('TAG'),
                 O.codons.indexOf('TGA')];

  /* ---------------- functional classes ---------------- */
  function makeGo(spec) {
    const n = spec.names.length;
    const base = logTable(spec.base), nterm = logTable(spec.nterm),
          period = logTable(spec.period), tmTable = logTable(spec.tm_table);
    const ntShape = spec.nterm_shape, pShape = spec.period_shape;
    const W = O.weights;

    function rates(i, m) {
      const stay = 1 - 1 / O.tm_segment_len;
      const expect = Math.max(spec.tm[i], 1e-3);
      const loop = Math.max(m - expect * O.tm_segment_len, 1);
      return [Math.min(Math.max(expect / loop, 1e-9), 0.5), stay];
    }

    function score(cods) {
      const c = [];
      for (const v of cods) if (v >= 0) c.push(v);
      const m = c.length, out = new Float64Array(n);
      if (!m) return out;
      for (let i = 0; i < n; i++) {
        const nl = spec.nterm_len[i], pn = spec.period_n[i], pw = spec.period_w[i];
        const flat = new Float64Array(m);
        for (let j = 0; j < m; j++) flat[j] = base[i * 64 + c[j]];
        let lp = flat;
        if (pn) {                       // the repeat's register is arbitrary: best phase
          let bestTot = -Infinity, bestArr = null;
          for (let off = 0; off < pn; off++) {
            const cand = new Float64Array(m);
            let tot = 0;
            for (let j = 0; j < m; j++) {
              const ph = (j + off) % pn;
              const v = (1 - pw) * flat[j] +
                        pw * period[(i * pShape[1] + ph) * 64 + c[j]];
              cand[j] = v; tot += v;
            }
            if (tot > bestTot) { bestTot = tot; bestArr = cand; }
          }
          lp = bestArr;
        } else { lp = flat.slice(); }
        if (nl) {
          const k = Math.min(nl, m);
          for (let j = 0; j < k; j++) lp[j] = nterm[(i * ntShape[1] + j) * 64 + c[j]];
        }
        let s;
        if (W.tm > 0) {
          const [enter, stay] = rates(i, m);
          let aLoop = 0, aTm = -60;
          const lStay = Math.log2(1 - enter), lEnter = Math.log2(enter);
          const tStay = Math.log2(stay), tExit = Math.log2(1 - stay);
          for (let j = 0; j < m; j++) {
            const nLoop = lse2(aLoop + lStay, aTm + tExit) + lp[j];
            const nTm = lse2(aLoop + lEnter, aTm + tStay) + tmTable[c[j]];
            aLoop = nLoop; aTm = nTm;
          }
          s = W.composition * lse2(aLoop, aTm);
        } else {
          let t = 0; for (let j = 0; j < m; j++) t += lp[j];
          s = W.composition * t;
        }
        const z = (Math.log(Math.max(m, 1)) - Math.log(spec.len_mean[i])) /
                  O.len_logsigma;
        s += W.length * (-0.5 * z * z - Math.log(O.len_logsigma)) / Math.LN2;
        s += (W.prior === undefined ? 1 : W.prior) *
             Math.log2(Math.max(spec.prior[i], 1e-9));
        out[i] = s;
      }
      return out;
    }

    function sample(i, nCodons, rand) {
      const r = rand || Math.random;
      const nl = spec.nterm_len[i], pn = spec.period_n[i], pw = spec.period_w[i];
      const [enter, stay] = rates(i, nCodons);
      const out = [];
      let inTm = false;
      for (let j = 0; j < nCodons; j++) {
        if (j >= nl) inTm = inTm ? (r() < stay) : (r() < enter);
        const p = new Float64Array(64);
        for (let k = 0; k < 64; k++) {
          let v;
          if (inTm && j >= nl) v = tmTable[k];
          else if (j < nl) v = nterm[(i * ntShape[1] + Math.min(j, ntShape[1] - 1)) * 64 + k];
          else if (pn) v = (1 - pw) * base[i * 64 + k] +
                          pw * period[(i * pShape[1] + (j % pn)) * 64 + k];
          else v = base[i * 64 + k];
          p[k] = Math.pow(2, v);
        }
        for (const st of STOPS) p[st] = 0;
        let tot = 0; for (let k = 0; k < 64; k++) tot += p[k];
        let u = r() * tot, pick = 63;
        for (let k = 0; k < 64; k++) { u -= p[k]; if (u <= 0) { pick = k; break; } }
        out.push(pick);
      }
      return out;
    }

    return { names: spec.names, labels: spec.labels, nparam: spec.nparam,
             lenMean: spec.len_mean, tm: spec.tm, score, sample,
             ranked: cods => rank(spec.names, spec.labels, score(cods)) };
  }

  /* ---------------- sequence ontology ---------------- */
  /* Windows of 19 residues holding at least 16 hydrophobics, counted the way
     qgo.tm_count does: non-overlapping, scanning left to right. */
  const HYDRO = new Set('AILMFVWC'.split(''));
  function tmCount(prot) {
    const w = 19, need = 16;
    if (prot.length < w) return 0;
    let hits = 0, i = 0;
    while (i + w <= prot.length) {
      let c = 0;
      for (let k = 0; k < w; k++) if (HYDRO.has(prot[i + k])) c++;
      if (c >= need) { hits++; i += w; } else i++;
    }
    return hits;
  }

  const PAIR = { 0: 3, 3: 0, 1: 2, 2: 1 };
  function hairpin(seq) {
    const n = seq.length;
    if (n < 30) return 0;
    let best = 0;
    for (let centre = 8; centre < n - 8; centre++) {
      for (const gap of [3, 5, 7, 9, 11]) {
        let i = centre - (gap >> 1) - 1, j = centre + (gap >> 1) + 1;
        let stem = 0, mism = 0;
        while (i >= 0 && j < n) {
          const a = seq[i], b = seq[j];
          if (a <= 3 && b <= 3 && PAIR[a] === b) stem++;
          else { mism++; if (mism > Math.max(2, Math.floor(stem / 5))) break; }
          i--; j++;
        }
        if (stem > best) best = stem;
      }
    }
    return best / n;
  }
  const WHERE = { start: 0, middle: 1, end: 2 };

  function makeSo(spec) {
    const mats = {};
    for (const k in spec.mats) {
      const rows = spec.mats[k].rows, lp = logTable(spec.mats[k].data);
      const m = new Float64Array(rows * 4);
      for (let i = 0; i < rows * 4; i++) m[i] = lp[i] - Math.log2(0.25);
      mats[k] = { rows, m };
    }
    const W = O.so_weights;

    function boxScore(seq, mat, where, win) {
      const n = seq.length, L = mat.rows;
      if (n < L) return -8;
      let a, b;
      const lo = win[0], hi = win[1];
      if (where === 0) { a = lo; b = Math.min(n - L, hi); }
      else if (where === 2) { a = Math.max(0, n - L - hi); b = Math.max(0, n - L - lo); }
      else { a = Math.max(0, lo); b = Math.min(n - L, hi); }
      if (b < a) { a = 0; b = n - L; }
      let best = -8;
      for (let s = a; s <= b; s++) {
        let v = 0, ok = true;
        for (let k = 0; k < L; k++) {
          const x = seq[s + k];
          if (x > 3) { ok = false; break; }
          v += mat.m[k * 4 + x];
        }
        if (ok && v > best) best = v;
      }
      return best;
    }

    function features(seq, nExons) {
      let gc = 0, tot = 0;
      for (const x of seq) if (x <= 3) { tot++; if (x === 1 || x === 2) gc++; }
      return { n: seq.length, gc: tot ? gc / tot : 0.5, exons: nExons,
               hairpin: hairpin(seq) };
    }

    function score(seq, nExons) {
      const f = features(seq, nExons), out = new Float64Array(spec.names.length);
      spec.names.forEach((nm, i) => {
        let s = Math.log2(Math.max(spec.prior[i], 1e-9));
        for (const [mat, where, win] of (spec.boxes[nm] || []))
          s += W.boxes * boxScore(seq, mats[mat], WHERE[where], win);
        const z = (Math.log(Math.max(f.n, 1)) - Math.log(spec.len_mean[i])) /
                  O.so_len_logsigma;
        s += W.length * (-0.5 * z * z) / Math.LN2;
        s += W.gc * (-40 * Math.pow(f.gc - spec.gc[i], 2));
        s += W.exons * (-0.8 * Math.abs(f.exons - spec.exons[i]));
        s += W.hairpin * (-18 * Math.pow(f.hairpin - spec.hairpin[i], 2));
        out[i] = s;
      });
      return out;
    }

    return { names: spec.names, nparam: spec.nparam, terms: O.so_terms,
             features, score, boxScore, mats,
             ranked: (seq, nExons) => rank(spec.names,
                 spec.names.map(n => O.so_terms[n] || ''), score(seq, nExons)) };
  }

  function rank(names, labels, scores) {
    const idx = Array.from(scores.keys()).sort((a, b) => scores[b] - scores[a]);
    const top = scores[idx[0]];
    let z = 0;
    for (const s of scores) z += Math.pow(2, Math.min(s - top, 0));
    return idx.map(i => ({ name: names[i], label: labels[i], score: scores[i],
                           posterior: Math.pow(2, Math.min(scores[i] - top, 0)) / z }));
  }

  return {
    encode, codonsOf, translate, hairpin, tmCount,
    go: makeGo(O.go), fam: makeGo(O.fam), so: makeSo(O.so),
    goCounted: makeGo(O.go_counted), famCounted: makeGo(O.fam_counted),
    soCounted: makeSo(O.so_counted),
    labelsGo: O.labels_go, labelsFam: O.labels_fam, soTerms: O.so_terms,
    AA: O.aa, CODONS: O.codons
  };
})();
