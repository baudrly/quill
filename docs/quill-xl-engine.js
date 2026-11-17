/* quill-xl2-nano in the browser: the JavaScript twin of qprot.py, qrna.py and qxl.py.

   describeProtein(prot)   written physicochemistry, signal peptide, transmembrane
                           helices and topology, targeting, disorder, coiled coils,
                           short linear motifs                              (qprot.describe)
   prosite(prot)           PROSITE pattern hits                             (qprot.prosite)
   describeRna(u5,c,u3)    written mRNA description                         (qrna.describe)
   predict(prot, rna)      every GO term's probability, graph-consistent    (qxl.XL.predict)
   annotate(text)          all of the above from pasted DNA, mRNA or protein
   Checked against Python by web/xlparity.js. */
window.QuillXL = (function () {
  const X = window.QUILL_XL;
  if (!X) throw new Error('quill-xl-nano.js must load first');
  const M = X.model, K = X.know, R = K.rules;
  const AA = K.AA, AIDX = {};
  for (let i = 0; i < AA.length; i++) AIDX[AA[i]] = i;
  const vec = s => AA.split('').map(a => s[a]);
  const KD = vec(K.KD), EIS = vec(K.EIS), DIS = vec(K.DIS), HEL = vec(K.HEL),
        SHE = vec(K.SHE), MASS = vec(K.MASS);

  // Python's round(x, n): round half to even on the decimal value
  function pyround(x, n) {
    const f = Math.pow(10, n), v = x * f, r = Math.round(v);
    if (Math.abs(v - Math.trunc(v)) === 0.5 && r % 2 !== 0) return (r - 1) / f;
    return r / f;
  }
  const mean = a => a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0;
  function smooth(x, w) {
    if (x.length < w) return [];
    const c = [0];
    for (const v of x) c.push(c[c.length - 1] + v);
    const out = [];
    for (let i = 0; i + w <= x.length; i++) out.push((c[i + w] - c[i]) / w);
    return out;
  }
  function runs(mask, minGap) {
    const out = [];
    let i = 0;
    const n = mask.length;
    while (i < n) {
      if (mask[i]) {
        let j = i;
        while (j < n && mask[j]) j++;
        if (out.length && i - out[out.length - 1][1] < (minGap || 0)) out[out.length - 1][1] = j;
        else out.push([i, j]);
        i = j;
      } else i++;
    }
    return out;
  }
  const searchsorted = (edges, v) => { let k = 0; for (const e of edges) if (e <= v) k++; return k; };
  const findall = (s, rx) => (s.match(new RegExp(rx.source, 'g')) || []).length;
  function countOverlap(s, sub) {
    let n = 0, i = s.indexOf(sub);
    while (i >= 0) { n++; i = s.indexOf(sub, i + 1); }
    return n;
  }

  /* ---------------- protein ---------------- */
  function isoelectric(prot) {
    const cnt = {};
    for (const k of 'KRHDECY') cnt[k] = prot.split(k).length - 1;
    const P = K.PKA;
    const charge = ph => {
      let pos = 1 / (1 + Math.pow(10, ph - P.Nterm));
      for (const k of 'KRH') pos += cnt[k] / (1 + Math.pow(10, ph - P[k]));
      let neg = 1 / (1 + Math.pow(10, P.Cterm - ph));
      for (const k of 'DECY') neg += cnt[k] / (1 + Math.pow(10, P[k] - ph));
      return pos - neg;
    };
    let lo = 0, hi = 14;
    for (let i = 0; i < 40; i++) { const mid = (lo + hi) / 2; if (charge(mid) > 0) lo = mid; else hi = mid; }
    return [pyround((lo + hi) / 2, 3), pyround(charge(7), 3)];
  }
  function tmHelices(x) {
    const h = smooth(x.map(i => KD[i]), R.TM_WINDOW);
    return runs(h.map(v => v >= R.TM_THRESHOLD), R.TM_MIN_GAP)
      .map(([a, b]) => [a, b + R.TM_WINDOW - 1]);
  }
  function topology(prot, hel) {
    if (!hel.length) return null;
    const [a, b] = hel[0];
    const nLoop = prot.slice(Math.max(0, a - 15), a);
    const cLoop = hel.length === 1 ? prot.slice(b, b + 15) : prot.slice(b, hel[1][0]).slice(0, 15);
    const kr = s => (s.match(/[KR]/g) || []).length;
    return kr(nLoop) >= kr(cLoop) ? 'N_in' : 'N_out';
  }
  function signalPeptide(prot, x) {
    const head = x.slice(0, R.SP_SEARCH);
    if (head.length < 20) return [false, null];
    const h = smooth(head.map(i => KD[i]), R.SP_H_WINDOW);
    for (let s = 1; s < Math.min(h.length, 20); s++) {
      if (h[s] >= R.SP_H_THRESHOLD) {
        let e = s + R.SP_H_WINDOW;
        while (e < head.length && KD[head[e]] > 0) e++;
        if (e - s > R.SP_MAX_H) return [false, null];
        for (let c = e + R.SP_C_MIN; c <= Math.min(e + R.SP_C_MAX, prot.length - 1); c++) {
          if (R.SP_SMALL.includes(prot[c - 1]) && R.SP_SMALL.includes(prot[c - 3]) &&
              prot[c] !== 'P' && prot[c - 1] !== 'P') return [true, c];
        }
      }
    }
    return [false, null];
  }
  function moment(x, angle, w) {
    if (x.length < w) return 0;
    const th = angle * Math.PI / 180;
    let best = 0;
    for (let s = 0; s + w <= x.length; s++) {
      let a = 0, b = 0;
      for (let k = 0; k < w; k++) { const h = EIS[x[s + k]]; a += h * Math.cos(th * k); b += h * Math.sin(th * k); }
      best = Math.max(best, Math.hypot(a, b) / w);
    }
    return best;
  }
  function coiledCoil(prot) {
    const n = prot.length, W = R.CC_WINDOW;
    if (n < W) return 0;
    const cov = new Uint8Array(n);
    for (let s = 0; s + W <= n; s += 7) {
      let best = 0;
      for (let reg = 0; reg < 7; reg++) {
        let ad = 0, nad = 0, eg = 0, neg = 0;
        for (let k = 0; k < W; k++) {
          const ph = (k + reg) % 7, c = prot[s + k];
          if (ph === 0 || ph === 3) { nad++; if (R.CC_A_D.includes(c)) ad++; }
          if (ph === 4 || ph === 6) { neg++; if (R.CC_E_G.includes(c)) eg++; }
        }
        best = Math.max(best, 0.6 * ad / nad + 0.4 * eg / neg);
      }
      if (best >= R.CC_MIN_SCORE) for (let k = 0; k < W; k++) cov[s + k] = 1;
    }
    return mean(Array.from(cov));
  }
  function lowComplexity(x) {
    const n = x.length, w = R.LC_WINDOW;
    if (n < w) return 0;
    const cov = new Uint8Array(n);
    for (let s = 0; s + w <= n; s++) {
      const c = new Array(20).fill(0);
      for (let k = 0; k < w; k++) c[x[s + k]]++;
      let H = 0;
      for (const v of c) if (v) { const p = v / w; H -= p * Math.log2(p); }
      if (H <= R.LC_MAX_BITS) for (let k = 0; k < w; k++) cov[s + k] = 1;
    }
    return mean(Array.from(cov));
  }
  const SLIM_RX = K.slims.map(([n, r]) => [n, new RegExp(r)]);
  function describeProtein(raw) {
    const prot = raw.toUpperCase().split('').filter(c => c in AIDX).join('');
    const x = prot.split('').map(c => AIDX[c]);
    const n = x.length;
    if (!n) return null;
    const comp = new Array(20).fill(0);
    for (const i of x) comp[i] += 1 / n;
    const [pi, charge] = isoelectric(prot);
    let hel = tmHelices(x);
    const [sp, cleave] = signalPeptide(prot, x);
    if (sp) hel = hel.filter(h => h[0] > cleave - 5);
    const head = prot.slice(0, R.MITO_SEARCH);
    const basic = (head.match(/[RK]/g) || []).length, acidic = (head.match(/[DE]/g) || []).length;
    const mom = moment(x.slice(0, R.MITO_SEARCH), R.MITO_MOMENT_ANGLE, R.MITO_MOMENT_WINDOW);
    const mito = basic >= R.MITO_MIN_BASIC && acidic <= R.MITO_MAX_ACIDIC &&
                 mom >= R.MITO_MOMENT_MIN && prot[0] === 'M';
    let gpi = false;
    if (sp && n >= R.GPI_TAIL) {
      const tail = x.slice(-R.GPI_TAIL), h = smooth(tail.map(i => KD[i]), R.GPI_H_WINDOW);
      const hyd = h.length && Math.max(...h.slice(-12)) >= R.GPI_H_THRESHOLD;
      const om = prot.slice(-R.GPI_TAIL, -R.GPI_TAIL + 12).split('').some(c => R.GPI_OMEGA.includes(c));
      gpi = !!(hyd && om);
    }
    const dis = smooth(x.map(i => DIS[i]), R.DIS_WINDOW), dmask = dis.map(v => v >= R.DIS_THRESHOLD);
    const longDis = runs(dmask).filter(([a, b]) => b - a >= R.DIS_LONG)
      .map(([a, b]) => [a, b + R.DIS_WINDOW - 1]);
    const slims = {};
    for (const [nm, rx] of SLIM_RX) slims[nm] = findall(prot, rx);
    let massSum = 0;
    for (const i of x) massSum += MASS[i];
    return {
      length: n, mass_kda: pyround((massSum + K.WATER) / 1000, 2), pI: pi, charge_pH7: charge,
      gravy: pyround(mean(x.map(i => KD[i])), 3),
      aromatic: pyround(comp[AIDX.F] + comp[AIDX.W] + comp[AIDX.Y], 4),
      cysteine: pyround(comp[AIDX.C], 4),
      helix_propensity: pyround(mean(x.map(i => HEL[i])), 3),
      sheet_propensity: pyround(mean(x.map(i => SHE[i])), 3),
      disorder_fraction: pyround(dmask.length ? mean(dmask.map(Number)) : 0, 3),
      long_disordered_regions: longDis,
      coiled_coil_fraction: pyround(coiledCoil(prot), 3),
      low_complexity_fraction: pyround(lowComplexity(x), 3),
      tm_helices: hel, topology: topology(prot, hel), signal_peptide: sp, cleavage_site: cleave,
      mito_presequence: mito, gpi_anchor: gpi, slims, sequence: prot
    };
  }
  const PROSITE = K.prosite.map(([a, n, d, rx]) => [a, n, d, new RegExp(rx)]);
  const prosite = prot => PROSITE.filter(p => p[3].test(prot)).map(p => p[0]);
  const prositeInfo = {};
  for (const [a, n, d] of K.prosite) prositeInfo[a] = [n, d];

  function proteinFeatures(d) {
    const f = [];
    for (const k in K.prot_bins) f.push(`${k}:${searchsorted(K.prot_bins[k], d[k])}`);
    const ntm = d.tm_helices.length;
    for (const [a, b] of K.tm_classes) if (a <= ntm && ntm <= b) { f.push(`tm:${a}-${b}`); break; }
    if (ntm) f.push(`topology:${d.topology}`);
    if (d.signal_peptide) {
      f.push('signal_peptide');
      f.push(ntm === 0 ? 'secreted_like' : (ntm === 1 ? 'type_I_membrane' : 'signal_multi_tm'));
    } else if (ntm && d.tm_helices[0][0] < 40) f.push('signal_anchor');
    if (d.gpi_anchor) f.push('gpi_anchor');
    if (d.mito_presequence) f.push('mito_presequence');
    if (d.long_disordered_regions.length) f.push('long_disorder:' + Math.min(d.long_disordered_regions.length, 3));
    const per100 = 100 / Math.max(d.length, 1);
    for (const nm in d.slims) {
      const c = d.slims[nm];
      if (!c) continue;
      f.push(`slim:${nm}:` + (c === 1 ? '1' : (c <= 3 ? '2-3' : '4+')));
      if (/^(phospho_|N_glyc|SUMO|D_box|ER_export)/.test(nm))
        f.push(`slimdens:${nm}:${searchsorted([0.5, 1, 2, 4], c * per100)}`);
    }
    if (d.signal_peptide && (d.slims.N_glycosylation || 0) >= 3) f.push('signal_and_glycosylated');
    return f;
  }

  /* ---------------- RNA ---------------- */
  const RBP = Object.entries(K.rbp).map(([k, r]) => [k, new RegExp(r)]);
  const G4 = new RegExp(K.g4), DRACH = new RegExp(K.drach), PRE = new RegExp(K.pre);
  const KOZAK = new RegExp('^' + K.kozak);
  function cai(cds) {
    let s = 0, n = 0;
    for (let i = 0; i + 3 <= cds.length; i += 3) {
      const c = cds.slice(i, i + 3);
      if (!(c in K.cai_w) || 'MW*'.includes(K.code[c])) continue;
      s += Math.log(Math.max(K.cai_w[c], 1e-4)); n++;
    }
    return n ? Math.exp(s / n) : 0;
  }
  function describeRna(u5, cds, u3, junctions, stopEnd) {
    const tail = u3 ? u3.slice(-60) : '';
    let top = 0;
    if (u5[0] === 'C') while (top + 1 < u5.length && 'CT'.includes(u5[top + 1])) top++;
    const ctx = (u5.length >= 3 ? u5.slice(-3) : '') + (cds.length >= 4 ? cds.slice(0, 4) : '');
    let gc3 = 0;
    for (let i = 2; i < cds.length; i += 3) if ('GC'.includes(cds[i])) gc3++;
    gc3 /= Math.max(Math.floor(cds.length / 3), 1);
    const gc = s => ((s.match(/[GC]/g) || []).length) / Math.max(s.length, 1);
    const rbp = {}, mir = {};
    for (const [k, rx] of RBP) rbp[k] = findall(u3, rx);
    for (const k in K.mir) mir[k] = countOverlap(u3, K.mir[k]);
    return {
      utr5_len: u5.length, utr3_len: u3.length, cds_len: cds.length,
      exons: (junctions || []).length + 1,
      utr5_gc: pyround(gc(u5), 3), utr3_gc: pyround(gc(u3), 3), top_tract: top,
      uorfs: findall(u5, /ATG/), kozak_strong: ctx.length === 7 ? KOZAK.test(ctx) : false,
      gc3: pyround(gc3, 3), cai_written: pyround(cai(cds), 4),
      g4_5utr: G4.test(u5), g4_3utr: G4.test(u3), ire_5utr: u5.includes(K.ire),
      are_count: countOverlap(u3, K.are), are_nonamer: u3.includes(K.are9),
      pre_count: findall(u3, PRE), cpe_count: countOverlap(u3, 'TTTTAT'),
      pas: K.pas.find(p => tail.includes(p)) || null,
      histone_stem_loop: (u3 ? u3.slice(0, 200) : '').includes(K.histone_sl) || tail.includes(K.histone_sl),
      m6a_per_kb: pyround(1000 * findall(u3, DRACH) / Math.max(u3.length, 1), 2),
      nmd_junction_in_3utr: (junctions || []).some(j => j - stopEnd > 55),
      rbp_3utr: rbp, mirna_sites: mir
    };
  }
  function rnaFeatures(d) {
    const f = [];
    for (const k in K.rna_bins) if (k in d) f.push(`rna:${k}:${searchsorted(K.rna_bins[k], d[k])}`);
    for (const k of ['kozak_strong', 'g4_5utr', 'g4_3utr', 'ire_5utr', 'are_nonamer',
                     'histone_stem_loop', 'nmd_junction_in_3utr']) if (d[k]) f.push(`rna:${k}`);
    if (d.top_tract >= K.top_min) f.push('rna:top_tract');
    f.push(`rna:pas:${d.pas === null ? 'None' : d.pas}`);
    for (const k of ['pre_count', 'cpe_count']) if (d[k]) f.push(`rna:${k}:` + (d[k] === 1 ? '1' : '2+'));
    for (const k in d.rbp_3utr) { const c = d.rbp_3utr[k]; if (c) f.push(`rna:rbp:${k}:` + (c === 1 ? '1' : (c <= 4 ? '2-4' : '5+'))); }
    for (const k in d.mirna_sites) { const c = d.mirna_sites[k]; if (c) f.push(`rna:mir:${k}:` + (c === 1 ? '1' : '2+')); }
    return f;
  }

  /* ---------------- the model ---------------- */
  const T = M.names.length;
  const VOC = {};
  for (const k in M.vocab) { VOC[k] = {}; M.vocab[k].forEach((f, i) => VOC[k][f] = i); }
  const NSIDX = { BP: [], MF: [], CC: [] };
  M.ns.forEach((n, i) => NSIDX[n].push(i));
  const TIDX = {};
  M.names.forEach((t, i) => TIDX['GO:' + t] = i);
  function featureExpert(name, feats) {
    const D = M.D[name], voc = VOC[name];
    const on = new Set(feats.map(f => voc[f]).filter(v => v !== undefined));
    const out = new Float64Array(T);
    const tau = M.tau[name];
    for (let t = 0; t < T; t++) {
      let S = 0;
      for (let k = D.indptr[t]; k < D.indptr[t + 1]; k++) if (on.has(D.indices[k])) S += D.data[k];
      const p = M.prior[t], lo = Math.log2(p / Math.max(1 - p, 1e-6));
      const z = Math.max(-30, Math.min(30, lo + S / tau));
      out[t] = 1 / (1 + Math.pow(2, -z));
    }
    return out;
  }
  function signatureExpert(prot) {
    const out = new Float64Array(T);
    for (const r of (window.QUILL_SIGNATURES || [])) {
      if (!r.go || !(r.go in TIDX)) continue;
      if (new RegExp(r.pattern).test(prot)) out[TIDX[r.go]] = Math.max(out[TIDX[r.go]], r.written);
    }
    return out;
  }
  function predict(prot, rnaDesc, protDesc, hits) {
    const d = protDesc || describeProtein(prot);
    const ph = hits || prosite(d.sequence);
    const ex = {
      protein: () => featureExpert('protein', proteinFeatures(d)),
      prosite: () => featureExpert('prosite', ph.map(a => 'prosite:' + a)),
      rna: () => featureExpert('rna', rnaDesc ? rnaFeatures(rnaDesc) : []),
      signatures: () => signatureExpert(d.sequence)
    };
    const L = new Float64Array(T);
    for (const e of M.experts) {
      const P = ex[e]();
      for (const ns of ['BP', 'MF', 'CC']) {
        const w = (M.weights[ns] || {})[e] || 0;
        if (!w) continue;
        for (const t of NSIDX[ns]) L[t] += w * Math.log1p(-Math.min(P[t], 0.999));
      }
    }
    const out = new Float64Array(T);
    for (let t = 0; t < T; t++) out[t] = Math.max(1 - Math.exp(L[t]), M.prior[t] * 0.999);
    for (const i of M.order)
      for (let k = M.parent_ptr[i]; k < M.parent_ptr[i + 1]; k++) {
        const j = M.parent_idx[k];
        if (out[i] > out[j]) out[j] = out[i];
      }
    return out;
  }
  function specific(P, ns, th, n) {
    // confident terms none of whose children is also confident
    // confident, and at least 1.2x as likely as the term's base rate among genes
    const conf = new Set(NSIDX[ns].filter(t => P[t] >= th && P[t] >= 1.2 * M.prior[t]));
    const covered = new Set();
    for (const i of conf)
      for (let k = M.parent_ptr[i]; k < M.parent_ptr[i + 1]; k++) covered.add(M.parent_idx[k]);
    return [...conf].filter(t => !covered.has(t))
      .sort((a, b) => P[b] / M.prior[b] - P[a] / M.prior[a]).slice(0, n)
      .map(t => ({ go: 'GO:' + M.names[t], name: M.term_names[t], p: P[t], prior: M.prior[t] }));
  }
  function ecNumbers(P, th) {
    const best = {};
    for (const i in M.ec) if (P[+i] >= th) for (const e of M.ec[i]) best[e] = Math.max(best[e] || 0, P[+i]);
    return Object.entries(best).sort((a, b) => b[1] - a[1]);
  }

  /* ---------------- from pasted text ---------------- */
  const COMP = { A: 'T', C: 'G', G: 'C', T: 'A', N: 'N' };
  const revcomp = s => s.split('').reverse().map(c => COMP[c] || 'N').join('');
  function longestOrf(dna) {
    let best = null;
    for (const [strand, s] of [['+', dna], ['-', revcomp(dna)]])
      for (let f = 0; f < 3; f++) {
        let i = f;
        while (i + 3 <= s.length) {
          if (s.slice(i, i + 3) === 'ATG') {
            let j = i;
            while (j + 3 <= s.length && !['TAA', 'TAG', 'TGA'].includes(s.slice(j, j + 3))) j += 3;
            const end = Math.min(j + 3, s.length);
            if (!best || (j - i) > best.len) best = { strand, start: i, end, len: j - i, s };
            i = j + 3;
          } else i += 3;
        }
      }
    return best;
  }
  function annotate(text) {
    const clean = text.replace(/^>.*$/gm, '').toUpperCase().replace(/[^A-Z]/g, '');
    const isDna = clean.length && /^[ACGTUN]+$/.test(clean);
    let prot, rna = null, orf = null;
    if (isDna) {
      const dna = clean.replace(/U/g, 'T');
      orf = longestOrf(dna);
      if (!orf || orf.len < 300) return { error: 'no open reading frame of 100 codons or more', dna };
      const cds = orf.s.slice(orf.start, orf.end);
      prot = '';
      for (let i = 0; i + 3 <= cds.length; i += 3) { const a = K.code[cds.slice(i, i + 3)]; if (a === '*') break; prot += a || 'X'; }
      rna = describeRna(orf.s.slice(0, orf.start), cds, orf.s.slice(orf.end), [], cds.length);
    } else prot = clean;
    const d = describeProtein(prot);
    const hits = prosite(d.sequence);
    const P = predict(prot, rna, d, hits);
    return {
      input: isDna ? 'nucleotide' : 'protein', orf, protein: d, rna,
      prosite: hits.map(a => ({ acc: a, name: prositeInfo[a][0], desc: prositeInfo[a][1] })),
      go: { MF: specific(P, 'MF', 0.3, 8), BP: specific(P, 'BP', 0.3, 8), CC: specific(P, 'CC', 0.3, 6) },
      ec: ecNumbers(P, 0.3).slice(0, 5), P
    };
  }

  return { describeProtein, prosite, proteinFeatures, describeRna, rnaFeatures, predict,
           annotate, specific, longestOrf, nterms: T, nparam: M.nparam, size: M.size };
})();
