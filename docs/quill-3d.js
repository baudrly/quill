/* Three-dimensional views of the model, drawn on a plain canvas (no library).
   helix     a gene the model has just written, wound as a double helix, every base pair
             coloured by the grammar state that wrote it; a reading head moves along it
   grammar   the 146 states as nodes on one ring per ribbon class, the written transitions
             as arcs, and the same gene's state path as a particle with a fading trail
   terrain   every single-base substitution in a window as a column whose height is the
             bits the model loses (or gains) under the structure it inferred
   Uses the page's live engine `Qu`, so switching quill-k / quill-s redraws everything. */
(function () {
  'use strict';
  const PG = window.QuillPage;   /* live handles to the page's engine and palette */
  const TAU = Math.PI * 2, BASES = 'ACGT';
  const $ = id => document.getElementById(id);
  const hex = c => { const n = parseInt(c.replace('#', ''), 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; };
  const rgba = (c, a, k) => `rgba(${c[0] * k | 0},${c[1] * k | 0},${c[2] * k | 0},${a})`;

  /* ---------- camera and canvas ---------- */
  function Stage(canvas, opt) {
    const st = { yaw: opt.yaw || 0.6, pitch: opt.pitch || 0.35, dist: opt.dist || 900, spin: opt.spin || 0.0025,
                 w: 0, h: 0, drag: null, idle: 0, ctx: canvas.getContext('2d') };
    const fit = () => {
      const r = canvas.getBoundingClientRect(), d = window.devicePixelRatio || 1;
      if (!r.width) return;
      st.w = r.width; st.h = r.height;
      canvas.width = r.width * d; canvas.height = r.height * d;
      st.ctx.setTransform(d, 0, 0, d, 0, 0);
    };
    canvas.addEventListener('pointerdown', e => { st.drag = [e.clientX, e.clientY]; canvas.setPointerCapture(e.pointerId); });
    canvas.addEventListener('pointermove', e => {
      if (!st.drag) return;
      st.yaw += (e.clientX - st.drag[0]) * 0.008;
      st.pitch = Math.max(-1.4, Math.min(1.4, st.pitch + (e.clientY - st.drag[1]) * 0.006));
      st.drag = [e.clientX, e.clientY]; st.idle = 0;
    });
    canvas.addEventListener('pointerup', () => { st.drag = null; });
    canvas.addEventListener('wheel', e => {
      e.preventDefault(); st.dist = Math.max(300, Math.min(2600, st.dist * (1 + Math.sign(e.deltaY) * 0.08)));
    }, { passive: false });
    window.addEventListener('resize', fit);
    st.fit = fit;
    st.step = () => { if (!st.drag && ++st.idle > 60) st.yaw += st.spin; st.cy = Math.cos(st.yaw); st.sy = Math.sin(st.yaw); st.cp = Math.cos(st.pitch); st.sp = Math.sin(st.pitch); };
    /* world -> [screen x, screen y, depth, scale] */
    st.project = (x, y, z) => {
      const x1 = x * st.cy + z * st.sy, z1 = -x * st.sy + z * st.cy;
      const y2 = y * st.cp - z1 * st.sp, z2 = y * st.sp + z1 * st.cp;
      const k = 700 / (z2 + st.dist);
      return [st.w / 2 + x1 * k, st.h / 2 + y2 * k, z2, k];
    };
    st.clear = () => {
      const g = st.ctx.createRadialGradient(st.w / 2, st.h / 2, 20, st.w / 2, st.h / 2, Math.max(st.w, st.h) * 0.7);
      g.addColorStop(0, '#152033'); g.addColorStop(1, '#070a10');
      st.ctx.fillStyle = g; st.ctx.fillRect(0, 0, st.w, st.h);
    };
    return st;
  }

  /* ---------- shared data: one gene written by the current model ---------- */
  let G = null;
  function writeGene() {
    const res = PG.Qu.samplePathGene(PG.META.idx.UTR5, 2200, 600);
    const g = PG.Qu.generateAlongPath(res.path, 1.0, -1, true, 0);
    /* keep the gene and 40 bases of flank: long intergenic runs make a dull picture */
    const ig = PG.RIBN.indexOf('intergenic'), full = g.seq.length;
    let a0 = 0, a1 = full;
    while (a0 < full && PG.Qu.ribbon[g.path[a0]] === ig) a0++;
    while (a1 > a0 && PG.Qu.ribbon[g.path[a1 - 1]] === ig) a1--;
    if (a1 - a0 < 60) { a0 = 0; a1 = full; }
    a0 = Math.max(0, a0 - 40); a1 = Math.min(full, a1 + 40);
    g.seq = g.seq.slice(a0, a1); g.path = g.path.slice(a0, a1);
    const n = g.seq.length, rib = new Uint8Array(n);
    for (let i = 0; i < n; i++) rib[i] = PG.Qu.ribbon[g.path[i]];
    const cols = PG.RIBN.map(nm => hex(PG.RIBC[nm] || '#888888'));
    /* terrain window: centred on the first splice site, else the stop codon, else the middle */
    const sp = PG.RIBN.indexOf('splice');
    let c = rib.indexOf(sp); if (c < 0) c = rib.indexOf(PG.RIBN.indexOf('stop')); if (c < 0) c = n >> 1;
    const a = Math.max(0, Math.min(n - 48, c - 24)), win = g.seq.slice(a, a + 48);
    const scan = PG.Qu.mutationScan(win, PG.Qu.gcBins(win));
    G = { model: PG.CURRENT, seq: g.seq, path: g.path, rib: rib, cols: cols, n: n, win: win, winStart: a, scan: scan.scores, t0: performance.now() };
    buildGraph();
  }

  /* ---------- scene 1: the helix ---------- */
  function drawHelix(st, now) {
    const ctx = st.ctx, n = G.n, R = 120, RISE = 13, TURN = TAU / 10.5, SPAN = 48, GROOVE = 2.3;
    const head = ((now - G.t0) / 90) % n, items = [];
    const lo = Math.max(0, Math.floor(head - SPAN)), hi = Math.min(n - 1, Math.ceil(head + SPAN));
    for (let i = lo; i <= hi; i++) {
      const a = i * TURN, y = (i - head) * RISE, c = G.cols[G.rib[i]];
      const p = st.project(R * Math.cos(a), y, R * Math.sin(a)), q = st.project(R * Math.cos(a + GROOVE), y, R * Math.sin(a + GROOVE));
      const fade = 1 - Math.abs(i - head) / SPAN, hot = Math.max(0, 1 - Math.abs(i - head) / 3);
      items.push({ z: (p[2] + q[2]) / 2, kind: 0, p: p, q: q, c: c, fade: fade, hot: hot, i: i });
      items.push({ z: p[2], kind: 1, p: p, c: c, fade: fade, hot: hot, b: G.seq[i] });
      items.push({ z: q[2], kind: 1, p: q, c: c, fade: fade, hot: hot, b: 3 - G.seq[i] });
    }
    /* the two sugar-phosphate backbones, as continuous strands behind the bases */
    ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.lineWidth = 3;
    for (const off of [0, GROOVE]) {
      for (let i = lo; i < hi; i++) {
        const u = st.project(R * Math.cos(i * TURN + off), (i - head) * RISE, R * Math.sin(i * TURN + off));
        const v = st.project(R * Math.cos((i + 1) * TURN + off), (i + 1 - head) * RISE, R * Math.sin((i + 1) * TURN + off));
        ctx.strokeStyle = `rgba(150,170,205,${0.5 * (1 - Math.abs(i - head) / SPAN) * (0.45 + 0.55 * (1 - (u[2] + R) / (2 * R)))})`;
        ctx.beginPath(); ctx.moveTo(u[0], u[1]); ctx.lineTo(v[0], v[1]); ctx.stroke();
      }
    }
    items.sort((u, v) => v.z - u.z);
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (const it of items) {
      const shade = 0.55 + 0.45 * (1 - (it.z + R) / (2 * R));
      if (it.kind === 0) {
        ctx.strokeStyle = rgba(it.c, 0.25 + 0.6 * it.fade, shade);
        ctx.lineWidth = (3.5 + 5 * it.hot) * it.p[3];
        ctx.beginPath(); ctx.moveTo(it.p[0], it.p[1]); ctx.lineTo(it.q[0], it.q[1]); ctx.stroke();
      } else {
        const r = (7 + 6 * it.hot) * it.p[3];
        if (it.hot > 0) { ctx.fillStyle = rgba(it.c, 0.25 * it.hot, 1); ctx.beginPath(); ctx.arc(it.p[0], it.p[1], r * 2.6, 0, TAU); ctx.fill(); }
        ctx.fillStyle = rgba(it.c, 0.35 + 0.65 * it.fade, shade);
        ctx.beginPath(); ctx.arc(it.p[0], it.p[1], r, 0, TAU); ctx.fill();
        if (it.fade > 0.55 && r > 6) {
          ctx.fillStyle = 'rgba(8,12,18,.85)'; ctx.font = `700 ${r * 1.15 | 0}px JetBrains Mono, monospace`;
          ctx.fillText(BASES[it.b], it.p[0], it.p[1] + 0.5);
        }
      }
    }
    const i = Math.floor(head), nm = PG.RIBN[G.rib[i]];
    hud(st, [`base ${i + 1} of ${n}`, `${BASES[G.seq[i]]}  written from: ${PG.PRETTY[nm] || nm}`, `state ${PG.Qu.stateNames[G.path[i]]}`], G.cols[G.rib[i]]);
    /* minimap of the whole path */
    const mw = st.w - 40, x0 = 20, y0 = st.h - 18;
    for (let x = 0; x < mw; x++) { const k = Math.floor(x / mw * n); ctx.fillStyle = rgba(G.cols[G.rib[k]], 0.9, 1); ctx.fillRect(x0 + x, y0, 1, 7); }
    ctx.fillStyle = '#fff'; ctx.fillRect(x0 + head / n * mw - 1, y0 - 3, 2, 13);
  }

  /* ---------- scene 2: the grammar ---------- */
  let NODES = null, EDGES = null, VISITS = null, RINGS = [];
  function buildGraph() {
    const S = PG.Qu.S, by = PG.RIBN.map(() => []);
    for (let s = 0; s < S; s++) by[PG.Qu.ribbon[s]].push(s);
    NODES = new Array(S); RINGS = [];
    const rings = by.filter(a => a.length).length; let r = 0;
    by.forEach((list, g) => {
      if (!list.length) return;
      const y = (r - (rings - 1) / 2) * 78, rad = 110 + 24 * Math.sqrt(list.length), ph = r * 0.7; r++;
      RINGS.push([y, rad, g]);
      list.forEach((s, k) => { const a = ph + TAU * k / list.length; NODES[s] = [rad * Math.cos(a), y, rad * Math.sin(a), g]; });
    });
    const E = PG.Qu.edges; EDGES = [];
    for (let k = 0; k < E.from.length; k++) if (E.from[k] !== E.to[k]) EDGES.push([E.from[k], E.to[k], Math.pow(2, E.logp[k])]);
    VISITS = new Float32Array(S);
    for (let i = 0; i < G.n; i++) VISITS[G.path[i]]++;
  }
  function drawGrammar(st, now) {
    const ctx = st.ctx, P = NODES.map(v => st.project(v[0], v[1], v[2]));
    ctx.lineWidth = 1; ctx.font = '600 11px Inter, system-ui, sans-serif'; ctx.textAlign = 'left';
    for (const [y, rad, g] of RINGS) {
      ctx.strokeStyle = rgba(G.cols[g], 0.22, 1); ctx.beginPath();
      for (let k = 0; k <= 48; k++) { const q = st.project(rad * Math.cos(TAU * k / 48), y, rad * Math.sin(TAU * k / 48)); k ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); }
      ctx.stroke();
      const t = st.project(rad + 14, y, 0); ctx.fillStyle = rgba(G.cols[g], 0.9, 1);
      ctx.fillText(PG.PRETTY[PG.RIBN[g]] || PG.RIBN[g], t[0], t[1]);
    }
    for (const [a, b, p] of EDGES) {
      ctx.strokeStyle = `rgba(140,170,210,${0.1 + 0.4 * Math.sqrt(p)})`;
      ctx.beginPath(); ctx.moveTo(P[a][0], P[a][1]); ctx.lineTo(P[b][0], P[b][1]); ctx.stroke();
    }
    /* the gene's path: one hop per changed state, with a fading trail */
    const hops = []; for (let i = 0; i < G.n; i++) if (!i || G.path[i] !== G.path[i - 1]) hops.push(i);
    const f = ((now - G.t0) / 260) % hops.length, k = Math.floor(f), frac = f - k;
    for (let d = Math.min(k, 26); d >= 1; d--) {
      const a = P[G.path[hops[k - d]]], b = P[G.path[hops[k - d + 1]]];
      ctx.strokeStyle = `rgba(94,234,212,${0.85 * (1 - d / 27)})`; ctx.lineWidth = 3 * (1 - d / 30);
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    }
    const order = NODES.map((_, s) => s).sort((u, v) => P[v][2] - P[u][2]);
    const cur = G.path[hops[k]], nxt = G.path[hops[Math.min(k + 1, hops.length - 1)]];
    for (const s of order) {
      const c = G.cols[NODES[s][3]], r = (4.5 + 2.4 * Math.log1p(VISITS[s])) * P[s][3];
      ctx.fillStyle = rgba(c, s === cur ? 1 : 0.8, 0.6 + 0.4 * (1 - (P[s][2] + 300) / 600));
      ctx.beginPath(); ctx.arc(P[s][0], P[s][1], Math.max(1.5, r), 0, TAU); ctx.fill();
    }
    const A = P[cur], B = P[nxt], x = A[0] + (B[0] - A[0]) * frac, y = A[1] + (B[1] - A[1]) * frac;
    ctx.fillStyle = 'rgba(255,255,255,.22)'; ctx.beginPath(); ctx.arc(x, y, 16, 0, TAU); ctx.fill();
    ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(x, y, 5, 0, TAU); ctx.fill();
    const nm = PG.RIBN[PG.Qu.ribbon[cur]];
    hud(st, [`${PG.Qu.S} states in this file, ${EDGES.length} transitions between them`, `now in: ${PG.Qu.stateNames[cur]} (${PG.PRETTY[nm] || nm})`,
             `base ${hops[k] + 1}; stays ${(hops[k + 1] || G.n) - hops[k]} bases`], G.cols[PG.Qu.ribbon[cur]]);
  }

  /* ---------- scene 3: the mutation terrain ---------- */
  function drawTerrain(st, now) {
    const ctx = st.ctx, n = G.win.length, DX = 21, DZ = 32, K = 26, cols = [];
    const grow = Math.min(1, (now - G.t0) / 1400);
    for (let t = 0; t < n; t++) for (let b = 0; b < 4; b++) {
      const v = b === G.win[t] ? 0 : G.scan[t * 4 + b];
      cols.push({ x: (t - n / 2) * DX, z: (b - 1.5) * DZ, h: Math.max(-90, Math.min(190, v * K)) * grow, v: v, t: t, b: b, ref: b === G.win[t] });
    }
    for (const c of cols) c.d = st.project(c.x, 0, c.z)[2];
    cols.sort((u, v) => v.d - u.d);
    for (const c of cols) {
      const rc = G.cols[G.rib[G.winStart + c.t]], w = 8;
      const quad = (pts, fill) => { ctx.fillStyle = fill; ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); ctx.closePath(); ctx.fill(); };
      const base = [[-w, -w], [w, -w], [w, w], [-w, w]].map(o => st.project(c.x + o[0], 60, c.z + o[1]));
      if (c.ref) { quad(base, rgba(rc, 0.95, 1)); continue; }
      const col = c.v >= 0 ? [239, 90, 90] : [80, 200, 170], top = [[-w, -w], [w, -w], [w, w], [-w, w]].map(o => st.project(c.x + o[0], 60 - c.h, c.z + o[1]));
      const a = 0.35 + 0.65 * Math.min(1, Math.abs(c.v) / 4);
      for (let e = 0; e < 4; e++) { const f = (e + 1) % 4; quad([base[e], base[f], top[f], top[e]], rgba(col, a, 0.55 + 0.12 * e)); }
      quad(top, rgba(col, a, 1));
    }
    ctx.font = '600 12px JetBrains Mono, monospace'; ctx.textAlign = 'center'; ctx.fillStyle = '#9aabc4';
    for (let b = 0; b < 4; b++) { const p = st.project(-(n / 2 + 2) * DX, 60, (b - 1.5) * DZ); ctx.fillText(BASES[b], p[0], p[1]); }
    let best = 0; for (let k = 0; k < n * 4; k++) if (G.scan[k] > G.scan[best]) best = k;
    hud(st, [`${n} bases x 3 substitutions, window at base ${G.winStart + 1}`, 'red: the model loses bits; green: it gains',
             `most damaging: ${BASES[G.win[best >> 2]]}${G.winStart + (best >> 2) + 1}${BASES[best & 3]}  (+${G.scan[best].toFixed(2)} bits)`], [239, 90, 90]);
  }

  function hud(st, lines, c) {
    const ctx = st.ctx; ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
    ctx.fillStyle = 'rgba(11,15,22,.72)'; ctx.fillRect(12, 12, 330, 20 + 18 * lines.length);
    ctx.fillStyle = rgba(c, 1, 1); ctx.fillRect(12, 12, 4, 20 + 18 * lines.length);
    lines.forEach((l, i) => { ctx.fillStyle = i === 1 ? '#e6edf7' : '#9aabc4'; ctx.font = `${i === 1 ? 600 : 400} 12.5px Inter, system-ui, sans-serif`; ctx.fillText(l, 26, 34 + 18 * i); });
  }

  /* ---------- wiring ---------- */
  const SCENES = { helix: { draw: drawHelix, cam: { yaw: 0.5, pitch: 0.18, dist: 820, spin: 0.004 },
                            text: 'A gene the model has just written, wound as a double helix. Each base pair takes the colour of the grammar state that wrote it; the bright pair is the reading head.' },
                   grammar: { draw: drawGrammar, cam: { yaw: 0.3, pitch: 0.42, dist: 1150, spin: 0.002 },
                              text: 'The whole grammar: one ring per kind of state, one arc per written transition. The white particle is the same gene, hopping from state to state as it is written; node size is time spent there.' },
                   terrain: { draw: drawTerrain, cam: { yaw: 0.45, pitch: 0.7, dist: 1050, spin: 0.0015 },
                              text: 'Every single-base substitution around the first splice site of that gene (the stop codon if it has no intron). Flat coloured tiles are the reference base; column height is the change in bits under the structure the model inferred.' } };
  let stage = null, scene = 'helix', raf = 0;
  function pick(name) {
    scene = name;
    document.querySelectorAll('#v3-pick button').forEach(b => b.classList.toggle('on', b.dataset.s === name));
    $('v3-text').textContent = SCENES[name].text;
    Object.assign(stage, SCENES[name].cam, { idle: 0 });
    G.t0 = performance.now();
  }
  function frame(now) {
    raf = 0;
    if (!$('t-v3').classList.contains('on')) return;
    if (!stage.w) stage.fit();
    stage.step(); stage.clear(); SCENES[scene].draw(stage, now);
    raf = requestAnimationFrame(frame);
  }
  function legend() {
    const used = new Set(G.rib);
    $('v3-legend').innerHTML = PG.RIBN.map((nm, i) => used.has(i) ? `<span class="chip"><i style="display:inline-block;width:9px;height:9px;border-radius:2px;background:${PG.RIBC[nm]};margin-right:6px"></i>${PG.PRETTY[nm] || nm}</span>` : '').join('');
  }
  window.Quill3D = {
    open() {
      if (!stage) {
        stage = Stage($('v3-canvas'), SCENES.helix.cam);
        document.querySelectorAll('#v3-pick button').forEach(b => b.onclick = () => pick(b.dataset.s));
        $('v3-new').onclick = () => { writeGene(); legend(); };
      }
      if (!G || G.model !== PG.CURRENT) { writeGene(); legend(); pick(scene); }
      stage.fit();
      if (!raf) raf = requestAnimationFrame(frame);
    }
  };
})();
