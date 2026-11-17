/* Load index.html under jsdom, run its scripts in order, and check that the page
   builds, that both models load, and that switching models changes the header.
   Canvas is stubbed: jsdom has no 2D backend, so drawing code runs against a no-op. */
const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

const here = __dirname;
const html = fs.readFileSync(path.join(here, 'index.html'), 'utf8');
const dom = new JSDOM(html, { runScripts: 'outside-only', pretendToBeVisual: true,
                              url: 'https://localhost/' });
const w = dom.window;

w.HTMLCanvasElement.prototype.getContext = function () {
  const noop = () => {};
  const self = this;
  return new Proxy({}, {
    get: (t, k) => {
      if (k === 'measureText') return () => ({ width: 10 });
      if (k === 'canvas') return self;
      if (k === 'createLinearGradient' || k === 'createRadialGradient')
        return () => ({ addColorStop: noop });
      if (k === 'getImageData') return () => ({ data: new Uint8ClampedArray(4) });
      if (k in t) return t[k];
      return noop;
    },
    set: () => true
  });
};
Object.defineProperty(w.HTMLCanvasElement.prototype, 'clientWidth', { get: () => 800 });
Object.defineProperty(w.HTMLCanvasElement.prototype, 'clientHeight', { get: () => 220 });
w.HTMLElement.prototype.getBoundingClientRect = function () {
  return { width: 800, height: 220, top: 0, left: 0, right: 800, bottom: 220, x: 0, y: 0 };
};

const errors = [];
w.addEventListener('error', e => errors.push(String(e.message || e)));

// run the page's scripts in document order, external ones from disk
const scripts = [...dom.window.document.querySelectorAll('script')];
for (const s of scripts) {
  const code = s.src ? fs.readFileSync(path.join(here, s.src.replace(/^.*\//, '')), 'utf8')
                     : s.textContent;
  try {
    w.eval(code);
  } catch (err) {
    errors.push(`${s.src || 'inline'}: ${err.message}`);
    break;
  }
}

function text(id) {
  const el = w.document.getElementById(id);
  return el ? (el.textContent || '').trim() : null;
}

const checks = [];
const push = (name, ok, detail) => checks.push({ name, ok, detail });

push('no script errors', errors.length === 0, errors.join(' | '));
push('both models present', !!(w.QUILL_K && w.QUILL_S),
     `k=${!!w.QUILL_K} s=${!!w.QUILL_S}`);
push('engine is a factory', typeof w.QuillInit === 'function', typeof w.QuillInit);
const hdr = text('hdr') || '';
push('header rendered', hdr.includes('parameters') && hdr.includes('bits/base'),
     hdr.slice(0, 90));
push('parameter count shown', text('fparam') === '1,406,988', text('fparam'));
push('results table filled', (text('abttab') || '').includes('bits/base'),
     (text('abttab') || '').slice(0, 60));
push('demo loci loaded', Array.isArray(w.QUILL_LOCI) && w.QUILL_LOCI.length > 0,
     w.QUILL_LOCI ? w.QUILL_LOCI.length + ' loci' : 'none');

// model switch
try {
  w.switchModel('quill-k');
  push('switch to quill-k', text('fparam') === '10,252', text('fparam'));
  const h2 = text('hdr') || '';
  push('header shows written provenance', h2.includes('written'), h2.slice(0, 90));
  w.switchModel('quill-s');
  push('switch back to quill-s', text('fparam') === '1,406,988', text('fparam'));
} catch (err) {
  push('model switching', false, err.message);
}

// the ontology tab
try {
  push('ontology engine present', !!w.QuillOnt, typeof w.QuillOnt);
  push('ontology examples loaded',
       (w.QUILL_ONT_EXAMPLES.coding || []).length >= 8 &&
       (w.QUILL_ONT_EXAMPLES.noncoding || []).length >= 6,
       `${w.QUILL_ONT_EXAMPLES.coding.length} coding, ` +
       `${w.QUILL_ONT_EXAMPLES.noncoding.length} non-coding`);
  push('functional table filled', (text('onttabgo') || '').length > 40,
       (text('onttabgo') || '').slice(0, 40));
  push('family table filled', (text('onttabfam') || '').length > 40,
       (text('onttabfam') || '').slice(0, 40));
  push('SO table filled', (text('sotab') || '').includes('SO:'),
       (text('sotab') || '').slice(0, 40));
  push('signature table filled', (text('sigtab') || '').length > 10,
       (text('sigtab') || '').slice(0, 50));
  push('all written signatures compile as JS regexes',
       (w.QUILL_SIGNATURES || []).every(r => { try { new RegExp(r.pattern); return true; } catch (e) { return false; } }),
       (w.QUILL_SIGNATURES || []).length + ' rules');
  push('generated protein shown', (text('genprot') || '').length > 80,
       (text('genprot') || '').length + ' residues');

  // the written weights should put each shipped example in its true class
  const O = w.QuillOnt;
  let right = 0, tried = 0;
  for (const e of w.QUILL_ONT_EXAMPLES.coding) {
    const axis = O.fam.names.includes(e.truth) ? 'fam' : 'go';
    const r = O[axis].ranked(O.codonsOf(O.encode(e.dna)));
    tried++;
    if (r[0].name === e.truth) right++;
  }
  push('written weights classify the shipped coding examples', right >= tried - 3,
       `${right}/${tried} top-1`);
  let rightSo = 0, triedSo = 0;
  for (const e of w.QUILL_ONT_EXAMPLES.noncoding) {
    const r = O.so.ranked(O.encode(e.dna), e.exons || 1);
    triedSo++;
    if (r[0].name === e.truth) rightSo++;
  }
  push('written weights classify the shipped RNA examples', rightSo >= triedSo - 4,
       `${rightSo}/${triedSo} top-1`);

  // a sampled collagen must put glycine at every third residue
  const ci = O.fam.names.indexOf('collagen');
  const prot = O.translate(O.fam.sample(ci, 600));
  const ph = [0, 1, 2].map(k => {
    const sub = prot.split('').filter((_, j) => j % 3 === k);
    return sub.filter(a => a === 'G').length / sub.length;
  });
  push('sampled collagen is glycine-periodic', Math.max(...ph) > 0.7,
       ph.map(v => v.toFixed(2)).join('/'));
  const oi = O.fam.names.indexOf('olfactory_receptor');
  const orp = O.translate(O.fam.sample(oi, 312));
  const runs = O.tmCount(orp);
  push('sampled olfactory receptor has membrane helices', runs >= 1,
       runs + ' segments by the 19/16 window rule');
} catch (err) {
  push('ontology tab', false, err.message);
}

// gene function tab
try {
  push('quill-xl2-nano loaded', !!w.QuillXL && w.QuillXL.nterms > 2000, w.QuillXL && w.QuillXL.nterms + ' terms');
  push('function tables filled', (text('fnmf') || '').length > 10 && (text('fnprot') || '').includes('isoelectric'),
       (text('fnmf') || '').slice(0, 50));
  push('protein map drawn', (w.document.getElementById('fnmap').innerHTML || '').includes('<svg'), '');
  const r = w.QuillXL.annotate(w.QUILL_XL_EXAMPLES[w.QUILL_XL_EXAMPLES.length - 1].text);
  push('protein input accepted', r.input === 'protein' && r.go.CC.length > 0, r.input);
} catch (err) { push('gene function tab', false, err.message); }

// the engine actually computes something on this page
try {
  const Q = w.QuillInit(w.QUILL_S);
  const seq = Q.encode('ACGTACGTGGCCATGGAAGCTTTACGTAAGTTTCCCAGGACGTACGTAAGGCCTTAA'.repeat(4));
  const r = Q.forward(seq, Q.gcBins(seq), false);
  const lp = typeof r === 'number' ? r : r.logp;
  push('forward runs in page context', isFinite(lp) && lp < 0, 'logp=' + lp.toFixed(3));
} catch (err) {
  push('forward runs in page context', false, err.message);
}

// 3D views: every scene builds its data from the live engine and draws without error
try {
  const frames = [];
  w.requestAnimationFrame = f => { frames.push(f); return frames.length; };
  w.document.querySelector('nav button[data-t=v3]').click();
  const picks = [...w.document.querySelectorAll('#v3-pick button[data-s]')];
  for (const b of picks) { b.click(); for (let k = 0; k < 3 && frames.length; k++) frames.shift()(1000 + 500 * k); }
  w.document.getElementById('v3-new').click(); frames.length && frames.shift()(5000);
  push('3D views draw', picks.length === 3 && (text('v3-text') || '').length > 40 &&
       w.document.getElementById('v3-legend').children.length > 2, picks.length + ' scenes, text ' + (text('v3-text') || '').length + ', legend ' + w.document.getElementById('v3-legend').children.length);
} catch (err) { push('3D views draw', false, err.message); }

let bad = 0;
for (const c of checks) {
  if (!c.ok) bad++;
  console.log(`${c.ok ? 'ok  ' : 'FAIL'}  ${c.name}${c.detail ? '  [' + c.detail + ']' : ''}`);
}
console.log(bad ? `${bad} CHECK(S) FAILED` : 'ALL CHECKS PASSED');
process.exit(bad ? 1 : 0);
