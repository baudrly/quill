/* Check the browser ontology engine against the Python reference in _ref-ontology.json. */
const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

const here = __dirname;
const ref = JSON.parse(fs.readFileSync(path.join(here, '_ref-ontology.json'), 'utf8'));
const win = new JSDOM('', { runScripts: 'outside-only' }).window;
win.eval(fs.readFileSync(path.join(here, 'quill-ontology.js'), 'utf8'));
win.eval(fs.readFileSync(path.join(here, 'quill-ontology-engine.js'), 'utf8'));
const Q = win.QuillOnt;

let worst = 0, checks = 0, fails = 0;
function cmp(label, js, py, tol) {
  checks++;
  let d = 0;
  for (let i = 0; i < py.length; i++) d = Math.max(d, Math.abs(js[i] - py[i]));
  worst = Math.max(worst, d);
  const ok = d <= tol;
  if (!ok) { fails++; console.log(`FAIL ${label}: max |d| = ${d.toExponential(3)}`); }
  return ok;
}

for (const arm of ['go', 'fam']) {
  for (const c of ref[arm]) {
    const cods = Q.codonsOf(Q.encode(c.dna));
    cmp(`${arm} ${c.symbol}`, Q[arm].score(cods), c.scores, 2e-3);
  }
  const names = Q[arm].names;
  if (JSON.stringify(names) !== JSON.stringify(ref[arm + '_names'])) {
    console.log(`FAIL ${arm} class order`); fails++;
  }
  checks++;
}
for (const c of ref.so) {
  const seq = Q.encode(c.dna);
  cmp(`so ${c.symbol}`, Q.so.score(seq, c.exons), c.scores, 2e-3);
  const f = Q.so.features(seq, c.exons);
  cmp(`so ${c.symbol} hairpin`, [f.hairpin], [c.hairpin], 1e-9);
  cmp(`so ${c.symbol} gc`, [f.gc], [c.gc], 1e-9);
}

console.log(`${checks - fails}/${checks} ontology checks passed; worst |d| = ` +
            `${worst.toExponential(2)}`);
console.log(`written parameters in the page: go ${Q.go.nparam.toLocaleString()}, ` +
            `fam ${Q.fam.nparam.toLocaleString()}, so ${Q.so.nparam.toLocaleString()}`);
console.log(fails ? 'ONTOLOGY PARITY FAIL' : 'ONTOLOGY PARITY OK');
process.exit(fails ? 1 : 0);
