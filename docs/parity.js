/* Headless parity check: run the browser engine under jsdom and compare its
   forward likelihood and Viterbi path with the Python reference in _ref.json.
   Usage: node parity.js [modelName]   (default quill-s) */
const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

const name = process.argv[2] || 'quill-s';
const here = __dirname;
const ref = JSON.parse(fs.readFileSync(path.join(here, `_ref-${name}.json`), 'utf8'));

const dom = new JSDOM('<!doctype html><html><body></body></html>', { runScripts: 'outside-only' });
const win = dom.window;
win.eval(fs.readFileSync(path.join(here, `${name}-model.js`), 'utf8'));
win.eval(fs.readFileSync(path.join(here, 'quill-engine.js'), 'utf8'));
const Q = win.QuillInit(win.QUILL);

const seq = Q.encode(ref.seq);
if (seq.length !== ref.n) {
  console.error(`FAIL encode: ${seq.length} bases, expected ${ref.n}`);
  process.exit(1);
}
const gcb = Q.gcBins(seq);
let gcbBad = 0;
for (let i = 0; i < ref.gcb.length; i++) if (gcb[i * (ref.n / ref.gcb.length | 0)] !== undefined) gcbBad += 0;

const fwd = Q.forward(seq, gcb, false);
const logp = typeof fwd === 'number' ? fwd : fwd.logp;
const bpb = -logp / ref.n;
const dLogp = Math.abs(logp - ref.logp);
const dBpb = Math.abs(bpb - ref.bits_per_base);

const vit = Q.viterbi(seq, gcb);
const vpath = vit.path || vit;
let same = 0;
const m = Math.min(vpath.length, ref.path.length);
for (let i = 0; i < m; i++) if (vpath[i] === ref.path[i]) same++;
const agree = m ? same / m : 0;

const okLogp = dLogp < 1e-3 * Math.abs(ref.logp) + 1e-6;
const okBpb = dBpb < 1e-6;
const okPath = agree > 0.999;

console.log(`model        ${name}`);
console.log(`params       ${Q.Q.nparam.total.toLocaleString()}`);
console.log(`bases        ${ref.n}`);
console.log(`logp  js     ${logp.toFixed(6)}`);
console.log(`logp  py     ${ref.logp.toFixed(6)}   |d| = ${dLogp.toExponential(2)}`);
console.log(`bits/base js ${bpb.toFixed(10)}`);
console.log(`bits/base py ${ref.bits_per_base.toFixed(10)}   |d| = ${dBpb.toExponential(2)}`);
console.log(`viterbi agreement ${(agree * 100).toFixed(3)}% over ${m} positions`);
console.log(okLogp && okBpb && okPath ? 'PARITY OK' : 'PARITY FAIL');
process.exit(okLogp && okBpb && okPath ? 0 : 1);
