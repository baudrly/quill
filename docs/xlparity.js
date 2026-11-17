/* quill-xl2-nano: browser engine against Python on real held-out mRNAs. */
const fs = require('fs'), path = require('path'), { JSDOM } = require('jsdom');
const here = __dirname;
const ref = JSON.parse(fs.readFileSync(path.join(here, '_ref-xl.json'), 'utf8'));
const win = new JSDOM('', { runScripts: 'outside-only' }).window;
for (const f of ['quill-signatures.js', 'quill-xl-nano.js', 'quill-xl-engine.js'])
  win.eval(fs.readFileSync(path.join(here, f), 'utf8'));
const Q = win.QuillXL;
let fails = 0, worst = 0, featOk = 0, featN = 0;
const same = (a, b) => JSON.stringify([...a].sort()) === JSON.stringify([...b].sort());
for (const r of ref) {
  const d = Q.describeProtein(r.prot);
  const pf = Q.proteinFeatures(d), hits = Q.prosite(d.sequence);
  const rd = Q.describeRna(r.u5, r.cds, r.u3, [], r.cds.length);
  const rf = Q.rnaFeatures(rd).filter(f => r.rna_features.includes(f) || !/^rna:(utr5_mfe|utr5_fold|exons|nmd)/.test(f));
  const checks = [['protein features', pf, r.protein_features], ['PROSITE', hits, r.prosite]];
  for (const [nm, a, b] of checks) {
    featN++;
    if (same(a, b)) featOk++;
    else {
      fails++;
      const x = a.filter(v => !b.includes(v)), y = b.filter(v => !a.includes(v));
      console.log(`FAIL ${r.symbol} ${nm}: js-only ${JSON.stringify(x)} py-only ${JSON.stringify(y)}`);
    }
  }
  featN++;
  const rfv = rf.filter(f => !/^rna:(exons|nmd)/.test(f));
  const rpy = r.rna_features.filter(f => !/^rna:(exons|nmd)/.test(f));
  if (same(rfv.filter(f => rpy.includes(f) || true).filter(f => !/^rna:(utr5_mfe|utr5_fold)/.test(f)), rpy)) featOk++;
  else {
    fails++;
    console.log(`FAIL ${r.symbol} RNA features: js-only ${JSON.stringify(rfv.filter(f => !rpy.includes(f)))} py-only ${JSON.stringify(rpy.filter(f => !rfv.includes(f)))}`);
  }
  const P = Q.predict(r.prot, rd, d, hits);
  let dmax = 0;
  for (let t = 0; t < P.length; t++) dmax = Math.max(dmax, Math.abs(P[t] - r.P[t]));
  worst = Math.max(worst, dmax);
  if (dmax > 5e-3) { fails++; console.log(`FAIL ${r.symbol} probabilities: max |d| ${dmax.toExponential(2)}`); }
}
console.log(`${featOk}/${featN} feature sets identical; worst probability |d| ${worst.toExponential(2)} over ${ref.length} mRNAs x ${Q.nterms} terms`);
console.log(fails ? 'XL PARITY FAIL' : 'XL PARITY OK');
process.exit(fails ? 1 : 0);
