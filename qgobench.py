"""Benchmark the function-conditioned model on held-out chromosomes.

Tasks, all on chromosomes 17, 19, 20, 21 and 22, which were never counted:
  classify  - zero-shot functional class from coding sequence, written vs counted
  generate  - sample a coding sequence per class and check the written signature shows up
  bits      - bits/base of coding sequence under the matching class vs the pooled model

Writes results/gobench.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import function as F
import qgo
from qgo import CODONS, GoModel

RES = HERE / 'results'
RES.mkdir(exist_ok=True)
OUT = RES / 'gobench.json'
AXES = (('go', 'cls', F.GO_CLASSES), ('fam', 'family', F.GENE_FAMILIES))


def save(d):
    cur = json.loads(OUT.read_text()) if OUT.exists() else {}
    cur.update(d)
    OUT.write_text(json.dumps(cur, indent=1))


def auroc(y, s):
    y = np.asarray(y, bool)
    s = np.asarray(s, float)
    if y.all() or not y.any():
        return float('nan')
    r = np.empty(len(s))
    o = np.argsort(s, kind='mergesort')
    sv = s[o]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    np_, nn = y.sum(), (~y).sum()
    return float((r[y].sum() - np_ * (np_ + 1) / 2.0) / (np_ * nn))


def score_matrix(model, recs):
    """One row of per-class log2 likelihoods per gene."""
    S = np.empty((len(recs), model.n))
    for i, r in enumerate(recs):
        S[i] = model.score(r['codons'], r['prot'])
    return S


def metrics(S, truth, names):
    """Accuracy, balanced accuracy, per-class one-vs-rest AUROC."""
    pred = [names[i] for i in S.argmax(1)]
    acc = float(np.mean([p == t for p, t in zip(pred, truth)]))
    per, aur, conf = {}, {}, {}
    # scores are not comparable in magnitude across genes, so rank within a gene
    Z = (S - S.mean(1, keepdims=True)) / np.maximum(S.std(1, keepdims=True), 1e-9)
    for j, nm in enumerate(names):
        y = [t == nm for t in truth]
        if not any(y):
            continue
        aur[nm] = round(auroc(y, Z[:, j]), 4)
        sel = [i for i, t in enumerate(truth) if t == nm]
        per[nm] = round(float(np.mean([pred[i] == nm for i in sel])), 4)
        conf[nm] = sorted(
            {names[k]: int(sum(1 for i in sel if pred[i] == names[k]))
             for k in range(len(names))}.items(), key=lambda kv: -kv[1])[:3]
    bal = round(float(np.mean(list(per.values()))), 4)
    return dict(accuracy=round(acc, 4), balanced_accuracy=bal,
                macro_auroc=round(float(np.mean([v for v in aur.values()
                                                 if v == v])), 4),
                per_class_recall=per, per_class_auroc=aur, confusions=conf,
                n=len(truth), majority_baseline=round(
                    max(sum(1 for t in truth if t == nm) for nm in names) / len(truth), 4))


def cmd_classify(_):
    held = qgo.load('held')
    out = {}
    for tag, key, specs in AXES:
        recs = [r for r in held if r[key]]
        truth = [r[key] for r in recs]
        for arm in ('quill-k', 'quill-s'):
            m = GoModel(HERE / 'models' / f'{arm}-{tag}.npz')
            S = score_matrix(m, recs)
            res = metrics(S, truth, m.names)
            res['parameters'] = m.nparam
            out[f'{tag}/{arm}'] = res
            print(f'{tag} {arm}: acc {res["accuracy"]:.3f}  balanced '
                  f'{res["balanced_accuracy"]:.3f}  macro AUROC '
                  f'{res["macro_auroc"]:.3f}  (majority {res["majority_baseline"]:.3f}, '
                  f'{res["n"]} genes)', flush=True)
        # composition-only ablation: drop length and transmembrane priors
        saved = dict(F.SCORE_WEIGHTS)
        F.SCORE_WEIGHTS.update(length=0.0, tm=0.0)
        m = GoModel(HERE / 'models' / f'quill-k-{tag}.npz')
        res = metrics(score_matrix(m, recs), truth, m.names)
        out[f'{tag}/quill-k-composition-only'] = res
        F.SCORE_WEIGHTS.update(saved)
        print(f'{tag} written, composition only: acc {res["accuracy"]:.3f}  '
              f'macro AUROC {res["macro_auroc"]:.3f}', flush=True)
    save({'classify': out})


def cmd_bits(_):
    """Does knowing the function help predict the bases?"""
    held = qgo.load('held')
    out = {}
    for tag, key, specs in AXES:
        recs = [r for r in held if r[key]]
        for arm in ('quill-k', 'quill-s'):
            m = GoModel(HERE / 'models' / f'{arm}-{tag}.npz')
            pooled = np.log2(np.maximum(
                (2.0 ** m.base).mean(0) / (2.0 ** m.base).mean(0).sum(), 1e-12))
            match, pool, nb = 0.0, 0.0, 0
            saved = dict(F.SCORE_WEIGHTS)
            F.SCORE_WEIGHTS.update(length=0.0, tm=0.0, period=1.0)
            for r in recs:
                c = np.asarray(r['codons'])
                c = c[c >= 0]
                if not len(c):
                    continue
                i = m.names.index(r[key])
                match += m.score(r['codons'], r['prot'])[i]
                pool += pooled[c].sum()
                nb += 3 * len(c)
            F.SCORE_WEIGHTS.update(saved)
            out[f'{tag}/{arm}'] = dict(
                bits_per_base_class_conditioned=round(float(-match / nb), 4),
                bits_per_base_pooled=round(float(-pool / nb), 4),
                gain=round(float((-pool / nb) - (-match / nb)), 4), n_bases=int(nb))
            o = out[f'{tag}/{arm}']
            print(f'{tag} {arm}: conditioned {o["bits_per_base_class_conditioned"]:.4f} '
                  f'vs pooled {o["bits_per_base_pooled"]:.4f}  '
                  f'gain {o["gain"]:+.4f} bits/base', flush=True)
    save({'bits': out})


def sample_cds(model, i, n_codons, rng):
    """Sample a coding sequence from one class's written tables.

    The loop/transmembrane chain is sampled along with the codons, so a membrane class
    produces hydrophobic helices of the written length rather than hydrophobic residues
    scattered through the protein.
    """
    nl, pn = int(model.nterm_len[i]), int(model.period_n[i])
    pw = model.period_w[i]
    base = 2.0 ** model.base[i]
    per = 2.0 ** model.period[i]
    nt = 2.0 ** model.nterm[i]
    tmt = 2.0 ** model.tm_table if model.tm_table is not None else None
    enter, stay = model._tm_rates(i, n_codons)
    stop = {CODONS.index(c) for c in qgo.STOP}
    out, in_tm = [], False
    for j in range(n_codons):
        if tmt is not None and j >= nl:
            in_tm = (rng.random() < stay) if in_tm else (rng.random() < enter)
        if tmt is not None and in_tm:
            p = tmt
        else:
            p = base
            if pn:
                p = (1.0 - pw) * p + pw * per[j % pn]
            if j < nl:
                p = nt[min(j, nt.shape[0] - 1)]
        p = np.array(p, dtype=np.float64)
        for s in stop:
            p[s] = 0.0
        p /= p.sum()
        out.append(int(rng.choice(64, p=p)))
    return out


def translate(cods):
    import knowledge as K
    return ''.join(K.GENETIC_CODE[CODONS[c]] for c in cods)


def cmd_generate(_):
    rng = np.random.default_rng(11)
    held = qgo.load('held')
    out = {}
    for tag, key, specs in AXES:
        m = GoModel(HERE / 'models' / f'quill-k-{tag}.npz')
        real = {}
        for r in held:
            if r[key]:
                real.setdefault(r[key], []).append(r['prot'])
        for i, nm in enumerate(m.names):
            n_cod = int(round(m.len_mean[i]))
            prots = [translate(sample_cds(m, i, n_cod, rng)) for _ in range(12)]
            gen = ''.join(prots)
            spec = specs[nm]
            want = F.composition(spec['aa'])
            got = np.array([gen.count(a) for a in F.AA], float)
            got = got / max(got.sum(), 1)
            js = 0.5 * (want + got)
            d = float(sum(
                0.5 * (want[k] * np.log2(want[k] / js[k]) if want[k] > 0 else 0) +
                0.5 * (got[k] * np.log2(got[k] / js[k]) if got[k] > 0 else 0)
                for k in range(len(F.AA)) if js[k] > 0))
            e = dict(composition_js_to_written=round(d, 5),
                     tm_segments_mean=round(float(np.mean([qgo.tm_count(p)
                                                           for p in prots])), 2),
                     tm_expected=float(spec['tm_expect']))
            if real.get(nm):
                rr = np.array([sum(p.count(a) for p in real[nm]) for a in F.AA], float)
                rr = rr / rr.sum()
                jr = 0.5 * (rr + got)
                e['composition_js_to_real_heldout'] = round(float(sum(
                    0.5 * (rr[k] * np.log2(rr[k] / jr[k]) if rr[k] > 0 else 0) +
                    0.5 * (got[k] * np.log2(got[k] / jr[k]) if got[k] > 0 else 0)
                    for k in range(len(F.AA)) if jr[k] > 0)), 5)
                e['n_real_heldout'] = len(real[nm])
            pr = spec.get('period')
            if pr:
                # phase profiles are computed per protein and then averaged: the proteins
                # are not multiples of the period, so concatenating them first would
                # shift the register and wash the periodicity out
                period = F.PERIOD_PROFILES[pr]['period']

                def by_phase(residues):
                    acc = np.zeros(period)
                    for q in prots:
                        for k in range(period):
                            sub = q[k::period]
                            if sub:
                                acc[k] += sum(sub.count(a) for a in residues) / len(sub)
                    return (acc / len(prots)).round(3).tolist()

                e['period'] = period
                if pr == 'collagen_GXY':
                    e['glycine_by_triplet_phase'] = by_phase('G')
                    e['proline_by_triplet_phase'] = by_phase('P')
                if pr == 'zinc_finger_C2H2':
                    cys = by_phase('C')
                    his = by_phase('H')
                    anc = F.PERIOD_PROFILES[pr]['anchors']
                    cpos = [k for k in anc if 'C' in anc[k]]
                    hpos = [k for k in anc if 'H' in anc[k]]
                    e['cysteine_at_anchored_positions'] = round(
                        float(np.mean([cys[k] for k in cpos])), 3)
                    e['cysteine_elsewhere'] = round(float(np.mean(
                        [v for k, v in enumerate(cys) if k not in cpos])), 3)
                    e['histidine_at_anchored_positions'] = round(
                        float(np.mean([his[k] for k in hpos])), 3)
                    e['histidine_elsewhere'] = round(float(np.mean(
                        [v for k, v in enumerate(his) if k not in hpos])), 3)
                    e['TGEKP_linkers_found'] = int(sum(q.count('TGEKP') for q in prots))
                if pr == 'coiled_coil_heptad':
                    hyd = by_phase('LIVMF')
                    e['hydrophobic_by_heptad_position'] = hyd
                    e['hydrophobic_at_a_and_d'] = round(
                        float(np.mean([hyd[0], hyd[3]])), 3)
                    e['hydrophobic_elsewhere'] = round(float(np.mean(
                        [hyd[k] for k in (1, 2, 4, 5, 6)])), 3)
            out[f'{tag}/{nm}'] = e
            print(f'{tag} {nm:22s} comp-JS to written {d:.4f}'
                  + (f'  to real {e.get("composition_js_to_real_heldout"):.4f}'
                     if 'composition_js_to_real_heldout' in e else '')
                  + f'  TM {e["tm_segments_mean"]:.1f} (want {spec["tm_expect"]})',
                  flush=True)
    save({'generate': out})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('task', nargs='?', default='all',
                    choices=['all', 'classify', 'bits', 'generate'])
    a = ap.parse_args()
    tasks = ['classify', 'bits', 'generate'] if a.task == 'all' else [a.task]
    for t in tasks:
        print(f'== {t}', flush=True)
        globals()[f'cmd_{t}'](a)
