"""Test the written protein signatures against held-out genes.

For each rule in domains.py: resolve its GO term names against the ontology, find the
held-out proteins that carry the pattern, and measure how often they have the term
(propagated) against how often any gene has it. Reports precision, lift and a Fisher
exact p-value next to the confidence that was written down beforehand.

Writes results/domains.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import domains
import qgox

RES = HERE / 'results'


def resolve(meta):
    """Rule -> first written term name that exists in the ontology, with its id."""
    by_name = {v['name']: t for t, v in meta.items()}
    out, missing = [], []
    for rule, rx in domains.compiled():
        hit = next(((n, by_name[n]) for n in rule['terms'] if n in by_name), None)
        if hit is None:
            missing.append(rule['name'])
        out.append((rule, rx, hit))
    return out, missing


def scan(prot, resolved):
    """Rules matched by one protein: [(rule name, term id, written confidence)]."""
    return [(rule['name'], hit[1], rule['confidence'])
            for rule, rx, hit in resolved if hit and rx.search(prot)]


def main():
    sets, meta = qgox.load_sets(), qgox.load_terms()
    resolved, missing = resolve(meta)
    held = sets['held']
    truth = [set(r['terms']) for r in held]
    n = len(held)
    rows = []
    for rule, rx, hit in resolved:
        if not hit:
            continue
        name, tid = hit
        m = np.array([bool(rx.search(r['prot'])) for r in held])
        y = np.array([tid in t for t in truth])
        a, b = int((m & y).sum()), int((m & ~y).sum())
        c, d = int((~m & y).sum()), int((~m & ~y).sum())
        base = y.mean()
        prec = a / max(a + b, 1)
        p = fisher_exact([[a, b], [c, d]], alternative='greater')[1] if a + b else 1.0
        rows.append(dict(rule=rule['name'], term=name, go=f'GO:{tid}', ns=meta[tid]['ns'],
                         matches=a + b, with_term=a, precision=round(prec, 4),
                         base_rate=round(float(base), 4),
                         lift=round(prec / base, 2) if base > 0 and a + b else None,
                         recall_of_term=round(a / max(a + c, 1), 4),
                         fisher_p=float(p), written_confidence=rule['confidence']))
    tested = [r for r in rows if r['matches'] >= 5]
    sig = [r for r in tested if r['fisher_p'] < 0.05 / max(len(tested), 1)]
    conf = np.array([r['written_confidence'] for r in tested])
    prec = np.array([r['precision'] for r in tested])
    rho = float(np.corrcoef(np.argsort(np.argsort(conf)),
                            np.argsort(np.argsort(prec)))[0, 1]) if len(tested) > 2 else 0.0
    any_match = np.array([any(rx.search(r['prot']) for rule, rx, hit in resolved if hit)
                          for r in held])
    summary = dict(
        rules_written=len(domains.RULES), rules_resolved=len(rows),
        unresolved=missing, held_out_genes=n,
        rules_with_5_or_more_matches=len(tested),
        rules_significant_bonferroni=len(sig),
        median_precision=round(float(np.median(prec)), 4) if len(tested) else None,
        median_lift=round(float(np.median([r['lift'] for r in tested])), 2)
        if tested else None,
        rules_with_precision_at_least_0_8=int((prec >= 0.8).sum()),
        mean_written_confidence=round(float(conf.mean()), 4) if len(tested) else None,
        mean_measured_precision=round(float(prec.mean()), 4) if len(tested) else None,
        spearman_confidence_vs_precision=round(rho, 4),
        genes_matching_any_rule=int(any_match.sum()),
        fraction_of_genes_matching_any_rule=round(float(any_match.mean()), 4))
    RES.mkdir(exist_ok=True)
    (RES / 'domains.json').write_text(json.dumps(dict(summary=summary, rules=rows),
                                                 indent=1))
    print(f'{"rule":38s} {"n":>4s} {"prec":>6s} {"base":>6s} {"lift":>7s} {"wrote":>6s}  term')
    for r in sorted(rows, key=lambda r: -(r['lift'] or 0) if r['matches'] >= 5 else 1):
        flag = '' if r['matches'] >= 5 else '  (too few to test)'
        print(f'{r["rule"][:38]:38s} {r["matches"]:4d} {r["precision"]:6.2f} '
              f'{r["base_rate"]:6.3f} {str(r["lift"]):>7s} {r["written_confidence"]:6.2f}  '
              f'{r["term"][:44]}{flag}')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
