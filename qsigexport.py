"""Ship the written protein signatures to the browser, with their measured precision."""
import json, sys
from pathlib import Path
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE))
import domains, qdomains, qgox

def main():
    meta = qgox.load_terms()
    resolved, _ = qdomains.resolve(meta)
    measured = {r['rule']: r for r in json.loads((HERE / 'results' / 'domains.json').read_text())['rules']}
    out = []
    for rule, rx, hit in resolved:
        m = measured.get(rule['name'], {})
        out.append(dict(name=rule['name'], pattern=rule['pattern'],
                        go=f'GO:{hit[1]}' if hit else None, term=hit[0] if hit else None,
                        written=rule['confidence'],
                        measured=m.get('precision') if m.get('matches', 0) >= 5 else None,
                        n=m.get('matches', 0)))
    p = HERE / 'web' / 'quill-signatures.js'
    p.write_text('window.QUILL_SIGNATURES = ' + json.dumps(out, separators=(',', ':')) + ';\n')
    print(f'{p.name}: {len(out)} rules, {p.stat().st_size/1024:.1f} KB')

if __name__ == '__main__':
    main()
