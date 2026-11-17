"""Entry point for the preprint figures: python3 qmakefigs.py [scaling brca1 generation]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import qfigs

if __name__ == '__main__':
    for w in sys.argv[1:] or ['scaling', 'brca1', 'generation']:
        getattr(qfigs, f'fig_{w}')()
