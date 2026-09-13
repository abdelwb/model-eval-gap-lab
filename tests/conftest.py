"""Put analysis/ on sys.path so tests can `import metrics` / `import
gap_analysis` directly -- it isn't packaged with __init__.py on purpose,
since it's meant to be run as a plain script from its own directory.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = str(ROOT / "analysis")
if ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, ANALYSIS_DIR)
