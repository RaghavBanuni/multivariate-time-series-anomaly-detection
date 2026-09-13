"""Make the package importable when the suite is run from the repository root.

Without this, ``pytest`` puts ``tests/`` on ``sys.path`` but not the project root, and ``import msad``
fails for reasons that have nothing to do with the code under test.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
