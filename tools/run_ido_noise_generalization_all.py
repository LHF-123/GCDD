"""Plan or explicitly execute the 3 datasets × IDN/GCCN × 3 seeds IDO matrix."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcdd.ido_batch import main_for_protocols


if __name__ == "__main__":
    main_for_protocols(protocols=("idn", "gccn"), description=__doc__)
