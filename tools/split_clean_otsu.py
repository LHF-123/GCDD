from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcdd.io_utils import ensure_dir, read_csv, write_csv
from gcdd.scoring import adaptive_otsu_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a clean split from a score column using class-wise Adaptive-Otsu.")
    parser.add_argument("--scores", required=True, help="Score CSV path.")
    parser.add_argument("--score-col", required=True, help="Score column used for clean selection.")
    parser.add_argument("--out", required=True, help="Output split CSV path.")
    parser.add_argument("--otsu-bins", type=int, default=256, help="Otsu histogram bins.")
    parser.add_argument("--clean-ratio-clip", default="0.3,0.9", help="low,high clip values.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_score_rows(Path(args.scores), args.score_col)
    split_info = build_split_info(rows, args.score_col, args.otsu_bins, parse_clip(args.clean_ratio_clip))
    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    write_split(out_path, rows, split_info["state"])
    print(f"Clean split written to {out_path}", flush=True)


def read_score_rows(path: Path, score_col: str) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Score CSV not found: {path}")
    rows = sorted(read_csv(path), key=lambda row: int(row["index"]))
    if not rows:
        raise ValueError(f"Score CSV is empty: {path}")
    required = {"index", "path", "web_label", score_col}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"Score CSV is missing fields: {sorted(missing)}")
    return rows


def build_split_info(rows: list[dict[str, str]], score_col: str, otsu_bins: int, clean_ratio_clip: tuple[float, float]) -> dict[str, np.ndarray]:
    labels = np.array([row["web_label"] for row in rows], dtype=object)
    scores = np.array([float(row[score_col]) for row in rows], dtype=np.float32)
    cfg = {"selection": {"otsu_bins": otsu_bins, "clean_ratio_clip": list(clean_ratio_clip)}}
    state, threshold_stats = adaptive_otsu_split(scores, labels, cfg)
    return {"state": state, **threshold_stats}


def write_split(path: Path, rows: list[dict[str, str]], state: np.ndarray) -> None:
    out_rows = []
    for i, row in enumerate(rows):
        out_rows.append({"index": row["index"], "path": row["path"], "web_label": row["web_label"], "state": state[i]})
    write_csv(path, out_rows, ["index", "path", "web_label", "state"])


def parse_clip(raw: str) -> tuple[float, float]:
    parts = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(parts) != 2:
        raise ValueError("--clean-ratio-clip must contain two comma-separated values.")
    if not (0.0 <= parts[0] <= parts[1] <= 1.0):
        raise ValueError("--clean-ratio-clip must satisfy 0 <= low <= high <= 1.")
    return parts[0], parts[1]


if __name__ == "__main__":
    main()
