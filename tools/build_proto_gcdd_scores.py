from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcdd.baselines import centroid_scores
from gcdd.io_utils import ensure_dir, read_csv, write_csv
from gcdd.scoring import percentile_by_class


PROTO_SCORE_FIELDS = [
    "index",
    "path",
    "web_label",
    "D_class",
    "R_class",
    "I_class_norm",
    "Q_same",
    "P_D_class",
    "P_R_class",
    "P_I_class",
    "P_Q_same",
    "centroid_score",
    "P_proto",
    "S_gcdd",
    "S_proto",
    "S_gcdd_proto",
    "S_gcdd_proto_noI",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build prototype-aware GCDD score table from existing V1 outputs.")
    parser.add_argument("--input-dir", default="outputs/v1_web_bird", help="V1 output directory.")
    parser.add_argument("--out", help="Output CSV. Defaults to <input-dir>/proto_gcdd/proto_gcdd_scores.csv.")
    parser.add_argument("--epsilon", type=float, default=1.0e-6, help="Score product epsilon.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_path = Path(args.out) if args.out else input_dir / "proto_gcdd" / "proto_gcdd_scores.csv"
    ensure_dir(out_path.parent)
    rows = build_proto_gcdd_rows(input_dir, args.epsilon)
    write_csv(out_path, rows, PROTO_SCORE_FIELDS)
    print(f"Prototype-aware GCDD scores written to {out_path}", flush=True)


def build_proto_gcdd_rows(input_dir: Path, epsilon: float = 1.0e-6) -> list[dict[str, Any]]:
    data = load_score_inputs(input_dir)
    labels = data["web_label"]

    p_d = percentile_by_class(data["D_class"], labels)
    p_r = percentile_by_class(data["R_class"], labels)
    p_i = percentile_by_class(data["I_class_norm"], labels)
    p_q = percentile_by_class(data["Q_same"], labels)
    p_proto = percentile_by_class(data["centroid_score"], labels)

    s_gcdd = np.power((p_d + epsilon) * (p_r + epsilon) * (p_i + epsilon) * (p_q + epsilon), 1.0 / 4.0)
    s_proto = p_proto
    s_gcdd_proto = np.power((p_d + epsilon) * (p_r + epsilon) * (p_i + epsilon) * (p_q + epsilon) * (p_proto + epsilon), 1.0 / 5.0)
    s_gcdd_proto_noi = np.power((p_d + epsilon) * (p_r + epsilon) * (p_q + epsilon) * (p_proto + epsilon), 1.0 / 4.0)

    rows = []
    for i in range(len(labels)):
        rows.append(
            {
                "index": int(data["index"][i]),
                "path": data["path"][i],
                "web_label": labels[i],
                "D_class": float(data["D_class"][i]),
                "R_class": float(data["R_class"][i]),
                "I_class_norm": float(data["I_class_norm"][i]),
                "Q_same": float(data["Q_same"][i]),
                "P_D_class": float(p_d[i]),
                "P_R_class": float(p_r[i]),
                "P_I_class": float(p_i[i]),
                "P_Q_same": float(p_q[i]),
                "centroid_score": float(data["centroid_score"][i]),
                "P_proto": float(p_proto[i]),
                "S_gcdd": float(s_gcdd[i]),
                "S_proto": float(s_proto[i]),
                "S_gcdd_proto": float(s_gcdd_proto[i]),
                "S_gcdd_proto_noI": float(s_gcdd_proto_noi[i]),
            }
        )
    return rows


def load_score_inputs(input_dir: Path) -> dict[str, np.ndarray]:
    gcdd_path = input_dir / "gcdd_scores.csv"
    if not gcdd_path.exists():
        raise FileNotFoundError(f"Missing GCDD scores: {gcdd_path}")
    gcdd_rows = sorted(read_csv(gcdd_path), key=lambda row: int(row["index"]))
    centroid_score = load_or_build_centroid_scores(input_dir, gcdd_rows)
    return {
        "index": np.array([int(row["index"]) for row in gcdd_rows], dtype=np.int64),
        "path": np.array([row["path"] for row in gcdd_rows], dtype=object),
        "web_label": np.array([row["web_label"] for row in gcdd_rows], dtype=object),
        "D_class": parse_float_array(gcdd_rows, "D_class"),
        "R_class": parse_float_array(gcdd_rows, "R_class"),
        "I_class_norm": parse_float_array(gcdd_rows, "I_class_norm"),
        "Q_same": parse_float_array(gcdd_rows, "Q_same"),
        "centroid_score": centroid_score,
    }


def load_or_build_centroid_scores(input_dir: Path, gcdd_rows: list[dict[str, str]]) -> np.ndarray:
    path = input_dir / "centroid_scores.csv"
    if path.exists():
        rows = sorted(read_csv(path), key=lambda row: int(row["index"]))
        if len(rows) != len(gcdd_rows):
            raise ValueError(f"{path} length does not match gcdd_scores.csv.")
        return parse_float_array(rows, "centroid_score")

    feature_path = input_dir / "features_cls.npy"
    labels_path = input_dir / "labels.npy"
    paths_path = input_dir / "paths.txt"
    if not feature_path.exists() or not labels_path.exists() or not paths_path.exists():
        raise FileNotFoundError("centroid_scores.csv is missing and features_cls.npy/labels.npy/paths.txt are not all available.")
    features = np.load(feature_path)
    labels = np.load(labels_path, allow_pickle=True).astype(str)
    paths = paths_path.read_text(encoding="utf-8").splitlines()
    scores = centroid_scores(features, labels)
    rows = [{"index": i, "path": paths[i], "web_label": labels[i], "centroid_score": float(scores[i])} for i in range(len(scores))]
    write_csv(path, rows, ["index", "path", "web_label", "centroid_score"])
    return scores


def parse_float_array(rows: list[dict[str, str]], field: str) -> np.ndarray:
    return np.array([float(row[field]) for row in rows], dtype=np.float32)


if __name__ == "__main__":
    main()
