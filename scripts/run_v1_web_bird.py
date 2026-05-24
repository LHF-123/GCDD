from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcdd.config import load_config
from gcdd.pipeline_v1 import run_v1_web_bird


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GCDD-Lite V1 on Web-Bird.")
    parser.add_argument("--config", default="configs/v1_web_bird.yaml", help="Path to YAML config.")
    parser.add_argument("--data-root", help="Dataset root. Overrides dataset.root.")
    parser.add_argument("--index-file", help="Optional CSV index. Overrides dataset.index_file.")
    parser.add_argument("--output-root", help="Output root. Overrides output.root.")
    parser.add_argument("--dataset-name", help="Dataset name. Overrides dataset.name.")
    parser.add_argument("--feature-backend", help="Feature backend: dinov2_vitb14 or random.")
    parser.add_argument("--device", help="Torch device: auto, cpu, cuda.")
    parser.add_argument("--batch-size", type=int, help="Feature extraction batch size.")
    parser.add_argument("--epochs", type=int, help="Linear classifier epochs.")
    parser.add_argument("--max-classes", type=int, help="Optional class cap for debugging.")
    parser.add_argument("--max-train-per-class", type=int, help="Optional train sample cap per class.")
    parser.add_argument("--max-eval-per-class", type=int, help="Optional eval sample cap per class.")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Dot-path override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {}
    if args.data_root:
        overrides["dataset.root"] = args.data_root
    if args.index_file:
        overrides["dataset.index_file"] = args.index_file
    if args.output_root:
        overrides["output.root"] = args.output_root
    if args.dataset_name:
        overrides["dataset.name"] = args.dataset_name
    if args.feature_backend:
        overrides["feature.backend"] = args.feature_backend
    if args.device:
        overrides["feature.device"] = args.device
    if args.batch_size is not None:
        overrides["feature.batch_size"] = args.batch_size
    if args.epochs is not None:
        overrides["train.epochs"] = args.epochs
    if args.max_classes is not None:
        overrides["dataset.max_classes"] = args.max_classes
    if args.max_train_per_class is not None:
        overrides["dataset.max_train_per_class"] = args.max_train_per_class
    if args.max_eval_per_class is not None:
        overrides["dataset.max_eval_per_class"] = args.max_eval_per_class

    cfg = load_config(Path(args.config), cli_overrides=overrides, set_overrides=args.set)
    run_v1_web_bird(cfg)


if __name__ == "__main__":
    main()

