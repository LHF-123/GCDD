from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcdd.noise_realization import prepare_cub_noise_realization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and audit one CUB cyclic-asym40 per-sample noise realization."
    )
    parser.add_argument("--base-input-dir", required=True)
    parser.add_argument("--noise-index", required=True)
    parser.add_argument("--validation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--noise-seed", type=int, required=True)
    parser.add_argument("--validation-seed", type=int, default=20250726)
    parser.add_argument("--peer-noise-index", action="append", default=[])
    parser.add_argument(
        "--create-validation-if-missing",
        action="store_true",
        help="Deterministically create fixed_clean_validation_v1 in --validation-dir when both files are absent.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = prepare_cub_noise_realization(
        base_input_dir=Path(args.base_input_dir),
        noise_index_path=Path(args.noise_index),
        validation_dir=Path(args.validation_dir),
        output_dir=Path(args.output_dir),
        noise_seed=int(args.noise_seed),
        validation_seed=int(args.validation_seed),
        peer_noise_indices=[Path(path) for path in args.peer_noise_index],
        create_validation_if_missing=bool(args.create_validation_if_missing),
    )
    print(
        "Prepared CUB noise realization: "
        f"seed={metadata['noise_seed']}, source_flipped={metadata['source_flipped_samples']}, "
        f"training_pool_clean_ratio={metadata['training_pool_clean_ratio']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
