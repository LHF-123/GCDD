"""Create an immutable IDN or GCCN realization plus a V1-compatible input adapter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcdd.noise_generalization import prepare_noise_generalization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, choices=("idn", "gccn"))
    parser.add_argument("--dataset", required=True, help="Dataset display name recorded in the artifact metadata.")
    parser.add_argument("--base-input-dir", required=True, help="Existing immutable V1 input directory.")
    parser.add_argument("--source-noise-index", required=True, help="Existing cyclic index used only to align source-train clean labels.")
    parser.add_argument("--validation-dir", required=True, help="Existing fixed_clean_validation_v1 manifest directory.")
    parser.add_argument("--output-dir", required=True, help="New, nonexistent output directory for this noise protocol.")
    parser.add_argument("--noise-rate", type=float, default=0.4)
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--validation-seed", type=int, default=20250726)
    parser.add_argument(
        "--idn-probability-modulation",
        type=float,
        default=0.5,
        help="IDN difficulty-rank modulation in [0, 1]; formal protocol fixes this at 0.5.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = prepare_noise_generalization(
        protocol=args.protocol,
        dataset=args.dataset,
        base_input_dir=Path(args.base_input_dir),
        source_noise_index=Path(args.source_noise_index),
        validation_dir=Path(args.validation_dir),
        output_dir=Path(args.output_dir),
        noise_rate=float(args.noise_rate),
        noise_seed=int(args.noise_seed),
        validation_seed=int(args.validation_seed),
        idn_probability_modulation=float(args.idn_probability_modulation),
    )
    print(
        f"Prepared {metadata['dataset']} {metadata['noise_protocol']}: "
        f"realized={metadata['realized_noise_rate']:.6f}, "
        f"manifest_sha256={metadata['manifest_sha256']}, "
        f"input={metadata['prepared_input']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
