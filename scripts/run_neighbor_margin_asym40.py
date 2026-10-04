"""Run a fresh, isolated cyclic-asym40 Neighbor-Margin experiment.

This launcher consumes raw CUB/Cars/Aircraft data, regenerates deterministic
noise indexes and V1 inputs, then runs validation-selected Neighbor-Margin
PGDF. It does not read checkpoints, selections, manifests, or V1 outputs from
an earlier experiment.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcdd.io_utils import ensure_dir, write_csv, write_yaml
from gcdd.pipeline_v1 import run_v1_web_bird
from tools import build_cub_asym_noise_index as cub_noise
from tools import build_folder_asym_noise_index as folder_noise


DEFAULT_CONFIG = ROOT / "configs" / "pgdf_neighbor_margin.yaml"
V1_REQUIRED_FILES = (
    "paths.txt",
    "labels.npy",
    "eval_paths.txt",
    "eval_labels.npy",
    "features_cls.npy",
    "features_gap.npy",
    "features_top.npy",
    "resolved_config.yaml",
)


class PreparedDataset:
    """Fresh V1 inputs and noise provenance for one dataset."""

    def __init__(self, *, key: str, spec: dict[str, Any], noise_index: Path, input_dir: Path) -> None:
        self.key = key
        self.spec = spec
        self.noise_index = noise_index
        self.input_dir = input_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate cyclic-asym40 inputs and run isolated multi-seed "
            "Neighbor-Margin PGDF for CUB, Cars, and Aircraft."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Complete standalone experiment YAML.")
    parser.add_argument("--seeds", help="Comma-separated training-seed override. Defaults to protocol.training.seeds in YAML.")
    parser.add_argument("--cub-root", type=Path, help="CUB root containing images.txt and images/.")
    parser.add_argument("--cars-root", type=Path, help="Cars root containing train/ and test/.")
    parser.add_argument("--aircraft-root", type=Path, help="Aircraft root containing train/ and test/.")
    parser.add_argument("--run-root", type=Path, required=True, help="New root for all fresh generated inputs and results.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), help="Override feature.device for V1 and LoRA.")
    parser.add_argument("--local-repo", help="Override feature.local_repo for an offline/local DINOv2 torch-hub checkout.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for the LoRA runner.")
    parser.add_argument("--dry-run", action="store_true", help="Print the complete plan without creating files or training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_standalone_config(args.config)
    apply_machine_overrides(cfg, args)
    validate_standalone_config(cfg)

    run_root = args.run_root.expanduser()
    bundle_root = build_protocol_bundle_root(run_root, cfg)
    variant_root = bundle_root / "variants" / str(cfg["protocol"]["variant_id"])
    if variant_root.exists():
        raise FileExistsError(f"Refusing to reuse an existing variant output: {variant_root}")

    if args.dry_run:
        print_plan(cfg, run_root, args)
        return

    validate_raw_roots(cfg)
    if bundle_root.exists():
        prepared = load_existing_prepared_bundle(bundle_root, cfg)
    else:
        ensure_dir(bundle_root)
        write_yaml(bundle_root / "input_bundle_config.yaml", input_bundle_config(cfg))
        prepared = []
        for key in ("cub", "cars", "aircraft"):
            spec = cfg["datasets"][key]
            noise_index = build_noise_index(key, spec, cfg, bundle_root)
            input_dir = rebuild_v1_input(spec, cfg, bundle_root, noise_index)
            verify_v1_input(input_dir)
            prepared.append(PreparedDataset(key=key, spec=spec, noise_index=noise_index, input_dir=input_dir))

    ensure_dir(variant_root)
    write_yaml(variant_root / "experiment_config.yaml", cfg)
    for item in prepared:
        run_lora_experiment(item, cfg, variant_root, args)

    write_provenance(variant_root, bundle_root, args.config, cfg, prepared)
    print(f"[DONE] Neighbor-Margin variant results: {variant_root}", flush=True)


def load_standalone_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Standalone config does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Standalone config must contain a YAML mapping.")
    return cfg


def apply_machine_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    roots = {
        "cub": args.cub_root,
        "cars": args.cars_root,
        "aircraft": args.aircraft_root,
    }
    for key, root in roots.items():
        if root is not None:
            cfg["datasets"][key]["data_root"] = str(root.expanduser())
    if args.device is not None:
        cfg["feature"]["device"] = args.device
    if args.local_repo is not None:
        cfg["feature"]["local_repo"] = args.local_repo
    if args.seeds is not None:
        cfg.setdefault("protocol", {}).setdefault("training", {})["seeds"] = [
            token.strip() for token in args.seeds.split(",") if token.strip()
        ]


def validate_standalone_config(cfg: dict[str, Any]) -> None:
    for section in ("protocol", "datasets", "v1", "feature", "lora", "lora_train", "pgdf"):
        if section not in cfg or not isinstance(cfg[section], dict):
            raise ValueError(f"Standalone config is missing mapping: {section}")
    if set(cfg["datasets"]) != {"cub", "cars", "aircraft"}:
        raise ValueError("Standalone config datasets must be exactly cub, cars, and aircraft.")
    protocol_name = str(cfg["protocol"].get("name", "")).strip()
    variant_id = str(cfg["protocol"].get("variant_id", "")).strip()
    for key, value in (("protocol.name", protocol_name), ("protocol.variant_id", variant_id)):
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError(f"{key} must be a non-empty directory-safe identifier.")
    cfg["protocol"]["name"] = protocol_name
    cfg["protocol"]["variant_id"] = variant_id

    noise = cfg["protocol"].get("noise", {})
    if float(noise.get("ratio", -1.0)) != 0.4 or int(noise.get("seed", -1)) != 42:
        raise ValueError("This formal launcher requires cyclic-asym40 noise ratio=0.4 and seed=42.")
    if str(noise.get("target_strategy", "")) != "adjacent_cyclic":
        raise ValueError("This formal launcher requires target_strategy=adjacent_cyclic.")
    raw_training_seeds = cfg["protocol"].get("training", {}).get("seeds", [])
    try:
        training_seeds = [int(seed) for seed in raw_training_seeds]
    except (TypeError, ValueError) as exc:
        raise ValueError("protocol.training.seeds must be a list of positive integers.") from exc
    if not training_seeds or any(seed <= 0 for seed in training_seeds) or len(set(training_seeds)) != len(training_seeds):
        raise ValueError("protocol.training.seeds must be a non-empty list of unique positive integers.")
    cfg["protocol"]["training"]["seeds"] = training_seeds
    validation = cfg["protocol"].get("validation", {})
    if float(validation.get("ratio", -1.0)) != 0.10 or int(validation.get("seed", -1)) != 20250726:
        raise ValueError("Formal validation must use ratio=0.10 and seed=20250726.")
    if not bool(validation.get("official_test_selected_only", False)):
        raise ValueError("Formal protocol requires official_test_selected_only: true.")
    pgdf_protocol = cfg["protocol"].get("pgdf", {})
    dynamic_ratio = float(pgdf_protocol.get("dynamic_ratio", -1.0))
    prototype_keep_ratio = float(pgdf_protocol.get("prototype_keep_ratio", -1.0))
    if not 0.0 < dynamic_ratio <= 1.0 or not 0.0 < prototype_keep_ratio <= 1.0:
        raise ValueError("protocol.pgdf dynamic_ratio and prototype_keep_ratio must lie in (0, 1].")
    warmup_epochs = int(pgdf_protocol.get("warmup_epochs", -1))
    update_interval = int(pgdf_protocol.get("update_interval", -1))
    if warmup_epochs < 0 or update_interval <= 0:
        raise ValueError("protocol.pgdf warmup_epochs must be non-negative and update_interval must be positive.")
    cfg["protocol"]["pgdf"]["dynamic_ratio"] = dynamic_ratio
    cfg["protocol"]["pgdf"]["prototype_keep_ratio"] = prototype_keep_ratio
    cfg["protocol"]["pgdf"]["warmup_epochs"] = warmup_epochs
    cfg["protocol"]["pgdf"]["update_interval"] = update_interval
    if cfg["pgdf"].get("geometry_mode") != "neighbor_margin":
        raise ValueError("Standalone config must set pgdf.geometry_mode: neighbor_margin.")
    if bool(cfg["pgdf"].get("neighbor_margin_use_fallback", True)):
        raise ValueError("Formal Neighbor-Margin requires neighbor_margin_use_fallback: false.")


def validate_raw_roots(cfg: dict[str, Any]) -> None:
    for key, spec in cfg["datasets"].items():
        root = Path(str(spec.get("data_root", ""))).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"{key}: raw data root is missing: {root}")
        if spec["layout"] == "cub_metadata":
            for name in ("images.txt", "classes.txt", "image_class_labels.txt", "train_test_split.txt"):
                if not (root / name).is_file():
                    raise FileNotFoundError(f"{key}: missing required CUB metadata: {root / name}")
            if not (root / "images").is_dir():
                raise FileNotFoundError(f"{key}: missing CUB image directory: {root / 'images'}")
        elif spec["layout"] == "folder":
            for split in (spec["train_split"], spec["eval_split"]):
                if not (root / split).is_dir():
                    raise FileNotFoundError(f"{key}: missing split directory: {root / split}")
        else:
            raise ValueError(f"{key}: unsupported dataset layout {spec['layout']!r}")


def build_protocol_bundle_root(run_root: Path, cfg: dict[str, Any]) -> Path:
    """Build a method-neutral protocol root shared by future tier variants."""
    protocol = cfg["protocol"]
    noise = protocol["noise"]
    pgdf_protocol = protocol["pgdf"]
    noise_percent = int(round(100.0 * float(noise["ratio"])))
    noise_tag = f"cyclic_asym{noise_percent}_noise{int(noise['seed'])}"
    validation_tag = f"fixedval_s{int(protocol['validation']['seed'])}"
    selection_tag = (
        f"r{compact_ratio_token(float(pgdf_protocol['dynamic_ratio']))}_"
        f"p{compact_ratio_token(float(pgdf_protocol['prototype_keep_ratio']))}_"
        f"w{int(pgdf_protocol['warmup_epochs'])}_u{int(pgdf_protocol['update_interval'])}"
    )
    return run_root / noise_tag / validation_tag / selection_tag


def compact_ratio_token(value: float) -> str:
    """Format 0.8 and 0.4 as 08 and 04 in directory-safe protocol tags."""
    text = format(value, ".12g")
    if text.startswith("0."):
        return "0" + text[2:]
    return text.replace(".", "p")


def input_bundle_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return exactly the configuration that determines reusable prepared inputs."""
    return {
        "noise": copy.deepcopy(cfg["protocol"]["noise"]),
        "datasets": copy.deepcopy(cfg["datasets"]),
        "v1": copy.deepcopy(cfg["v1"]),
        "feature": copy.deepcopy(cfg["feature"]),
    }


def expected_prepared_paths(bundle_root: Path, key: str, spec: dict[str, Any]) -> tuple[Path, Path]:
    noise_index = bundle_root / "prepared_inputs" / "noise_indices" / key / f"{spec['noise_prefix']}_index.csv"
    input_dir = bundle_root / "prepared_inputs" / "v1_inputs" / spec["dataset_name"] / spec["input_version"]
    return noise_index, input_dir


def load_existing_prepared_bundle(bundle_root: Path, cfg: dict[str, Any]) -> list[PreparedDataset]:
    config_path = bundle_root / "input_bundle_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Existing protocol bundle is incomplete: missing {config_path}. "
            "Choose a new --run-root or repair/remove the incomplete bundle explicitly."
        )
    with config_path.open("r", encoding="utf-8") as handle:
        recorded = yaml.safe_load(handle) or {}
    if recorded != input_bundle_config(cfg):
        raise ValueError(
            f"Existing prepared inputs at {bundle_root} were built with a different noise/data/V1/feature configuration. "
            "Choose a different --run-root rather than mixing input bundles."
        )

    prepared: list[PreparedDataset] = []
    for key in ("cub", "cars", "aircraft"):
        spec = cfg["datasets"][key]
        noise_index, input_dir = expected_prepared_paths(bundle_root, key, spec)
        if not noise_index.is_file():
            raise FileNotFoundError(f"Existing protocol bundle is missing noise index: {noise_index}")
        verify_v1_input(input_dir)
        prepared.append(PreparedDataset(key=key, spec=spec, noise_index=noise_index, input_dir=input_dir))
    print(f"[prepared-inputs] Reusing verified immutable bundle: {bundle_root / 'prepared_inputs'}", flush=True)
    return prepared


def build_noise_index(key: str, spec: dict[str, Any], cfg: dict[str, Any], bundle_root: Path) -> Path:
    noise_cfg = cfg["protocol"]["noise"]
    output_dir = bundle_root / "prepared_inputs" / "noise_indices" / key
    ensure_dir(output_dir)
    prefix = str(spec["noise_prefix"])
    root = Path(str(spec["data_root"])).expanduser().resolve()
    ratio = float(noise_cfg["ratio"])
    seed = int(noise_cfg["seed"])
    strategy = str(noise_cfg["target_strategy"])

    if spec["layout"] == "cub_metadata":
        cub_noise.validate_config(root, ratio, strategy)
        metadata = cub_noise.load_cub_metadata(root)
        target_map = cub_noise.build_target_map(metadata["classes"], strategy)
        noisy_ids = cub_noise.sample_noisy_train_ids(metadata["rows"], ratio, seed)
        rows = cub_noise.build_index_rows(root, metadata["rows"], target_map, noisy_ids, ratio, seed, strategy)
        write_csv(output_dir / f"{prefix}_index.csv", rows, cub_noise.index_fieldnames())
        write_csv(
            output_dir / f"{prefix}_mapping.csv",
            cub_noise.build_mapping_rows(metadata["classes"], target_map),
            ["class_id", "class_name", "target_label", "target_class_name"],
        )
        write_csv(output_dir / f"{prefix}_summary.csv", cub_noise.build_summary_rows(rows), cub_noise.summary_fieldnames())
        resolved_noise_cfg = {
            "dataset": {"cub_root": str(root)},
            "noise": {"ratio": ratio, "seed": seed, "target_strategy": strategy},
            "output": {"dir": str(output_dir), "prefix": prefix},
        }
    else:
        train_split = str(spec["train_split"])
        eval_split = str(spec["eval_split"])
        folder_noise.validate_config(root, train_split, eval_split, ratio, strategy)
        records = folder_noise.discover_folder_records(root, train_split, eval_split)
        class_names = sorted({record["clean_label"] for record in records})
        target_map = folder_noise.build_target_map(class_names, strategy)
        noisy_keys = folder_noise.sample_noisy_train_keys(records, ratio, seed, train_split)
        rows = folder_noise.build_index_rows(records, target_map, noisy_keys, ratio, seed, strategy)
        write_csv(output_dir / f"{prefix}_index.csv", rows, folder_noise.index_fieldnames())
        write_csv(output_dir / f"{prefix}_mapping.csv", folder_noise.build_mapping_rows(target_map), ["class_name", "target_class_name"])
        write_csv(output_dir / f"{prefix}_summary.csv", folder_noise.build_summary_rows(rows, train_split), folder_noise.summary_fieldnames())
        resolved_noise_cfg = {
            "dataset": {"name": spec["dataset_name"], "root": str(root), "train_split": train_split, "test_split": eval_split},
            "noise": {"ratio": ratio, "seed": seed, "target_strategy": strategy},
            "output": {"dir": str(output_dir), "prefix": prefix},
        }

    write_yaml(output_dir / f"{prefix}_resolved_config.yaml", resolved_noise_cfg)
    index_path = output_dir / f"{prefix}_index.csv"
    print(f"[noise] {spec['dataset_name']}: {index_path}", flush=True)
    return index_path


def rebuild_v1_input(spec: dict[str, Any], cfg: dict[str, Any], bundle_root: Path, noise_index: Path) -> Path:
    feature_cfg = copy.deepcopy(cfg["feature"])
    feature_cfg["reuse"] = False
    v1_cfg = {
        "dataset": {
            "name": spec["dataset_name"],
            "root": str(Path(str(spec["data_root"])).expanduser()),
            "index_file": str(noise_index),
            "train_split": spec["train_split"],
            "eval_split": spec["eval_split"],
            "max_classes": None,
            "max_train_per_class": None,
            "max_eval_per_class": None,
            "verify_images": True,
        },
        "output": {"root": str(bundle_root / "prepared_inputs" / "v1_inputs"), "version": spec["input_version"]},
        "feature": feature_cfg,
        "graph": copy.deepcopy(spec["v1_graph"]),
        "selection": copy.deepcopy(cfg["v1"]["selection"]),
        "train": copy.deepcopy(cfg["v1"]["linear_train"]),
    }
    v1_cfg["feature"]["batch_size"] = int(cfg["v1"]["feature_batch_size"])
    input_dir = Path(v1_cfg["output"]["root"]) / spec["dataset_name"] / spec["input_version"]
    print(f"[v1] rebuilding {spec['dataset_name']} -> {input_dir}", flush=True)
    run_v1_web_bird(v1_cfg)
    return input_dir


def verify_v1_input(input_dir: Path) -> None:
    missing = [name for name in V1_REQUIRED_FILES if not (input_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"V1 preparation is incomplete at {input_dir}: missing {missing}")


def run_lora_experiment(item: PreparedDataset, cfg: dict[str, Any], variant_root: Path, args: argparse.Namespace) -> None:
    protocol = cfg["protocol"]
    pgdf_protocol = protocol["pgdf"]
    validation = protocol["validation"]
    output_dir = variant_root / item.key
    command = [
        args.python,
        "scripts/run_lora_checkpoint_validation.py",
        "--input-dir",
        str(item.input_dir),
        "--noise-index",
        str(item.noise_index),
        "--config",
        str(args.config),
        "--output-dir",
        str(output_dir),
        "--run-layout",
        "direct_seed",
        "--methods",
        "pgdf_dynamic_proto",
        "--seeds",
        ",".join(str(seed) for seed in protocol["training"]["seeds"]),
        "--validation-ratio",
        str(validation["ratio"]),
        "--validation-seed",
        str(validation["seed"]),
        "--dynamic-ratio",
        str(pgdf_protocol["dynamic_ratio"]),
        "--fixed-p",
        str(pgdf_protocol["prototype_keep_ratio"]),
        "--warmup-epochs",
        str(pgdf_protocol["warmup_epochs"]),
        "--update-interval",
        str(pgdf_protocol["update_interval"]),
        "--official-test-selected-only",
        "--no-posthoc-oracle-test",
        "--device",
        str(cfg["feature"]["device"]),
    ]
    local_repo = str(cfg["feature"].get("local_repo", ""))
    if local_repo:
        command.extend(["--local-repo", local_repo])
    print(f"[lora] {item.spec['dataset_name']} -> {output_dir}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def write_provenance(variant_root: Path, bundle_root: Path, config_path: Path, cfg: dict[str, Any], prepared: list[PreparedDataset]) -> None:
    payload: dict[str, Any] = {
        "protocol": cfg["protocol"],
        "standalone_config": str(config_path),
        "standalone_config_sha256": sha256_file(config_path),
        "noise_indices": {item.key: {"path": str(item.noise_index), "sha256": sha256_file(item.noise_index)} for item in prepared},
        "input_dirs": {item.key: str(item.input_dir) for item in prepared},
        "input_bundle_root": str(bundle_root),
        "git_commit": git_commit(),
    }
    (variant_root / "provenance.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def print_plan(cfg: dict[str, Any], run_root: Path, args: argparse.Namespace) -> None:
    protocol = cfg["protocol"]
    bundle_root = build_protocol_bundle_root(run_root, cfg)
    variant_root = bundle_root / "variants" / str(protocol["variant_id"])
    print(
        "[DRY RUN] standalone experiment: "
        f"noise_seed={protocol['noise']['seed']}, train_seeds={protocol['training']['seeds']}, "
        f"validation_seed={protocol['validation']['seed']}, "
        f"r={protocol['pgdf']['dynamic_ratio']}, p={protocol['pgdf']['prototype_keep_ratio']}, "
        f"warmup={protocol['pgdf']['warmup_epochs']}, interval={protocol['pgdf']['update_interval']}",
        flush=True,
    )
    for key in ("cub", "cars", "aircraft"):
        spec = cfg["datasets"][key]
        noise_index, input_dir = expected_prepared_paths(bundle_root, key, spec)
        output_dir = variant_root / key
        print(
            f"[DRY RUN] {spec['dataset_name']}: raw={spec['data_root']}; "
            f"noise -> {noise_index}; V1 -> {input_dir}; "
            f"LoRA -> {output_dir}",
            flush=True,
        )
    print(f"[DRY RUN] Python runner: {args.python}; device={cfg['feature']['device']}", flush=True)


if __name__ == "__main__":
    main()
