"""Audit reuse of the published random-derangement asym40 mapping artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", action="append", required=True,
        metavar="NAME=CYCLIC_INDEX=RANDOM_MANIFEST=MAPPING_JSON=VALIDATION_DIR=NEW_VALIDATION_DIR",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse(value: str) -> tuple[str, Path, Path, Path, Path, Path]:
    parts = value.split("=", 5)
    if len(parts) != 6 or not all(parts):
        raise ValueError("Invalid --dataset value")
    return parts[0], *(Path(part) for part in parts[1:])


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite mapping audit: {args.output}")
    audit: list[dict[str, Any]] = []
    for raw in args.dataset:
        dataset, cyclic_path, random_path, mapping_path, validation_dir, new_validation_dir = parse(raw)
        required = [cyclic_path, random_path, mapping_path, validation_dir / "validation_manifest.json", new_validation_dir / "validation_manifest.json"]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{dataset}: required audit artifacts missing: {missing}")
        cyclic = [row for row in read_csv(cyclic_path) if row.get("split", "train").lower() == "train"]
        random = [row for row in read_csv(random_path) if row.get("split", "train").lower() == "train"]
        key = lambda row: str(row.get("path") or row.get("abs_path") or row.get("index"))
        cyclic_by_path, random_by_path = ({key(row): row for row in rows} for rows in (cyclic, random))
        if len(cyclic_by_path) != len(cyclic) or len(random_by_path) != len(random):
            raise ValueError(f"{dataset}: duplicate or missing source path in mapping manifests")
        source_path_match = set(cyclic_by_path) == set(random_by_path)
        if not source_path_match:
            raise RuntimeError(f"{dataset}: source training-pool identities differ between cyclic and random mapping")
        paired = [(row, random_by_path[path]) for path, row in cyclic_by_path.items()]
        flip_identity_match = all(str(a["is_noisy"]) == str(b["is_noisy"]) for a, b in paired)
        clean_match = all(str(a["clean_label"]) == str(b["clean_label"]) for a, b in paired)
        unchanged_unflipped = all(
            str(a["web_label"]) == str(b["web_label"])
            for a, b in paired if str(a["is_noisy"]) in {"0", "false", "False"}
        )
        payload = json.loads(mapping_path.read_text(encoding="utf-8"))
        mapping_seed = int(payload.get("mapping_seed", -1))
        mapping = {str(source): str(target) for source, target in payload.get("mapping", {}).items()}
        mapped_targets_match = all(
            str(b["web_label"]) == mapping.get(str(a["clean_label"]), "")
            for a, b in paired if str(a["is_noisy"]) in {"1", "true", "True"}
        )
        mapping_is_derangement = bool(mapping) and all(source != target for source, target in mapping.items())
        reference = json.loads((validation_dir / "validation_manifest.json").read_text(encoding="utf-8"))
        candidate = json.loads((new_validation_dir / "validation_manifest.json").read_text(encoding="utf-8"))
        same_validation = reference == candidate
        row = {
            "dataset": dataset, "num_source_samples": len(cyclic),
            "num_flipped": sum(str(item["is_noisy"]) in {"1", "true", "True"} for item in cyclic),
            "source_path_match": "yes" if source_path_match else "no",
            "flipped_identity_match": "yes" if flip_identity_match else "no",
            "clean_label_match": "yes" if clean_match else "no",
            "unflipped_observed_label_match": "yes" if unchanged_unflipped else "no",
            "flipped_target_matches_fixed_mapping": "yes" if mapped_targets_match else "no",
            "mapping_is_derangement": "yes" if mapping_is_derangement else "no",
            "mapping_seed": mapping_seed, "mapping_type": payload.get("mapping_type", ""),
            "mapping_sha256": sha256(mapping_path),
            "validation_manifest_hash": sha256(validation_dir / "validation_manifest.json"),
            "new_validation_manifest_hash": sha256(new_validation_dir / "validation_manifest.json"),
            "validation_manifest_match": "yes" if same_validation else "no",
        }
        if not (
            source_path_match and flip_identity_match and clean_match and unchanged_unflipped
            and mapped_targets_match and mapping_is_derangement and mapping_seed == 20260815 and same_validation
        ):
            raise RuntimeError(f"{dataset}: random-derangement mapping control audit failed: {row}")
        audit.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(audit[0])
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit)
    print(f"Random mapping audit PASS: {args.output}")


if __name__ == "__main__":
    main()
