"""WebFG-only raw-byte recovery and fixed corrupt-image manifest generation."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
from typing import Any

from .dataset import WebFGRecord, parse_list_file


# These are only basenames supplied by the maintenance request.  Full relative
# paths are resolved against the actual official list files, never guessed.
REQUESTED_CORRUPT_BASENAMES: dict[str, tuple[str, ...]] = {
    "web-bird": ("062.Herring_Gull_00090.jpg", "151.Black_capped_Vireo_00081.jpg"),
    "web-aircraft": (
        "747-400_00017.jpg",
        "777-200_00102.jpg",
        "ERJ 145_00109.jpg",
        "Tornado_00108.jpg",
        "Tornado_00111.jpg",
    ),
}


def prepare_corrupt_image_resolution(data_root: Path, output_root: Path) -> dict[str, Path]:
    """Recover valid archive bytes or exclude only confirmed unreadable files.

    No source file is changed.  A recoverable original member is copied to the
    WebFG-only recovery directory and selected by an explicit manifest.  An
    archive-invalid member is excluded by that same manifest.  Any other bad
    image remains in the manifest and therefore makes subsequent preflight
    fail rather than being silently skipped.
    """
    data_root = Path(data_root)
    resolution_rows: list[dict[str, Any]] = []
    resolution_by_relative: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset, basenames in REQUESTED_CORRUPT_BASENAMES.items():
        dataset_root = data_root / dataset
        source_records = parse_list_file(dataset_root / "train-list.txt")
        for basename in basenames:
            matches = [record for record in source_records if Path(record.relative_path).name == basename]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected exactly one {dataset} list-file match for {basename}, found {len(matches)}."
                )
            record = matches[0]
            source_path = dataset_root / record.relative_path
            local = inspect_file(source_path)
            archive = inspect_archive_member(data_root / f"{dataset}.tar.gz", f"{dataset}/{record.relative_path}")
            row: dict[str, Any] = {
                "dataset": dataset,
                "relative_path": record.relative_path,
                "observed_label": record.label,
                "original_file_size_bytes": local["size_bytes"],
                "original_sha256": local["sha256"],
                "original_decode_status": local["decode_status"],
                "original_decode_failure_reason": local["decode_failure_reason"],
                "archive_path": str(data_root / f"{dataset}.tar.gz"),
                "archive_member": archive["member"],
                "archive_file_size_bytes": archive["size_bytes"],
                "archive_sha256": archive["sha256"],
                "archive_decode_status": archive["decode_status"],
                "archive_decode_failure_reason": archive["decode_failure_reason"],
                "recovery_attempt_result": "",
                "final_action": "",
                "effective_path": "",
            }
            if local["decode_status"] == "PASS":
                row.update({"recovery_attempt_result": "not_needed_local_original_decodes", "final_action": "use_original", "effective_path": str(source_path)})
            elif archive["decode_status"] == "PASS" and archive["bytes"] is not None:
                recovery_path = output_root / "recovered_originals" / dataset / record.relative_path
                write_bytes_once(recovery_path, archive["bytes"])
                recovered = inspect_file(recovery_path)
                if recovered["decode_status"] != "PASS" or recovered["sha256"] != archive["sha256"]:
                    raise RuntimeError(f"WebFG recovery verification failed for {recovery_path}")
                row.update({"recovery_attempt_result": "archive_member_restored_and_revalidated", "final_action": "use_recovered_original", "effective_path": str(recovery_path)})
            else:
                row.update({"recovery_attempt_result": "no_valid_original_archive_member", "final_action": "exclude_confirmed_unrecoverable", "effective_path": ""})
            resolution_rows.append(row)
            resolution_by_relative[(dataset, record.relative_path)] = row
    manifests = write_verified_manifests(data_root, output_root, resolution_by_relative)
    excluded = [row for row in resolution_rows if row["final_action"] == "exclude_confirmed_unrecoverable"]
    excluded_path = output_root / "excluded_corrupt_images.csv"
    write_csv(excluded_path, excluded, resolution_fields())
    write_csv(output_root / "corrupt_image_resolution.csv", resolution_rows, resolution_fields())
    write_resolution_report(output_root / "corrupt_image_resolution_report.md", resolution_rows, manifests)
    return {"excluded": excluded_path, "resolution": output_root / "corrupt_image_resolution.csv", **manifests}


def inspect_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"size_bytes": "", "sha256": "", "decode_status": "FAIL", "decode_failure_reason": f"missing_file:{path}", "bytes": None}
    payload = path.read_bytes()
    return {"size_bytes": len(payload), "sha256": sha256(payload), **decode_bytes(payload), "bytes": payload}


def inspect_archive_member(archive_path: Path, expected_member: str) -> dict[str, Any]:
    if not archive_path.is_file():
        return {"member": "", "size_bytes": "", "sha256": "", "decode_status": "FAIL", "decode_failure_reason": f"archive_missing:{archive_path}", "bytes": None}
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            normalized = expected_member.replace("\\", "/")
            member = next((item for item in archive.getmembers() if item.name.lstrip("./") == normalized), None)
            if member is None or not member.isfile():
                return {"member": "", "size_bytes": "", "sha256": "", "decode_status": "FAIL", "decode_failure_reason": "archive_member_missing", "bytes": None}
            handle = archive.extractfile(member)
            if handle is None:
                return {"member": member.name, "size_bytes": "", "sha256": "", "decode_status": "FAIL", "decode_failure_reason": "archive_extractfile_none", "bytes": None}
            payload = handle.read()
    except (OSError, tarfile.TarError) as exc:
        return {"member": "", "size_bytes": "", "sha256": "", "decode_status": "FAIL", "decode_failure_reason": f"archive_error:{type(exc).__name__}:{exc}", "bytes": None}
    return {"member": member.name, "size_bytes": len(payload), "sha256": sha256(payload), **decode_bytes(payload), "bytes": payload}


def decode_bytes(payload: bytes) -> dict[str, str]:
    """Require a full PIL decode without enabling truncated-image tolerance."""
    from PIL import Image

    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
        with Image.open(io.BytesIO(payload)) as image:
            image.convert("RGB")
    except Exception as exc:  # noqa: BLE001 - evidence is persisted for every requested file.
        return {"decode_status": "FAIL", "decode_failure_reason": f"{type(exc).__name__}: {exc}"}
    return {"decode_status": "PASS", "decode_failure_reason": ""}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_bytes_once(path: Path, payload: bytes) -> None:
    """Write only WebFG recovery artifacts, refusing a conflicting overwrite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_bytes()
        if sha256(existing) != sha256(payload):
            raise FileExistsError(f"Recovery destination contains different content: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_verified_manifests(
    data_root: Path,
    output_root: Path,
    resolution: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Path]:
    manifests: dict[str, Path] = {}
    for dataset in ("web-bird", "web-car", "web-aircraft"):
        records = parse_list_file(data_root / dataset / "train-list.txt")
        rows: list[dict[str, Any]] = []
        for record in records:
            decision = resolution.get((dataset, record.relative_path))
            if decision is not None and decision["final_action"] == "exclude_confirmed_unrecoverable":
                continue
            effective_path = decision["effective_path"] if decision is not None else str(data_root / dataset / record.relative_path)
            rows.append({
                "relative_path": record.relative_path,
                "observed_label": record.label,
                "effective_path": effective_path,
                "resolution": decision["final_action"] if decision is not None else "use_original_verified_by_effective_preflight",
            })
        manifest = output_root / "manifests" / f"{dataset}_effective_train.csv"
        write_csv(manifest, rows, ["relative_path", "observed_label", "effective_path", "resolution"])
        manifests[dataset] = manifest
    return manifests


def resolution_fields() -> list[str]:
    return [
        "dataset", "relative_path", "observed_label", "original_file_size_bytes", "original_sha256", "original_decode_status", "original_decode_failure_reason",
        "archive_path", "archive_member", "archive_file_size_bytes", "archive_sha256", "archive_decode_status", "archive_decode_failure_reason",
        "recovery_attempt_result", "final_action", "effective_path",
    ]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_resolution_report(path: Path, rows: list[dict[str, Any]], manifests: dict[str, Path]) -> None:
    recovered = sum(row["final_action"] == "use_recovered_original" for row in rows)
    excluded = sum(row["final_action"] == "exclude_confirmed_unrecoverable" for row in rows)
    lines = [
        "# WebFG-496 Corrupt Image Resolution Report",
        "",
        "- Original data were never modified or deleted.",
        "- Archive members were read as raw bytes, SHA-256 compared, and fully decoded with PIL; truncated-image tolerance was not enabled.",
        f"- Requested files: {len(rows)}; recovered original bytes: {recovered}; fixed exclusions: {excluded}.",
        "",
        "| dataset | relative path | original decode | archive decode | action |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {row['dataset']} | `{row['relative_path']}` | {row['original_decode_status']} | {row['archive_decode_status']} | {row['final_action']} |")
    lines.extend(["", "## Effective training manifests", ""])
    for dataset, manifest in manifests.items():
        lines.append(f"- {dataset}: `{manifest}`")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, path)
