from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcdd.io_utils import ensure_dir, read_csv, write_csv, write_json


METRICS = ["S_clean", "Q_same", "centroid_score", "R_class", "I_class_norm", "loss", "confidence"]


class AssetContext:
    def __init__(self, asset_dir: Path, copy_assets: bool, path_maps: list[tuple[str, str]]):
        self.asset_dir = asset_dir
        self.copy_assets = copy_assets
        self.path_maps = path_maps
        self.cache: dict[str, Path] = {}
        ensure_dir(asset_dir)

    def html_src(self, image_path: str, html_file: Path, index: int) -> str:
        source = Path(image_path)
        if not self.copy_assets:
            return source.as_uri() if source.is_absolute() else image_path
        asset = self.copy_image(source, index)
        if asset is None:
            return source.as_uri() if source.is_absolute() else image_path
        return os.path.relpath(asset, html_file.parent).replace("\\", "/")

    def copy_image(self, source: Path, index: int) -> Path | None:
        resolved_source = self.resolve_source(source)
        if resolved_source is None:
            return None
        key = str(resolved_source)
        if key in self.cache:
            return self.cache[key]
        target = self.asset_dir / f"{index:06d}_{safe_name(resolved_source.name)}"
        suffix_count = 1
        while target.exists() and not same_file(target, resolved_source):
            target = self.asset_dir / f"{index:06d}_{safe_name(resolved_source.stem)}_{suffix_count}{resolved_source.suffix}"
            suffix_count += 1
        if not target.exists():
            shutil.copy2(resolved_source, target)
        self.cache[key] = target
        return target

    def resolve_source(self, source: Path) -> Path | None:
        if source.exists():
            return source
        raw = str(source).replace("\\", "/")
        for old, new in self.path_maps:
            old_norm = old.replace("\\", "/").rstrip("/")
            if raw == old_norm or raw.startswith(old_norm + "/"):
                suffix = raw[len(old_norm) :].lstrip("/")
                mapped = Path(new).expanduser() / Path(*suffix.split("/"))
                if mapped.exists():
                    return mapped
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze samples removed by GCDD Q_same AND centroid gate.")
    parser.add_argument("--input-dir", default="outputs/v1_web_bird", help="V1 output directory.")
    parser.add_argument("--gate-dir", help="V1.6 gated split directory. Defaults to <input-dir>/v1_6_gated_splits.")
    parser.add_argument("--output-dir", help="Analysis output directory. Defaults to <gate-dir>/qp_gate_deleted_analysis.")
    parser.add_argument("--loss-samples", type=int, default=30, help="Neighbor HTML count for highest-loss deleted samples.")
    parser.add_argument("--confidence-samples", type=int, default=30, help="Neighbor HTML count for lowest-confidence deleted samples.")
    parser.add_argument("--top-class-count", type=int, default=20, help="Number of high-deletion-ratio classes to report.")
    parser.add_argument("--samples-per-top-class", type=int, default=5, help="Samples per high-deletion-ratio class for HTML.")
    parser.add_argument("--copy-assets", action="store_true", default=True, help="Copy visualization images into figures/assets.")
    parser.add_argument("--no-copy-assets", action="store_false", dest="copy_assets", help="Do not copy images; HTML uses original paths.")
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW", help="Map original image root to local root before copying assets.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    gate_dir = Path(args.gate_dir) if args.gate_dir else input_dir / "v1_6_gated_splits"
    output_dir = Path(args.output_dir) if args.output_dir else gate_dir / "qp_gate_deleted_analysis"
    ensure_dir(output_dir)
    ensure_dir(output_dir / "figures" / "neighbors")
    ensure_dir(output_dir / "figures" / "assets")

    data = load_data(input_dir, gate_dir)
    deleted_mask = data["full_gcdd_clean"] & ~data["qp_gate_clean"]
    if int(deleted_mask.sum()) == 0:
        raise ValueError("No samples were deleted by qp_gate.")

    class_rows, class_stats = write_per_class_distribution(
        output_dir / "qp_gate_deleted_per_class.csv",
        output_dir / "qp_gate_deleted_top20_classes.csv",
        data,
        deleted_mask,
        args.top_class_count,
    )
    deleted_rows = write_deleted_samples(output_dir / "deleted_by_qp_gate.csv", data, deleted_mask, class_stats)
    metric_rows = write_metric_summary(output_dir / "deleted_metric_summary.csv", data, deleted_mask)

    asset_context = AssetContext(output_dir / "figures" / "assets", args.copy_assets, parse_path_maps(args.path_map))
    viz_rows = write_neighbor_visualizations(
        output_dir / "figures" / "neighbors",
        output_dir / "visualization_index.csv",
        data,
        deleted_mask,
        class_rows,
        asset_context,
        args.loss_samples,
        args.confidence_samples,
        args.samples_per_top_class,
    )
    write_manual_review_template(output_dir / "manual_qp_gate_review.csv", viz_rows)
    write_manual_review_html(output_dir / "manual_review.html", viz_rows)
    write_summary(output_dir / "analysis_summary.md", data, deleted_mask, class_rows, metric_rows, viz_rows)
    write_json(
        output_dir / "analysis_metadata.json",
        {
            "input_dir": str(input_dir),
            "gate_dir": str(gate_dir),
            "output_dir": str(output_dir),
            "deleted_count": int(deleted_mask.sum()),
            "visualization_count": len(viz_rows),
        },
    )
    print(f"QP-gate deleted-sample analysis written to {output_dir}", flush=True)


def load_data(input_dir: Path, gate_dir: Path) -> dict[str, Any]:
    required = [
        input_dir / "full_gcdd_clean_split.csv",
        gate_dir / "gcdd_qp_gate_split.csv",
        input_dir / "gcdd_scores.csv",
        input_dir / "centroid_scores.csv",
        input_dir / "linear_all_train_scores.csv",
        input_dir / "class_knn_indices.npy",
        input_dir / "global_knn_indices.npy",
        input_dir / "labels.npy",
        input_dir / "paths.txt",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required file is missing: {path}")

    score_rows = sorted(read_csv(input_dir / "gcdd_scores.csv"), key=lambda row: int(row["index"]))
    centroid_rows = sorted(read_csv(input_dir / "centroid_scores.csv"), key=lambda row: int(row["index"]))
    train_score_rows = sorted(read_csv(input_dir / "linear_all_train_scores.csv"), key=lambda row: int(row["index"]))
    labels = np.load(input_dir / "labels.npy", allow_pickle=True).astype(str)
    paths = (input_dir / "paths.txt").read_text(encoding="utf-8").splitlines()
    n = len(labels)
    if len(score_rows) != n or len(centroid_rows) != n or len(train_score_rows) != n or len(paths) != n:
        raise ValueError("Input files have inconsistent lengths.")

    return {
        "index": np.array([int(row["index"]) for row in score_rows], dtype=np.int64),
        "path": np.array([row["path"] for row in score_rows], dtype=object),
        "labels": labels,
        "paths": paths,
        "S_clean": float_array(score_rows, "S_clean"),
        "Q_same": float_array(score_rows, "Q_same"),
        "R_class": float_array(score_rows, "R_class"),
        "I_class_norm": float_array(score_rows, "I_class_norm"),
        "centroid_score": float_array(centroid_rows, "centroid_score"),
        "loss": float_array(train_score_rows, "loss"),
        "confidence": float_array(train_score_rows, "confidence"),
        "full_gcdd_clean": read_clean_mask(input_dir / "full_gcdd_clean_split.csv", n),
        "qp_gate_clean": read_clean_mask(gate_dir / "gcdd_qp_gate_split.csv", n),
        "class_knn": np.load(input_dir / "class_knn_indices.npy"),
        "global_knn": np.load(input_dir / "global_knn_indices.npy"),
    }


def float_array(rows: list[dict[str, str]], field: str) -> np.ndarray:
    return np.array([float(row[field]) for row in rows], dtype=np.float32)


def read_clean_mask(path: Path, n: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for row in read_csv(path):
        idx = int(row["index"])
        if idx >= n:
            raise ValueError(f"Index {idx} in {path} exceeds expected length {n}.")
        mask[idx] = row["state"] == "clean"
    return mask


def write_per_class_distribution(
    path: Path,
    top_path: Path,
    data: dict[str, Any],
    deleted_mask: np.ndarray,
    top_class_count: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = []
    labels = data["labels"]
    for class_id in sorted(set(labels.tolist())):
        class_idx = labels == class_id
        gcdd_clean_count = int(np.sum(data["full_gcdd_clean"] & class_idx))
        deleted_count = int(np.sum(deleted_mask & class_idx))
        rows.append(
            {
                "class_id": class_id,
                "gcdd_clean_count": gcdd_clean_count,
                "deleted_count": deleted_count,
                "deleted_ratio": safe_ratio(deleted_count, gcdd_clean_count),
            }
        )
    sorted_rows = sorted(rows, key=lambda row: (float(row["deleted_ratio"]), int(row["deleted_count"])), reverse=True)
    write_csv(path, rows, ["class_id", "gcdd_clean_count", "deleted_count", "deleted_ratio"])
    write_csv(top_path, sorted_rows[:top_class_count], ["class_id", "gcdd_clean_count", "deleted_count", "deleted_ratio"])
    return rows, {row["class_id"]: row for row in rows}


def write_deleted_samples(path: Path, data: dict[str, Any], deleted_mask: np.ndarray, class_stats: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for i in np.where(deleted_mask)[0]:
        label = data["labels"][i]
        class_row = class_stats[label]
        rows.append(
            {
                "index": int(data["index"][i]),
                "path": data["path"][i],
                "web_label": label,
                "S_clean": float(data["S_clean"][i]),
                "Q_same": float(data["Q_same"][i]),
                "centroid_score": float(data["centroid_score"][i]),
                "R_class": float(data["R_class"][i]),
                "I_class_norm": float(data["I_class_norm"][i]),
                "loss": float(data["loss"][i]),
                "confidence": float(data["confidence"][i]),
                "class_deleted_count": int(class_row["deleted_count"]),
                "class_deleted_ratio": float(class_row["deleted_ratio"]),
            }
        )
    write_csv(
        path,
        rows,
        [
            "index",
            "path",
            "web_label",
            "S_clean",
            "Q_same",
            "centroid_score",
            "R_class",
            "I_class_norm",
            "loss",
            "confidence",
            "class_deleted_count",
            "class_deleted_ratio",
        ],
    )
    return rows


def write_metric_summary(path: Path, data: dict[str, Any], deleted_mask: np.ndarray) -> list[dict[str, Any]]:
    groups = {
        "qp_deleted": deleted_mask,
        "qp_kept_gcdd_clean": data["full_gcdd_clean"] & ~deleted_mask,
        "full_gcdd_clean": data["full_gcdd_clean"],
        "all_train": np.ones(len(deleted_mask), dtype=bool),
    }
    rows = []
    for group, mask in groups.items():
        row: dict[str, Any] = {"group": group, "count": int(mask.sum())}
        for metric in METRICS:
            values = data[metric][mask]
            row[f"{metric}_mean"] = finite_mean(values)
            row[f"{metric}_median"] = finite_percentile(values, 50)
            row[f"{metric}_p25"] = finite_percentile(values, 25)
            row[f"{metric}_p75"] = finite_percentile(values, 75)
        rows.append(row)
    fieldnames = ["group", "count"]
    for metric in METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_median", f"{metric}_p25", f"{metric}_p75"])
    write_csv(path, rows, fieldnames)
    return rows


def write_neighbor_visualizations(
    output_dir: Path,
    index_path: Path,
    data: dict[str, Any],
    deleted_mask: np.ndarray,
    class_rows: list[dict[str, Any]],
    asset_context: AssetContext,
    loss_samples: int,
    confidence_samples: int,
    samples_per_top_class: int,
) -> list[dict[str, Any]]:
    deleted_idx = np.where(deleted_mask)[0]
    selections: list[tuple[int, str]] = []
    selections.extend(select_by_metric(deleted_idx, data["loss"], loss_samples, largest=True, rule="highest_loss"))
    selections.extend(select_by_metric(deleted_idx, data["confidence"], confidence_samples, largest=False, rule="lowest_confidence"))
    selections.extend(select_from_high_deleted_ratio_classes(deleted_idx, data, class_rows, samples_per_top_class))

    rows = []
    seen: set[tuple[int, str]] = set()
    for idx, rule in selections:
        key = (idx, rule)
        if key in seen:
            continue
        seen.add(key)
        filename = f"{int(data['index'][idx]):06d}_{safe_name(rule)}.html"
        html_path = output_dir / filename
        write_sample_html(html_path, data, idx, rule, asset_context)
        rows.append(
            {
                "index": int(data["index"][idx]),
                "path": data["path"][idx],
                "web_label": data["labels"][idx],
                "selection_rule": rule,
                "html": filename,
                "S_clean": float(data["S_clean"][idx]),
                "Q_same": float(data["Q_same"][idx]),
                "centroid_score": float(data["centroid_score"][idx]),
                "loss": float(data["loss"][idx]),
                "confidence": float(data["confidence"][idx]),
            }
        )
    write_csv(index_path, rows, ["index", "path", "web_label", "selection_rule", "html", "S_clean", "Q_same", "centroid_score", "loss", "confidence"])
    return rows


def write_manual_review_template(path: Path, viz_rows: list[dict[str, Any]]) -> None:
    rows = []
    for row in viz_rows:
        rows.append(
            {
                "index": row["index"],
                "path": row["path"],
                "web_label": row["web_label"],
                "selection_rule": row["selection_rule"],
                "html": f"figures/neighbors/{row['html']}",
                "manual_label": "",
                "looks_like_class_neighbors": "",
                "looks_like_global_neighbors": "",
                "has_duplicate": "",
                "note": "",
                "S_clean": row["S_clean"],
                "Q_same": row["Q_same"],
                "centroid_score": row["centroid_score"],
                "loss": row["loss"],
                "confidence": row["confidence"],
            }
        )
    write_csv(path, rows, manual_review_fieldnames())


def write_manual_review_html(path: Path, viz_rows: list[dict[str, Any]]) -> None:
    rows = []
    for row in viz_rows:
        item = dict(row)
        item["html"] = f"figures/neighbors/{row['html']}"
        rows.append(item)
    rows_json = json.dumps(rows, ensure_ascii=False)
    fields_json = json.dumps(manual_review_fieldnames(), ensure_ascii=False)
    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>QP Gate Manual Review</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #1f2933; }}
    .layout {{ display: grid; grid-template-columns: 380px 1fr; height: 100vh; }}
    .panel {{ border-right: 1px solid #d0d7de; padding: 12px; overflow: auto; }}
    .viewer {{ height: 100vh; }}
    iframe {{ width: 100%; height: 100%; border: 0; }}
    .row {{ padding: 8px; border: 1px solid #d0d7de; margin-bottom: 8px; cursor: pointer; border-radius: 6px; }}
    .row.active {{ border-color: #0969da; background: #eef6ff; }}
    .row.done {{ background: #f0fff4; }}
    label {{ display: block; margin-top: 8px; font-size: 13px; }}
    select, textarea {{ width: 100%; box-sizing: border-box; margin-top: 4px; }}
    textarea {{ height: 64px; }}
    button {{ margin-top: 8px; margin-right: 6px; padding: 6px 10px; }}
    .metrics {{ font-size: 12px; color: #57606a; line-height: 1.5; }}
    .toolbar {{ position: sticky; top: 0; background: #fff; padding-bottom: 8px; border-bottom: 1px solid #d0d7de; z-index: 1; }}
  </style>
</head>
<body>
  <div class="layout">
    <div class="panel">
      <div class="toolbar">
        <h2>QP Gate Review</h2>
        <div id="progress"></div>
        <button onclick="prevItem()">Prev</button>
        <button onclick="nextItem()">Next</button>
        <button onclick="exportCsv()">Export CSV</button>
        <button onclick="clearAnnotations()">Clear saved annotations</button>
        <label>manual_label
          <select id="manual_label" onchange="saveCurrent()">
            <option value=""></option>
            <option value="noise">noise</option>
            <option value="clean_like">clean_like</option>
            <option value="uncertain">uncertain</option>
          </select>
        </label>
        <label>looks_like_class_neighbors
          <select id="looks_like_class_neighbors" onchange="saveCurrent()">
            <option value=""></option><option value="yes">yes</option><option value="no">no</option><option value="uncertain">uncertain</option>
          </select>
        </label>
        <label>looks_like_global_neighbors
          <select id="looks_like_global_neighbors" onchange="saveCurrent()">
            <option value=""></option><option value="yes">yes</option><option value="no">no</option><option value="uncertain">uncertain</option>
          </select>
        </label>
        <label>has_duplicate
          <select id="has_duplicate" onchange="saveCurrent()">
            <option value=""></option><option value="yes">yes</option><option value="no">no</option><option value="uncertain">uncertain</option>
          </select>
        </label>
        <label>note
          <textarea id="note" oninput="saveCurrent()"></textarea>
        </label>
      </div>
      <div id="list"></div>
    </div>
    <div class="viewer">
      <iframe id="frame"></iframe>
    </div>
  </div>
  <script>
    const rows = {rows_json};
    const fields = {fields_json};
    const storageKey = "qp_gate_manual_review:" + location.pathname;
    let current = 0;
    let annotations = JSON.parse(localStorage.getItem(storageKey) || "{{}}");

    function defaultAnnotation(row) {{
      return {{
        index: row.index,
        path: row.path,
        web_label: row.web_label,
        selection_rule: row.selection_rule,
        html: row.html,
        manual_label: "",
        looks_like_class_neighbors: "",
        looks_like_global_neighbors: "",
        has_duplicate: "",
        note: "",
        S_clean: row.S_clean,
        Q_same: row.Q_same,
        centroid_score: row.centroid_score,
        loss: row.loss,
        confidence: row.confidence
      }};
    }}

    function getAnnotation(row) {{
      const ann = Object.assign(defaultAnnotation(row), annotations[row.index] || {{}});
      if (ann.manual_label === "hard_clean" || ann.manual_label === "boundary") {{
        ann.manual_label = "clean_like";
      }}
      if (ann.manual_label === "duplicate") {{
        ann.manual_label = "clean_like";
        ann.has_duplicate = ann.has_duplicate || "yes";
      }}
      return ann;
    }}

    function renderList() {{
      const list = document.getElementById("list");
      list.innerHTML = "";
      rows.forEach((row, i) => {{
        const ann = getAnnotation(row);
        const div = document.createElement("div");
        div.className = "row" + (i === current ? " active" : "") + (ann.manual_label ? " done" : "");
        div.onclick = () => selectItem(i);
        div.innerHTML = `<b>${{row.index}}</b> ${{row.web_label}}<br>${{row.selection_rule}}<br>` +
          `<span class="metrics">label=${{ann.manual_label || "-"}} | loss=${{Number(row.loss).toFixed(3)}} | conf=${{Number(row.confidence).toFixed(3)}} | Q=${{Number(row.Q_same).toFixed(3)}} | C=${{Number(row.centroid_score).toFixed(3)}}</span>`;
        list.appendChild(div);
      }});
      const reviewedEntryCount = rows.filter(row => getAnnotation(row).manual_label).length;
      const uniqueIds = [...new Set(rows.map(row => String(row.index)))];
      const reviewedUniqueCount = uniqueIds.filter(index => {{
        const row = rows.find(item => String(item.index) === index);
        return row && getAnnotation(row).manual_label;
      }}).length;
      document.getElementById("progress").textContent =
        `Reviewed unique samples ${{reviewedUniqueCount}} / ${{uniqueIds.length}}; ` +
        `reviewed HTML entries ${{reviewedEntryCount}} / ${{rows.length}}`;
    }}

    function selectItem(i) {{
      current = Math.max(0, Math.min(rows.length - 1, i));
      const row = rows[current];
      const ann = getAnnotation(row);
      document.getElementById("frame").src = row.html;
      for (const field of ["manual_label", "looks_like_class_neighbors", "looks_like_global_neighbors", "has_duplicate", "note"]) {{
        document.getElementById(field).value = ann[field] || "";
      }}
      renderList();
    }}

    function saveCurrent() {{
      const row = rows[current];
      const ann = getAnnotation(row);
      for (const field of ["manual_label", "looks_like_class_neighbors", "looks_like_global_neighbors", "has_duplicate", "note"]) {{
        ann[field] = document.getElementById(field).value;
      }}
      annotations[row.index] = ann;
      localStorage.setItem(storageKey, JSON.stringify(annotations));
      renderList();
    }}

    function nextItem() {{ selectItem(current + 1); }}
    function prevItem() {{ selectItem(current - 1); }}

    function csvEscape(value) {{
      const text = value === undefined || value === null ? "" : String(value);
      return /[",\\n\\r]/.test(text) ? `"${{text.replaceAll('"', '""')}}"` : text;
    }}

    function exportCsv() {{
      const lines = [fields.join(",")];
      rows.forEach(row => {{
        const ann = getAnnotation(row);
        lines.push(fields.map(field => csvEscape(ann[field])).join(","));
      }});
      const blob = new Blob([lines.join("\\n")], {{type: "text/csv;charset=utf-8"}});
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "manual_qp_gate_review.csv";
      a.click();
      URL.revokeObjectURL(a.href);
    }}

    function clearAnnotations() {{
      if (!confirm("Clear saved annotations for this review page?")) {{
        return;
      }}
      annotations = {{}};
      localStorage.removeItem(storageKey);
      selectItem(current);
    }}

    renderList();
    selectItem(0);
  </script>
</body>
</html>
"""
    path.write_text(page, encoding="utf-8")


def manual_review_fieldnames() -> list[str]:
    return [
        "index",
        "path",
        "web_label",
        "selection_rule",
        "html",
        "manual_label",
        "looks_like_class_neighbors",
        "looks_like_global_neighbors",
        "has_duplicate",
        "note",
        "S_clean",
        "Q_same",
        "centroid_score",
        "loss",
        "confidence",
    ]


def select_by_metric(idx: np.ndarray, values: np.ndarray, count: int, largest: bool, rule: str) -> list[tuple[int, str]]:
    valid = idx[np.isfinite(values[idx])]
    order = np.argsort(-values[valid] if largest else values[valid], kind="mergesort")
    return [(int(i), rule) for i in valid[order[: min(count, len(valid))]]]


def select_from_high_deleted_ratio_classes(
    deleted_idx: np.ndarray,
    data: dict[str, Any],
    class_rows: list[dict[str, Any]],
    samples_per_class: int,
) -> list[tuple[int, str]]:
    selected: list[tuple[int, str]] = []
    rows = [row for row in class_rows if int(row["deleted_count"]) > 0]
    top_classes = sorted(rows, key=lambda row: (float(row["deleted_ratio"]), int(row["deleted_count"])), reverse=True)[:20]
    for row in top_classes:
        class_id = row["class_id"]
        class_deleted = deleted_idx[data["labels"][deleted_idx] == class_id]
        # Show the strongest GCDD-scored deleted samples within high-deletion classes first.
        order = np.argsort(-data["S_clean"][class_deleted], kind="mergesort")
        for i in class_deleted[order[: min(samples_per_class, len(class_deleted))]]:
            selected.append((int(i), f"high_deleted_ratio_class_{safe_name(class_id)}"))
    return selected


def write_sample_html(path: Path, data: dict[str, Any], idx: int, rule: str, asset_context: AssetContext) -> None:
    class_neighbors = [j for j in data["class_knn"][idx, :5].tolist() if j >= 0]
    global_neighbors = [j for j in data["global_knn"][idx, :5].tolist() if j >= 0]
    body = [
        "<html><body>",
        f"<h1>{int(data['index'][idx])} - {html.escape(str(data['labels'][idx]))}</h1>",
        "<ul>",
        f"<li>selection_rule: {html.escape(rule)}</li>",
        f"<li>S_clean: {data['S_clean'][idx]:.4f}</li>",
        f"<li>Q_same: {data['Q_same'][idx]:.4f}</li>",
        f"<li>centroid_score: {data['centroid_score'][idx]:.4f}</li>",
        f"<li>R_class / I_class_norm: {data['R_class'][idx]:.4f} / {data['I_class_norm'][idx]:.4f}</li>",
        f"<li>loss / confidence: {data['loss'][idx]:.4f} / {data['confidence'][idx]:.4f}</li>",
        "</ul>",
        image_block("Query", data, idx, path, asset_context),
        neighbor_block("Top-5 Class Neighbors", class_neighbors, data, path, asset_context),
        neighbor_block("Top-5 Global Neighbors", global_neighbors, data, path, asset_context),
        "</body></html>",
    ]
    path.write_text("\n".join(body), encoding="utf-8")


def image_block(title: str, data: dict[str, Any], idx: int, html_file: Path, asset_context: AssetContext) -> str:
    source = str(data["path"][idx])
    src = asset_context.html_src(source, html_file, int(data["index"][idx]))
    return f"<h2>{html.escape(title)}</h2><div><img src=\"{html.escape(src)}\" style=\"max-width:260px\"><p>{html.escape(source)}</p></div>"


def neighbor_block(title: str, indices: list[int], data: dict[str, Any], html_file: Path, asset_context: AssetContext) -> str:
    parts = [f"<h2>{html.escape(title)}</h2><div style='display:flex;gap:12px;flex-wrap:wrap'>"]
    for j in indices:
        src = asset_context.html_src(str(data["path"][j]), html_file, int(data["index"][j]))
        parts.append(
            "<div style='width:180px'>"
            f"<img src=\"{html.escape(src)}\" style=\"max-width:160px\"><br>"
            f"idx={int(data['index'][j])}<br>"
            f"{html.escape(str(data['labels'][j]))}<br>"
            f"S={data['S_clean'][j]:.3f}, Q={data['Q_same'][j]:.3f}<br>"
            f"C={data['centroid_score'][j]:.3f}, loss={data['loss'][j]:.3f}"
            "</div>"
        )
    parts.append("</div>")
    return "\n".join(parts)


def write_summary(
    path: Path,
    data: dict[str, Any],
    deleted_mask: np.ndarray,
    class_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    viz_rows: list[dict[str, Any]],
) -> None:
    metric_map = {row["group"]: row for row in metric_rows}
    deleted = metric_map["qp_deleted"]
    kept = metric_map["qp_kept_gcdd_clean"]
    top_classes = sorted([row for row in class_rows if int(row["deleted_count"]) > 0], key=lambda row: (float(row["deleted_ratio"]), int(row["deleted_count"])), reverse=True)[:20]
    lines = [
        "# QP-Gate Deleted Sample Analysis",
        "",
        f"- Deleted by qp_gate: {int(deleted_mask.sum())}",
        f"- Kept Full GCDD-clean after qp_gate: {int((data['full_gcdd_clean'] & ~deleted_mask).sum())}",
        "",
        "## Metric Comparison",
        "| metric | deleted mean | kept-GCDD mean | direction |",
        "| --- | ---: | ---: | --- |",
        metric_line("loss", deleted, kept, higher_bad=True),
        metric_line("confidence", deleted, kept, higher_bad=False),
        metric_line("Q_same", deleted, kept, higher_bad=False),
        metric_line("centroid_score", deleted, kept, higher_bad=False),
        metric_line("R_class", deleted, kept, higher_bad=False),
        metric_line("I_class_norm", deleted, kept, higher_bad=False),
        "",
        "## Deleted-Ratio Top Classes",
        "| class_id | gcdd_clean_count | deleted_count | deleted_ratio |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in top_classes:
        lines.append(f"| {row['class_id']} | {row['gcdd_clean_count']} | {row['deleted_count']} | {float(row['deleted_ratio']):.4f} |")
    lines.extend(
        [
            "",
            "## Visualization",
            f"- HTML samples generated: {len(viz_rows)}",
            "- Rules: highest_loss, lowest_confidence, high_deleted_ratio_class_*.",
            "- Use neighbor HTML to label each case as noise, clean_like, or uncertain; use neighbor-likeness fields and duplicate flags for finer diagnosis.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def metric_line(metric: str, deleted: dict[str, Any], kept: dict[str, Any], higher_bad: bool) -> str:
    deleted_mean = float(deleted[f"{metric}_mean"])
    kept_mean = float(kept[f"{metric}_mean"])
    if higher_bad:
        direction = "deleted higher" if deleted_mean > kept_mean else "deleted not higher"
    else:
        direction = "deleted lower" if deleted_mean < kept_mean else "deleted not lower"
    return f"| {metric} | {deleted_mean:.4f} | {kept_mean:.4f} | {direction} |"


def finite_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


def finite_percentile(values: np.ndarray, percentile: float) -> float:
    values = values[np.isfinite(values)]
    return float(np.percentile(values, percentile)) if len(values) else float("nan")


def safe_ratio(num: int | float, denom: int | float) -> float:
    return float(num / denom) if denom else 0.0


def parse_path_maps(items: list[str]) -> list[tuple[str, str]]:
    maps = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--path-map must use OLD=NEW format, got: {item}")
        old, new = item.split("=", 1)
        maps.append((old, new))
    return maps


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def same_file(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False


if __name__ == "__main__":
    main()
