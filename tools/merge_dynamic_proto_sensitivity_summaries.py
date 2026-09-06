"""Join one-dimensional PGDF-DynamicProto sensitivity summaries without tuning."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p-summary", type=Path, required=True)
    parser.add_argument("--r-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite sensitivity report: {args.output}")
    p_rows, r_rows = read_rows(args.p_summary), read_rows(args.r_summary)
    if {float(row["p"]) for row in p_rows} != {0.4, 0.5, 0.6, 0.8}:
        raise ValueError("p sensitivity summary does not contain exactly p=0.4/0.5/0.6/0.8")
    if {float(row["r"]) for row in r_rows} != {0.7, 0.8, 0.9}:
        raise ValueError("r sensitivity summary does not contain exactly r=0.7/0.8/0.9")
    lines = [
        "PGDF-DynamicProto parameter sensitivity",
        "Descriptive one-dimensional sensitivity only; the pre-specified main configuration is r=0.8, p=0.4.",
        "",
        "p sensitivity (r=0.8):",
    ]
    for row in sorted(p_rows, key=lambda item: float(item["p"])):
        lines.append(
            "p={p}, scheduler={sched}, CUB={cub:.4f} +/- {cubstd:.4f}, "
            "Cars={cars:.4f} +/- {carsstd:.4f}, Aircraft={air:.4f} +/- {airstd:.4f}, Avg={avg:.4f}, main={main}".format(
                p=row["p"], sched=row["scheduler_estimate"], cub=float(row["cub_mean_pct"]),
                cubstd=float(row["cub_sample_std_pct"]), cars=float(row["cars_mean_pct"]),
                carsstd=float(row["cars_sample_std_pct"]), air=float(row["aircraft_mean_pct"]),
                airstd=float(row["aircraft_sample_std_pct"]), avg=float(row["avg_dataset_mean_pct"]),
                main=row["pre_specified_main_configuration"],
            )
        )
    lines.extend(["", "r sensitivity (p=0.4):"])
    for row in sorted(r_rows, key=lambda item: float(item["r"])):
        lines.append(
            "r={r}, scheduler={sched}, CUB={cub:.4f} +/- {cubstd:.4f}, "
            "Cars={cars:.4f} +/- {carsstd:.4f}, Aircraft={air:.4f} +/- {airstd:.4f}, Avg={avg:.4f}, main={main}".format(
                r=row["r"], sched=row["scheduler_estimate"], cub=float(row["cub_mean_pct"]),
                cubstd=float(row["cub_sample_std_pct"]), cars=float(row["cars_mean_pct"]),
                carsstd=float(row["cars_sample_std_pct"]), air=float(row["aircraft_mean_pct"]),
                airstd=float(row["aircraft_sample_std_pct"]), avg=float(row["avg_dataset_mean_pct"]),
                main=row["pre_specified_main_configuration"],
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote combined sensitivity summary: {args.output}")


if __name__ == "__main__":
    main()
