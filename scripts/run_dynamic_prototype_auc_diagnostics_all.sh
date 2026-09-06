#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# This is an exact-artifact, post-hoc diagnostic.  It reads the dynamic
# prototype scores saved at each selection update and performs no training.
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="outputs/analysis/dynamic_prototype_auc_asym40_s42_validation"
VAL_SEED=20250726

ROOT_CUB="outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42/checkpoint_validation_pgdf_dynamic_proto_r08_p04_s${VAL_SEED}"
ROOT_CARS="outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42/checkpoint_validation_pgdf_dynamic_proto_r08_p04_s${VAL_SEED}"
ROOT_AIR="outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42/checkpoint_validation_pgdf_dynamic_proto_r08_p04_s${VAL_SEED}"

for required in "$ROOT_CUB/pgdf_dynamic_proto/seed1/selection_rows.csv" "$ROOT_CARS/pgdf_dynamic_proto/seed1/selection_rows.csv" "$ROOT_AIR/pgdf_dynamic_proto/seed1/selection_rows.csv"; do
  [[ -f "$required" ]] || { echo "[ERROR] missing exact dynamic selection artifact: $required" >&2; exit 1; }
done
[[ ! -e "$OUT_DIR" ]] || { echo "[ERROR] destination exists: $OUT_DIR" >&2; exit 1; }

command=("$PYTHON_BIN" tools/dynamic_proto_auc_diagnostics.py --output-dir "$OUT_DIR"
  --dataset "CUB-200-2011=$ROOT_CUB=outputs/CUB_200_2011/noise_indices/cub_asym_r0p4_s42_index.csv"
  --dataset "Stanford Cars=$ROOT_CARS=outputs/Stanford_Cars/noise_indices/stanford_cars_asym_r0p4_s42_index.csv"
  --dataset "FGVC-Aircraft=$ROOT_AIR=outputs/FGVC_Aircraft/noise_indices/fgvc_aircraft_asym_r0p4_s42_index.csv")
echo "+ ${command[*]}"
if (( ! DRY_RUN )); then "${command[@]}"; fi
