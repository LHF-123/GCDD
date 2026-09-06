#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
DRY_RUN=0; [[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1
PYTHON_BIN="${PYTHON_BIN:-python}"; SEEDS="1,42,88"; VAL_SEED=20250726; OUT_SUMMARY="outputs/analysis/budget_matched_dynamic_dynamic_pgdf_asym40_s42_validation"
DATASETS=(
  "CUB-200-2011|outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42|outputs/CUB_200_2011/noise_indices/cub_asym_r0p4_s42_index.csv"
  "Stanford Cars|outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42|outputs/Stanford_Cars/noise_indices/stanford_cars_asym_r0p4_s42_index.csv"
  "FGVC-Aircraft|outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42|outputs/FGVC_Aircraft/noise_indices/fgvc_aircraft_asym_r0p4_s42_index.csv"
)
run() {
  echo "+ $*"
  if (( ! DRY_RUN )); then "$@"; fi
}
[[ ! -e "$OUT_SUMMARY" ]] || { echo "[ERROR] summary exists: $OUT_SUMMARY" >&2; exit 1; }
for item in "${DATASETS[@]}"; do
  IFS='|' read -r dataset input noise <<<"$item"; source="$input/checkpoint_validation_pgdf_dynamic_proto_r08_p04_s${VAL_SEED}/pgdf_dynamic_proto"; output="$input/checkpoint_validation_budget_matched_dynamic_proto_r08_p04_s${VAL_SEED}"
  [[ -f "$noise" && -f "$source/seed1/selection_per_class.csv" && -f "$source/seed42/selection_per_class.csv" && -f "$source/seed88/selection_per_class.csv" && ! -e "$output" ]] || { echo "[ERROR] incomplete source or existing destination for $dataset" >&2; exit 1; }
  echo "[RUN] experiment=budget_matched_dynamic_dynamic_pgdf dataset=$dataset seeds=$SEEDS noise=cyclic-asym40 r=0.8 p=0.4"
  run "$PYTHON_BIN" scripts/run_lora_checkpoint_validation.py --input-dir "$input" --noise-index "$noise" --output-dir "$output" --methods dynamic_budget_matched_dynamic_proto --pgdf-budget-root "$source" --seeds "$SEEDS" --validation-ratio 0.10 --validation-seed "$VAL_SEED" --dynamic-ratio 0.8 --fixed-p 0.4 --warmup-epochs 5 --update-interval 5 --official-test-selected-only --no-posthoc-oracle-test
done
run "$PYTHON_BIN" tools/audit_dynamic_proto_budget_match.py --dataset "CUB-200-2011=outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42/checkpoint_validation_pgdf_dynamic_proto_r08_p04_s${VAL_SEED}=outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42/checkpoint_validation_budget_matched_dynamic_proto_r08_p04_s${VAL_SEED}" --dataset "Stanford Cars=outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42/checkpoint_validation_pgdf_dynamic_proto_r08_p04_s${VAL_SEED}=outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42/checkpoint_validation_budget_matched_dynamic_proto_r08_p04_s${VAL_SEED}" --dataset "FGVC-Aircraft=outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42/checkpoint_validation_pgdf_dynamic_proto_r08_p04_s${VAL_SEED}=outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42/checkpoint_validation_budget_matched_dynamic_proto_r08_p04_s${VAL_SEED}" --output-dir "$OUT_SUMMARY"
