#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
DRY_RUN=0; [[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1
PYTHON_BIN="${PYTHON_BIN:-python}"; SEEDS="1,42,88"; VAL_SEED=20250726
OUTPUT_ROOT="outputs_major_revision/pgdf_exact_budget_ablation"
OUT_SUMMARY="$OUTPUT_ROOT/analysis"
# Default to the complete three-control matrix because this worktree has no
# verified historical Small-loss-only results.  A server may override this to
# the two new methods only after independently verifying compliant L-only
# outputs; the controls remain in a fresh, isolated output root either way.
CONTROL_METHODS="${CONTROL_METHODS:-dynamic_budget_matched_dynamic_proto,pgdf_exact_budget_proto,pgdf_exact_budget_random}"
IFS=',' read -r -a CONTROL_METHOD_ARRAY <<<"$CONTROL_METHODS"
AUDIT_METHOD_ARGS=()
for method in "${CONTROL_METHOD_ARRAY[@]}"; do
  [[ -n "$method" ]] || { echo "[ERROR] empty method key in CONTROL_METHODS" >&2; exit 1; }
  AUDIT_METHOD_ARGS+=(--method "$method")
done
DATASETS=(
  "CUB-200-2011|outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42|outputs/CUB_200_2011/noise_indices/cub_asym_r0p4_s42_index.csv|outputs_major_revision/pgdf_dynamic_ref_cub40/pgdf_dynamic_proto|$OUTPUT_ROOT/cub40"
  "Stanford Cars|outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42|outputs/Stanford_Cars/noise_indices/stanford_cars_asym_r0p4_s42_index.csv|outputs_major_revision/pgdf_dynamic_ref_cars40/pgdf_dynamic_proto|$OUTPUT_ROOT/cars40"
  "FGVC-Aircraft|outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42|outputs/FGVC_Aircraft/noise_indices/fgvc_aircraft_asym_r0p4_s42_index.csv|outputs_major_revision/pgdf_dynamic_ref_aircraft40/pgdf_dynamic_proto|$OUTPUT_ROOT/aircraft40"
)
run() {
  echo "+ $*"
  if (( ! DRY_RUN )); then "$@"; fi
}
[[ ! -e "$OUT_SUMMARY" ]] || { echo "[ERROR] summary exists: $OUT_SUMMARY" >&2; exit 1; }
for item in "${DATASETS[@]}"; do
  IFS='|' read -r dataset input noise source output <<<"$item"
  [[ -d "$input" && -f "$input/paths.txt" && -f "$input/labels.npy" && -f "$input/resolved_config.yaml" && -f "$noise" && -f "$source/validation_manifest.json" && -f "$source/resolved_config.yaml" && -f "$source/seed1/selection_per_class.csv" && -f "$source/seed1/selection_updates.csv" && -f "$source/seed42/selection_per_class.csv" && -f "$source/seed42/selection_updates.csv" && -f "$source/seed88/selection_per_class.csv" && -f "$source/seed88/selection_updates.csv" && ! -e "$output" ]] || { echo "[ERROR] incomplete V1/source budget asset or existing destination for $dataset" >&2; exit 1; }
  echo "[RUN] experiment=dynamic_pgdf_actual_budget_controls dataset=$dataset seeds=$SEEDS noise=cyclic-asym40 r=0.8 p=0.4"
  run "$PYTHON_BIN" scripts/run_lora_checkpoint_validation.py --input-dir "$input" --noise-index "$noise" --output-dir "$output" --methods "$CONTROL_METHODS" --pgdf-budget-root "$source" --seeds "$SEEDS" --validation-ratio 0.10 --validation-seed "$VAL_SEED" --dynamic-ratio 0.8 --fixed-p 0.4 --warmup-epochs 5 --update-interval 5 --official-test-selected-only --no-posthoc-oracle-test
done
run "$PYTHON_BIN" tools/audit_dynamic_proto_budget_match.py "${AUDIT_METHOD_ARGS[@]}" --dataset "CUB-200-2011=outputs_major_revision/pgdf_dynamic_ref_cub40=outputs_major_revision/pgdf_exact_budget_ablation/cub40" --dataset "Stanford Cars=outputs_major_revision/pgdf_dynamic_ref_cars40=outputs_major_revision/pgdf_exact_budget_ablation/cars40" --dataset "FGVC-Aircraft=outputs_major_revision/pgdf_dynamic_ref_aircraft40=outputs_major_revision/pgdf_exact_budget_ablation/aircraft40" --output-dir "$OUT_SUMMARY"
