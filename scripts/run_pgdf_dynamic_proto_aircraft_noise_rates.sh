#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1
PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS="1,42,88"
VAL_SEED=20250726
OUT_SUMMARY="outputs/analysis/pgdf_dynamic_proto_aircraft_noise_rates"
MAIN_ASYM40="outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42/checkpoint_validation_pgdf_dynamic_proto_r08_p04_s${VAL_SEED}"

SPECS=(
  "0.2|outputs/FGVC_Aircraft/FGVC-Aircraft-asym20-s42/v1_aircraft_asym_r0p2_s42|outputs/FGVC_Aircraft/noise_indices/fgvc_aircraft_asym_r0p2_s42_index.csv|checkpoint_validation_pgdf_dynamic_proto_r08_p04_s${VAL_SEED}"
  "0.6|outputs/FGVC_Aircraft/FGVC-Aircraft-asym60-s42/v1_aircraft_asym_r0p6_s42|outputs/FGVC_Aircraft/noise_indices/fgvc_aircraft_asym_r0p6_s42_index.csv|checkpoint_validation_pgdf_dynamic_proto_r08_p04_s${VAL_SEED}"
)

run() {
  echo "+ $*"
  if (( ! DRY_RUN )); then "$@"; fi
}

[[ -f "$MAIN_ASYM40/checkpoint_validation_results.csv" && -f "$MAIN_ASYM40/run_index.csv" && ! -e "$OUT_SUMMARY" ]] || {
  echo "[ERROR] main asym40 PGDF-DynamicProto result missing or destination exists" >&2; exit 1;
}
for spec in "${SPECS[@]}"; do
  IFS='|' read -r rate input noise name <<<"$spec"; output="$input/$name"
  [[ -f "$noise" && -f "$input/checkpoint_validation_s${VAL_SEED}/validation_manifest.json" && ! -e "$output" ]] || {
    echo "[ERROR] invalid inputs or existing destination for Aircraft asym${rate}" >&2; exit 1;
  }
  echo "[RUN] experiment=pgdf_dynamic_proto_aircraft_noise_rates dataset=FGVC-Aircraft seeds=$SEEDS noise=cyclic-asym${rate#0.}0 r=0.8 p=0.4"
  run "$PYTHON_BIN" scripts/run_lora_checkpoint_validation.py \
    --input-dir "$input" --noise-index "$noise" --output-dir "$output" --methods pgdf_dynamic_proto \
    --seeds "$SEEDS" --validation-ratio 0.10 --validation-seed "$VAL_SEED" --dynamic-ratio 0.8 --fixed-p 0.4 \
    --warmup-epochs 5 --update-interval 5 --official-test-selected-only --no-posthoc-oracle-test
done
run "$PYTHON_BIN" tools/summarize_dynamic_proto_variants.py --mode noise_rates --output-dir "$OUT_SUMMARY" \
  --entry "0.2|FGVC-Aircraft|outputs/FGVC_Aircraft/FGVC-Aircraft-asym20-s42/v1_aircraft_asym_r0p2_s42/checkpoint_validation_pgdf_dynamic_proto_r08_p04_s${VAL_SEED}" \
  --entry "0.4|FGVC-Aircraft|$MAIN_ASYM40" \
  --entry "0.6|FGVC-Aircraft|outputs/FGVC_Aircraft/FGVC-Aircraft-asym60-s42/v1_aircraft_asym_r0p6_s42/checkpoint_validation_pgdf_dynamic_proto_r08_p04_s${VAL_SEED}"
