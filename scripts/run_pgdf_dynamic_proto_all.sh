#!/usr/bin/env bash
set -Eeuo pipefail

# Controlled ablation: formal PGDF dynamic small-loss plus a current
# LoRA-adapted dynamic prototype gate. Runs sequentially: 3 x 3 = 9 runs.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
VALIDATION_RATIO="0.10"
VALIDATION_SEED="20250726"
DYNAMIC_RATIO="0.8"
FIXED_P="0.4"
SEEDS="1,42,88"
WARMUP_EPOCHS="5"
UPDATE_INTERVAL="5"
SUMMARY_DIR="outputs/analysis/pgdf_dynamic_proto_asym40_s42_validation"

DATASETS=(
  "CUB-200-2011|outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42|outputs/CUB_200_2011/noise_indices/cub_asym_r0p4_s42_index.csv"
  "Stanford Cars|outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42|outputs/Stanford_Cars/noise_indices/stanford_cars_asym_r0p4_s42_index.csv"
  "FGVC-Aircraft|outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42|outputs/FGVC_Aircraft/noise_indices/fgvc_aircraft_asym_r0p4_s42_index.csv"
)

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

preflight() {
  [[ -f scripts/run_lora_checkpoint_validation.py ]] || die "Missing unified validation runner."
  [[ -f tools/summarize_dynamic_prototype_multiseed.py ]] || die "Missing Dynamic Prototype summary tool."
  [[ ! -e "$SUMMARY_DIR" ]] || die "Refusing to overwrite existing summary directory: $SUMMARY_DIR"
  for item in "${DATASETS[@]}"; do
    IFS='|' read -r name input_dir noise_index <<< "$item"
    [[ -d "$input_dir" ]] || die "$name input directory is missing: $input_dir"
    [[ -f "$noise_index" ]] || die "$name noise index is missing: $noise_index"
    [[ -f "$input_dir/checkpoint_validation_s${VALIDATION_SEED}/validation_manifest.csv" ]] || die "$name validation manifest CSV is missing."
    [[ -f "$input_dir/checkpoint_validation_s${VALIDATION_SEED}/validation_manifest.json" ]] || die "$name validation manifest JSON is missing."
    [[ ! -e "$input_dir/checkpoint_validation_pgdf_dynamic_proto_r08_p04_s${VALIDATION_SEED}" ]] || die "$name output already exists; refusing overwrite."
  done
}

preflight
echo "[PGDF-DynamicProto] fixed_clean_validation_v1; r=$DYNAMIC_RATIO; p=$FIXED_P; warmup=$WARMUP_EPOCHS; interval=$UPDATE_INTERVAL"
echo "[PGDF-DynamicProto] sequential plan: 3 datasets x seeds $SEEDS = 9 runs"

for item in "${DATASETS[@]}"; do
  IFS='|' read -r name input_dir noise_index <<< "$item"
  output_dir="$input_dir/checkpoint_validation_pgdf_dynamic_proto_r08_p04_s${VALIDATION_SEED}"
  mkdir -p "$output_dir"
  cp "$input_dir/checkpoint_validation_s${VALIDATION_SEED}/validation_manifest.csv" "$output_dir/validation_manifest.csv"
  cp "$input_dir/checkpoint_validation_s${VALIDATION_SEED}/validation_manifest.json" "$output_dir/validation_manifest.json"

  echo "[RUN] $name -> $output_dir"
  "$PYTHON_BIN" scripts/run_lora_checkpoint_validation.py \
    --input-dir "$input_dir" \
    --noise-index "$noise_index" \
    --output-dir "$output_dir" \
    --methods pgdf_dynamic_proto \
    --seeds "$SEEDS" \
    --validation-ratio "$VALIDATION_RATIO" \
    --validation-seed "$VALIDATION_SEED" \
    --dynamic-ratio "$DYNAMIC_RATIO" \
    --fixed-p "$FIXED_P" \
    --warmup-epochs "$WARMUP_EPOCHS" \
    --update-interval "$UPDATE_INTERVAL" \
    --official-test-selected-only \
    --no-posthoc-oracle-test
done

"$PYTHON_BIN" tools/summarize_dynamic_prototype_multiseed.py \
  --method-key pgdf_dynamic_proto \
  --dataset "CUB-200-2011=outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42/checkpoint_validation_pgdf_dynamic_proto_r08_p04_s${VALIDATION_SEED}" \
  --dataset "Stanford Cars=outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42/checkpoint_validation_pgdf_dynamic_proto_r08_p04_s${VALIDATION_SEED}" \
  --dataset "FGVC-Aircraft=outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42/checkpoint_validation_pgdf_dynamic_proto_r08_p04_s${VALIDATION_SEED}" \
  --seeds "$SEEDS" \
  --output-dir "$SUMMARY_DIR"

echo "[DONE] 9 PGDF-DynamicProto runs completed; summary: $SUMMARY_DIR"
