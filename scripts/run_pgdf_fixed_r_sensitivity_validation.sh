#!/usr/bin/env bash
set -Eeuo pipefail

# Run the fixed-p PGDF retention-ratio sensitivity grid under
# fixed_clean_validation_v1.  The six jobs run sequentially in one shell/GPU
# instance: 3 datasets x {r=0.7, r=0.9}, with seeds 1/42/88 per job.
#
# Usage from the repository root:
#   bash scripts/run_pgdf_fixed_r_sensitivity_validation.sh
#
# Background usage:
#   mkdir -p logs
#   nohup bash scripts/run_pgdf_fixed_r_sensitivity_validation.sh \
#     > logs/pgdf_fixed_r_sensitivity_p04.log 2>&1 &
#
# Print commands without training:
#   DRY_RUN=1 bash scripts/run_pgdf_fixed_r_sensitivity_validation.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"
VALIDATION_RATIO="0.10"
VALIDATION_SEED="20250726"
FIXED_P="0.4"
SEEDS="1,42,88"
WARMUP_EPOCHS="5"
UPDATE_INTERVAL="5"

DATASETS=(
  "CUB|outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42|outputs/CUB_200_2011/noise_indices/cub_asym_r0p4_s42_index.csv"
  "Stanford-Cars|outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42|outputs/Stanford_Cars/noise_indices/stanford_cars_asym_r0p4_s42_index.csv"
  "FGVC-Aircraft|outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42|outputs/FGVC_Aircraft/noise_indices/fgvc_aircraft_asym_r0p4_s42_index.csv"
)

RATIOS=(
  "0.7|r07"
  "0.9|r09"
)

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

print_command() {
  printf ' %q' "$@"
  printf '\n'
}

preflight() {
  [[ -f scripts/run_lora_checkpoint_validation.py ]] || \
    die "Run this script from a checkout containing scripts/run_lora_checkpoint_validation.py."

  for dataset_item in "${DATASETS[@]}"; do
    IFS="|" read -r dataset_name input_dir noise_index <<< "$dataset_item"
    [[ -d "$input_dir" ]] || die "$dataset_name input directory is missing: $input_dir"
    [[ -f "$noise_index" ]] || die "$dataset_name noise index is missing: $noise_index"

    for ratio_item in "${RATIOS[@]}"; do
      IFS="|" read -r _ ratio_tag <<< "$ratio_item"
      output_dir="$input_dir/checkpoint_validation_pgdf_fixed_${ratio_tag}_p04_s${VALIDATION_SEED}"
      [[ ! -e "$output_dir" ]] || \
        die "Refusing to overwrite existing output: $output_dir"
    done
  done
}

if [[ "$DRY_RUN" != "1" ]]; then
  preflight
fi

echo "[PGDF r-sensitivity] protocol=fixed_clean_validation_v1 p=$FIXED_P seeds=$SEEDS"
echo "[PGDF r-sensitivity] grid=3 datasets x 2 ratios x 3 seeds = 18 runs"

job_index=0
total_jobs=$((${#DATASETS[@]} * ${#RATIOS[@]}))
for dataset_item in "${DATASETS[@]}"; do
  IFS="|" read -r dataset_name input_dir noise_index <<< "$dataset_item"

  for ratio_item in "${RATIOS[@]}"; do
    IFS="|" read -r retention_ratio ratio_tag <<< "$ratio_item"
    job_index=$((job_index + 1))
    output_dir="$input_dir/checkpoint_validation_pgdf_fixed_${ratio_tag}_p04_s${VALIDATION_SEED}"

    command=(
      "$PYTHON_BIN" scripts/run_lora_checkpoint_validation.py
      --input-dir "$input_dir"
      --noise-index "$noise_index"
      --output-dir "$output_dir"
      --methods pgdf_fixed
      --seeds "$SEEDS"
      --validation-ratio "$VALIDATION_RATIO"
      --validation-seed "$VALIDATION_SEED"
      --dynamic-ratio "$retention_ratio"
      --fixed-p "$FIXED_P"
      --warmup-epochs "$WARMUP_EPOCHS"
      --update-interval "$UPDATE_INTERVAL"
      --no-posthoc-oracle-test
    )

    echo
    echo "[$job_index/$total_jobs] dataset=$dataset_name r=$retention_ratio p=$FIXED_P"
    echo "output=$output_dir"
    print_command "${command[@]}"

    if [[ "$DRY_RUN" != "1" ]]; then
      "${command[@]}"
    fi
  done
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo
  echo "[DRY RUN] Commands were printed; no training was started."
else
  echo
  echo "[DONE] All 18 runs completed. Use validation_selected_test_top1 from each checkpoint_validation_results.csv."
fi
