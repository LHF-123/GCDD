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
SUMMARY_ROOT="outputs/analysis/pgdf_dynamic_proto_parameter_sensitivity"
DATASETS=(
  "CUB-200-2011|outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42|outputs/CUB_200_2011/noise_indices/cub_asym_r0p4_s42_index.csv"
  "Stanford Cars|outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42|outputs/Stanford_Cars/noise_indices/stanford_cars_asym_r0p4_s42_index.csv"
  "FGVC-Aircraft|outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42|outputs/FGVC_Aircraft/noise_indices/fgvc_aircraft_asym_r0p4_s42_index.csv"
)

run() {
  echo "+ $*"
  if (( ! DRY_RUN )); then "$@"; fi
}

[[ ! -e "$SUMMARY_ROOT" ]] || { echo "[ERROR] destination exists: $SUMMARY_ROOT" >&2; exit 1; }
for item in "${DATASETS[@]}"; do
  IFS='|' read -r dataset input noise <<<"$item"
  main="$input/checkpoint_validation_pgdf_dynamic_proto_r08_p04_s${VAL_SEED}/run_index.csv"
  [[ -f "$noise" && -f "$main" ]] || { echo "[ERROR] main PGDF-DynamicProto result missing for $dataset" >&2; exit 1; }
  [[ "$(grep -c ',complete' "$main")" -eq 3 ]] || { echo "[ERROR] main PGDF-DynamicProto is incomplete for $dataset" >&2; exit 1; }
done

# One-dimensional p sensitivity: r remains 0.8.  The existing p=0.4 main
# configuration is deliberately reused and never retrained.
for p in 0.5 0.6 0.8; do
  p_tag="${p/./}"
  for item in "${DATASETS[@]}"; do
    IFS='|' read -r dataset input noise <<<"$item"; output="$input/checkpoint_validation_pgdf_dynamic_proto_r08_p${p_tag}_s${VAL_SEED}"
    [[ ! -e "$output" ]] || { echo "[ERROR] destination exists: $output" >&2; exit 1; }
    echo "[RUN] experiment=pgdf_dynamic_proto_p_sensitivity dataset=$dataset seeds=$SEEDS noise=cyclic-asym40 r=0.8 p=$p"
    run "$PYTHON_BIN" scripts/run_lora_checkpoint_validation.py --input-dir "$input" --noise-index "$noise" --output-dir "$output" \
      --methods pgdf_dynamic_proto --seeds "$SEEDS" --validation-ratio 0.10 --validation-seed "$VAL_SEED" \
      --dynamic-ratio 0.8 --fixed-p "$p" --warmup-epochs 5 --update-interval 5 --official-test-selected-only --no-posthoc-oracle-test
  done
done

# One-dimensional r sensitivity: p remains 0.4.  The existing r=0.8 main
# configuration is deliberately reused and never retrained.
for r in 0.7 0.9; do
  r_tag="${r/./}"
  for item in "${DATASETS[@]}"; do
    IFS='|' read -r dataset input noise <<<"$item"; output="$input/checkpoint_validation_pgdf_dynamic_proto_r${r_tag}_p04_s${VAL_SEED}"
    [[ ! -e "$output" ]] || { echo "[ERROR] destination exists: $output" >&2; exit 1; }
    echo "[RUN] experiment=pgdf_dynamic_proto_r_sensitivity dataset=$dataset seeds=$SEEDS noise=cyclic-asym40 r=$r p=0.4"
    run "$PYTHON_BIN" scripts/run_lora_checkpoint_validation.py --input-dir "$input" --noise-index "$noise" --output-dir "$output" \
      --methods pgdf_dynamic_proto --seeds "$SEEDS" --validation-ratio 0.10 --validation-seed "$VAL_SEED" \
      --dynamic-ratio "$r" --fixed-p 0.4 --warmup-epochs 5 --update-interval 5 --official-test-selected-only --no-posthoc-oracle-test
  done
done

run "$PYTHON_BIN" tools/summarize_dynamic_proto_variants.py --mode p_sensitivity --output-dir "$SUMMARY_ROOT/p" \
  --entry "0.4|CUB-200-2011|outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42/checkpoint_validation_pgdf_dynamic_proto_r08_p04_s${VAL_SEED}" \
  --entry "0.4|Stanford Cars|outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42/checkpoint_validation_pgdf_dynamic_proto_r08_p04_s${VAL_SEED}" \
  --entry "0.4|FGVC-Aircraft|outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42/checkpoint_validation_pgdf_dynamic_proto_r08_p04_s${VAL_SEED}" \
  --entry "0.5|CUB-200-2011|outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42/checkpoint_validation_pgdf_dynamic_proto_r08_p05_s${VAL_SEED}" \
  --entry "0.5|Stanford Cars|outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42/checkpoint_validation_pgdf_dynamic_proto_r08_p05_s${VAL_SEED}" \
  --entry "0.5|FGVC-Aircraft|outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42/checkpoint_validation_pgdf_dynamic_proto_r08_p05_s${VAL_SEED}" \
  --entry "0.6|CUB-200-2011|outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42/checkpoint_validation_pgdf_dynamic_proto_r08_p06_s${VAL_SEED}" \
  --entry "0.6|Stanford Cars|outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42/checkpoint_validation_pgdf_dynamic_proto_r08_p06_s${VAL_SEED}" \
  --entry "0.6|FGVC-Aircraft|outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42/checkpoint_validation_pgdf_dynamic_proto_r08_p06_s${VAL_SEED}" \
  --entry "0.8|CUB-200-2011|outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42/checkpoint_validation_pgdf_dynamic_proto_r08_p08_s${VAL_SEED}" \
  --entry "0.8|Stanford Cars|outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42/checkpoint_validation_pgdf_dynamic_proto_r08_p08_s${VAL_SEED}" \
  --entry "0.8|FGVC-Aircraft|outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42/checkpoint_validation_pgdf_dynamic_proto_r08_p08_s${VAL_SEED}"
run "$PYTHON_BIN" tools/summarize_dynamic_proto_variants.py --mode r_sensitivity --output-dir "$SUMMARY_ROOT/r" \
  --entry "0.7|CUB-200-2011|outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42/checkpoint_validation_pgdf_dynamic_proto_r07_p04_s${VAL_SEED}" \
  --entry "0.7|Stanford Cars|outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42/checkpoint_validation_pgdf_dynamic_proto_r07_p04_s${VAL_SEED}" \
  --entry "0.7|FGVC-Aircraft|outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42/checkpoint_validation_pgdf_dynamic_proto_r07_p04_s${VAL_SEED}" \
  --entry "0.8|CUB-200-2011|outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42/checkpoint_validation_pgdf_dynamic_proto_r08_p04_s${VAL_SEED}" \
  --entry "0.8|Stanford Cars|outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42/checkpoint_validation_pgdf_dynamic_proto_r08_p04_s${VAL_SEED}" \
  --entry "0.8|FGVC-Aircraft|outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42/checkpoint_validation_pgdf_dynamic_proto_r08_p04_s${VAL_SEED}" \
  --entry "0.9|CUB-200-2011|outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42/checkpoint_validation_pgdf_dynamic_proto_r09_p04_s${VAL_SEED}" \
  --entry "0.9|Stanford Cars|outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42/checkpoint_validation_pgdf_dynamic_proto_r09_p04_s${VAL_SEED}" \
  --entry "0.9|FGVC-Aircraft|outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42/checkpoint_validation_pgdf_dynamic_proto_r09_p04_s${VAL_SEED}"
run "$PYTHON_BIN" tools/merge_dynamic_proto_sensitivity_summaries.py --p-summary "$SUMMARY_ROOT/p/p_sensitivity_summary.csv" --r-summary "$SUMMARY_ROOT/r/r_sensitivity_summary.csv" --output "$SUMMARY_ROOT/parameter_sensitivity_summary.txt"
