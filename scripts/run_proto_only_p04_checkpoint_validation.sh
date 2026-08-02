#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASETS=(
  "outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42|outputs/CUB_200_2011/noise_indices/cub_asym_r0p4_s42_index.csv"
  "outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42|outputs/Stanford_Cars/noise_indices/stanford_cars_asym_r0p4_s42_index.csv"
  "outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42|outputs/FGVC_Aircraft/noise_indices/fgvc_aircraft_asym_r0p4_s42_index.csv"
)

for ITEM in "${DATASETS[@]}"; do
  IFS='|' read -r INPUT_DIR NOISE_INDEX <<< "$ITEM"

  MANIFEST_DIR="$INPUT_DIR/checkpoint_validation_s20250726"
  OUTPUT_DIR="$INPUT_DIR/checkpoint_validation_proto_only_p04_s20250726"

  test -f "$NOISE_INDEX"
  test -f "$MANIFEST_DIR/validation_manifest.csv"
  test -f "$MANIFEST_DIR/validation_manifest.json"

  if [[ -e "$OUTPUT_DIR" ]]; then
    echo "拒绝覆盖已有目录：$OUTPUT_DIR" >&2
    exit 1
  fi

  mkdir -p "$OUTPUT_DIR"
  cp "$MANIFEST_DIR/validation_manifest.csv" "$OUTPUT_DIR/validation_manifest.csv"
  cp "$MANIFEST_DIR/validation_manifest.json" "$OUTPUT_DIR/validation_manifest.json"

  python scripts/run_lora_checkpoint_validation.py \
    --input-dir "$INPUT_DIR" \
    --noise-index "$NOISE_INDEX" \
    --output-dir "$OUTPUT_DIR" \
    --methods proto_only \
    --seeds 1,42,88 \
    --validation-ratio 0.10 \
    --validation-seed 20250726 \
    --fixed-p 0.4 \
    --warmup-epochs 5 \
    --update-interval 5 \
    --official-test-selected-only \
    --no-posthoc-oracle-test
done
