#!/usr/bin/env bash
set -euo pipefail

CURRENT_DATASET="尚未进入数据集循环"

report_error() {
  local exit_code=$1
  local line_number=$2
  local failed_command=$3
  trap - ERR
  echo "[ERROR] Prototype gate only 实验失败。" >&2
  echo "[ERROR] 当前数据集：$CURRENT_DATASET" >&2
  echo "[ERROR] 退出码：$exit_code；行号：$line_number；命令：$failed_command" >&2
}

require_file() {
  local path=$1
  local description=$2
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] 缺少${description}：$path" >&2
    echo "[ERROR] 请确认 outputs 已同步到 /root/GCDD，或修正脚本中的数据路径。" >&2
    exit 1
  fi
}

trap 'report_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[INFO] 仓库目录：$REPO_ROOT"
echo "[INFO] 将顺序运行 3 个数据集 × 3 个 seed，共 9 个 run。"

DATASETS=(
  "outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42|outputs/CUB_200_2011/noise_indices/cub_asym_r0p4_s42_index.csv"
  "outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42|outputs/Stanford_Cars/noise_indices/stanford_cars_asym_r0p4_s42_index.csv"
  "outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42|outputs/FGVC_Aircraft/noise_indices/fgvc_aircraft_asym_r0p4_s42_index.csv"
)

for ITEM in "${DATASETS[@]}"; do
  IFS='|' read -r INPUT_DIR NOISE_INDEX <<< "$ITEM"
  CURRENT_DATASET="$INPUT_DIR"

  MANIFEST_DIR="$INPUT_DIR/checkpoint_validation_s20250726"
  OUTPUT_DIR="$INPUT_DIR/checkpoint_validation_proto_only_p04_s20250726"

  echo "[INFO] 检查数据集：$INPUT_DIR"
  require_file "$NOISE_INDEX" "noise index"
  require_file "$MANIFEST_DIR/validation_manifest.csv" "validation manifest CSV"
  require_file "$MANIFEST_DIR/validation_manifest.json" "validation manifest JSON"

  if [[ -e "$OUTPUT_DIR" ]]; then
    echo "[ERROR] 拒绝覆盖已有目录：$OUTPUT_DIR" >&2
    echo "[ERROR] 如该目录来自失败的空 run，请先人工核对内容，再改名或移走后重试。" >&2
    exit 1
  fi

  echo "[INFO] 创建新输出目录：$OUTPUT_DIR"
  mkdir -p "$OUTPUT_DIR"
  cp "$MANIFEST_DIR/validation_manifest.csv" "$OUTPUT_DIR/validation_manifest.csv"
  cp "$MANIFEST_DIR/validation_manifest.json" "$OUTPUT_DIR/validation_manifest.json"

  echo "[INFO] 启动 proto_only p=0.4，seeds=1,42,88"
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

  echo "[INFO] 数据集完成：$INPUT_DIR"
done

echo "[INFO] 全部 9 个 run 已完成。"
