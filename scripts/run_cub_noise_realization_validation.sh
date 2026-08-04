#!/usr/bin/env bash
set -Eeuo pipefail

CURRENT_STAGE="初始化"

report_error() {
  local exit_code=$1
  local line_number=$2
  local failed_command=$3
  trap - ERR
  echo "[ERROR] CUB noise-realization sensitivity 运行失败。" >&2
  echo "[ERROR] 阶段：${CURRENT_STAGE}" >&2
  echo "[ERROR] 退出码：${exit_code}；行号：${line_number}" >&2
  echo "[ERROR] 失败命令：${failed_command}" >&2
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

require_file() {
  local path=$1
  [[ -f "$path" ]] || die "缺少文件：${path}"
}

trap 'report_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

if [[ $# -ne 1 ]]; then
  echo "用法：bash scripts/run_cub_noise_realization_validation.sh <noise-seed>" >&2
  echo "示例：bash scripts/run_cub_noise_realization_validation.sh 17" >&2
  exit 2
fi

NOISE_SEED=$1
if [[ ! "$NOISE_SEED" =~ ^[0-9]+$ ]]; then
  die "noise seed 必须是非负整数；收到：${NOISE_SEED}"
fi
# Normalize leading zeros so e.g. 042 is the same realization as 42.
NOISE_SEED=$((10#$NOISE_SEED))
REFERENCE_NOISE_SEEDS=(17 42 73)
NOISE_SEEDS=("${REFERENCE_NOISE_SEEDS[@]}")
IS_REFERENCE_SEED=no
for REFERENCE_SEED in "${REFERENCE_NOISE_SEEDS[@]}"; do
  if [[ "$NOISE_SEED" == "$REFERENCE_SEED" ]]; then
    IS_REFERENCE_SEED=yes
    break
  fi
done
if [[ "$IS_REFERENCE_SEED" == no ]]; then
  NOISE_SEEDS+=("$NOISE_SEED")
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BASE_INPUT="outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42"
BASE_VALIDATION_DIR="$BASE_INPUT/checkpoint_validation_s20250726"
NOISE_INDEX_DIR="outputs/CUB_200_2011/noise_indices"
REALIZATION_ROOT="outputs/CUB_200_2011/noise_realization_sensitivity/cub_asym_r0p4_s${NOISE_SEED}"
INPUT_DIR="$REALIZATION_ROOT/v1_cub_asym_r0p4_s${NOISE_SEED}"
BASE_METHODS_DIR="$REALIZATION_ROOT/fixed_clean_validation_v1_s20250726/base_methods"
MATCHED_DIR="$REALIZATION_ROOT/fixed_clean_validation_v1_s20250726/budget_matched"

[[ ! -e "$REALIZATION_ROOT" ]] || die "拒绝覆盖已有 realization 目录：${REALIZATION_ROOT}"
require_file "$BASE_INPUT/paths.txt"
require_file "$BASE_INPUT/eval_paths.txt"
require_file "$BASE_INPUT/eval_labels.npy"
require_file "$BASE_INPUT/features_cls.npy"
require_file "$BASE_INPUT/features_gap.npy"
require_file "$BASE_INPUT/features_top.npy"
require_file "$BASE_INPUT/resolved_config.yaml"

BASE_MANIFEST_CSV="$BASE_VALIDATION_DIR/validation_manifest.csv"
BASE_MANIFEST_JSON="$BASE_VALIDATION_DIR/validation_manifest.json"
if [[ -f "$BASE_MANIFEST_CSV" && -f "$BASE_MANIFEST_JSON" ]]; then
  VALIDATION_SOURCE_DIR="$BASE_VALIDATION_DIR"
  CREATE_VALIDATION_ARGS=()
elif [[ ! -e "$BASE_MANIFEST_CSV" && ! -e "$BASE_MANIFEST_JSON" ]]; then
  VALIDATION_SOURCE_DIR="$REALIZATION_ROOT/fixed_validation_manifest_source"
  CREATE_VALIDATION_ARGS=(--create-validation-if-missing)
else
  die "旧 validation manifest 仅存在 CSV/JSON 之一；请先人工核对：${BASE_VALIDATION_DIR}"
fi

if [[ -n "${CUB_ROOT:-}" ]]; then
  RESOLVED_CUB_ROOT="$CUB_ROOT"
elif [[ -f "/root/autodl-tmp/CUB_200_2011/images.txt" ]]; then
  RESOLVED_CUB_ROOT="/root/autodl-tmp/CUB_200_2011"
elif [[ -f "dataset/CUB_200_2011/CUB_200_2011/images.txt" ]]; then
  RESOLVED_CUB_ROOT="dataset/CUB_200_2011/CUB_200_2011"
else
  die "未找到 CUB metadata；请设置 CUB_ROOT，例如 CUB_ROOT=/root/autodl-tmp/CUB_200_2011"
fi

mkdir -p "$REALIZATION_ROOT"
exec > >(tee -a "$REALIZATION_ROOT/launcher.log") 2>&1

echo "[INFO] 仓库目录：${REPO_ROOT}"
echo "[INFO] noise seed：${NOISE_SEED}；training seed：42；validation seed：20250726"
echo "[INFO] 本实例将顺序训练 4 个 run：Dynamic r=0.8、JAL-CE、PGDF r=0.8/p=0.4、Budget-matched Dynamic。"
echo "[INFO] 用于交叉审计的 noise index：${NOISE_SEEDS[*]}；仅训练当前 noise seed。"
if [[ ${#CREATE_VALIDATION_ARGS[@]} -eq 0 ]]; then
  echo "[INFO] 复用正式 validation manifest：${VALIDATION_SOURCE_DIR}"
else
  echo "[INFO] 未发现旧 manifest；将在新 realization 目录中确定性重建 fixed_clean_validation_v1。"
fi

CURRENT_STAGE="生成并核对当前及参考 noise index"
mkdir -p "$NOISE_INDEX_DIR"
for SEED in "${NOISE_SEEDS[@]}"; do
  PREFIX="cub_asym_r0p4_s${SEED}"
  EXPECTED=(
    "$NOISE_INDEX_DIR/${PREFIX}_index.csv"
    "$NOISE_INDEX_DIR/${PREFIX}_mapping.csv"
    "$NOISE_INDEX_DIR/${PREFIX}_summary.csv"
    "$NOISE_INDEX_DIR/${PREFIX}_resolved_config.yaml"
  )
  PRESENT=0
  for FILE in "${EXPECTED[@]}"; do
    [[ -e "$FILE" ]] && PRESENT=$((PRESENT + 1))
  done
  if [[ $PRESENT -eq 0 ]]; then
    echo "[INFO] 生成 noise seed ${SEED} 的 cyclic-asym40 index。"
    python tools/build_cub_asym_noise_index.py \
      --cub-root "$RESOLVED_CUB_ROOT" \
      --noise-ratio 0.4 \
      --seed "$SEED" \
      --output-dir "$NOISE_INDEX_DIR" \
      --prefix "$PREFIX"
  elif [[ $PRESENT -eq ${#EXPECTED[@]} ]]; then
    echo "[INFO] 复用已有 noise seed ${SEED} index。"
  else
    die "noise seed ${SEED} 的产物不完整；为避免覆盖，请先人工核对 ${NOISE_INDEX_DIR}/${PREFIX}_*"
  fi
done

NOISE_INDEX="$NOISE_INDEX_DIR/cub_asym_r0p4_s${NOISE_SEED}_index.csv"
PEER_ARGS=()
for PEER_SEED in "${NOISE_SEEDS[@]}"; do
  if [[ "$PEER_SEED" != "$NOISE_SEED" ]]; then
    PEER_ARGS+=(--peer-noise-index "$NOISE_INDEX_DIR/cub_asym_r0p4_s${PEER_SEED}_index.csv")
  fi
done

CURRENT_STAGE="构造并审计 realization 输入"
python tools/prepare_cub_noise_realization.py \
  --base-input-dir "$BASE_INPUT" \
  --noise-index "$NOISE_INDEX" \
  --validation-dir "$VALIDATION_SOURCE_DIR" \
  --output-dir "$INPUT_DIR" \
  --noise-seed "$NOISE_SEED" \
  --validation-seed 20250726 \
  "${CREATE_VALIDATION_ARGS[@]}" \
  "${PEER_ARGS[@]}"

CURRENT_STAGE="运行 Dynamic、JAL-CE、PGDF"
mkdir -p "$BASE_METHODS_DIR"
cp "$INPUT_DIR/validation_manifest.csv" "$BASE_METHODS_DIR/validation_manifest.csv"
cp "$INPUT_DIR/validation_manifest.json" "$BASE_METHODS_DIR/validation_manifest.json"

python scripts/run_lora_checkpoint_validation.py \
  --input-dir "$INPUT_DIR" \
  --noise-index "$NOISE_INDEX" \
  --output-dir "$BASE_METHODS_DIR" \
  --methods dynamic_r08,jal_ce,pgdf_fixed \
  --seeds 42 \
  --validation-ratio 0.10 \
  --validation-seed 20250726 \
  --dynamic-ratio 0.8 \
  --fixed-p 0.4 \
  --warmup-epochs 5 \
  --update-interval 5 \
  --official-test-selected-only \
  --no-posthoc-oracle-test

CURRENT_STAGE="运行 Budget-matched Dynamic"
require_file "$BASE_METHODS_DIR/pgdf_fixed/seed42/selection_per_class.csv"
mkdir -p "$MATCHED_DIR"
cp "$INPUT_DIR/validation_manifest.csv" "$MATCHED_DIR/validation_manifest.csv"
cp "$INPUT_DIR/validation_manifest.json" "$MATCHED_DIR/validation_manifest.json"

python scripts/run_lora_checkpoint_validation.py \
  --input-dir "$INPUT_DIR" \
  --noise-index "$NOISE_INDEX" \
  --output-dir "$MATCHED_DIR" \
  --methods dynamic_budget_matched \
  --seeds 42 \
  --pgdf-budget-root "$BASE_METHODS_DIR/pgdf_fixed" \
  --validation-ratio 0.10 \
  --validation-seed 20250726 \
  --dynamic-ratio 0.8 \
  --fixed-p 0.4 \
  --warmup-epochs 5 \
  --update-interval 5 \
  --official-test-selected-only \
  --no-posthoc-oracle-test

CURRENT_STAGE="完成"
echo "[INFO] noise seed ${NOISE_SEED} 的 4 个 run 已全部完成。"
echo "[INFO] 基础三方法输出：${BASE_METHODS_DIR}"
echo "[INFO] Budget-matched 输出：${MATCHED_DIR}"
echo "[INFO] realization 审计：${INPUT_DIR}/noise_realization_audit.json"
