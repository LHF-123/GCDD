#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Reuses the already materialized fixed random-derangement mapping.  This
# script never calls mapping generation: cyclic and random runs therefore use
# the same source sample identities for their flips.
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1
PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS="1,42,88"
VAL_SEED=20250726
OUT_ROOT="outputs/pgdf_dynamic_proto_random_derangement_asym40_map20260815_noise42"

DATASETS=(
  "CUB-200-2011|outputs/random_derangement_asym40_map20260815_noise42/prepared_inputs/CUB_200_2011|outputs/noise_manifests/random_derangement_asym40_seed42_map20260815/cub200_random_derangement_asym40.csv|outputs/CUB_200_2011/noise_indices/cub_asym_r0p4_s42_index.csv|outputs/noise_mappings/random_derangement_seed20260815/cub200_random_derangement.json|outputs/random_derangement_asym40_map20260815_noise42/CUB_200_2011/checkpoint_validation_s20250726|$OUT_ROOT/CUB_200_2011"
  "Stanford Cars|outputs/random_derangement_asym40_map20260815_noise42/prepared_inputs/Stanford_Cars|outputs/noise_manifests/random_derangement_asym40_seed42_map20260815/stanford_cars_random_derangement_asym40.csv|outputs/Stanford_Cars/noise_indices/stanford_cars_asym_r0p4_s42_index.csv|outputs/noise_mappings/random_derangement_seed20260815/stanford_cars_random_derangement.json|outputs/random_derangement_asym40_map20260815_noise42/Stanford_Cars/checkpoint_validation_s20250726|$OUT_ROOT/Stanford_Cars"
  "FGVC-Aircraft|outputs/random_derangement_asym40_map20260815_noise42/prepared_inputs/FGVC_Aircraft|outputs/noise_manifests/random_derangement_asym40_seed42_map20260815/fgvc_aircraft_random_derangement_asym40.csv|outputs/FGVC_Aircraft/noise_indices/fgvc_aircraft_asym_r0p4_s42_index.csv|outputs/noise_mappings/random_derangement_seed20260815/fgvc_aircraft_random_derangement.json|outputs/random_derangement_asym40_map20260815_noise42/FGVC_Aircraft/checkpoint_validation_s20250726|$OUT_ROOT/FGVC_Aircraft"
)

run() {
  echo "+ $*"
  if (( ! DRY_RUN )); then "$@"; fi
}

[[ ! -e "$OUT_ROOT" ]] || { echo "[ERROR] destination already exists: $OUT_ROOT" >&2; exit 1; }

for item in "${DATASETS[@]}"; do
  IFS='|' read -r dataset input random_manifest cyclic_index mapping old_validation new_base <<<"$item"
  output="$new_base/checkpoint_validation_pgdf_dynamic_proto_r08_p04_s${VAL_SEED}"
  [[ -d "$input" && -f "$random_manifest" && -f "$cyclic_index" && -f "$mapping" && -f "$old_validation/validation_manifest.json" && ! -e "$output" ]] || {
    echo "[ERROR] invalid mapping input or existing destination for $dataset" >&2; exit 1;
  }
  echo "[RUN] experiment=pgdf_dynamic_proto_random_derangement dataset=$dataset seeds=$SEEDS noise=asym40 mapping_seed=20260815"
  run "$PYTHON_BIN" scripts/run_lora_checkpoint_validation.py \
    --input-dir "$input" --noise-index "$random_manifest" --output-dir "$output" \
    --methods pgdf_dynamic_proto --seeds "$SEEDS" --validation-ratio 0.10 --validation-seed "$VAL_SEED" \
    --dynamic-ratio 0.8 --fixed-p 0.4 --warmup-epochs 5 --update-interval 5 \
    --official-test-selected-only --no-posthoc-oracle-test
done

if (( ! DRY_RUN )); then
  "$PYTHON_BIN" tools/audit_dynamic_proto_random_mapping.py \
    --dataset "CUB-200-2011=outputs/CUB_200_2011/noise_indices/cub_asym_r0p4_s42_index.csv=outputs/noise_manifests/random_derangement_asym40_seed42_map20260815/cub200_random_derangement_asym40.csv=outputs/noise_mappings/random_derangement_seed20260815/cub200_random_derangement.json=outputs/random_derangement_asym40_map20260815_noise42/CUB_200_2011/checkpoint_validation_s${VAL_SEED}=outputs/pgdf_dynamic_proto_random_derangement_asym40_map20260815_noise42/CUB_200_2011/checkpoint_validation_pgdf_dynamic_proto_r08_p04_s${VAL_SEED}" \
    --dataset "Stanford Cars=outputs/Stanford_Cars/noise_indices/stanford_cars_asym_r0p4_s42_index.csv=outputs/noise_manifests/random_derangement_asym40_seed42_map20260815/stanford_cars_random_derangement_asym40.csv=outputs/noise_mappings/random_derangement_seed20260815/stanford_cars_random_derangement.json=outputs/random_derangement_asym40_map20260815_noise42/Stanford_Cars/checkpoint_validation_s${VAL_SEED}=outputs/pgdf_dynamic_proto_random_derangement_asym40_map20260815_noise42/Stanford_Cars/checkpoint_validation_pgdf_dynamic_proto_r08_p04_s${VAL_SEED}" \
    --dataset "FGVC-Aircraft=outputs/FGVC_Aircraft/noise_indices/fgvc_aircraft_asym_r0p4_s42_index.csv=outputs/noise_manifests/random_derangement_asym40_seed42_map20260815/fgvc_aircraft_random_derangement_asym40.csv=outputs/noise_mappings/random_derangement_seed20260815/fgvc_aircraft_random_derangement.json=outputs/random_derangement_asym40_map20260815_noise42/FGVC_Aircraft/checkpoint_validation_s${VAL_SEED}=outputs/pgdf_dynamic_proto_random_derangement_asym40_map20260815_noise42/FGVC_Aircraft/checkpoint_validation_pgdf_dynamic_proto_r08_p04_s${VAL_SEED}" \
    --output "$OUT_ROOT/mapping_audit.csv"
  "$PYTHON_BIN" tools/summarize_dynamic_prototype_multiseed.py --method-key pgdf_dynamic_proto \
    --dataset "CUB-200-2011=$OUT_ROOT/CUB_200_2011/checkpoint_validation_pgdf_dynamic_proto_r08_p04_s${VAL_SEED}" \
    --dataset "Stanford Cars=$OUT_ROOT/Stanford_Cars/checkpoint_validation_pgdf_dynamic_proto_r08_p04_s${VAL_SEED}" \
    --dataset "FGVC-Aircraft=$OUT_ROOT/FGVC_Aircraft/checkpoint_validation_pgdf_dynamic_proto_r08_p04_s${VAL_SEED}" \
    --seeds "$SEEDS" --output-dir "$OUT_ROOT/summary"
else
  echo "+ $PYTHON_BIN tools/audit_dynamic_proto_random_mapping.py ... --output $OUT_ROOT/mapping_audit.csv"
  echo "+ $PYTHON_BIN tools/summarize_dynamic_prototype_multiseed.py ... --output-dir $OUT_ROOT/summary"
fi
