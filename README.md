# GCDD-Lite

This repository implements the GCDD-Lite experiment plan.

Detailed commands and parameter descriptions are in [RUNNING.md](RUNNING.md).

## V0 Smoke Test

V0 is an engineering smoke test, not a paper result. It samples a small subset of Web-Bird, verifies images, skips bad WebFG-496 images, extracts features, builds the RRF graphs, computes `S_clean`, runs Adaptive-Otsu, and trains a linear classifier for 1 epoch.

For V0, KNN is exact numpy KNN because the sampled set is tiny. The graph module is isolated so the formal V1 implementation can replace this with FAISS without changing data indexing or scoring.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run with DINOv2:

```bash
python scripts/run_v0_smoke.py --data-root /path/to/Web-Bird
```

Run only the pipeline shape check without DINOv2 weights:

```bash
python scripts/run_v0_smoke.py --data-root /path/to/Web-Bird --feature-backend random
```

The resolved config is saved to `outputs/<dataset>/v0_smoke/resolved_config.yaml`. Hyperparameter precedence is:

```text
command line > YAML config > default parameters
```

Expected V0 outputs include:

- `debug_index.csv`
- `bad_images.csv`
- `debug_features.npy`
- `debug_scores.csv`
- `debug_split.csv`
- `debug_train_log.csv`
- `run_summary.md`

## V1 Web-Bird Minimal Closed Loop

V1 runs the first real Web-Bird check: train split selection with GCDD, aligned filtering baselines, and validation split evaluation.

```bash
python scripts/run_v1_web_bird.py --data-root /path/to/web-bird
```

For a quick implementation check:

```bash
python scripts/run_v1_web_bird.py \
  --data-root /path/to/web-bird \
  --feature-backend random \
  --max-classes 3 \
  --max-train-per-class 10 \
  --max-eval-per-class 5 \
  --epochs 2
```

V1 outputs are written to `outputs/<dataset>/v1_web_bird/`. Important files:

- `gcdd_scores.csv`
- `sample_split.csv`
- `clean_thresholds.csv`
- `q_same_top_bottom.csv`
- `s_clean_distribution.csv`
- `neighbor_sanity_samples.csv`
- `baseline_compare_web_bird.csv`
- `train_log.csv`
- `run_summary.md`

## V1.6 Gated Split Check

V1.6 reads an existing V1 output directory and checks whether low `Q_same` / low `centroid_score` samples inside `Full GCDD-clean` should be gated out.

Generate statistics and splits only:

```bash
python scripts/run_v1_6_gated_splits.py --input-dir outputs/Web-Bird/v1_web_bird
```

Train the three no-supplement gated splits:

```bash
python scripts/run_v1_6_gated_splits.py --input-dir outputs/Web-Bird/v1_web_bird --train
```

Outputs are written to `outputs/Web-Bird/v1_web_bird/v1_6_gated_splits/`.

## V1.7 QP-Risk Soft Weighting

V1.7 keeps the original `Full GCDD-clean` training set and down-weights the samples that `gcdd_qp_gate` would delete.

```bash
python scripts/run_v1_7_qp_soft_weighting.py --input-dir outputs/Web-Bird/v1_web_bird
```

Outputs are written to `outputs/Web-Bird/v1_web_bird/qp_soft_weighting/`.

## V1.8 Partial-Label Recovery

V1.8 keeps `Full GCDD-clean` CE training and adds partial-label loss for selected non-clean samples whose global neighbors indicate concentrated alternative labels.

```bash
python scripts/run_v1_8_recover_partial_label.py --input-dir outputs/Web-Bird/v1_web_bird
```

Outputs are written to `outputs/Web-Bird/v1_web_bird/recover_partial_label/`.

## Prototype-Aware GCDD

This experiment changes only the clean score by adding the class prototype percentile as a virtual super-node signal. Training remains clean-only CE on frozen DINOv2 features.

```bash
python scripts/run_proto_gcdd_web_bird.py --input-dir outputs/Web-Bird/v1_web_bird
```

Outputs are written to `outputs/Web-Bird/v1_web_bird/proto_gcdd/`. The main score variants are:

- `S_proto`: prototype percentile only with Adaptive-Otsu.
- `S_gcdd_proto`: original GCDD score plus prototype percentile.
- `S_gcdd_proto_noI`: prototype-aware score without `I_class_norm`.

Current Web-Bird best Top-1: `GCDD + Proto = 0.843631`, slightly above `Centroid filtering = 0.843286` and `Full GCDD-clean = 0.842423`.

## Multi-Seed Verification

Run only the three key methods across seeds 1, 2, and 3:

```bash
python scripts/run_multiseed_web_bird.py --input-dir outputs/Web-Bird/v1_web_bird
```

Outputs are written to `outputs/Web-Bird/v1_web_bird/multiseed/`. Current Web-Bird mean best Top-1:

- `Centroid filtering`: `0.844207`
- `Full GCDD-clean`: `0.839144`
- `GCDD + Proto`: `0.843516`

The prototype-aware score consistently improves over original GCDD, but it does not yet beat the centroid baseline on the three-seed mean.

## LoRA Route

LoRA trains on image data instead of frozen `.npy` features. It tests whether `GCDD + Proto` hard-clean samples help when DINOv2 can adapt.

Current main method:

```bash
python scripts/run_lora_web_bird.py --input-dir outputs/Web-Bird/v1_web_bird --methods gcdd_proto --seeds 1
```

Recommended table:

```bash
python scripts/run_lora_web_bird.py --input-dir outputs/Web-Bird/v1_web_bird --methods all,full_gcdd,both_only,gcdd_proto,centroid --seeds 1
```

Outputs are written to `outputs/Web-Bird/v1_web_bird/lora/`. If `DINOv2 LoRA + GCDD+Proto-only added` exceeds `DINOv2 LoRA + Centroid filtering` by `0.3-0.5 pp`, the hard-clean contribution is meaningful.
