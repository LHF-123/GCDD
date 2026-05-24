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
python scripts/run_v1_6_gated_splits.py --input-dir outputs/v1_web_bird
```

Train the three no-supplement gated splits:

```bash
python scripts/run_v1_6_gated_splits.py --input-dir outputs/v1_web_bird --train
```

Outputs are written to `outputs/v1_web_bird/v1_6_gated_splits/`.
