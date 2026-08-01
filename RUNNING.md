# 运行说明

## Validation-Selected Checkpoint Protocol

以下命令运行统一的 `fixed_clean_validation_v1` 协议。`--methods all` 按固定顺序展开 13 个方法：`all_noisy`、`full_gcdd`、`centroid`、`both_only`、`gcdd_proto`、`fine`、`dynamic_r08`、`dynamic_r09`、`jal_ce`、`coteaching`、`jocor`、`pgdf_auto`、`pgdf_fixed`。旧 key `dynamic` 仍可使用，并等价于 `dynamic_r08`。

- checkpoint 只按训练集划出的固定 clean validation Top-1 选择；官方 test 不参与训练期选择。
- `--fixed-p 0.4` 是预先固定的全局值，不能根据本轮 test 结果调整。
- validation 不参与训练、静态/动态筛选、graph budget、prototype reference 或 class centroid。
- Co-teaching/JoCoR 以两个分支 Top-1 的算术均值选择 checkpoint，不使用 ensemble prediction。
- 以下命令显式关闭 post-hoc oracle test；official test 只在训练结束后评估 validation-selected、final 和 last-5 checkpoint。
- 任一运行报错会直接退出，不会跳过失败 seed 继续写完整结果。
- 输出使用新的 `checkpoint_validation_all_methods_s20250726` 目录，不覆盖旧 `checkpoint_validation_s20250726` 原始结果。
- 三个数据集共 `13 × 3 × 3 = 117` 个 run；Co-teaching 和 JoCoR 的每个 run 各训练两个分支模型。

### CUB-200-2011 asym40

```bash
python scripts/run_lora_checkpoint_validation.py \
  --input-dir outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42 \
  --noise-index outputs/CUB_200_2011/noise_indices/cub_asym_r0p4_s42_index.csv \
  --output-dir outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42/checkpoint_validation_all_methods_s20250726 \
  --methods all --seeds 1,42,88 \
  --validation-ratio 0.10 --validation-seed 20250726 \
  --dynamic-ratio 0.8 --fixed-p 0.4 \
  --warmup-epochs 5 --update-interval 5 \
  --no-posthoc-oracle-test
```

### Stanford Cars asym40

```bash
python scripts/run_lora_checkpoint_validation.py \
  --input-dir outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42 \
  --noise-index outputs/Stanford_Cars/noise_indices/stanford_cars_asym_r0p4_s42_index.csv \
  --output-dir outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42/checkpoint_validation_all_methods_s20250726 \
  --methods all --seeds 1,42,88 \
  --validation-ratio 0.10 --validation-seed 20250726 \
  --dynamic-ratio 0.8 --fixed-p 0.4 \
  --warmup-epochs 5 --update-interval 5 \
  --no-posthoc-oracle-test
```

### FGVC-Aircraft asym40

```bash
python scripts/run_lora_checkpoint_validation.py \
  --input-dir outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42 \
  --noise-index outputs/FGVC_Aircraft/noise_indices/fgvc_aircraft_asym_r0p4_s42_index.csv \
  --output-dir outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42/checkpoint_validation_all_methods_s20250726 \
  --methods all --seeds 1,42,88 \
  --validation-ratio 0.10 --validation-seed 20250726 \
  --dynamic-ratio 0.8 --fixed-p 0.4 \
  --warmup-epochs 5 --update-interval 5 \
  --no-posthoc-oracle-test
```

每个输出目录优先查看：

```text
validation_manifest.csv
validation_manifest.json
checkpoint_validation_results.csv
checkpoint_validation_summary.csv
checkpoint_validation_summary.json
run_index.csv
run_summary.md
<method>/seed<seed>/train_log.csv
<method>/seed<seed>/checkpoints/{best_val.pt,last.pt,last5/}
<method>/seed<seed>/{selection_rows.csv|static_selection.csv|selection_history.csv}
```
## 参数优先级

所有运行脚本统一使用：

```text
命令行参数 > YAML 配置文件 > 默认参数
```

最终生效配置会写入输出目录的 `resolved_config.yaml`。

## V1 五个方法在哪里控制

V1 当前固定运行 5 个方法，控制位置在 `gcdd/pipeline_v1.py`：

- 方法列表：`V1_METHODS`
- baseline 样本选择：`baseline_masks`
- 训练循环：`for method in V1_METHODS[1:]`

当前方法：

```text
DINOv2 Linear all
Confidence filtering
Loss filtering
Centroid filtering
Full GCDD-clean
```

这 5 个方法不是 V4 消融实验，而是 V1 的最小公平对比 baseline。一次 V1 运行会按顺序跑完这 5 个方法，并在同一个 `baseline_compare_web_bird.csv` 中汇总结果；当前不需要、也暂时不支持在命令行里只选择其中一个方法。

说明：

- `DINOv2 Linear all` 使用全部 train 样本。
- `Full GCDD-clean` 使用 `S_clean + Adaptive-Otsu` 得到的 clean 样本。
- `Confidence filtering`、`Loss filtering`、`Centroid filtering` 都按每类 `Full GCDD-clean` 的保留数量对齐后再训练，避免 clean ratio 不一致导致不公平。
- `Confidence filtering` 和 `Loss filtering` 依赖 `DINOv2 Linear all` 的训练预测，因此它必须先跑。

## V1 正式全量运行命令

你的数据结构是：

```text
web-bird/
  train/
    class_001/
    class_002/
  val/
    class_001/
    class_002/
```

所以 `--data-root` 写到 `web-bird` 这一层：

```powershell
python scripts/run_v1_web_bird.py --data-root "E:\下载\dataset\webfg496\web-bird"
```

这条命令会使用 `configs/v1_web_bird.yaml` 中的默认设置：

```text
feature backend: dinov2_vitb14
train split: train
eval split: val
epochs: 50
methods: 5 个方法全部运行
```

V1 默认读取：

```text
train split: train
eval split: val
```

输出目录：

```text
outputs/Web-Bird/v1_web_bird/
```

## V1 调试运行命令

下面的命令只是调试用，不是正式结果。它用于快速检查路径、坏图、shape、日志和训练循环：

```powershell
python scripts/run_v1_web_bird.py `
  --data-root "E:\下载\dataset\webfg496\web-bird" `
  --feature-backend random `
  --max-classes 3 `
  --max-train-per-class 10 `
  --max-eval-per-class 5 `
  --epochs 2
```

用 DINOv2 但只跑少量类别：

```powershell
python scripts/run_v1_web_bird.py `
  --data-root "E:\下载\dataset\webfg496\web-bird" `
  --max-classes 5 `
  --max-train-per-class 20 `
  --max-eval-per-class 10 `
  --epochs 5
```

## 常用命令行参数

| 参数 | 作用 |
| --- | --- |
| `--data-root` | 数据集根目录，写到 `web-bird` 这一层。 |
| `--index-file` | 可选 CSV 索引文件；如果不用目录结构推断类别，就传这个。 |
| `--output-root` | 输出根目录，默认 `outputs`。 |
| `--dataset-name` | 数据集名，默认 `Web-Bird`。 |
| `--feature-backend` | `dinov2_vitb14` 或 `random`。 |
| `--device` | `auto`、`cpu` 或 `cuda`。 |
| `--batch-size` | DINOv2 特征提取 batch size。 |
| `--epochs` | 线性分类器训练轮数。 |
| `--max-classes` | 只跑前 N 个类别，调试用。 |
| `--max-train-per-class` | 每类最多取多少 train 样本，调试用。 |
| `--max-eval-per-class` | 每类最多取多少 val 样本，调试用。 |
| `--set KEY=VALUE` | 任意点路径覆盖配置，例如 `--set graph.k_class=10`。 |

## YAML 配置文件

V1 默认配置在：

```text
configs/v1_web_bird.yaml
```

关键字段和含义：

```yaml
dataset:
  # 数据集根目录；命令行 --data-root 会覆盖它。
  root: ""

  # 可选 CSV 索引文件；为空时按目录结构扫描图片。
  index_file: ""

  # 训练集目录名。你的结构中是 web-bird/train，所以这里是 train。
  train_split: train

  # 评估集目录名。你的结构中是 web-bird/val，所以这里是 val。
  eval_split: val

  # 调试用：只取前 N 个类别；正式实验应留空。
  max_classes:

  # 调试用：每类最多取多少 train 样本；正式实验应留空。
  max_train_per_class:

  # 调试用：每类最多取多少 val 样本；正式实验应留空。
  max_eval_per_class:

  # 是否在索引阶段检查坏图；WebFG-496 建议保持 true。
  verify_images: true

feature:
  # 特征后端。正式 V1 用 dinov2_vitb14；random 只用于流程调试。
  backend: dinov2_vitb14

  # 可选：DINOv2 本地 torch hub repo 路径。为空时优先使用 ~/.cache/torch/hub/facebookresearch_dinov2_main。
  local_repo: ""

  # DINOv2 特征提取 batch size；32GB 显存建议先用 128，OOM 再调小。
  batch_size: 128

  # 输入图片 resize 尺寸。
  input_size: 224

  # Top-patch 特征使用响应最高的 patch 比例。
  top_patch_ratio: 0.2

  # 是否复用已有特征缓存；只有 paths.txt 与当前索引完全一致才会复用。
  reuse: true

graph:
  # class-wise KNN 的候选池大小。
  k_pool_class: 100

  # global KNN 的候选池大小。
  k_pool_global: 300

  # RRF 后保留的类内邻居数。
  k_class: 20

  # RRF 后保留的全局邻居数。
  k_global: 50

  # Reciprocal Rank Fusion 的平滑参数。
  rrf_k0: 20

selection:
  # Otsu 阈值直方图 bin 数。
  otsu_bins: 256

  # 每类 clean ratio 的上下限，防止 Otsu 过低或过高。
  clean_ratio_clip: [0.3, 0.9]

train:
  # 线性分类器训练轮数。V1 正式默认 50。
  epochs: 50

  # 线性分类器 batch size。
  batch_size: 256

  # 线性分类器学习率。
  lr: 0.05

  # 最小学习率，scheduler 衰减到这个值。
  min_lr: 0.0

  # 学习率调度。正式 V1 默认 cosine；可选 none、linear、cosine。
  scheduler: cosine

  # 训练和抽样随机种子。
  seed: 42

  # 用哪种特征训练线性分类器；当前默认 cls。
  feature: cls
```

epoch 建议：

- V1 正式默认先跑 `50`，因为 5 个方法都会训练，直接改成 `100` 会让耗时接近翻倍。
- 如果 `run_summary.md` 或 `baseline_compare_web_bird.csv` 中出现 `best_top1` 明显高于 `final_top1`，或者 `last10_std` 偏高，说明后期不稳定，可以再跑 `--epochs 100`。
- 如果 50 epoch 下 `final_top1` 接近 `best_top1` 且 `last10_std` 很小，不需要延长。

## V1 主要输出文件

```text
resolved_config.yaml
bad_images.csv
dataset_index.csv
eval_index.csv
features_cls.npy
features_gap.npy
features_top.npy
eval_features_cls.npy
eval_features_gap.npy
eval_features_top.npy
class_knn_indices.npy
class_knn_weights.npy
global_knn_indices.npy
global_knn_weights.npy
gcdd_scores.csv
sample_split.csv
clean_thresholds.csv
q_same_top_bottom.csv
s_clean_distribution.csv
neighbor_sanity_samples.csv
baseline_compare_web_bird.csv
train_log.csv
eval_log.csv
run_summary.md
```

优先查看：

```text
run_summary.md
baseline_compare_web_bird.csv
bad_images.csv
q_same_top_bottom.csv
s_clean_distribution.csv
```

## 注意事项

- WebFG-496 可能有坏图，坏图会写入 `bad_images.csv` 并跳过。
- `feature.reuse: true` 时会复用已有特征，但只有缓存的 `paths.txt` 与当前索引完全一致才会复用。
- 5090 32GB 显存建议 V1 从 `feature.batch_size=128` 开始；如果 OOM，降到 `64` 或 `32`。
- 如果 GPU 占用很低，先看终端进度：可能仍在图片读取、坏图检查、特征缓存校验或 KNN 构建阶段。
- DINOv2 默认优先从本地 torch hub cache 加载：`~/.cache/torch/hub/facebookresearch_dinov2_main`，避免重复访问 GitHub。
- 如果本地 repo 不在默认位置，用 `--set feature.local_repo="/path/to/facebookresearch_dinov2_main"` 指定。
- 如果 DINOv2 权重缓存损坏，先删除 torch hub checkpoints 中对应的 DINOv2 权重，再重新运行。
- `random` 后端只能检查流程，不代表实验效果。

## V1.5 GCDD vs Centroid 诊断

V1.5 不训练、不重新筛样本，只分析 V1 已生成的 `Full GCDD-clean` 和 `Centroid filtering` 差异。

运行命令：

```powershell
python tools/analyze_gcdd_centroid_diff.py --input-dir outputs/Web-Bird/v1_web_bird
```

默认会把 HTML 可视化涉及到的图片复制到：

```text
<input-dir>/gcdd_centroid_analysis/figures/assets/
```

HTML 使用相对路径引用这些 asset，所以下载整个 `gcdd_centroid_analysis` 文件夹后也能看图。

如果 V1 是在 autodl 跑的，CSV 里的图片路径通常是 `/root/autodl-tmp/web-bird/...`。在本地重新生成 assets 时，需要把远端根目录映射到本地数据集根目录：

```powershell
python tools/analyze_gcdd_centroid_diff.py `
  --input-dir outputs/Web-Bird/v1_web_bird `
  --path-map "/root/autodl-tmp/web-bird=E:\下载\dataset\webfg496\web-bird"
```

如果直接在 autodl 上生成分析目录，并且原始图片仍在 `/root/autodl-tmp/web-bird`，不需要 `--path-map`。

如果你的结果目录不同，替换 `--input-dir`：

```powershell
python tools/analyze_gcdd_centroid_diff.py --input-dir outputs/Web-Bird/v1_web_bird
```

默认输出：

```text
<input-dir>/gcdd_centroid_analysis/
```

主要输出：

```text
overlap_summary.csv
per_class_overlap.csv
per_class_overlap_top_low_jaccard.csv
per_class_gcdd_only_top_classes.csv
per_class_centroid_only_top_classes.csv
group_metric_summary.csv
hard_clean_group_distribution.csv
visualization_index.csv
class_visualization_index.csv
analysis_summary.md
figures/distributions/
figures/neighbors/
figures/classes/
```

说明：

- 如果当前 V1 目录没有 `linear_all_train_scores.csv`，脚本会跳过基于 loss/confidence 的 hard-clean 统计。
- 新版本 V1 后续会自动写 `linear_all_train_scores.csv`，以后再跑 V1.5 时会自动启用 loss/confidence 分析。
- 如不想复制图片，可加 `--no-copy-assets`，此时 HTML 会直接引用原始图片路径。

## V1.6 Gated GCDD Split

V1.6 用来验证一个更直接的问题：`Full GCDD-clean` 里是否存在一批 `Q_same` 低、`centroid_score` 低的可疑样本。它读取 V1 已有输出，不重新提特征、不重新计算 GCDD、不改原始 V1 结果。

第一步只统计并生成 3 个 gated split，不训练：

```powershell
python scripts/run_v1_6_gated_splits.py --input-dir outputs/Web-Bird/v1_web_bird
```

默认阈值定义：

```text
low_Q_same: 每个类别内 Q_same 排后 30%
low_centroid: 每个类别内 centroid_score 排后 30%
```

生成的 split：

```text
gcdd_qgate_split.csv      # Full GCDD clean 中 low_Q_same 改为 ignored
gcdd_pgate_split.csv      # Full GCDD clean 中 low_centroid 改为 ignored
gcdd_qp_gate_split.csv    # Full GCDD clean 中 low_Q_same 且 low_centroid 改为 ignored
```

第一轮不补齐被删样本，目的是直接判断这些可疑样本是否有害。正式训练 3 个 gated split：

```powershell
python scripts/run_v1_6_gated_splits.py --input-dir outputs/Web-Bird/v1_web_bird --train
```

常用参数：

| 参数 | 作用 |
| --- | --- |
| `--low-ratio 0.3` | 每类低分样本比例，默认 30%。 |
| `--output-dir` | 指定 V1.6 输出目录，默认 `<input-dir>/v1_6_gated_splits`。 |
| `--train` | 训练 `qgate`、`pgate`、`qp_gate` 三个 split。 |
| `--epochs` | 覆盖线性分类器训练轮数。正式对比默认沿用 V1 的 50。 |
| `--feature cls` | 使用哪个缓存特征训练，默认沿用 V1 配置。 |

主要输出：

```text
v1_6_gated_splits/
  suspicious_ratio_summary.csv
  per_class_gate_summary.csv
  gated_sample_flags.csv
  gcdd_qgate_split.csv
  gcdd_pgate_split.csv
  gcdd_qp_gate_split.csv
  gated_compare_web_bird.csv        # 使用 --train 时生成
  combined_compare_web_bird.csv     # 使用 --train 时生成，合并 V1 关键结果
  train_log.csv                     # 使用 --train 时生成
  run_summary.md
```

优先看：

```text
suspicious_ratio_summary.csv
run_summary.md
combined_compare_web_bird.csv
```

判断规则：

- 如果 `gcdd_qp_gate` 高于 `Full GCDD-clean`，说明原始 GCDD-clean 中确实有局部连通噪声。
- 如果 `gcdd_qp_gate` 还能接近或超过 `Centroid filtering`，说明 GCDD 仍有继续改的价值。
- 如果三个 gate 都不涨，先不要进入 V2，应重新检查 score 设计或承认 centroid 当前更强。

## V1.6 QP-Gate 删除样本分析

如果 `gcdd_qp_gate` 没有提升，下一步先分析它删除的样本，不继续补齐训练。该脚本只读已有 V1/V1.6 结果，不改 split、不训练。

```powershell
python tools/analyze_qp_gate_deleted.py --input-dir outputs/Web-Bird/v1_web_bird
```

如果 V1 是在 AutoDL 跑的，且 CSV 中图片路径是 `/root/autodl-tmp/web-bird/...`，在本地生成可视化时加路径映射：

```powershell
python tools/analyze_qp_gate_deleted.py `
  --input-dir outputs/Web-Bird/v1_web_bird `
  --path-map "/root/autodl-tmp/web-bird=E:\下载\dataset\webfg496\web-bird"
```

主要输出：

```text
outputs/Web-Bird/v1_web_bird/v1_6_gated_splits/qp_gate_deleted_analysis/
  deleted_by_qp_gate.csv
  qp_gate_deleted_per_class.csv
  qp_gate_deleted_top20_classes.csv
  deleted_metric_summary.csv
  visualization_index.csv
  manual_qp_gate_review.csv
  manual_review.html
  analysis_summary.md
  figures/neighbors/
  figures/assets/
```

重点看：

- `deleted_by_qp_gate.csv`：233 个被删样本的明细。
- `qp_gate_deleted_per_class.csv`：是否集中在少数类别。
- `analysis_summary.md`：loss、confidence、Q_same、centroid_score 是否相对 kept-GCDD 明显异常。
- `figures/neighbors/*.html`：人工判断 hard clean、真实噪声、背景相似簇、类别边界样本。

人工标注最便捷方式：

1. 打开 `manual_review.html`。
2. 左侧选择样本，右侧查看 query、top-5 class neighbors 和 top-5 global neighbors。
3. 填 `manual_label`、`looks_like_class_neighbors`、`looks_like_global_neighbors`、`has_duplicate` 和 `note`。
4. 点击 `Export CSV` 导出标注结果。

字段含义：

```text
manual_label:
  noise        明显不是有效鸟图，或是无关/错误图
  clean_like   是有效鸟图，但不强行判断是否属于 web_label 的细粒度类别
  uncertain    看不清、主体太小、遮挡严重，或无法判断是否是有效鸟图

looks_like_class_neighbors:
  query 是否更像 top-5 class neighbors

looks_like_global_neighbors:
  query 是否更像 top-5 global neighbors

has_duplicate:
  邻居中是否存在重复或近重复图片
```

当前 Web-Bird 人工检查不要求区分具体细粒度类别，所以不要强行标 `hard_clean` 或 `boundary`。只需要判断是否是有效鸟图，并记录它更像 class neighbors 还是 global neighbors。

## V1.7 QP-Risk Soft Weighting

V1.7 验证 `qp_gate` 命中的高风险样本是否应该硬删除，还是保留但降低 CE loss 权重。它复用 V1 和 V1.6 输出，不重新提特征、不重新筛样本。

定义：

```text
qp_risk = Full GCDD-clean 中被 gcdd_qp_gate 删除的样本
```

默认实验：

```text
qp_soft_0.3
qp_soft_0.5
qp_soft_0.7
```

运行：

```powershell
python scripts/run_v1_7_qp_soft_weighting.py --input-dir outputs/Web-Bird/v1_web_bird
```

可选指定 alpha：

```powershell
python scripts/run_v1_7_qp_soft_weighting.py `
  --input-dir outputs/Web-Bird/v1_web_bird `
  --alphas 0.3,0.5,0.7
```

主要输出：

```text
outputs/Web-Bird/v1_web_bird/qp_soft_weighting/
  qp_risk_indices.csv
  sample_weight_alpha_0.3.csv
  sample_weight_alpha_0.5.csv
  sample_weight_alpha_0.7.csv
  qp_soft_results.csv
  combined_compare_web_bird.csv
  train_log.csv
  run_summary.md
```

`qp_soft_results.csv` 重点字段：

```text
method
alpha
train_samples
effective_train_weight
num_qp_risk
qp_risk_ratio
best_top1
final_top1
last10_mean
last10_std
best_epoch
```

判断：

- 如果 `qp_soft_x > Full GCDD-clean`，说明 hard delete 太激进，soft weighting 有效。
- 如果 `qp_gate delete < qp_soft_x < Full GCDD-clean`，说明降权比删除好，但这批样本整体仍有正贡献。
- 如果 `qp_soft_x <= qp_gate delete`，说明保留这些风险样本即使降权也没有帮助。

## V1.8 Partial-Label Recovery

V1.8 验证：对 GCDD 判为 non-clean、但 global graph 指向其他候选类别的样本，不使用原 web label CE，而使用候选标签集合 partial-label loss。

训练组成：

```text
Full GCDD-clean:
  原 web label CE

recoverable non-clean:
  -log sum_{c in candidate_labels} p(c)
```

默认 recoverable 定义：

```text
recover_top_qalt:
  non-clean
  Q_alt >= 0.30
  candidate_size in [2, 6]

safe_recover:
  recover_top_qalt
  neighbor_entropy <= 0.70
  D_global_percentile >= 50
```

候选标签集合：

```text
candidate_labels = {c | q_i(c) >= 0.10} union {original web label}
```

默认运行：

```powershell
python scripts/run_v1_8_recover_partial_label.py --input-dir outputs/Web-Bird/v1_web_bird
```

主要输出：

```text
outputs/Web-Bird/v1_web_bird/recover_partial_label/
  recover_candidate_stats.csv
  recover_top_qalt_samples.csv
  safe_recover_samples.csv
  recover_results.csv
  combined_compare_web_bird.csv
  train_log.csv
  run_summary.md
```

判断：

- 如果 `recover_* > Full GCDD-clean`，说明 non-clean recovery 有价值。
- 如果 `recover_* >= Centroid filtering`，说明 recovery 方向可以作为后续主线。
- 如果低于 `Full GCDD-clean`，说明当前候选标签不够可靠，应暂停 recovery 方向。

## Prototype-Aware GCDD

该实验把 `centroid_score` 当作类别 prototype super-node 连接强度，只改 clean score，不改训练逻辑。训练仍然是：

```text
选 clean 样本 -> DINOv2 frozen feature -> linear classifier clean-only CE
```

一次运行会生成并训练 3 个 score 版本：

```text
Proto only (Otsu):     S_proto = P_proto
GCDD + Proto:          S_gcdd_proto = (P_D * P_R * P_I * P_Q * P_proto)^(1/5)
GCDD + Proto no-I:     S_gcdd_proto_noI = (P_D * P_R * P_Q * P_proto)^(1/4)
```

运行：

```powershell
python scripts/run_proto_gcdd_web_bird.py --input-dir outputs/Web-Bird/v1_web_bird
```

只生成 score 表：

```powershell
python tools/build_proto_gcdd_scores.py --input-dir outputs/Web-Bird/v1_web_bird
```

手动从任意 score 列生成 Adaptive-Otsu split：

```powershell
python tools/split_clean_otsu.py `
  --scores outputs/Web-Bird/v1_web_bird/proto_gcdd/proto_gcdd_scores.csv `
  --score-col S_gcdd_proto `
  --out outputs/Web-Bird/v1_web_bird/proto_gcdd/splits/split_gcdd_proto.csv
```

主要输出：

```text
outputs/Web-Bird/v1_web_bird/proto_gcdd/
  proto_gcdd_scores.csv
  splits/split_full_gcdd_rebuilt.csv
  splits/split_proto_only.csv
  splits/split_gcdd_proto.csv
  splits/split_gcdd_proto_noI.csv
  proto_gcdd_split_summary.csv
  proto_gcdd_results.csv
  combined_compare_web_bird.csv
  train_log.csv
  run_summary.md
```

注意：`Proto only (Otsu)` 使用 Adaptive-Otsu 选样，不等同于 V1 中按 `Full GCDD-clean` 每类保留数量对齐的 `Centroid filtering` baseline。

当前 Web-Bird 结果：

```text
Centroid filtering:  0.843286
Full GCDD-clean:     0.842423
GCDD + Proto:        0.843631
GCDD + Proto no-I:   0.840525
```

结论：`GCDD + Proto` 略高于当前 centroid baseline，说明 prototype anchor 对图筛选有正收益；`no-I` 明显更低，说明加入 prototype 后暂时不应直接去掉 `I_class_norm`。

## Multi-Seed Verification

多 seed 验证只重复训练下面 3 个方法，不重新提特征、不重新筛样本：

```text
Centroid filtering
Full GCDD-clean
GCDD + Proto
```

默认 seeds：

```text
1,2,3
```

运行：

```powershell
python scripts/run_multiseed_web_bird.py --input-dir outputs/Web-Bird/v1_web_bird
```

主要输出：

```text
outputs/Web-Bird/v1_web_bird/multiseed/
  multiseed_summary.csv
  multiseed_results.csv
  train_log.csv
  run_summary.md
```

当前 Web-Bird best Top-1 多 seed 结果：

```text
method              seed1     seed2     seed3     mean      std
Centroid filtering  0.843977  0.843114  0.845530  0.844207  0.001224
Full GCDD-clean     0.838799  0.838108  0.840525  0.839144  0.001245
GCDD + Proto        0.843631  0.842423  0.844494  0.843516  0.001040
```

结论：`GCDD + Proto` 多 seed 下稳定高于 `Full GCDD-clean`，但均值仍略低于 `Centroid filtering`，差距约 `0.069 pp`。因此 prototype anchor 是有效修正，但 Web-Bird 上尚不能声称稳定超过 centroid baseline。

## GCDD+Proto vs Centroid Difference Analysis

该分析只比较 `GCDD + Proto` 和 `Centroid filtering` 的 clean 样本集合，不训练、不重新筛样本。目标是判断 `GCDD + Proto` 是否只是 centroid 的换皮版本。

运行：

```powershell
python tools/analyze_proto_gcdd_centroid_diff.py --input-dir outputs/Web-Bird/v1_web_bird
```

主要输出：

```text
outputs/Web-Bird/v1_web_bird/proto_gcdd_vs_centroid_analysis/
  proto_gcdd_vs_centroid_overlap.csv
  group_metric_summary.csv
  per_class_proto_gcdd_vs_centroid_overlap.csv
  per_class_low_jaccard_top20.csv
  per_class_gcdd_proto_only_top20.csv
  per_class_centroid_only_top20.csv
  sample_groups.csv
  analysis_summary.md
```

当前 overlap：

```text
Centroid clean:      9305
GCDD+Proto clean:    9260
Overlap:             7665
GCDD+Proto only:     1595
Centroid only:       1640
Neither:             7486
Jaccard:             0.7032
```

当前四组均值：

```text
group            count  S_gcdd_proto  S_clean  centroid  R       I       Q       loss    confidence
both             7665   0.6719        0.6532   0.7034    0.7987  0.9624  0.2870  0.5528  0.6910
gcdd_proto_only  1595   0.5382        0.5938   0.4453    0.7500  0.9370  0.1767  1.3491  0.4412
centroid_only    1640   0.3142        0.2706   0.6012    0.3719  0.6223  0.1631  0.9717  0.5409
neither          7486   0.2224        0.2396   0.2750    0.3728  0.5279  0.0757  2.1655  0.2861
```

结论：Jaccard=0.7032，说明 `GCDD + Proto` 不是 centroid 的换皮。`gcdd_proto_only` 比 `centroid_only` 有明显更强的 `R_class` 和 `I_class_norm`，但 loss 更高、confidence 更低、centroid_score 更低；这符合“图结构保留较难样本，centroid 保留更典型 easy clean”的模式。

## LoRA Route

LoRA 路线用于验证一个新问题：`GCDD + Proto` 额外保留的 hard clean 样本，在 backbone 可以适配时是否能贡献收益。

训练数据：

```text
DINOv2 LoRA all noisy samples:
  使用全部 train 样本，不做筛选

DINOv2 LoRA + GCDD+Proto-only added:
  使用 both only + GCDD+Proto-only 样本，即 proto_gcdd/splits/split_gcdd_proto.csv 中 state=clean 的样本

DINOv2 LoRA + Full GCDD-clean:
  使用 full_gcdd_clean_split.csv 中 state=clean 的样本，即原 GCDD clean

DINOv2 LoRA + both only:
  使用 GCDD+Proto clean 与 centroid clean 的交集样本，即 easy clean 上限/保守集

DINOv2 LoRA + Centroid filtering:
  使用 centroid_filtering_split.csv 中 state=clean 的样本，即 centroid clean
```

当前新主方法：

```text
DINOv2 LoRA + GCDD+Proto-only added
```

最小对照应包含：

```text
DINOv2 LoRA all noisy samples
DINOv2 LoRA + Full GCDD-clean
DINOv2 LoRA + both only
DINOv2 LoRA + Centroid filtering
```

作用：

```text
验证 GCDD+Proto-only 额外保留的 hard clean 是否在 LoRA 微调下有贡献。
```

最小运行当前主方法：

```powershell
python scripts/run_lora_web_bird.py `
  --input-dir outputs/Web-Bird/v1_web_bird `
  --methods gcdd_proto `
  --seeds 1
```

推荐运行完整最小表：

```powershell
python scripts/run_lora_web_bird.py `
  --input-dir outputs/Web-Bird/v1_web_bird `
  --methods all,full_gcdd,both_only,gcdd_proto,centroid `
  --seeds 1
```

如果 V1 输出里的图片路径来自 AutoDL，但你在本地跑，需要加路径映射：

```powershell
python scripts/run_lora_web_bird.py `
  --input-dir outputs/Web-Bird/v1_web_bird `
  --methods all,full_gcdd,both_only,gcdd_proto,centroid `
  --seeds 1 `
  --path-map "/root/autodl-tmp/web-bird=E:\下载\dataset\webfg496\web-bird"
```

默认 LoRA 配置：

```text
target_modules: qkv
rank: 8
alpha: 16
dropout: 0.05
epochs: 10
batch_size: 32
eval_batch_size: 64
lora_lr: 1e-4
head_lr: 1e-3
weight_decay: 0.05
scheduler: cosine
warmup_ratio: 0.1
amp: true
```

常用覆盖参数：

```powershell
python scripts/run_lora_web_bird.py `
  --input-dir outputs/Web-Bird/v1_web_bird `
  --methods all,full_gcdd,both_only,gcdd_proto,centroid `
  --epochs 20 `
  --batch-size 48 `
  --eval-batch-size 96 `
  --rank 8 `
  --alpha 16
```

输出：

```text
outputs/Web-Bird/v1_web_bird/lora/
  lora_results.csv
  lora_summary.csv
  train_log.csv
  lora_modules.csv
  run_summary.md
  checkpoints/
```

判断标准：

```text
如果 DINOv2 LoRA + GCDD+Proto-only added > DINOv2 LoRA + Centroid filtering：
  说明 GCDD+Proto-only 额外保留的 hard clean 在可微调 backbone 下有贡献。

如果 DINOv2 LoRA + GCDD+Proto-only added > DINOv2 LoRA + Full GCDD-clean：
  说明 prototype anchor 在 LoRA 场景下也优于原始 GCDD-clean。

如果 DINOv2 LoRA + GCDD+Proto-only added > DINOv2 LoRA + both only：
  说明 GCDD+Proto-only 额外样本不只是噪声，可能给 LoRA 提供了有效 hard clean。

如果 DINOv2 LoRA + GCDD+Proto-only added > DINOv2 LoRA all noisy samples：
  说明筛选后的 hard clean 训练优于直接使用全量噪声 web 数据。

如果提升达到 0.3-0.5 pp：
  这个结果很有价值，可以作为后续主线。

如果 DINOv2 LoRA + GCDD+Proto-only added 仍低于 DINOv2 LoRA + Centroid filtering：
  说明当前 hard clean 额外样本对 backbone 微调也没有明显正收益，应继续分析样本质量或改 score。
```
