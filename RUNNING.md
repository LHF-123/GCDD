# 运行说明

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

  # DINOv2 特征提取 batch size；显存不足时调小。
  batch_size: 16

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
- 如果 DINOv2 权重缓存损坏，先删除 torch hub checkpoints 中对应的 DINOv2 权重，再重新运行。
- `random` 后端只能检查流程，不代表实验效果。
