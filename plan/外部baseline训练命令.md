# 外部 Baseline 训练命令

以下命令默认在仓库根目录 `D:\实验\GCDD` 或服务器上的项目根目录执行。所有路径均使用仓库相对路径。

## 0. 约定

- 主 smoke seed：`42`
- 3seed：`1,42,88`
- 主输入尺寸：`448`
- 主训练 epoch：`30`
- Co-teaching / JoCoR 双模型默认用 `--batch-size 16 --grad-accum-steps 2`，降低 OOM 风险。
- FINE 主配置使用 `nocenter p=0.6`，因为当前 selector purity 明显强于 `center p=0.6`。

## 0.1 流程总览

FINE 是两步流程：

```text
Step 1: FINE selector
        用 DINOv2 frozen feature 生成 fine_selection_*.csv

Step 2: Static LoRA training
        读取 fine_selection_*.csv 中 state == clean 的样本训练 DINOv2-LoRA
```

Co-teaching 和 JoCoR 都是一步训练流程：

```text
Co-teaching: 直接训练两个 DINOv2-LoRA 网络，mini-batch 内 small-loss 互教
JoCoR:       直接训练两个 DINOv2-LoRA 网络，joint loss small-loss 选择
```

因此执行时不要把 `FINE selector` 当成一次训练结果；它只负责生成选择文件。FINE 的 Top-1 来自第二步 static LoRA training。

## 0.2 学习率控制

所有 LoRA 训练脚本的学习率都由两个参数控制：

```bash
--lora-lr 1e-4
--head-lr 1e-3
```

如果命令里不显式写这两个参数，脚本会先读取 `<input-dir>/resolved_config.yaml` 里的 `lora_train.lora_lr` 和 `lora_train.head_lr`；如果配置里也没有，就使用默认值 `1e-4 / 1e-3`。因此超参优先级仍然是：

```text
命令行 > resolved_config.yaml > 代码默认值
```

双网络 Co-teaching / JoCoR 的命令里默认写的是：

```bash
--batch-size 16 --grad-accum-steps 2
```

这样显存里的单步 batch 变小，但有效 batch size 仍约等于 `32`。这种情况下第一轮不要改学习率，保持和 PGDF / Dynamic 同一套 LoRA LR，更方便公平比较。

如果 448 下仍然 OOM，可以先改成：

```bash
--batch-size 8 --grad-accum-steps 4
```

有效 batch size 仍约等于 `32`，学习率也先不变。只有在无法使用梯度累积、实际有效 batch size 确实变小的时候，才建议降一档学习率，例如：

```bash
--lora-lr 5e-5 --head-lr 5e-4
```

主表第一版建议不显式写 LR 参数，默认继承当前 DINOv2-LoRA 配置；如果后续为了稳定性改 LR，需要在结果表备注。

## 0.3 当前结果状态

截至 2026-06-07，当前 workspace 中已找到以下训练结果：

| 数据集 | FINE static LoRA | Co-teaching | JoCoR |
| --- | --- | --- | --- |
| CUB asym40 | 3seed 已完成，best Top-1 77.87 ± 0.15 | seed42 已完成，best Top-1 61.51 | seed42 已完成，best Top-1 63.42 |
| Stanford Cars asym40 | 3seed 已完成，best Top-1 58.26 ± 0.20 | seed42 已完成，best Top-1 60.92 | seed42 已完成，best Top-1 62.32 |
| FGVC-Aircraft asym40 | 3seed 已完成，best Top-1 60.31 ± 0.13 | seed42 已完成，best Top-1 59.77 | seed42 已完成，best Top-1 59.44 |

对应汇总文件：

```text
outputs/analysis/external_baselines_seed42/external_baseline_seed42_summary.csv
outputs/analysis/external_baselines_seed42/external_baseline_seed42_summary.md
outputs/analysis/external_baselines_seed42/external_baseline_seed42_summary.json
outputs/analysis/fine_3seed_summary/fine_3seed_per_seed.csv
outputs/analysis/fine_3seed_summary/fine_3seed_summary.csv
outputs/analysis/fine_3seed_summary/fine_3seed_summary.md
outputs/analysis/fine_3seed_summary/fine_3seed_summary.json
```

## 1. FINE-DINOv2 两步流程

### 1.1 第一步：FINE-DINOv2 Selector

这一步只生成 selection，不训练 LoRA。

```bash
python tools/run_fine_dinov2_selector.py --input-dir outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42 --noise-index outputs/CUB_200_2011/noise_indices/cub_asym_r0p4_s42_index.csv
```

```bash
python tools/run_fine_dinov2_selector.py --input-dir outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42 --noise-index outputs/Stanford_Cars/noise_indices/stanford_cars_asym_r0p4_s42_index.csv
```

```bash
python tools/run_fine_dinov2_selector.py --input-dir outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42 --noise-index outputs/FGVC_Aircraft/noise_indices/fgvc_aircraft_asym_r0p4_s42_index.csv
```

### 1.2 第二步：FINE-DINOv2 Feature Static Training

主配置为 `nocenter p=0.6`。

```bash
python scripts/run_lora_static_selection.py --input-dir outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42 --selection-file outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42/fine_dinov2/fine_selection_nocenter_p0p6.csv --method-name "FINE-DINOv2 feature nocenter p=0.6" --output-dir outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42/fine_lora_nocenter_p0p6_seed42 --seeds 42 --epochs 30 --input-size 448
```

```bash
python scripts/run_lora_static_selection.py --input-dir outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42 --selection-file outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42/fine_dinov2/fine_selection_nocenter_p0p6.csv --method-name "FINE-DINOv2 feature nocenter p=0.6" --output-dir outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42/fine_lora_nocenter_p0p6_seed42 --seeds 42 --epochs 30 --input-size 448
```

```bash
python scripts/run_lora_static_selection.py --input-dir outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42 --selection-file outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42/fine_dinov2/fine_selection_nocenter_p0p6.csv --method-name "FINE-DINOv2 feature nocenter p=0.6" --output-dir outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42/fine_lora_nocenter_p0p6_seed42 --seeds 42 --epochs 30 --input-size 448
```

`nocenter p=0.6` 的 seed 1/42/88 已完成。以下命令可用于一次性复现 3seed；当前原始结果分别保存在 `fine_lora_nocenter_p0p6_seed1`、`fine_lora_nocenter_p0p6_seed42` 和 `fine_lora_nocenter_p0p6_seed88`：

```bash
python scripts/run_lora_static_selection.py --input-dir outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42 --selection-file outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42/fine_dinov2/fine_selection_nocenter_p0p6.csv --method-name "FINE-DINOv2 feature nocenter p=0.6" --output-dir outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42/fine_lora_nocenter_p0p6_3seed --seeds 1,42,88 --epochs 30 --input-size 448
```

```bash
python scripts/run_lora_static_selection.py --input-dir outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42 --selection-file outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42/fine_dinov2/fine_selection_nocenter_p0p6.csv --method-name "FINE-DINOv2 feature nocenter p=0.6" --output-dir outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42/fine_lora_nocenter_p0p6_3seed --seeds 1,42,88 --epochs 30 --input-size 448
```

```bash
python scripts/run_lora_static_selection.py --input-dir outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42 --selection-file outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42/fine_dinov2/fine_selection_nocenter_p0p6.csv --method-name "FINE-DINOv2 feature nocenter p=0.6" --output-dir outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42/fine_lora_nocenter_p0p6_3seed --seeds 1,42,88 --epochs 30 --input-size 448
```

## 2. Co-teaching-DINOv2+LoRA 一步训练

主配置：`fixed remember_rate=0.8`。

```bash
python scripts/run_lora_coteaching.py --input-dir outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42 --output-dir outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42/coteaching_r0p8_seed42 --seeds 42 --remember-mode fixed --remember-rate 0.8 --warmup-epochs 5 --noise-index outputs/CUB_200_2011/noise_indices/cub_asym_r0p4_s42_index.csv --epochs 30 --input-size 448 --batch-size 16 --grad-accum-steps 2
```

```bash
python scripts/run_lora_coteaching.py --input-dir outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42 --output-dir outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42/coteaching_r0p8_seed42 --seeds 42 --remember-mode fixed --remember-rate 0.8 --warmup-epochs 5 --noise-index outputs/Stanford_Cars/noise_indices/stanford_cars_asym_r0p4_s42_index.csv --epochs 30 --input-size 448 --batch-size 16 --grad-accum-steps 2
```

```bash
python scripts/run_lora_coteaching.py --input-dir outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42 --output-dir outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42/coteaching_r0p8_seed42 --seeds 42 --remember-mode fixed --remember-rate 0.8 --warmup-epochs 5 --noise-index outputs/FGVC_Aircraft/noise_indices/fgvc_aircraft_asym_r0p4_s42_index.csv --epochs 30 --input-size 448 --batch-size 16 --grad-accum-steps 2
```

如果 `fixed r=0.8` 明显偏弱，再跑 schedule oracle-ish 版本：

```bash
python scripts/run_lora_coteaching.py --input-dir outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42 --output-dir outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42/coteaching_schedule_final0p6_seed42 --seeds 42 --remember-mode schedule --remember-rate 0.8 --final-remember-rate 0.6 --warmup-epochs 5 --noise-index outputs/CUB_200_2011/noise_indices/cub_asym_r0p4_s42_index.csv --epochs 30 --input-size 448 --batch-size 16 --grad-accum-steps 2
```

## 3. JoCoR-DINOv2+LoRA 一步训练

主配置：`remember_rate=0.8, lambda_cor=0.1`。

```bash
python scripts/run_lora_jocor.py --input-dir outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42 --output-dir outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42/jocor_r0p8_lam0p1_seed42 --seeds 42 --remember-mode fixed --remember-rate 0.8 --lambda-cor 0.1 --warmup-epochs 5 --noise-index outputs/CUB_200_2011/noise_indices/cub_asym_r0p4_s42_index.csv --epochs 30 --input-size 448 --batch-size 16 --grad-accum-steps 2
```

```bash
python scripts/run_lora_jocor.py --input-dir outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42 --output-dir outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42/jocor_r0p8_lam0p1_seed42 --seeds 42 --remember-mode fixed --remember-rate 0.8 --lambda-cor 0.1 --warmup-epochs 5 --noise-index outputs/Stanford_Cars/noise_indices/stanford_cars_asym_r0p4_s42_index.csv --epochs 30 --input-size 448 --batch-size 16 --grad-accum-steps 2
```

```bash
python scripts/run_lora_jocor.py --input-dir outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42 --output-dir outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42/jocor_r0p8_lam0p1_seed42 --seeds 42 --remember-mode fixed --remember-rate 0.8 --lambda-cor 0.1 --warmup-epochs 5 --noise-index outputs/FGVC_Aircraft/noise_indices/fgvc_aircraft_asym_r0p4_s42_index.csv --epochs 30 --input-size 448 --batch-size 16 --grad-accum-steps 2
```

如果 JoCoR 异常偏弱，先只在 CUB 上扫描 lambda：

```bash
python scripts/run_lora_jocor.py --input-dir outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42 --output-dir outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42/jocor_r0p8_lam0p01_seed42 --seeds 42 --remember-mode fixed --remember-rate 0.8 --lambda-cor 0.01 --warmup-epochs 5 --noise-index outputs/CUB_200_2011/noise_indices/cub_asym_r0p4_s42_index.csv --epochs 30 --input-size 448 --batch-size 16 --grad-accum-steps 2
```

```bash
python scripts/run_lora_jocor.py --input-dir outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42 --output-dir outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42/jocor_r0p8_lam1p0_seed42 --seeds 42 --remember-mode fixed --remember-rate 0.8 --lambda-cor 1.0 --warmup-epochs 5 --noise-index outputs/CUB_200_2011/noise_indices/cub_asym_r0p4_s42_index.csv --epochs 30 --input-size 448 --batch-size 16 --grad-accum-steps 2
```

## 4. 参数含义

通用路径与运行参数：

| 参数 | 适用脚本 | 含义 |
| --- | --- | --- |
| `--input-dir` | 全部 | V1 输出目录，里面应有 `features_cls.npy`、`labels.npy`、`paths.txt`、`eval_paths.txt`、`eval_labels.npy` 和 `resolved_config.yaml` 等文件。 |
| `--output-dir` | 全部 | 当前方法的输出目录；不写时脚本会按默认规则放到 `input-dir` 下。 |
| `--seeds` | 训练脚本 | 训练随机种子，逗号分隔；`42` 是 smoke test，`1,42,88` 是 3seed。 |
| `--noise-index` | FINE selector、Co-teaching、JoCoR | 合成噪声的 GT 索引 CSV，用来统计 selected purity / clean recall；WebFG 没有 GT purity 时不要传。 |
| `--path-map` | 训练脚本 | 路径根目录映射，格式 `OLD=NEW`；当 `paths.txt` 里的图片根目录和当前机器不一致时使用。 |
| `--no-save-checkpoints` | 训练脚本 | 不保存 best checkpoint，只保留日志和结果表，适合快速 smoke test。 |

FINE selector 参数：

| 参数 | 含义 |
| --- | --- |
| `--feature-file` | 相对 `input-dir` 的 feature 文件名，默认 `features_cls.npy`。 |
| `--labels-file` | 相对 `input-dir` 的 noisy label 文件名，默认 `labels.npy`。 |
| `--paths-file` | 相对 `input-dir` 的训练图片路径文件名，默认 `paths.txt`。 |
| `--keep-ratios` | 类内保留比例，默认 `0.6,0.8`；会分别生成 `p0p6` 和 `p0p8` selection。 |
| `--score-modes` | FINE 打分方式，默认 `center,nocenter`；当前优先训练 `nocenter p=0.6`。 |
| `--min-class-size` | 小类保护阈值，默认 `3`；类内样本数小于该值时不做 SVD。 |
| `--small-class-policy` | 小类处理策略，当前只支持 `keep_all`。 |

Static selection / FINE 第二步训练参数：

| 参数 | 含义 |
| --- | --- |
| `--selection-file` | 第一步生成的 selection CSV；训练时只使用其中 `state == clean` 的样本。 |
| `--method-name` | 写入结果表的实验名称，例如 `FINE-DINOv2 feature nocenter p=0.6`。 |

Co-teaching / JoCoR 双网络参数：

| 参数 | 含义 |
| --- | --- |
| `--remember-mode` | 保留率模式；`fixed` 表示 warmup 后固定使用 `remember-rate`，`schedule` 表示 warmup 后逐步下降到 `final-remember-rate`。 |
| `--remember-rate` | small-loss 保留比例；主配置用 `0.8`，和 Dynamic r=0.8 口径一致。 |
| `--final-remember-rate` | schedule 模式最终保留率；oracle-ish 版本可用 `0.6`。 |
| `--warmup-epochs` | 前多少个 epoch 不筛样本，默认主实验用 `5`。 |
| `--lambda-cor` | JoCoR 的 symmetric KL co-regularization 权重；只对 JoCoR 有效，主配置 `0.1`。 |
| `--grad-accum-steps` | 梯度累积步数；双网络默认 `2`，用于在显存 batch 变小时保持有效 batch。 |

LoRA 训练超参：

| 参数 | 含义 |
| --- | --- |
| `--epochs` | LoRA 训练 epoch 数；主实验用 `30`。 |
| `--input-size` | 输入图像尺寸；主实验用 `448`。 |
| `--batch-size` | 训练 dataloader 的单步 batch size；双网络默认 `16`。 |
| `--eval-batch-size` | 验证 dataloader batch size；不写时继承配置或默认值。 |
| `--num-workers` | dataloader worker 数。 |
| `--lora-lr` | LoRA 参数学习率；不写时继承 `resolved_config.yaml`，再没有则默认 `1e-4`。 |
| `--head-lr` | 分类头学习率；不写时继承 `resolved_config.yaml`，再没有则默认 `1e-3`。 |
| `--weight-decay` | AdamW weight decay。 |
| `--rank` | LoRA rank。 |
| `--alpha` | LoRA alpha。 |
| `--dropout` | LoRA dropout。 |
| `--target-modules` | 注入 LoRA 的模块名模式，默认 `qkv`。 |
| `--scheduler` | 学习率调度器，可选 `none`、`linear`、`cosine`。 |
| `--warmup-ratio` | 学习率 warmup 占总训练步数的比例。 |
| `--device` | 运行设备，可选 `auto`、`cpu`、`cuda`。 |
| `--local-repo` | 本地 DINOv2 torch hub repo 路径；服务器无法联网时使用。 |

## 5. 建议执行顺序

1. 三个数据集的 FINE `nocenter p=0.6` seed 1/42/88 已完成；Co-teaching、JoCoR seed42 已完成。
2. FINE 3seed 确认了明显的数据集依赖性：CUB 为 77.87 ± 0.15，Cars 为 58.26 ± 0.20，Aircraft 为 60.31 ± 0.13。
3. PGDF auto 3seed 在 CUB、Cars、Aircraft 上分别领先 FINE 3seed `2.40`、`12.99`、`4.77 pp`；Co-teaching / JoCoR 暂不优先补多 seed。
