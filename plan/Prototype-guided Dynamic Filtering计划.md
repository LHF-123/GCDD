# Prototype-guided Dynamic Filtering 实验计划

记录日期：2026-06-02

## 1. 当前动机

当前结果已经形成明确矛盾：

| 方法 | 强项 | 弱点 |
| --- | --- | --- |
| Dynamic loss r=0.8 | Web-Car、Stanford Cars、FGVC-Aircraft 上明显强 | CUB asym40 明显崩 |
| Centroid / Proto | CUB asym40 稳定 | WebFG 448 平均低于 dynamic loss |
| GCDD / GCDD+Proto | 有诊断价值 | 不是稳定主方法 |

因此下一步不应继续强化 GCDD，而应验证：

```text
用 prototype / centroid 的离线稳定性约束 dynamic loss 的在线高上限。
```

建议方法名：

```text
Prototype-guided Dynamic Filtering
简称 PGDF
```

## 2. 相对原草案的修改

根据当前进度，需要修改三点：

1. CUB 静态 LoRA 30 epoch 已完成，结果仍是 centroid / both only 明显强于 dynamic loss，因此 CUB 是必须保留的失败修复验证集。
2. 合成非对称噪声已经不是“dynamic 一定失败”：Stanford Cars 和 FGVC-Aircraft 上 dynamic r=0.8 明显最强，所以不能把方法目标写成“让 dynamic 在所有合成噪声上都更稳”，而应写成“降低 dynamic 的数据集依赖风险”。
3. 第一轮不要加 soft weight、recovery、GCDD score 融合；先只做 proto gate，验证最小假设。

## 3. 方法定义

原始 dynamic loss 每次更新时按类选择：

```text
loss_rank_i <= r
```

其中 `r=0.8` 表示每个 web label 类别保留 small-loss top 80%。

PGDF v1 增加 class-wise prototype gate：

```text
loss_rank_i <= r
and
proto_rank_i <= p
```

其中：

```text
r = dynamic retention ratio
p = proto_keep_ratio
```

`proto_keep_ratio=0.9` 表示每类只剔除 prototype score 最低的 10%；`proto_keep_ratio=0.8` 表示每类剔除最低的 20%。

prototype score 使用当前已有 centroid / proto filtering 同口径：

```text
s_i_proto = z_i^T m_{web_label_i}
```

只使用 web label 计算，不能使用真实 clean label。CUB / Stanford Cars / FGVC-Aircraft 的 `is_noisy` 只能用于训练后 audit。

## 4. 第一轮实验

优先只跑两个数据集：

| 数据集 | 原因 | 当前关键基线 |
| --- | --- | --- |
| Web-Car 448 | dynamic r=0.8 当前最强，检查 proto gate 是否伤害上限 | dynamic r=0.8 = 88.00, centroid = 86.13 |
| CUB asym40 448 | dynamic r=0.8 明显失败，检查 proto gate 是否能救回来 | dynamic r=0.8 = 64.50, centroid = 75.41 |

第一轮方法：

| 方法 | 说明 |
| --- | --- |
| Dynamic r=0.8 | 已有基线 |
| Centroid | 已有静态强基线 |
| PGDF r=0.8 p=0.9 | 剔除 prototype 最低 10% 的 dynamic clean |
| PGDF r=0.8 p=0.8 | 剔除 prototype 最低 20% 的 dynamic clean |

第一轮先不补齐样本数量。原因是当前问题是判断 proto-low 样本是否有害，直接剔除更容易解释。必须记录每次更新后的实际 selected samples，避免误判为单纯样本数变化带来的收益。

## 5. 输出文件

建议输出目录：

```text
outputs/<dataset>/<v1_dir>/proto_guided_dynamic/
```

建议文件：

```text
pgdf_results.csv
pgdf_summary.csv
pgdf_update_summary.csv
pgdf_selection_r0.8_p0.9_seed42_epoch_005.csv
pgdf_selection_r0.8_p0.8_seed42_epoch_005.csv
pgdf_per_class_summary.csv
run_summary.md
```

`pgdf_update_summary.csv` 至少包含：

```text
method,seed,epoch,
retention_ratio,proto_keep_ratio,
num_candidates,num_loss_selected,num_proto_pass,num_selected,
selected_ratio,
mean_loss_selected,mean_loss_rejected,
overlap_with_previous_selection,
overlap_with_centroid,
proto_reject_count
```

合成噪声数据集额外记录 audit：

```text
selected_clean_precision
selected_noise_ratio
rejected_noise_ratio
```

## 6. 判断标准

### Web-Car 448

目标不是必须超过 dynamic r=0.8，而是确认 proto gate 不明显破坏 dynamic 上限：

```text
PGDF >= 87.5:
    可以继续。

PGDF < centroid 86.13:
    proto gate 伤害过大，v1 不成立。
```

### CUB asym40 448

目标是显著修复 dynamic loss：

```text
PGDF 明显高于 64.50:
    proto gate 有修复价值。

PGDF 接近或超过 75.41:
    方向很有潜力。

PGDF 仍低于 68:
    简单 proto gate 不足，需要停止或改为 soft weight。
```

## 7. 第二轮扩展条件

只有当第一轮满足下面任一条件时再扩展：

```text
1. Web-Car 不明显下降，且 CUB 明显提升。
2. Web-Car 仍高于 centroid，且 CUB 至少提升到 70+。
```

第二轮扩展数据集：

```text
Web-Bird 448
Web-Aircraft 448
Stanford Cars asym40 448
FGVC-Aircraft asym40 448
```

第二轮可加入 soft weight：

```text
Dynamic selected 且 proto-low:
    weight = beta

beta = 0.3, 0.5
```

但 soft weight 不进入第一轮，避免一次性变量太多。

## 8. 实现注意事项

1. 不需要重跑 V1 特征和 proto split，优先复用已有 `features_cls.npy`、`labels.npy`、`paths.txt`、`proto_gcdd/proto_gcdd_scores.csv`。
2. 如果没有 `proto_gcdd_scores.csv`，从 `features_cls.npy` 和 `labels.npy` 重新计算 `centroid_score`。
3. `proto_keep_ratio` 必须按类别内部计算，不能全局筛。
4. 每类至少保留 1 个样本，避免极小类在 gate 后为空。
5. 训练集标签仍使用 web label，CUB 等合成噪声的真实标签只能用于 audit。
6. 超参优先级仍保持：命令行 > YAML 配置文件 > 默认参数。

## 9. 当前结论预期

如果 PGDF 成功，论文主线可以从“GCDD clean selection”转向：

```text
Prototype-guided dynamic clean selection for VFM-LoRA under noisy fine-grained labels.
```

如果 PGDF 失败，当前最稳妥的论文方向仍是：

```text
Rethinking static prototype filtering and dynamic loss filtering under VFM-LoRA.
```

即把 dynamic loss 的数据集依赖性作为分析重点，而不是强行包装一个统一最优方法。

## 10. 第一轮结果更新

当前 CUB 与 Web-Car seed42 结果如下。

### CUB asym40

| method | best Top-1 | final Top-1 | final selected |
| --- | ---: | ---: | ---: |
| Dynamic r=0.8 | 64.50 | 64.36 | 4794 |
| PGDF r=0.8 p=0.9 | 66.31 | 66.31 | 4528 |
| PGDF r=0.8 p=0.8 | 69.66 | 69.31 | 4195 |
| PGDF r=0.8 p=0.7 | 73.39 | 73.21 | 3783 |
| PGDF r=0.8 p=0.6 | 77.56 | 77.41 | 3347 |
| PGDF r=0.8 p=0.5 | 79.70 | 79.70 | 2832 |
| PGDF r=0.8 p=0.4 | 80.51 | 80.46 | 2282 |
| PGDF auto2 r=0.8, J=0.495 -> p=0.4 | 80.08 | 80.03 | 2287 |
| Centroid | 75.41 | 75.34 | 3181 |

结论：

```text
CUB 上 PGDF r=0.8 p=0.4 已超过 centroid。
更严格 proto gate 呈现连续提升趋势：p=0.9 -> 0.8 -> 0.7 -> 0.6 -> 0.5 -> 0.4。
```

### Web-Car 448

| method | best Top-1 | final Top-1 | final selected |
| --- | ---: | ---: | ---: |
| Dynamic r=0.8 | 88.00 | 87.95 | 17085 |
| PGDF r=0.8 p=0.9 | 88.06 | 87.82 | 16770 |
| PGDF r=0.8 p=0.8 | 88.14 | 87.89 | 15879 |
| PGDF r=0.8 p=0.7 | 87.75 | 87.60 | 14189 |
| PGDF r=0.8 p=0.6 | 87.58 | 87.39 | 12273 |
| PGDF auto2 r=0.8, J=0.810 -> p=0.8 | 87.75 | 87.45 | 15665 |
| Centroid | 86.13 | 86.11 | 19394 |

结论：

```text
Web-Car 上 PGDF p=0.8 不伤 dynamic loss 上限，略高于 dynamic r=0.8。
但 p=0.7 和 p=0.6 均低于 p=0.8，说明更强 gate 在 Web-Car 上过筛。
```

### Auto2 自适应 p 结果

Auto2 使用 dynamic clean set 与 prototype/centroid clean set 的初始 Jaccard 自动选择 `p`：

```text
J >= 0.75        -> p = 0.8
0.60 <= J < 0.75 -> p = 0.6
0.50 <= J < 0.60 -> p = 0.5
J < 0.50         -> p = 0.4
```

当前 seed42 结果：

| 数据集 | J | auto p | best Top-1 | final Top-1 | final selected |
| --- | ---: | ---: | ---: | ---: | ---: |
| CUB asym40 | 0.495 | 0.4 | 80.08 | 80.03 | 2287 |
| Web-Bird 448 | 0.563 | 0.5 | 86.90 | 86.69 | 8848 |
| Web-Aircraft 448 | 0.823 | 0.8 | 83.08 | 82.96 | 9708 |
| Web-Car 448 | 0.810 | 0.8 | 87.75 | 87.45 | 15665 |
| Stanford Cars asym40 | 0.489 | 0.4 | 71.57 | 69.97 | 2972 |
| FGVC-Aircraft asym40 | 0.506 | 0.5 | 65.14 | 64.54 | 3090 |

判断：

```text
Auto p 能把低 J 数据集路由到严格 gate，把高 J 数据集路由到宽松 gate，方向符合人工扫参观察。
CUB / Stanford Cars / FGVC-Aircraft 上，auto p 都明显修复或增强 dynamic loss。
Web-Bird 上，auto p 也超过 dynamic r=0.8 和 centroid。
Web-Car auto p 低于手动 p=0.8，需要补多 seed，并检查 auto 与手动运行的调度/选择数量是否完全可比。
Web-Aircraft 是当前主要反例：J 很高且 p=0.8 选择合理，但 PGDF 仍低于 centroid，说明该数据集可能应直接 route 到 centroid。
```

### 下一步

当前最优先不再是固定一个 p，而是确认 gate 强度如何自适应。

```text
CUB asym40: PGDF r=0.8 p=0.3（可选）
Web-Car: 暂停继续扫低 p，当前 p=0.8 为最佳候选
Auto p: 优先补 CUB / Web-Bird / Stanford Cars / FGVC-Aircraft seed1 和 seed88
Method-level routing: 增加 PGDF vs centroid 的数据集级选择规则
```

目的：

```text
CUB p=0.3 用于判断 CUB 是否继续随更严格 gate 上升，还是开始过筛。
Web-Car 已显示 p=0.7/0.6 低于 p=0.8，不建议继续向更低 p 扫描。
```

当前方法结论：

```text
PGDF 方向成立，但固定 proto_keep_ratio 不通用。
CUB 当前最优 p=0.4，Web-Car 当前最优 p=0.8。
Auto p 的 Jaccard 路由已初步成立，但还不能作为最终主结果。
下一步需要验证 adaptive proto gate 的多 seed 稳定性，并增加 PGDF vs centroid 的 method-level routing。
```

如果 adaptive 规则初步成立，再扩展：

```text
Web-Bird 448
Web-Aircraft 448
Stanford Cars asym40
FGVC-Aircraft asym40
```
