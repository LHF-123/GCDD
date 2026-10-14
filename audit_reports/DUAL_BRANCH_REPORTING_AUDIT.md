# Co-teaching / JoCoR 双分支报告规则最终对齐

## 最终判定

**结果规则与投稿 Table 1：CONFIRMED。** 用户确认的历史规则为：

1. 以 A/B 平均 validation Top-1 选择最佳 epoch；
2. 在该 epoch 按 validation Top-1 选择较优 branch；
3. A/B validation 完全相同时固定选择 A；
4. 报告该 branch 的 official-test Top-1；
5. 对 seed 1/42/88 计算 mean 与 sample SD（`ddof=1`）。

按此规则，保存的 18 个 seed 级 A/B validation/test 字段重建投稿 Table 1 六组数值，
均值与 sample SD 在论文显示的两位精度全部一致。此前“投稿数字与当前历史汇总不一致”
的原因已经对齐为**不同报告口径**：投稿使用 validation-selected branch，而 `master`
及其根目录汇总使用 A/B test Top-1 算术平均。

原始论文汇总脚本的 Git 来源则单独判为 **UNVERIFIABLE / 未归档**：用户已确认其实际
规则，审计不应以脚本未找到为由否定该研究事实；但仓库没有可提交地证明“这是投稿时所
执行脚本”的 path、commit 或 blob。

## 原始脚本定位结果

已检查当前工作树、`master`、`big_paper` 的可达 Git 历史，以及包含
`selected_branch`、`validation branch`、`val_a >= val_b`、
`validation_selected_test_model_a_top1` 的脚本／日志文本。

| 发现 | Git / 路径证据 | 结论 |
|---|---|---|
| `master` 双分支实现 | `master:gcdd/lora_noisy_baselines.py:547-555,566-590` | 用 branch-mean validation 选 epoch，报告 A/B test 算术均值（规则 B）；不是投稿的 branch-selected 汇总脚本。 |
| validation-selected 历史实现提交 | `c0f17826640231f3dfba9caa0e8adb08db961f39`，2026-08-01，`validation-selected Test :新增缺少的比对实验` | 包含前述 Rule-B 代码与保存 A/B test 字段；仍不含 branch-selected Table-1 汇总。 |
| 后续 verifier | `big_paper` 的 `966f39e` 新增 `scripts/verify_results.py:349`，其中有 `A if val_a >= val_b else B` | 这是后续验证器，且不在 `master`；不能反向充作投稿期脚本来源。 |
| 原始投稿汇总脚本 | 未找到 | 没有名为 branch-selected／Table-1 exporter 的提交、可达脚本或历史结果文件。应记为“历史代码来源未归档”，而非“规则不存在”。 |

作为可复算的原始数值来源，三个固定 clean validation 根目录仍完整保存：

```
outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42/checkpoint_validation_missing9_s20250726/
outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42/checkpoint_validation_missing9_s20250726/
outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42/checkpoint_validation_missing9_s20250726/
```

每个 `<method>/seed{1,42,88}/result.json` 保存 `best_val_epoch`、
`best_mean_ab_top1`、`best_model_a_top1`、`best_model_b_top1`、
`validation_selected_test_model_a_top1`、`validation_selected_test_model_b_top1` 与
`validation_selected_test_top1`。每个目录亦有 `train_log.csv`、
`selection_policy.json` 和 `checkpoints/best_val.pt`；18/18 个 seed 的这些 required
artifact 均存在。

## `major-revision` 中的新独立汇总脚本

为让论文、结果和以后代码可复现一致，已在独立 worktree 的 `major-revision` 分支新增：

`scripts/summarize_dual_branch_results.py`

该 branch 从 `master` 的 `2e6c93e0b042ca4d9e64914bbdb8eff8241f24cd` 建立；脚本尚未
提交，且仅新增该汇总工具与本审计输出，不修改训练代码、checkpoint 或原始 CSV/JSON。

脚本的可定位实现证据：

- `:76-78` 实现 `val_a >= val_b`，因此 A/B validation 平局固定选 A；
- `:168-208` 要求 `selection_mode=two_branch_mean_validation_top1`，从 `result.json`
  读取 A/B validation/test，核对 `best_val_epoch` 与 `train_log.csv` 的首次最大 mean-val
  epoch，以及 Rule-B mean 字段的算术一致性；
- `:210-235` 写出完整 18 条 seed 选择记录；
- `:238-264` 以 `statistics.stdev`（sample SD）自动核对六组 Table-1 数值，任一显示
  精度不一致即非零退出。

本次只读运行命令为：

```powershell
python scripts\summarize_dual_branch_results.py `
  --dataset-root "CUB-200-2011=D:\实验\GCDD\outputs\CUB_200_2011\CUB-200-2011-asym40-s42\v1_cub_asym_r0p4_s42\checkpoint_validation_missing9_s20250726" `
  --dataset-root "Stanford Cars=D:\实验\GCDD\outputs\Stanford_Cars\Stanford-Cars-asym40-s42\v1_cars_asym_r0p4_s42\checkpoint_validation_missing9_s20250726" `
  --dataset-root "FGVC-Aircraft=D:\实验\GCDD\outputs\FGVC_Aircraft\FGVC-Aircraft-asym40-s42\v1_aircraft_asym_r0p4_s42\checkpoint_validation_missing9_s20250726" `
  --seed-output audit_reports\DUAL_BRANCH_SEED_RECONSTRUCTION.csv
```

运行输出为六项 `PASS`，其未舍入重建值如下：

| 方法 | CUB-200-2011 | Stanford Cars | FGVC-Aircraft |
|---|---:|---:|---:|
| Co-teaching | 62.96744±1.02752 → **62.97±1.03** | 61.10766±.33861 → **61.11±.34** | 60.07601±.12126 → **60.08±.12** |
| JoCoR | 62.24830±1.18353 → **62.25±1.18** | 60.81333±.70339 → **60.81±.70** | 60.26603±.77796 → **60.27±.78** |

逐 seed 选择、A/B validation、A/B test、选中分支、mean-val epoch 平局与源文件路径见
`audit_reports/DUAL_BRANCH_SEED_RECONSTRUCTION.csv`。此前 Cars seed=88 的两个
validation branch 平局现按已确认规则选择 A；该记录不再标为 Missing。

## 与 Rule B 的分离核对

当前历史根目录 `checkpoint_validation_summary.csv` 的 Rule-B 字段
`mean_validation_selected_test_top1` / `std_validation_selected_test_top1` 为：

| 方法 | CUB | Cars | Aircraft |
|---|---:|---:|---:|
| Co-teaching | 62.69±0.74 | 60.74±0.14 | 59.83±0.46 |
| JoCoR | 62.39±1.06 | 60.82±0.49 | 60.05±1.42 |

这些是 A/B test 的算术均值，与投稿 Table 1 不同；它们不构成训练差异、seed 差异或
测试集反向选择的证据。

## checkpoint 和官方测试集时序

每个 `selection_policy.json` 均写明：

- `selection_mode="two_branch_mean_validation_top1"`；
- `validation_samples_are_training_candidates=false`；
- `official_test_used_for_checkpoint_selection=false`。

根目录 `run_summary.md` 同样说明 checkpoint selection 使用 validation Top-1，official
test 在 fitting 后评估。结合用户确认的 branch 规则（只比较该 epoch 的 validation Top-1），
此次对齐后的论文规则不使用 test 选择 epoch 或 A/B branch。保存的 `result.json` 已有
两分支 selected-checkpoint test Top-1，故无需重新执行测试推理来重建表格。

仍应区分：历史结果未保存 prediction hash 以逐字节绑定 `best_val.pt` 和测试值；这是
artifact provenance 的 Partial，不影响已存数值按规则的重建。若未来需独立复核该绑定，
可获批后做 test-only 推理；本轮未执行。

## 修订处置

| 事项 | 结论 |
|---|---|
| 投稿 Table 1 数值 | **无需修改。** 已由用户确认的规则与 18 条保存记录精确重建。 |
| 投稿规则文字 | 保留 Rule A，并在 Appendix/代码说明中明确“validation tie → A”。 |
| `master` 与论文规则 | 仍不一致；`master` 是 Rule B。以后只在 `major-revision` 继续维护 branch-selected 汇总规则及日志字段。 |
| 现有 checkpoint 重评估 | 不需要用于当前表格；A/B selected-checkpoint test 字段已经保存。 |
| Co-teaching/JoCoR 重新训练 | **不需要。** |
| 历史原始脚本来源 | 标为未归档；应在 revision release 中发布新脚本、完整命令、seed CSV 和 source-artifact paths，不能编造原脚本 commit。 |

本轮未修改训练实现、模型 checkpoint、原始实验结果或投稿 Table 1，仅新增可复算汇总脚本
和 branch 内审计输出。等待人工验收后再决定是否提交／合并 `major-revision` 分支。
