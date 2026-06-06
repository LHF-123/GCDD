# SNSCL-DINOv2+LoRA (adapted) 训练命令

该 baseline 独立运行，不与 Dynamic、PGDF、FINE 或 DivideMix 组合。`--noise-index` 仅用于计算机制指标，不参与 reliability、软标签、queue 或其他训练决策。

## CUB 7-epoch smoke

```bash
python scripts/run_lora_snscl.py \
  --input-dir outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42 \
  --config configs/cub_asym40_snscl.yaml \
  --output-dir outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42/snscl_smoke_stable \
  --noise-index outputs/CUB_200_2011/noise_indices/cub_asym_r0p4_s42_index.csv \
  --path-map /root/autodl-tmp/CUB_200_2011=dataset/CUB_200_2011/CUB_200_2011 \
  --seeds 42 \
  --epochs 7
```

验收时检查：

- Epoch 1-5 的 `loss_ntcl`、`loss_kl` 和 queue fill 均为 0。
- `reliability/snscl_seed42_epoch_005.csv` 存在。
- Epoch 6 开始出现 queue 写入，随后出现有效 NTCL anchors。
- loss 均为有限值，gamma 不全部为 0 或 1。
- 提供 noise index 时，`gamma_clean_auc > 0.5` 且 clean gamma mean 高于 noisy gamma mean。

## 正式训练

```bash
python scripts/run_lora_snscl.py --input-dir outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42 --config configs/cub_asym40_snscl.yaml --output-dir outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42/snscl_seed42_stable --noise-index outputs/CUB_200_2011/noise_indices/cub_asym_r0p4_s42_index.csv --seeds 42
python scripts/run_lora_snscl.py --input-dir outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42 --config configs/cars_asym40_snscl.yaml --output-dir outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42/snscl_seed42_stable --noise-index outputs/Stanford_Cars/noise_indices/stanford_cars_asym_r0p4_s42_index.csv --seeds 42
python scripts/run_lora_snscl.py --input-dir outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42 --config configs/aircraft_asym40_snscl.yaml --output-dir outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42/snscl_seed42_stable --noise-index outputs/FGVC_Aircraft/noise_indices/fgvc_aircraft_asym_r0p4_s42_index.csv --seeds 42
```

配置优先级固定为：

```text
CLI > SNSCL method YAML > V1 resolved_config.yaml > code defaults
```

正式输出包括 `snscl_train_log.csv`、`snscl_results.csv`、`snscl_summary.csv/json`、`snscl_queue_stats.csv`、`snscl_reliability_summary.csv`、逐样本 reliability CSV、`resolved_config.yaml`、`run_summary.md` 和 best checkpoint。

## 长跑健康检查与实时日志

训练每完成一个 epoch，立即更新以下 seed 独立文件，无需等待 30 epochs 结束：

- `snscl_seed{seed}_train_log.csv`
- `snscl_seed{seed}_queue_stats.csv`
- `snscl_seed{seed}_reliability_summary.csv`
- `run_status_seed{seed}.json`

默认强制终止条件：

- 任意 batch 的 logits 或 loss 出现 NaN/Inf。
- 连续 2 次 reliability GMM 拟合失败并 fallback。
- 从 Epoch 7 起，queue 仍为空、有效 NTCL anchor 为 0，或 gamma 完全退化。

新 best checkpoint 会在该 epoch 结束时立即保存。健康检查失败时，当前 epoch 的实时日志和失败原因会先写盘，再抛出错误停止训练。

数值稳定性设置：

- DINOv2 主干和分类分支保留 AMP；projection、stochastic sampling、NTCL 和 KL 强制使用 FP32。
- optimizer step 前检查并裁剪梯度，默认 `max_grad_norm=1.0`。
- projection 和 stochastic head 使用独立的 `1e-4` 学习率。
- `label_ma_alpha=0.9`，允许低可靠样本在 30 epochs 内实际改变 corrected hard label。
- `checkpoints/snscl_seed{seed}_latest.pt` 每个健康 epoch 覆盖保存，供失败定位；best checkpoint 仍独立保存。
- 实时训练日志记录梯度范数、`mu/logvar` 范围、参数最大绝对值以及 corrected label 变化数量。

调试时可使用 `--warn-only-health-checks` 只记录失败而不停止；正式实验不建议使用该参数。
