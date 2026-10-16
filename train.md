# 正式训练命令

## 动态原型融合（已有实验）

三条命令互相独立：按数据集分别启动。每条均先完成 `pgdf_dynamic_proto` 的三个 seed；只有 reference 成功后才继续对应的三种 fusion 方法。

### 启动 CUB-200-2011

```bash
REF=outputs_major_revision/pgdf_dynamic_ref_cub40;FUS=outputs_major_revision/fusion_dynamic_cub40;python scripts/run_lora_checkpoint_validation.py --input-dir outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42 --noise-index outputs/CUB_200_2011/noise_indices/cub_asym_r0p4_s42_index.csv --output-dir "$REF" --methods pgdf_dynamic_proto --seeds 1,42,88 --epochs 30 --validation-ratio 0.10 --validation-seed 20250726 --dynamic-ratio 0.8 --fixed-p 0.4 --warmup-epochs 5 --update-interval 5 --official-test-selected-only --no-posthoc-oracle-test --device cuda --revision-profiling-mode basic --revision-selection-logging --revision-output-root "${REF}_instrumentation" && python scripts/run_lora_checkpoint_validation.py --input-dir outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42 --noise-index outputs/CUB_200_2011/noise_indices/cub_asym_r0p4_s42_index.csv --output-dir "$FUS" --methods pgdf_dynamic_fusion_weighted,pgdf_dynamic_fusion_union,pgdf_dynamic_fusion_soft --pgdf-budget-root "$REF/pgdf_dynamic_proto" --seeds 1,42,88 --epochs 30 --validation-ratio 0.10 --validation-seed 20250726 --dynamic-ratio 0.8 --fixed-p 0.4 --warmup-epochs 5 --update-interval 5 --official-test-selected-only --no-posthoc-oracle-test --device cuda --revision-profiling-mode basic --revision-selection-logging --revision-output-root "${FUS}_instrumentation"
```

### 启动 Stanford Cars

```bash
REF=outputs_major_revision/pgdf_dynamic_ref_cars40;FUS=outputs_major_revision/fusion_dynamic_cars40;python scripts/run_lora_checkpoint_validation.py --input-dir outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42 --noise-index outputs/Stanford_Cars/noise_indices/stanford_cars_asym_r0p4_s42_index.csv --output-dir "$REF" --methods pgdf_dynamic_proto --seeds 1,42,88 --epochs 30 --validation-ratio 0.10 --validation-seed 20250726 --dynamic-ratio 0.8 --fixed-p 0.4 --warmup-epochs 5 --update-interval 5 --official-test-selected-only --no-posthoc-oracle-test --device cuda --revision-profiling-mode basic --revision-selection-logging --revision-output-root "${REF}_instrumentation" && python scripts/run_lora_checkpoint_validation.py --input-dir outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42 --noise-index outputs/Stanford_Cars/noise_indices/stanford_cars_asym_r0p4_s42_index.csv --output-dir "$FUS" --methods pgdf_dynamic_fusion_weighted,pgdf_dynamic_fusion_union,pgdf_dynamic_fusion_soft --pgdf-budget-root "$REF/pgdf_dynamic_proto" --seeds 1,42,88 --epochs 30 --validation-ratio 0.10 --validation-seed 20250726 --dynamic-ratio 0.8 --fixed-p 0.4 --warmup-epochs 5 --update-interval 5 --official-test-selected-only --no-posthoc-oracle-test --device cuda --revision-profiling-mode basic --revision-selection-logging --revision-output-root "${FUS}_instrumentation"
```

### 启动 FGVC-Aircraft

```bash
REF=outputs_major_revision/pgdf_dynamic_ref_aircraft40;FUS=outputs_major_revision/fusion_dynamic_aircraft40;python scripts/run_lora_checkpoint_validation.py --input-dir outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42 --noise-index outputs/FGVC_Aircraft/noise_indices/fgvc_aircraft_asym_r0p4_s42_index.csv --output-dir "$REF" --methods pgdf_dynamic_proto --seeds 1,42,88 --epochs 30 --validation-ratio 0.10 --validation-seed 20250726 --dynamic-ratio 0.8 --fixed-p 0.4 --warmup-epochs 5 --update-interval 5 --official-test-selected-only --no-posthoc-oracle-test --device cuda --revision-profiling-mode basic --revision-selection-logging --revision-output-root "${REF}_instrumentation" && python scripts/run_lora_checkpoint_validation.py --input-dir outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42 --noise-index outputs/FGVC_Aircraft/noise_indices/fgvc_aircraft_asym_r0p4_s42_index.csv --output-dir "$FUS" --methods pgdf_dynamic_fusion_weighted,pgdf_dynamic_fusion_union,pgdf_dynamic_fusion_soft --pgdf-budget-root "$REF/pgdf_dynamic_proto" --seeds 1,42,88 --epochs 30 --validation-ratio 0.10 --validation-seed 20250726 --dynamic-ratio 0.8 --fixed-p 0.4 --warmup-epochs 5 --update-interval 5 --official-test-selected-only --no-posthoc-oracle-test --device cuda --revision-profiling-mode basic --revision-selection-logging --revision-output-root "${FUS}_instrumentation"
```

## IDN/GCCN 全量实验（126 次训练）

### 前置条件

- 当前分支包含 `cbbd2c6` 及前序 IDN/GCCN 核心提交 `4492a9d`。
- `configs/noise_generalization_all.yaml` 已按本文件上方动态融合命令的既有相对路径填写。其固定 validation manifest 指向 `outputs_major_revision/pgdf_dynamic_ref_{cub40,cars40,aircraft40}`，因此三个 dynamic PGDF reference 输出必须完整存在。
- 三个数据集均已有完整 V1 输入、对齐的 source noise-index CSV、固定干净 validation manifest，以及服务器可访问的图像路径；若服务器图像根不同，才修改该 YAML 的 `path_maps`。
- Python 环境具备 CUDA PyTorch、仓库 `requirements.txt` 依赖、可解析的本地 DINOv2 仓库/缓存与权重。
- 三个新的输出根目录（noise assets、training、instrumentation）可写且互不重叠。

正式配置已在 `configs/noise_generalization_all.yaml`；`formal` 块是锁定的正式参数，不要修改。

### 正式启动命令（唯一）

```bash
python tools/run_noise_generalization_all.py --config configs/noise_generalization_all.yaml --resume
```

这条命令顺序生成 6 组 IDN/GCCN 噪声资产，并完成 CUB、Cars、Aircraft × IDN/GCCN × 7 方法 × seeds 1/42/88 共 126 次训练。主 PGDF 是 `pgdf_dynamic_proto`，不是 `pgdf_fixed`。

首次运行前可选预检（不启动 GPU 训练）：

```bash
python tools/run_noise_generalization_all.py --config configs/noise_generalization_all.yaml --dry-run
```

中断后仍使用同一条带 `--resume` 的正式启动命令；仅通过结果、checkpoint、epoch、噪声与验证协议审计的任务会被跳过。仅重建汇总：

```bash
python tools/run_noise_generalization_all.py --config configs/noise_generalization_all.yaml --summarize-only
```
