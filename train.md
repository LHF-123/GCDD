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
- `configs/noise_generalization_all.yaml` 复用上方命令中的 V1 输入与 source noise-index，但不依赖动态融合或任何历史训练输出。调度器会在 `outputs_major_revision/noise_generalization/fixed_validation/<dataset>/` 独立创建或校验固定 clean validation manifest，并将同一份 manifest 写入 IDN/GCCN 的训练根目录。
- 三个数据集均已有完整 V1 输入、对齐的 source noise-index CSV，以及服务器可访问的图像路径；若服务器图像根不同，才修改该 YAML 的 `path_maps`。
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

## IDO-style DINOv2-LoRA controlled adaptation（待独立预检，未启动训练）

该实现接入固定 clean-validation 协议，但必须表述为
`IDO-style DINOv2-B/14+LoRA controlled adaptation`，不是 IDO 官方
ViT-B/16、ResNet 或 ConvNeXt 的原版等价复现。

- BMM：按 observed-label class 拟合，与论文主方法和官方 `stage_2_class.py` 一致；不使用官方启动脚本实际调用的全局 BMM 路径。
- BMM 退化：类别 wrong-event 全相同、非有限、component collapse、参数边界，或 EM 的两个分量发生严格均值反转时，记录类别、阶段、原因、统计量和可用的最后 BMM 参数，然后 **fail-closed 中止该 run**；不静默重排分量、改用全局 BMM，也不把 observed label 当作 clean-safe 回退。任何替代退化策略均须另行人工批准。
- Label-wave：Stage 1 候选 checkpoint 均为 LoRA/head 张量的 deep clone，避免历史快照与后续优化共享存储。
- 损失：保留官方 `utils/lnl_methods.py::WELoss` 的双视图 CE、logit MSE 与 confidence-weighted entropy。官方源码将 `confidence.unsqueeze(1)` 与 `[B]` entropy 相乘，导致跨样本 `[B,B]` 广播；本适配改为逐样本 `[B]`，因此准确名称是 `official-source WELoss adaptation with per-sample entropy broadcast fix`，不可称未修改 official reproduction，也不可替换为 BMM-weighted CE。
- 预算：30 epoch 时按官方 Stage 1:Stage 2 = 5:10 固定比例缩放的候选为 10+20；这仍是 **PROTOCOL APPROVAL REQUIRED**，不是已冻结正式协议。总 epoch 不是 3 的倍数时，必须显式以 `--ido-stage1-epochs` 预注册 Stage 1 预算。

单数据集、单 seed 的预检/正式入口（按实际数据集替换路径；不与正在执行的 IDN/GCCN 调度器共享输出目录）：

```bash
python scripts/run_lora_checkpoint_validation.py --input-dir <v1_input_dir> --noise-index <synthetic_noise_index.csv> --output-dir outputs_major_revision/ido_controlled_<dataset> --methods ido --seeds 1 --epochs 30 --validation-ratio 0.10 --validation-seed 20250726 --official-test-selected-only --no-posthoc-oracle-test --device cuda
```

每个 `seed*/` 会额外落盘 `stage1_wrong_event.csv`（不含 clean-label oracle）和
`bmm_class_fits.csv`（每个 observed class、每轮 BMM 的状态及 fail-closed 原因）。解析后的
`result.json` 还区分完整 Stage 1/2 成本、optimizer-loop、BMM、validation、official-test、checkpoint
及 end-to-end D1 时间；其中 `d1_total_wall_seconds` 仅为 end-to-end 兼容别名，不能当作纯优化时间。

在不访问 official test 的单 seed GPU pilot 中，必须显式使用下列 IDO 专用安全门；该输出不能作为
正式批量完成结果或纳入三 seed 汇总：

```bash
python scripts/run_lora_checkpoint_validation.py --input-dir <v1_input_dir> --noise-index <synthetic_noise_index.csv> --output-dir <isolated_pilot_output> --methods ido --seeds 1 --epochs 30 --validation-ratio 0.10 --validation-seed 20250726 --no-posthoc-oracle-test --ido-pilot-no-official-test --device cuda --revision-profiling-mode basic --revision-output-root <isolated_pilot_d1_output>
```

正式批量任务为每一个 `ido/seedN` 写入不可变 `ido_task_contract.json`；完成判定会校验该契约、输入/验证划分/源码 SHA256、结果 schema、official-test 模式，以及三个可读取且 metadata 一致的 checkpoint。CPU 层还要求 checkpoint 仅含 DINOv2 `backbone.*.lora_*` 与 `head.*` 的有限 tensor、三份 key/shape layout 一致、分类 head 输出维度与 V1 observed-label 类别数一致；这不是实际 DINOv2 `load_state_dict` 的替代，后者仍由 GPU pilot 验收。
`resolved_config.yaml` 固化架构差异、两阶段预算、BMM 退化策略和 WELoss 来源。

完整矩阵只可通过显式执行授权启动；默认是不会启动训练的 dry-run：

```bash
python tools/run_ido_cyclic_all.py --config configs/ido_batch_all.yaml --dry-run
python tools/run_ido_noise_generalization_all.py --config configs/ido_batch_all.yaml --dry-run
python tools/run_ido_all.py --config configs/ido_batch_all.yaml --dry-run
```

`configs/ido_batch_all.yaml` 默认 `protocol.protocol_frozen: false`。只有人工冻结协议、在目标服务器通过 preflight 后，才可使用 `--execute`。预检会逐样本核对 V1 `paths.txt`/`labels.npy` 与 noise index 的 path/observed label；IDN/GCCN 还核对 `source_train_index`。中断任务不会覆盖或伪恢复，必须以 `--new-attempt --execute` 创建隔离 attempt；新 attempt 只会复用同一冻结 experiment signature 下唯一已完整验收的 seed，重跑未完成 seed。若发现多个已完成重复 seed，调度器会 fail-closed，而不会按成绩或目录顺序选择。
