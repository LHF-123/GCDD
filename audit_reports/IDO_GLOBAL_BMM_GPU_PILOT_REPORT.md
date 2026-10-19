# IDO Global-BMM 受控变体与 CUB GPU Pilot 验收报告

## 当前判定

`NOT EXECUTED`

本轮已完成可运行的 `IDO-style global-BMM DINOv2-B/14+LoRA controlled adaptation` 实现与本地 CPU 回归；当前工作区不能访问 Linux GPU 服务器，故没有启动 DINOv2、CUDA、official test 或任何 27 次正式任务。CPU 证据不能替代 GPU pilot 证据。

本变体不是论文 class-wise IDO 的等价复现，不能称为 faithful reproduction、改进版 IDO 或 PGDF+IDO。

## 实际修改

| 路径 | 修改前 | 修改后 | 对 class-wise 分支的影响 |
|---|---|---|---|
| `gcdd/ido.py` | 仅有按 observed-label 分组的 BMM 路由 | 增加显式 `classwise` / `global` 模式和单个全训练池 `GlobalBMM`；记录各轮参数、后验/CDF、软支持与 Kish ESS | 未指定模式仍为 `classwise`；原 class-wise 拟合、fail-closed、20/2 EM 和 WELoss 未变 |
| `scripts/run_lora_checkpoint_validation.py` | 无 BMM 范围参数 | 增加 `--ido-bmm-mode {classwise,global}`，将模式、scope、初始化、epsilon、边界与后验分母写入 resolved config、结果和 D1 元数据 | 无参数时回退到 class-wise；pilot 使用显式 `global` |
| `gcdd/ido_batch.py` | 契约/完成审计未区分 BMM 范围 | BMM 模式和 scope 纳入实验签名、checkpoint/结果/策略验证；global 审计要求每个消耗 BMM 的 Stage-2 轮次恰有一条全池记录 | 现有正式 YAML 明确固定为 `bmm_mode: classwise`，不会混入 global 结果 |
| `configs/ido_batch_all*.yaml` | 未显式声明模式 | 声明 `formal.bmm_mode: classwise` | `protocol_frozen` 保持 `false` |
| `tests/test_ido.py`、`tests/test_ido_batch.py` | 无 global 分支覆盖 | 覆盖单 BMM、全池输入、元数据/签名隔离、WELoss 后验语义、末轮不重拟合和 class-wise 回归 | 仅新增测试 |
| `tests/test_noise_generalization_all_runner.py` | 测试未传已有的两个 keyword-only 参数 | 显式传入 `False` | 仅修复过期测试调用；不改噪声生成逻辑 |

未修改 `gcdd/ido.py` 以外的正式训练数学，也未修改噪声标签、validation 划分、原 class-wise pilot、126 次 IDN/GCCN 调度环境或正式结果。

## Global BMM 定义与责任度口径

- Stage 2 仅对 actual observed training pool 的全部 wrong-event 拟合一个 BMM；不分 observed class，不纳入 validation/official test，也不读 clean/noisy oracle。
- 每次 BMM fit 从固定初值开始：alpha `[1, 2]`、beta `[2, 1]`、mixture weights `[0.5, 0.5]`；epsilon `1e-6`，component boundary `1e-5`；初始 fit 为 20 EM，后续实际消耗的 Stage-2 更新为 2 EM。末轮不会执行无消费者的额外 refit。
- 模型保持非有限、权重塌缩、参数触边和 component order reversal 的 fail-closed 检查；没有新增 shape 下界或其他数值补丁。
- 当前 BMM 的 responsibility/prediction 分母是 `component_density_sum + epsilon`。因此两个 posterior 不强制严格互补；审计中记录 `posterior_mass_deficit`，但不重归一化。WELoss 沿用 `prob1*cdf1 + (1-prob2)*(1-cdf2)` 与 `(1-prob1)`，不会错误地把 `prob2` 当作 `1-prob1`。

## 既有离线 CPU 证据

来源：`outputs_major_revision/analysis/ido_global_bmm_cpu_scan/cub_cyclic_seed1/global_bmm_scan_summary.json`。

| 项目 | 实测 |
|---|---:|
| Stage-1 checkpoint SHA256 | `90454f148b3484a4fcd5150e13474c3c7e850647a3b7914d1e2c9ff9d2bc87cf` |
| training-pool / observed classes | 5,400 / 200 |
| EM iterations / epsilon | 20 / `1e-6` |
| Component 1 alpha / beta / weight / mean | 3.1773419190 / 11.6669693010 / 0.5878280335 / 0.2140444155 |
| Component 2 alpha / beta / weight / mean | 1.1 / 1.7584599666 / 0.4121719665 / 0.3848226013 |
| Component 1 soft assignment / Kish ESS | 3143.8623 / 4393.1285 |
| Component 2 soft assignment / Kish ESS | 2255.1451 / 3738.9709 |

该固定配置的离线全池 fit 参数有限、无 parameter boundary、无 weight collapse、无 component order reversal，posterior/CDF 均在有效区间。两分量 soft assignment 和为 5399.0074，而非 5400；约 0.9926 的差额来自既有 `+ epsilon` 分母语义，不是额外修改，也不会在本轮被强制补齐。

这仅表明 global 方案值得进入一次隔离 GPU pilot；不证明 20 个 Stage-2 epoch 都稳定，更不批准正式 27 次实验。

## 本地 CPU 验证

| 命令 | 结果 |
|---|---|
| `python -m py_compile gcdd/ido.py gcdd/ido_batch.py scripts/run_lora_checkpoint_validation.py` | PASS |
| `python -m unittest discover -s tests -p test_ido.py -v` | 16 PASS |
| `python -m unittest discover -s tests -p test_ido_batch.py -v` | 16 PASS |
| `python -m unittest discover -s tests -p test_noise_generalization_all_runner.py -v` | 6 PASS |
| `python -m unittest discover -s tests -q` | 90 PASS，18.55 s |

测试覆盖 global 只建立一个 BMM、全池输入、无 clean-label oracle、有效 posterior/CDF、责任度 epsilon 语义、WELoss 调用、class-wise 行为保持、模式进入契约/签名、global/class-wise 不可混合汇总、以及末个 Stage-2 epoch 无额外 refit。测试未初始化 CUDA，也未训练 DINOv2。

## 服务器部署与隔离 pilot 命令

先由维护者审核并发布本轮代码；本轮按约束没有 commit/push。服务器先确认处于包含 global-BMM 实现的受审 commit，且没有占用 GPU 的 126 次任务。以下命令只启动一个新目录的 CUB pilot，不运行 batch `--execute`：

```bash
cd /root/GCDD-ido
git status --short
git rev-parse HEAD
nvidia-smi --query-gpu=index,name,memory.free,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
test ! -e /root/GCDD-ido/outputs_major_revision/ido_gpu_pilot/cub_global_seed1_attempt1

python scripts/run_lora_checkpoint_validation.py \
  --input-dir /root/GCDD/outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42 \
  --noise-index /root/GCDD/outputs/CUB_200_2011/noise_indices/cub_asym_r0p4_s42_index.csv \
  --output-dir /root/GCDD-ido/outputs_major_revision/ido_gpu_pilot/cub_global_seed1_attempt1 \
  --methods ido --seeds 1 --epochs 30 --ido-stage1-epochs 10 \
  --ido-bmm-mode global \
  --validation-ratio 0.10 --validation-seed 20250726 \
  --device cuda --revision-profiling-mode basic \
  --revision-output-root /root/GCDD-ido/outputs_major_revision/ido_gpu_pilot_instrumentation/cub_global_seed1_attempt1 \
  --ido-pilot-no-official-test
```

若 `nvidia-smi --query-compute-apps` 显示 126 次任务正在运行或空闲显存不足，应停止，不抢占或中断任务。若输出目录已存在，也应停止而不是覆盖。该命令没有 `--official-test-selected-only`，不会传递 test paths/labels；pilot 会写 `not_run_ido_pilot`，不能通过正式 completion audit 或进入三 seed/27 run 汇总。

## GPU 验收门槛

仅当运行日志证明 Stage 1=10、Stage 2=20、每个实际消耗的 global BMM 成功、loss/gradient 有限、checkpoint 可加载、validation-selected checkpoint 存在、official-test 调用数为 0，且 D1/CUDA/BMM 审计完整时，才可改判 `GPU PILOT PASS`。任一 BMM 触边、塌缩、非有限或提前退出只能记为 `GPU PILOT FAIL` 或 `PARTIAL PILOT`，不得自动改算法重跑。

正式 `protocol_frozen` 仍为 `false`；即使本次 isolated pilot 成功，也不能自动启动 27 次正式训练。

## 本轮文件状态

- 新增本报告：`audit_reports/IDO_GLOBAL_BMM_GPU_PILOT_REPORT.md`。
- 已更新：`audit_reports/IDO_IMPLEMENTATION_ACCEPTANCE_REPORT.md`。
- 未执行 Git add、commit 或 push。
- 完成前应运行：`git diff --check` 与 `git status --short`；预期可见本轮代码、测试、配置和两份报告，且保留用户原有未提交文件。
