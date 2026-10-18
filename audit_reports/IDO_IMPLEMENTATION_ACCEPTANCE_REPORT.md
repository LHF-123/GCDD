# IDO-style DINOv2-B/14+LoRA 实现独立验收报告

> **修复复验附录（2026-10-17）。** 下文“审查范围、边界与结论”至文末保留的是修复前的独立验收记录，故其中的 `REVISION REQUIRED`、`NOT IMPLEMENTED` 与旧 `clean_safe` 描述是历史发现，不能当作当前实现状态。本附录记录本轮实际修复、重新执行的 CPU 证据和仍然有效的阻塞条件；没有删除或改写原始发现。

## 修复复验（当前状态）

### 当前总体验收：**PASS WITH CONDITIONS**

已修复并以源码和 CPU 测试复验的已知工程阻塞包括：WELoss 跨样本广播、`clean_safe` 静默目标改写、BMM component-collapse 诊断、Stage-1 wrong-event 证据命名、IDO D1 记录，以及 9+18+27 批量调度能力。当前实现应称为 **`IDO-style DINOv2-B/14+LoRA controlled adaptation`**；WELoss 更准确地称为 **`official-source WELoss adaptation with per-sample entropy broadcast fix`**，不是未修改的官方复现。

这不是“正式 27 次训练 READY”的无条件结论。正式执行仍被下列条件阻塞：

1. **PROTOCOL APPROVAL REQUIRED：** 30 epoch 的 10+20 只是 5:10 官方预算比例的候选缩放，尚未人工冻结；配置明确保持 `protocol_frozen: false`。
2. **SERVER PREFLIGHT REQUIRED：** 本工作树缺少 9 个 dataset/protocol V1 输入契约；实际 `--dry-run` 计划了 27 个任务，但 preflight 报告 `blocked=9`，没有启动训练子进程。IDN/GCCN 的服务器资产和与 126 次任务的 SHA256 一致性尚未在目标服务器实测。
3. **BMM FAIL-CLOSED RISK：** 常量、非有限、component collapse 与接近数值边界的 class-wise BMM 现在会按要求中止而不改写损失。小类/离散 wrong-event 是否会在真实 10+20 轨迹触发该策略，必须由批准后的最小 GPU smoke 观察；不得擅自加入 pseudo-count、global BMM 或中性权重。
4. **GPU D1 UNVERIFIED：** CPU mock 已验证字段和汇总路径，但 CUDA peak allocated/reserved 和真实 DINOv2+LoRA 计时仍须 GPU smoke 验收。

### 第三轮独立复核后的整改增补（2026-10-17）

下表保留本报告原有发现，并记录针对第二轮外部源码复核（`IDO-001` 至 `IDO-008`）的当前整改状态。该增补不改变“10+20 尚待人工冻结、服务器资产尚未预检、真实 GPU smoke 尚未执行”的结论。

| 外部复核项 | 当前状态 | 实际整改与 CPU 证据 | 仍然限制 |
|---|---|---|---|
| IDO-001：仅凭文件存在误判完成 | PASS ON CPU | `gcdd/ido_batch.py::is_complete/validate_checkpoints` 现在要求不可变 task contract、结果/策略 metadata、连续日志、fitted BMM、有限且范围合法的 validation-selected official-test 指标，并以 `torch.load(..., map_location="cpu")` 验证 `stage1_base.pt`、`best_val.pt`、`last.pt` 的 state_dict、seed、类别、预算、best epoch/score 与契约签名。`tests/test_ido_batch.py::test_completion_rejects_corrupt_checkpoint_missing_score_and_signature` 覆盖文本伪 checkpoint、缺失 score 和签名不一致。 | 尚无真实 DINOv2 checkpoint 的 GPU/服务器证据。 |
| IDO-002：不同 attempt 可混合三 seed 汇总 | PASS ON CPU | 每个 seed 写入包含输入、验证划分、批量 YAML 与运行源码 SHA256 的 `ido_task_contract.json`；汇总按 `ido_experiment_signature` 分组，并将同一签名下的重复 seed 标为冲突。`test_summary_never_merges_a_seed_with_a_mismatched_experiment_signature` 证明跨签名不会产生 `COMPLETE_3_SEEDS`。 | 只有签名逐项一致的显式新 attempt 才可合并；正式运行仍需人工冻结协议。 |
| IDO-003：末轮无消费者的 BMM refit | PASS ON CPU | `gcdd/ido.py::train_ido_lora` 只在 `stage_epoch < resolved_stage2_epochs` 时刷新 BMM；末轮写审计日志但不再因无后续 loss 使用的诊断拟合而 fail-closed。CPU smoke 断言 1 个 initial fit + 1 个非末轮 refresh。 | 这是训练控制流修订；真实 BMM 稳定性仍未由 GPU 数据证明。 |
| IDO-004：BMM 合法离散输入的稳定性 | PASS WITH CONDITIONS | 常量/非有限/权重塌缩/参数边界继续 fail-closed；新增严格分量均值反转检测，防止 EM 标签交换倒置 `prob1/prob2` 的 WELoss 语义。`test_bmm_component_label_swap_is_fail_closed` 覆盖该路径。 | 小类离散 wrong-event 的实际退化频率仍是 GPU pilot 阻塞条件；不得未批准地加入 global/pseudo-count/clean-safe 回退。 |
| IDO-005：预检未统计实际 training pool | PASS ON CPU | `inspect_input_contract` 先读取不可变 validation manifest，再以 `partition == training_pool` 统计样本数、min/P5/median/P95/max/empty；source-pool 数字仅保留为对照。批量 dry-run fixture 输出两套明确字段。 | 服务器的 9 组真实资产尚未核验。 |
| IDO-006：D1 时间边界含混 | PASS ON CPU / GPU UNVERIFIED | 结果现区分 `d1_optimizer_loop_wall_seconds`、stage1/stage2 完整阶段成本、`d1_bmm_fit_seconds`、`d1_validation_seconds`、`d1_official_test_seconds`、`d1_checkpoint_seconds`、`d1_end_to_end_wall_seconds`。旧 `d1_total_wall_seconds` 保留为 end-to-end 兼容别名，文档禁止将其称为纯优化时间。 | CUDA 峰值与真实吞吐仍需 GPU smoke；跨方法比较必须使用同一边界。 |
| IDO-007：结果缺少 IDO 源码/配置签名 | PASS ON CPU | task contract 包含 `gcdd/ido.py`、`gcdd/ido_batch.py`、入口、LoRA、数据/特征、validation、profiling 等运行依赖 SHA256；`build_code_provenance` 也记录 IDO 模块 SHA256。completion audit 会重算并比对。 | 旧 attempt 不带新契约，按 fail-closed 视为不完整，不可直接汇总。 |
| IDO-008：第二轮源码包遗漏依赖 | PENDING REPACKAGE | 后续独立复核包必须加入 `gcdd/data.py` 与 `revision_experiments/offline_diagnostics.py`，以及本轮变更后的完整依赖闭包与 manifest；旧 `IDO_SECOND_REVIEW_BUNDLE.zip` 保持为历史审查对象，不被覆盖。 | 需要在本轮最终 CPU 回归后生成新的、另名的第三轮包。 |

另新增 `--ido-pilot-no-official-test`：它仅允许 `--methods ido`、单 seed、无 batch task contract、无任何 official-test 选项，并向 IDO trainer 传入 `None` test loader。该 pilot 输出显式标为 `not_run_ido_pilot`，不能通过正式 completion audit 或进入三-seed 汇总。

**本轮实际 CPU 证据（未调用 GPU、未启动真实训练）：**

| 命令 | 结果 | 说明 |
|---|---|---|
| `python -m unittest discover -s tests -p test_ido.py -v` | 14 PASS，0 fail/error/skip | WELoss、BMM 退化/分量反转、Stage 1/2、D1 字段、无 official-test pilot 安全门。 |
| `python -m unittest discover -s tests -p test_ido_batch.py -v` | 6 PASS，0 fail/error/skip | 伪 checkpoint/score/signature 负例、contract 路径闭环、跨 attempt 不混合、training-pool 预检、27 task dry-run。 |
| `python -m unittest discover -s tests -v` | 78 PASS，0 fail/error/skip | 全仓 CPU 回归；不构成 GPU 或真实 BMM 稳定性证据。 |
| `python tools/run_ido_all.py --config configs/ido_batch_all.yaml --dry-run` | `planned=27, blocked=9` | 仅本地缺失 9 个 V1 输入契约；未启动训练子进程。 |

### 修复矩阵

| ID | 修复项 | 修复前 | 当前实际行为与证据 | 状态 | 剩余影响 |
|---|---|---|---|---|---|
| R1 | WELoss `[B,B]` 广播 | 官方 `conf.unsqueeze(1)` 与 `[B]` entropy 相乘，跨样本混合 | `gcdd/ido.py:512-571` 强制所有 BMM 项和所有三项 loss 为 `[B]`，confidence 保持 `[B]`；`tests/test_ido.py:91-157` 覆盖 B=1/2/8、人工参考、置换不变、其他样本不交叉加权、有限梯度 | PASS | 是受控 WELoss 修正，不能称原样 official source |
| R2 | BMM clean-safe | 退化类返回 `p1=1,p2=0,cdf=.5`，隐式信任 observed label | `gcdd/ido.py:69-137,196-305,344-415` 仅构造含类别/epoch/reason/statistics/可用参数的 `BMMDegenerateError` 并中止；`scripts/run_lora_checkpoint_validation.py:526-537` 落盘 `bmm_class_fits.csv` 和 `bmm_degeneracy.json` | PASS | 真实小类若退化，run 会 BLOCKED，而非产生成绩 |
| R3 | BMM 边界/塌缩 | 近 `epsilon` alpha/beta 或塌缩权重仍可能被标作 fitted | `gcdd/ido.py:228-238` 分别检测非有限 shape、非有限 weight、component weight collapse、parameter boundary，并保留最后参数；`tests/test_ido.py:33-60` 覆盖全零、常量非零、极偏斜、双值、n=24、NaN、normal fit 和受控 component collapse | PASS | 阈值本身是 fail-closed 数值保护，不是替代 BMM 策略 |
| R4 | Stage-1 统计证据 | `stage1_wrong_event.csv` 在 Stage 2 后写出，文件名与内容不一致 | `gcdd/ido.py:737-740` 在 Stage-1 边界 copy；`scripts/run_lora_checkpoint_validation.py:539-544` 写该副本；`tests/test_ido.py:286-289` 比较其与 `stage1_base.pt`，并确认最终计数单调继承 | PASS | 仅支持审计 checkpoint，不支持中断恢复 |
| R5 | Stage-1 snapshot / 阶段状态 | 原官方快照共享 state_dict 风险；无明确恢复语义 | `gcdd/ido.py:435-474,1071-1210,1354-1389` clone LoRA/head，纯 label-wave 状态机保存首个局部最小候选，保存 selected Stage-1 state、wrong-event、seed、budget 和 `resume_support=not_implemented_fail_closed_new_attempt_required`；CPU 覆盖 Stage 1→2 与局部最小/epoch-1 fallback | PASS WITH CONDITIONS | 仍不提供 epoch 级 resume；这是刻意 fail-closed，部分任务须新 attempt |
| R6 | IDO D1 | 没有 Stage 1/2/BMM 分段记录 | `revision_experiments/profiling.py:38-47,420-427` 新阶段；`gcdd/ido.py:452-509,616-1016` 记录 stage wall-time、BMM、steps、samples、额外 view forward、validation、CUDA peaks；入口 `scripts/run_lora_checkpoint_validation.py:493-567` 写 isolated profiler 输出和 result fields | PASS ON CPU / GPU UNVERIFIED | 不将 30 epoch 等同于相同计算成本 |
| R7 | 27 次调度 | 三个指定脚本不存在 | `tools/run_ido_cyclic_all.py`、`tools/run_ido_noise_generalization_all.py`、`tools/run_ido_all.py` 共用 `gcdd/ido_batch.py:109-509`；`configs/ido_batch_all.yaml` | PASS WITH CONDITIONS | 正式 `--execute` 仍要求人工冻结及服务器 preflight |
| R8 | 126 次 IDN/GCCN 隔离 | 无 IDO 批量器 | 新调度器仅调用单次入口的 `--methods ido`；`scripts/run_lora_checkpoint_validation.py:65-84,1531-1554` 保持 `--methods all` 的 13-method 集合不变；`tests/test_ido.py:186-190` 验证 | PASS | 未修改原 126 调度器 |

### 关键代码差异与理由

**1. WELoss。** 官方 `utils/lnl_methods.py:73-80` 的 `conf.unsqueeze(1)` 使 `[B,1] * [B]` 形成 `[B,B]`。当前实现保留官方 logits-MSE 与
`p1*cdf1 + (1-p2)*(1-cdf2)` consistency weight，但把 entropy 的 confidence 保持为 `[B]`：

```python
# gcdd/ido.py:544-561
mse = functional.mse_loss(output_s, output_w, reduction="none").mean(dim=1)
mse_weight = prob_tensor * cdf1_tensor + (1.0 - prob2_tensor) * (1.0 - cdf2_tensor)
y_pred_avg = (y_pred_s + y_pred_w) / 2.0
confidence = torch.max(y_pred_avg, dim=1)[0]
entropy = -confidence * torch.sum(y_pred_avg * torch.log(y_pred_avg), dim=1)
terms = {
    "ce_per_sample": prob_tensor * (ce_s + ce_w),
    "mse_per_sample": mse_weight * mse,
    "entropy_per_sample": (1.0 - prob_tensor) * entropy,
}
```

所有返回项在 `gcdd/ido.py:535-539` 被检查为严格 `[B]` 和 finite；最终才分别 mean。该改动是修复明确的跨样本广播，不根据准确率选择损失定义。

**2. BMM。** 不再有返回 posterior/CDF 的退化 fallback；`ClasswiseBMM.fit/predict` 在 observed-label class 级别附加 context 后抛错：

```python
# gcdd/ido.py:358-368
values = wrong_event[label_ids == label_id]
try:
    metadata = self.models[label_id].fit(values, max_iter=max_iter)
except BMMDegenerateError as exc:
    raise exc.add_context(
        observed_label_id=label_id,
        observed_label=observed_label,
        stage2_epoch=stage2_epoch,
        fit_round=fit_round,
    ) from exc
```

因此退化样本的 clean/noisy/consistency loss 不存在“默认权重”：训练在损失计算前终止。它既不使用 clean/noisy oracle，也不自动切换 global BMM；这是受控 fail-closed 数值策略，不是将退化类别视为 clean 的算法适配。

**3. Stage 1/2 时序。** Stage 1 以 original index 更新一次；Stage 2 与官方 `stage_2_class.py` 一样，弱、强两个视图各更新一次，再以当前 batch 的累计计数查询上一轮拟合的 BMM，epoch 末 refit：`gcdd/ido.py:821-823,902-911,1098-1135`。数据 shuffle 不改变 original index 对齐。label-wave 不读取 validation/official test；deep clone 与 Stage-1 boundary checkpoint 的 CPU 证据见 `tests/test_ido.py:173-190,263-297`。

**4. D1。** result 字段包含 Stage 1/2 wall-clock、BMM time、total time、optimizer attempted/actual steps、CUDA allocated/reserved peak、两阶段样本数、Stage-2 extra view forwards 和 validation time：`gcdd/ido.py:452-509`；没有 profiling 时明确写 `off`，绝不填造数值。

### 批量能力、preflight 与恢复策略

`gcdd/ido_batch.py` 使用固定 task ID `<dataset>__<protocol>__ido__seed<seed>` 和 attempt 隔离目录。`build_plan`（218-231）生成 cyclic 9、IDN/GCCN 18、完整 27；`main_for_protocols`（109-153）默认 dry-run、仅 `--execute` 才会调用子进程，且要求 YAML `protocol_frozen: true`。`inspect_input_contract`（264-323）只读检查：V1 文件、observed labels 的 class count/min/P5/median/P95/max/empty、noise strategy、noise-index SHA256、CSV/JSON validation-manifest SHA256、全池 index/path 一致性、训练/validation 分区互斥、path map，以及 IDN/GCCN 与 126 次 reference resolved config 的 SHA256 一致性。

`is_complete`（421-468）不把仅有权重文件的任务误判完成：必须有 result/log/selection policy、Stage-1 wrong-event、BMM fits、Stage-1 base/best/last checkpoints、未变动的 canonical manifests、连续 1..epochs log、fitted BMM、fail-closed policy、D1 complete 与匹配的阶段预算。`summarize_all`（471-509）仅在三 seed 都通过该审计后写 mean 和 **sample SD**。

恢复策略是刻意保守：没有保存完整 epoch-level optimizer/scheduler/scaler/RNG/BMM/label-wave 状态，故不声称可恢复训练。任一不完整 run 立即 BLOCKED，不能覆盖；只有显式 `--new-attempt --execute` 创建新的 `attempt-NNN`。`--summarize-only` 只读汇总，不启动训练。

目标服务器仅做预检的命令（不启动 GPU）：

```bash
python tools/run_ido_all.py --config /absolute/path/to/ido_batch_all.yaml --dry-run
```

将服务器路径和必要 `path_maps` 填入单独复制的 YAML 后，预检必须为 `planned=27, blocked=0`。仅在人工冻结 10+20、审阅 WELoss/BMM 策略、完成最小 GPU smoke 后，才可把该 YAML 的 `protocol.protocol_frozen` 显式改为 `true` 并使用 `--execute`。

### 本轮实际测试证据

| 命令 | 结果 | 覆盖 |
|---|---|---|
| `python -m unittest discover -s tests -p test_ido.py -v` | 12 PASS, 0 fail/error/skip | WELoss B=1/2/8/manual/permutation/finite grad；BMM 常量、NaN/Inf、偏斜、双值、n=24、component collapse；snapshot；label-wave 局部最小/fallback；Stage 1→2；D1 CPU mock；`all` 不含 IDO |
| `python -m unittest discover -s tests -p test_ido_batch.py -v` | 3 PASS, 0 fail/error/skip | 9/18/27 task counts；dry-run 无 subprocess；冻结门；incomplete/manifest mismatch 不算完成；sample SD |
| `python -m unittest discover -s tests -v` | **73 PASS, 0 fail/error/skip** | 全仓 CPU 回归；包含旧 126-run scheduler 的测试 |
| `python tools/run_ido_all.py --config configs/ido_batch_all.yaml --dry-run` | `planned=27, blocked=9; no training subprocess was started` | 真实配置的安全 dry-run；本地 V1/IDN/GCCN 资产缺失被如实 BLOCKED |

CPU 结果不是 DINOv2 GPU 训练、服务器 asset 或正式 27-run 成绩的替代证据。

### 仍需人工/服务器处理的清单

1. 冻结或否决 10+20，连同 config hash 和不允许临时 `--ido-stage1-epochs` override 的规则写入正式协议。
2. 在服务器仅执行上述 dry-run，核验全部 9 个 dataset/protocol 输入契约；尤其确认 IDN/GCCN SHA256 与原 126-run 一致。
3. 在正式 27 次前做一个已批准的最小 GPU smoke，确认 DINOv2+LoRA、CUDA D1 fields、BMM 不退化和 validation-selected checkpoint；禁止据 official-test 调参。
4. 若真实离散 wrong-event 触发 fail-closed，不得在本分支偷偷改为 pseudo-count/global/clean-safe；必须单独决定是中止该协议还是预注册新的算法适配。

## 审查范围、边界与结论

**审查日期：** 2026-10-17
**工作树：** `D:\实验\GCDD\.worktrees\major-revision`
**分支：** `major-revision`
**审查性质：** 只读源码、配置、已有资产和 CPU 测试验收。本报告没有修改 `gcdd/ido.py`、训练入口、噪声资产、validation manifest、126 次 IDN/GCCN 调度器、论文或审稿回复；没有启动 GPU、真实数据训练或正式 27 次实验。

初始工作树状态中已存在以下修改/未跟踪项：`.gitignore`、`plan/当前实验结果.md`、`requirements.txt`、`scripts/run_lora_checkpoint_validation.py`、`train.md`、`gcdd/ido.py`、`tests/test_ido.py`，以及与本次 IDO 无关的既有输出、分析脚本、审稿回复和审计材料。初始 `git diff --check` 无空白错误（仅报告 `train.md` 的 LF→CRLF Git 警告）。本次审查新增的唯一文件是本报告。

### A. 总体验收结论：**REVISION REQUIRED**

当前代码不是“原版等价复现”，而是源码明确标注的 **`IDO-style DINOv2-B/14+LoRA controlled adaptation`**。单次运行路径中，稳定 sample ID、Stage 1/2 wrong-event 累计、observed-label class-wise BMM、深拷贝 snapshot、官方源码风格 WELoss、验证集选 checkpoint，以及 official-test 后置调用均有实际源码证据；CPU 测试也能执行该小型路径。

但不能验收为正式实验 READY，原因如下：

1. 30 epoch 的 **10+20** 仅是“按官方 5:10 比例缩放”的代码默认值，尚未获得题述要求的人工协议批准；它必须保持 **PROTOCOL APPROVAL REQUIRED**，不能以 CPU/official-test 结果倒推确认。
2. `clean_safe` fallback 将退化类别当成 observed label 可靠样本（CE 权重 1、entropy 0、MSE 1）；这不是中性的数值保护，而是 **实质性算法变更**，尚无预注册/人工批准。
3. WELoss 保留了官方源码的 batch 广播行为：entropy 中 `[B,1] * [B] -> [B,B]`。这可称为 `official-source WELoss adaptation`，但不能同时声称逐样本论文公式完全一致；本审查未取得论文 Eq. (5)/(8) 的可直接逐式核对文本，故论文等价性为 **UNVERIFIED**。
4. BMM 对常量/非有限输入可 fallback，但对两点和极端不平衡输入会将 alpha/beta 推到接近 `1e-6` 仍标记 `fitted`；没有 component-collapse/近边界参数检测。
5. 指定的三个 27 次调度脚本均不存在，故 cyclic 9、IDN/GCCN 18 和合计 27 都是 **NOT IMPLEMENTED**。
6. IDO 未接入 D1 profiler，无法给出分阶段时间、BMM 时间、optimizer steps、峰值 CUDA allocated/reserved 或额外 forward 计量；Stage 1 base snapshot/optimizer/scheduler 也不能持久恢复。

因此，在最小 GPU smoke 前应先取得协议与算法适配裁决；在正式 27 次训练前还必须实现、测试和预检专用调度与 D1 计量。

状态含义：PASS=已有充分源码/测试证据；PARTIAL=局部实现但缺少必要部分或覆盖；UNVERIFIED=本审查没有充分证据；NOT IMPLEMENTED=不存在；`PROTOCOL APPROVAL REQUIRED`=代码可解析但尚非获批实验协议。

## B. 逐项审查矩阵

| ID | 检查项 | 状态 | 实际代码路径与行号 | 证据 | 问题严重程度 |
|---|---|---|---|---|---|
| A1 | Stage 1 stable ID 与 wrong-event 累计 | PASS | `gcdd/lora_training.py:49-66`; `gcdd/ido.py:459-471, 490, 760-782` | dataset 返回原始 `idx`；shuffle 后以 `original_indices` 更新全长 `wrong_event`；只比较传入的 `train_labels` | — |
| A2 | label-wave 判定、无局部最小值处理 | PARTIAL | `gcdd/ido.py:752-805, 822-835` | 与官方 `stage_1.py:80-108` 的首个非下降锁定逻辑一致；不使用 validation/test | 未持久保存逐 epoch prediction；无多 epoch/local-min/fallback 单测 |
| A3 | 历史 checkpoint 的独立快照与 Stage 2 加载 | PASS WITH CONDITIONS | `gcdd/ido.py:341-347, 497-514`; `tests/test_ido.py:57-69` | tensor `detach().cpu().clone()`；CPU 测试在模型权重变化后验证 snapshot 未变 | 修正官方共享引用，属于受控语义修正；未持久保存 Stage 1 base state，不能恢复 |
| A4 | 30 epoch 10+20 阶段预算 | PROTOCOL APPROVAL REQUIRED | `gcdd/ido.py:324-338`; `scripts/run_lora_checkpoint_validation.py:280-298`; `train.md:47-57` | 默认 30→10+20，显式 CLI 可改；文档同写 5:10 缩放 | 无人工作为最终协议批准；显式 CLI 不检查预注册记录 |
| B1 | observed-label class-wise BMM 路由 | PASS | `gcdd/ido.py:264-321, 516-520, 625-632` | 一个 observed class 一个模型；按 batch observed label 路由；没有 clean label 参数 | — |
| B2 | clean-safe fallback 的定义与损失影响 | PARTIAL | `gcdd/ido.py:114-151, 207-212, 251-261, 350-380` | 常量、极小、非有限和拟合异常都可触发；CPU 测试验证有限项 | fallback 实质上提高 observed-label 信任；需要方法学批准 |
| B3 | BMM 稳定性与异常审计 | PARTIAL | `gcdd/ido.py:118-148, 153-188, 190-205, 304-321`; `tests/test_ido.py:25-48` | 常量、极小、NaN/Inf 可 fallback；每类每轮写 status/reason | 无近边界 alpha/beta/component-collapse 检测；predict 时切换 fallback 未按类别/次数单独落盘 |
| B4 | cyclic/IDN/GCCN observed-class 分布 | PARTIAL | `transfer_to_new_server/v1_inputs/*/labels.npy`; `outputs_major_revision/pgdf_ref_*/validation_manifest.csv` | cyclic 训练池三数据集已只读统计；本地没有 IDN/GCCN V1 label asset | IDN/GCCN 分布 UNVERIFIED |
| C1 | WELoss 来源及论文一致性声明 | PASS WITH CONDITIONS | `gcdd/ido.py:350-386`; `train.md:47-57` | 实现与官方 `utils/lnl_methods.py:53-84` 同一三项计算/广播结构 | 应严格称 `official-source WELoss adaptation`；论文 Eq. (5)/(8) 一致性 UNVERIFIED |
| C2 | 完整损失、梯度与数值有限性 | PARTIAL | `gcdd/ido.py:364-385, 560-573`; `tests/test_ido.py:50-56` | 双视图 observed CE、logit MSE、confidence entropy 都在，loss backward 有限 | 没有 pseudo-label/独立 noisy-distribution 项（官方 WELoss 本身也没有）；无非有限 loss/gradient guard |
| C3 | Stage 2 wrong-event/BMM 时序 | PASS WITH CONDITIONS | `gcdd/ido.py:545-573, 625-632` | 两视图各计一次、BMM query 在当前 batch 更新后、每 epoch 末 refit，符合官方 `stage_2_class.py:49-85, 191-202` | 训练步内使用上一轮 BMM 参数和当前累计计数；shuffle 影响 batch 内时点，未单测但无未来数据 |
| D1 | 冻结 DINOv2、LoRA、head、optimizer/scheduler | PASS WITH CONDITIONS | `gcdd/ido.py:698-732`; `gcdd/lora_training.py:69-91, 443-472, 529-544` | 先 freeze backbone，再注入 qkv LoRA；head 可训练；分组 AdamW，阶段各自 scheduler | 无真实 DINOv2 GPU smoke；无 gradient clipping（源码未宣称有） |
| D2 | 不混入 PGDF 机制 | PASS | `scripts/run_lora_checkpoint_validation.py:386-413, 484-524`; `gcdd/ido.py:405-695` | `ido` 不触发 graph/reference；Stage 2 全 training pool，无 prototype/dynamic subset 参数 | — |
| E1 | clean/noisy oracle 与 validation/test 数据流 | PASS WITH CONDITIONS | `scripts/run_lora_checkpoint_validation.py:343-379, 489-509, 982-993`; `gcdd/checkpoint_validation.py:39-142` | `labels.npy` 进入训练；clean label 只生成/标记固定 validation；test 在拟合后评估 | clean validation 是显式协议 oracle，不进入 BMM；缺少 sentinel 型 oracle-isolation 单测 |
| E2 | `--methods ido` 与 `--methods all` 隔离 | PASS | `scripts/run_lora_checkpoint_validation.py:65-83, 1481-1500`; `tests/test_ido.py:71-74`; `tests/test_checkpoint_validation.py` 的 all-method 测试 | IDO 只在 `CONTROLLED_ADAPTATION_METHODS`；`all` 仍返回 `METHODS` 13 项 | — |
| F1 | checkpoint 产物与恢复性 | PARTIAL | `gcdd/ido.py:979-1024`; `scripts/run_lora_checkpoint_validation.py:1360-1365` | 保存 Stage 2 best/final/last5，并含 seed/stage/BMM metadata | 不保存 Stage 1 base、wrong-event、optimizer、scheduler/scaler；没有 resume，已有目录被拒绝 |
| F2 | IDO D1 计算开销记录 | NOT IMPLEMENTED | `scripts/run_lora_checkpoint_validation.py:728-825`（仅 dynamic 分支实例化 `RunProfiler`）；`gcdd/ido.py:405-695` | IDO 调用不接收 profiler；日志只含普通 epoch loss/样本数 | 无阶段耗时、BMM 耗时、steps、峰值内存、额外 forward |
| G1 | `tools/run_ido_cyclic_all.py` | NOT IMPLEMENTED | 路径不存在（`Test-Path` 返回 `False`） | 无文件 | BLOCKER |
| G2 | `tools/run_ido_noise_generalization_all.py` | NOT IMPLEMENTED | 路径不存在（`Test-Path` 返回 `False`） | 无文件 | BLOCKER |
| G3 | `tools/run_ido_all.py` | NOT IMPLEMENTED | 路径不存在（`Test-Path` 返回 `False`） | 无文件 | BLOCKER |
| G4 | 27 次计划的 dry-run/preflight/resume/汇总/SHA/output 隔离 | NOT IMPLEMENTED | 仅有单次入口 `scripts/run_lora_checkpoint_validation.py:139-230, 421-425` | 单次入口 preflight 反而要求 fresh run dir；无 IDO 3×3×3 聚合器 | BLOCKER |
| T1 | IDO CPU 单测 | PASS WITH CONDITIONS | `tests/test_ido.py:1-128` | 本次重新运行 7/7 PASS | 未覆盖多 epoch label-wave、BMM collapse、oracle sentinel、checkpoint resume、27-task plan |
| T2 | 全仓 CPU 回归 | PASS | `python -m unittest discover -s tests -v` | 本次 65/65 PASS，0 fail/error/skip | 不构成 GPU、真实 DINOv2 或正式数据训练证据 |

## C. 关键问题详细说明

### C-1. clean-safe fallback 不是常规数值保护，而是实质性算法变更（HIGH）

**实际代码。** `ClassConditionalBetaMixture.fit` 在少于两个样本、非有限输入、常量输入、无效归一化区间、拟合异常或非有限参数时调用 `_activate_fallback`（`gcdd/ido.py:118-151, 207-212`）。`predict` 若无有效范围或预测非有限也切入 fallback（`153-188`）。其完整返回值在 `251-261`：

```python
# gcdd/ido.py:251-261
# p1=1 gives CE weight 1 and entropy weight 0; cdf1=cdf2=0.5 gives
# p1*cdf1 + (1-p2)*(1-cdf2) = 1 for the consistency term.
return (
    np.ones(length, dtype=np.float64),
    np.zeros(length, dtype=np.float64),
    np.full(length, 0.5, dtype=np.float64),
    np.full(length, 0.5, dtype=np.float64),
    0,
)
```

代入 `official_we_loss`（`350-380`）后：

* observed-label 双视图 CE 系数 `prob1=1`；
* 没有独立命名的“noisy-distribution loss”（官方 WELoss 中不存在该项），所以该项为 **N/A**，不能写成一个已实现的权重；
* consistency/logit MSE 系数 `1×0.5+(1-0)×(1-0.5)=1`；
* entropy 系数 `1-prob1=0`；
* 因而退化类样本被完全按 observed noisy label 训练，BMM 可靠性估计被跳过。

**期望与当前行为。** 防 NaN 本身是合理数值防护；但“fallback 后以 CE=1、MSE=1、entropy=0 训练”改变了该类的目标与噪声标签信任程度，不只是避免除零。它不使用 clean-label oracle，但这并不使其算法中性。

**影响。** 会影响任何训练池中 wrong-event 常量、极小或数值退化的 observed class；CUB/Cars/Aircraft 的实际训练池最小 class size 分别为 24/27/57，仍可能在早期 epoch 形成常量 wrong-event。

**建议。** 在启动最小 GPU 测试前人工确认此为“受控算法适配”，在正式协议与论文中如实披露；若希望更接近原实现，需另行预注册退化时的停止、全局模型或其他策略，并重新验证。任何替换 fallback 的措施都会改变 IDO 算法，不能作为无影响修补。

### C-2. BMM 对参数边界/组件塌缩的监控不完整（HIGH）

**实际代码。** `fit` 的成功判据仅为参数有限且全部 `>0`（`gcdd/ido.py:146-150`），没有 alpha/beta 接近 `epsilon`、权重接近 0/1 或两组件无法辨识的判据。`bmm_class_fits.csv` 会记录每类、每个 initial/epoch-update 的 `status`、`reason`、alpha/beta/weight（`190-205, 285-297`），但没有 prediction-time fallback 的 class/reason/event-count 专项记录。

**本次独立 CPU 证据。** 两点输入 `[0,1]` 拟合后为 `alpha=[1e-06,1.1]`、`beta=[1.1,3.99997e-06]`；99:1 输入为 `alpha=[1e-06,1.1]`、`beta=[1.1,2.03999e-06]`、`weights=[0.98999997,0.01000003]`。两者都保持有限输出且状态是 `fitted`，而不是 fallback。

**为什么有问题。** 这不是 NaN，但说明离散小支持集可给出几乎边界的 Beta 参数；现有代码将它静默当作正常拟合。它影响类别 BMM 的数值可信度与审查可解释性。

**建议。** 在正式训练前先确定并预注册“何为退化”（例如参数/权重阈值或拟合诊断）及对应行为，并把 prediction-time fallback 的 observed class、reason、次数落盘。此改变 BMM 的适用集合，属于受控算法适配；不可事后按结果启用。

### C-3. WELoss 是官方源码适配，但不是已证实的逐样本论文公式（HIGH）

**实际代码。** `gcdd/ido.py:364-380` 与官方 `utils/lnl_methods.py:53-83` 同构：双视图 observed-label CE、logits MSE、平均概率的 confidence-weighted entropy。

```python
# gcdd/ido.py:368-380
ce_term = (prob_tensor * (ce_s + ce_w)).mean()
mse = functional.mse_loss(output_s, output_w, reduction="none").mean(dim=1)
mse_weight = prob_tensor * cdf1_tensor + (1.0 - prob2_tensor) * (1.0 - cdf2_tensor)
mse_term = (mse_weight * mse).mean()
y_pred_avg = (y_pred_s + y_pred_w) / 2.0
confidence = torch.max(y_pred_avg, dim=1)[0].unsqueeze(1)
entropy_avg = -confidence * torch.sum(y_pred_avg * torch.log(y_pred_avg), dim=1)
entropy_term = ((1.0 - prob_tensor) * entropy_avg).mean()
total = ce_term + mse_term + entropy_term
```

**当前行为。** 对 batch size `B`，`confidence` 是 `[B,1]`，熵和为 `[B]`，故 `entropy_avg` 是 `[B,B]`；`(1-prob_tensor)` 也是 `[B]`，仍广播到 `[B,B]`。本次独立 CPU 检查实际得到 `entropy_shape=(2,2)`、`weighted_entropy_shape=(2,2)`，loss 与梯度有限。这不是 adapter 独有的新增错误，而是官方源码同样存在的行为。

**期望。** 当前分支应该准确标注为 `official-source WELoss adaptation`。本审查未把不可直接取得的论文 Eq. (5)/(8) 误判为已逐式一致；因此论文公式一致性为 **UNVERIFIED**。没有混用两套实现：代码只调用 `official_we_loss`。

**建议。** 在最小 GPU 前由维护者选择并冻结：保留官方源码的广播行为（并在方法说明中承认）或改成逐样本 entropy（这是方法变更，须重新测试/预注册）。不可同时宣称官方源码和论文逐样本公式都完全一致。

### C-4. 10+20 是可运行设置，不是已验收正式协议（MEDIUM）

**实际代码。** `resolve_stage_budget` 在总 epoch 可被 3 整除时给 `total//3` 与余数（`gcdd/ido.py:324-338`），所以 30→10+20；入口将规则和数值写入 resolved config（`scripts/run_lora_checkpoint_validation.py:280-298`），而 `train.md:47-57` 使用同一表述。显式 `--ido-stage1-epochs` 只验证正数（`1475-1476`），并未读取不可变的协议批准文件。

**影响。** Stage 1 多 5 epoch 会直接增加 wrong-event 累计尺度，BMM 输入与 label-wave 候选也随之变化；Stage 2 继承同一 `wrong_event`（`490, 497-520`）。两个阶段各自创建新的 AdamW/scheduler（`715-732; 752; 521-524`），没有错误地跨阶段复用状态，但 10+20 的科学公平性尚未获批。

**建议。** 在配置/实验计划中由人工明确登记 10+20、理由、不可更改的 config hash 和是否允许 `--ido-stage1-epochs` override；在此之前禁止 27 次正式运行。登记本身不改变 IDO；调整比例会改变实验协议。

### C-5. checkpoint 不能恢复 Stage 1/Stage 2 训练（MEDIUM）

**实际代码。** Stage 1 的 `base_state` 仅内存中传给 `model.load_state_dict(base_state, strict=False)`（`gcdd/ido.py:497-514`）。最终写出的 best/final/last5 checkpoint 只含 Stage 2 的 trainable `state_dict` 和 metadata（`979-1024`）；没有 `wrong_event`、BMM 参数、Stage 1 base state、optimizer、scheduler 或 GradScaler。入口的 `preflight_run_dirs` 又要求 run dir 全新（`scripts/run_lora_checkpoint_validation.py:1360-1365`）。

**影响。** 中断后无法从 Stage 1 selected base 或 Stage 2 epoch 恢复；同一结果目录也无法安全 resume。虽然这不改变一次性训练的目标，却影响可复现性、服务器故障恢复和 27 次任务成本。

**建议。** 在正式批量前设计版本化的阶段状态与 resume 语义，并测试中断恢复同 seed/config 的等价性。保存完整状态是工程扩展，不必改变 IDO 的数学目标。

### C-6. IDO D1 开销测量未实现（MEDIUM）

`RunProfiler` 虽被入口导入，但只在 dynamic-loss 分支 `scripts/run_lora_checkpoint_validation.py:728-825` 创建并传入。IDO dispatch `484-524` 没有 profiler 参数，`train_ido_lora` 也没有对应参数。因此未实现：Stage 1/Stage 2 wall time、BMM fit time、optimizer steps、peak CUDA allocated/reserved、每阶段额外 forward 的 D1 数据。普通 `train_log.csv` 仅记录 loss/epoch、样本数和 wrong-event（`gcdd/ido.py:867-911`）。

**建议。** D1 若要用于 IDO 比较，必须在启动正式 GPU 实验前将同一 profiler 语义接到两个阶段和 BMM 拟合，并经 CPU/GPU smoke 验证；这属于测量基础设施变更，不应在本审查轮直接修改。

### C-7. 27 次批量能力完全缺失（BLOCKER）

本次在 worktree 执行：

```powershell
Test-Path tools\run_ido_cyclic_all.py
Test-Path tools\run_ido_noise_generalization_all.py
Test-Path tools\run_ido_all.py
```

三个结果均为 `False`；`tools/` 也没有任何 `*ido*` 文件。单次入口可以运行显式 `--methods ido`，但不等于存在 3 dataset × 3 seed × cyclic/IDN/GCCN 的计划器。故下列能力均未实现：27-task 生成、dry-run、preflight、resume、summarize-only、fail-fast、已完成跳过、不完整 run 处理、manifest SHA256、输出隔离及 3-seed mean±sample SD 汇总。

**建议。** 另行开发并独立验收三个指定脚本及共享调度/汇总模块，保证不改动现有 126 次 IDN/GCCN 调度器。此为运行基础设施，不是 IDO 核心算法变更。

## D. 关键源码证据包

以下片段来自本审查工作树的真实源码；行号以审查当时文件为准。长函数应结合指明的连续范围阅读。

### D1. Stable ID、Stage 1 wrong-event 与 label-wave

`gcdd/lora_training.py:49-66` 使用 `self.indices` 映射 dataset position 到原始 V1 index；`gcdd/ido.py:459-471` 将 `train_idx=np.where(train_mask)[0]` 传给两个 shuffle loader。因此 batch 的 `original_indices` 不会随 shuffle 错位。

```python
# gcdd/ido.py:760-805
for stage_epoch in range(1, stage1_epochs + 1):
    current_prediction = np.full(len(wrong_event), -1, dtype=np.int64)
    for images, labels, original_indices in progress:
        ...
        predicted = torch.argmax(logits.detach(), dim=1)
        indices_np = original_indices.detach().cpu().numpy().astype(np.int64)
        predicted_np = predicted.detach().cpu().numpy().astype(np.int64)
        labels_np = labels.detach().cpu().numpy().astype(np.int64)
        wrong_event[indices_np[predicted_np != labels_np]] += 1
        current_prediction[indices_np] = predicted_np
        ...
    if previous_prediction is None:
        base_state = snapshot_trainable_state(model)
        base_stage_epoch = stage_epoch
        previous_prediction = current_prediction
    else:
        prediction_changes = int(np.count_nonzero(
            current_prediction[train_idx] != previous_prediction[train_idx]
        ))
        if not label_wave_saved:
            if prediction_changes < previous_change:
                base_state = snapshot_trainable_state(model)
                base_stage_epoch = stage_epoch
                selected_this_epoch = True
                previous_change = prediction_changes
                previous_prediction = current_prediction
                if stage_epoch == stage1_epochs:
                    label_wave_saved = True
            else:
                label_wave_saved = True
```

首 epoch 必定是 base fallback；首次较前一候选“不再严格下降”时锁定上一个最小点；若严格下降直到末 epoch，则选择末 epoch。全程不调用 validation 或 official test。完整 prediction 向量只在内存保留当前/前一候选，日志只写 `stage1_prediction_changes`（`822-826`），没有逐 epoch prediction 文件。

### D2. Deep-copy snapshot 与 Stage 1→2 切换

```python
# gcdd/ido.py:341-347, 497-520
def snapshot_trainable_state(model: Any) -> dict[str, Any]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if "lora_" in name or name.startswith("head.")
    }

base_state, base_stage_epoch = _run_stage1(...)
model.load_state_dict(base_state, strict=False)
model.to(device)
pool_label_ids = np.asarray([
    label_to_id[str(train_labels[index])] for index in train_idx
], dtype=np.int64)
bmm = ClasswiseBMM(classes)
bmm_fit_rows = bmm.fit(
    wrong_event[train_idx], pool_label_ids,
    stage2_epoch=0, fit_round="stage2_initial", max_iter=20
)
```

`tests/test_ido.py:57-69` 在修改模型权重后验证 snapshot tensor 保持不变，故“历史 tensor 与后续 optimizer step 共享存储”的问题有直接 CPU 证据。官方 `stage_1.py:88-108` 使用未 clone 的 `model.state_dict()`；本实现修正该问题，故应称受控修正而非原样复现。

### D3. Class-wise BMM、posterior/CDF 与 fallback

```python
# gcdd/ido.py:271-318
def fit(self, wrong_event, label_ids, *, stage2_epoch, fit_round, max_iter):
    ...
    for label_id, observed_label in enumerate(self.observed_label_names):
        values = wrong_event[label_ids == label_id]
        metadata = self.models[label_id].fit(values, max_iter=max_iter)
        rows.append({"fit_round": fit_round, "stage2_epoch": int(stage2_epoch),
                     "observed_label_id": int(label_id),
                     "observed_label": observed_label, **metadata})
    return rows

def predict(self, wrong_event, label_ids):
    ...
    for label_id, model in self.models.items():
        positions = np.where(label_ids == label_id)[0]
        ...
        prob1, prob2, cdf1, cdf2, clipped = model.predict(wrong_event[positions])
        outputs["prob1"][positions] = prob1
        outputs["prob2"][positions] = prob2
        outputs["cdf1"][positions] = cdf1
        outputs["cdf2"][positions] = cdf2
```

`gcdd/ido.py:114-151` 为 fit 的退化判定；`153-188` 用 `scipy.stats.beta.pdf/cdf` 计算 posterior/CDF，并对预测非有限转换 fallback。完整 clean-safe 返回见本报告 C-1。BMM 的 observed label 来自 `train_labels` 建立的 `label_to_id`（`436-449, 516`），不存在 clean label 入参。

### D4. Stage 2 wrong-event、BMM 时序与 WELoss

```python
# gcdd/ido.py:545-573, 625-632
for weak_images, strong_images, labels, original_indices in progress:
    ...
    weak_logits = model(weak_images)
    strong_logits = model(strong_images)
    weak_predicted = torch.argmax(weak_logits.detach(), dim=1).cpu().numpy()
    strong_predicted = torch.argmax(strong_logits.detach(), dim=1).cpu().numpy()
    wrong_event[index_np[weak_predicted != label_np]] += 1
    wrong_event[index_np[strong_predicted != label_np]] += 1
    bmm_terms = bmm.predict(wrong_event[index_np], label_np)
    loss, parts = official_we_loss(torch, strong_logits, weak_logits,
                                   bmm_terms["prob1"], bmm_terms["prob2"],
                                   bmm_terms["cdf1"], bmm_terms["cdf2"], labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
...
bmm_fit_rows.extend(bmm.fit(wrong_event[train_idx], pool_label_ids,
                             stage2_epoch=stage_epoch,
                             fit_round="stage2_epoch_update", max_iter=2))
```

每个 view 各累计一次，更新发生在本 batch BMM query 前；BMM 初始拟合发生在 Stage 1 后，之后每个 Stage 2 epoch 结束 refit 一次。这与官方 `stage_2_class.py` 的更新/query/refit 顺序对应。没有使用未来 epoch 数据；但是 batch 顺序会影响同一 epoch 内的累计时点。

### D5. 模型、优化器和增强

```python
# gcdd/ido.py:698-732
model = DINOv2LoRAClassifier.make(torch, cfg, num_classes).to(device)
freeze_all(model.backbone)
modules = inject_lora(torch, model.backbone,
    target_modules=parse_target_modules(str(cfg["lora"].get("target_modules", "qkv"))),
    rank=int(cfg["lora"].get("rank", 8)),
    alpha=float(cfg["lora"].get("alpha", 16.0)),
    dropout=float(cfg["lora"].get("dropout", 0.05)))
for parameter in model.head.parameters():
    parameter.requires_grad_(True)
...
optimizer = torch.optim.AdamW([
    {"params": lora_parameters(model), "lr": float(train_cfg["lora_lr"])},
    {"params": model.head.parameters(), "lr": float(train_cfg["head_lr"])}],
    weight_decay=float(train_cfg.get("weight_decay", 0.05)))
total_steps = max(1, int(stage_epochs) * max(1, loader_length))
warmup_steps = int(total_steps * float(train_cfg.get("warmup_ratio", 0.1)))
```

`gcdd/ido.py:389-402` 定义 weak `RandomResizedCrop+Flip` 和 strong `RandomResizedCrop+Flip+RandAugment(2,10)`；`gcdd/lora_training.py:443-472` 证明未匹配 target 会抛错、base 参数冻结而新增 LoRA 参数参与训练。Stage 2 使用同一模型加载 Stage 1 selected state，不会重置 classifier；优化器/scheduler 则是两阶段各自新的实例。代码没有 gradient clipping 或 pseudo-label 分支。

### D6. 训练入口、validation-selected checkpoint 和 official test

```python
# scripts/run_lora_checkpoint_validation.py:489-516
result = train_ido_lora(
    data["train_paths"], data["train_labels"], validation_paths,
    validation_labels, split.train_mask, run_cfg, method=method_name,
    seed=seed, stage1_epochs=int(ido_stage_budget[0]), ...,
    test_paths=data["test_paths"], test_labels=data["test_labels"],
    checkpoint_path=checkpoints / "best_val.pt", ...,
    official_test_selected_only=bool(args.official_test_selected_only))
write_csv(run_dir / "bmm_class_fits.csv", result.bmm_fit_rows, ido_bmm_fit_fields())
write_ido_wrong_event(run_dir / "stage1_wrong_event.csv", data,
                      split.train_mask, result.wrong_event)
```

`load_data` 读取 V1 `paths.txt/labels.npy/eval_paths.txt/eval_labels.npy`（`982-993`）。`clean_labels` 仅通过 `load_clean_labels_from_noise_index` 创建固定验证 split，训练调用仍传 `data["train_labels"]`（`343-379,489-496`）。`gcdd/checkpoint_validation.py:39-142` 也明确标注 clean label 只能用于 validation split。此处是固定 clean-validation 协议，并非 BMM/训练期 clean-label oracle。

```python
# gcdd/ido.py:613-617, 931-976
if best_row is None or float(row["top1"]) > float(best_row["top1"]):
    best_row = row
    best_state = snapshot_trainable_state(model)
...
selected_top1, selected_top5 = evaluate_state_lora(...best_state, test_loader...)
if official_test_selected_only:
    metrics.update({"final_test_top1": "", "final_test_top5": "", ...})
    return metrics
final_top1, final_top5 = evaluate_state_lora(...final_state, test_loader...)
```

best state 只由 validation Top-1 决定。official test 在 Stage 2 训练完成后才调用；若正式命令带 `--official-test-selected-only`，只测该 validation-selected checkpoint 一次。未带该参数时还会评估 final/last5，但没有反向改写 best state。

### D7. Method dispatch 与 D1 缺口

`scripts/run_lora_checkpoint_validation.py:65-83,1481-1500` 将 `ido` 放入 `CONTROLLED_ADAPTATION_METHODS=("ido",)`；`--methods all` 仅返回 `METHODS`，不含 IDO。`386-413` 显示仅 IDO 时跳过 PGDF graph/static reference。D1 profiler 的实例化和传入仅出现在 dynamic 分支 `728-825`，不在 IDO dispatch `484-524` 中。

## E. 测试与独立 CPU 检查证据

所有以下命令在 `D:\实验\GCDD\.worktrees\major-revision` 执行，未请求 CUDA、未启动真实训练。

| 命令 | 收集/检查 | PASS | FAIL | ERROR | SKIP | 结果 |
|---|---:|---:|---:|---:|---:|---|
| `python -m unittest discover -s tests -p test_ido.py -v` | 7 | 7 | 0 | 0 | 0 | `Ran 7 tests in 4.802s`, `OK` |
| `python -m unittest discover -s tests -v` | 65 | 65 | 0 | 0 | 0 | `Ran 65 tests in 3.648s`, `OK` |

测试输出中的 `torch.cuda.amp.*` FutureWarning 和 joblib CPU-core 探测 warning 均不是失败；没有 skipped 行。

已覆盖的正式 IDO 测试（`tests/test_ido.py`）包括：30→10+20 budget、全零常量 fallback、两类 class-wise BMM 的有限输出与路由、三项 WELoss 的 backward 有限性、snapshot clone、`--methods ido` 解析、以及 1+2 epoch CPU fake-backbone smoke（含 `stage1_wrong_event` 返回、BMM rows、validation-selected checkpoint、last5 checkpoint）。全仓另有 `test_methods_all_expands_all_thirteen_once`，确认 `all` 主方法集未改变。

本审查另行运行未提交的只读/内存 CPU harness：

* all-zero、same-nonzero、one-sample、NaN、Inf：均进入 `clean_safe_fallback`，原因分别为 `constant_wrong_event`、`fewer_than_two_samples` 或 `nonfinite_wrong_event`，返回项全有限；
* two-values、99:1 imbalance、near-boundary discrete support：保持 `fitted` 且 posterior/CDF 有限，但前两者产生接近边界 beta/alpha（见 C-2）；
* WELoss B=2：`loss_finite=True`，双输出梯度有限，且实测 entropy 中间张量 `(2,2)`；
* 已有 V1 labels 只读分布统计：见下一节。

未覆盖，不能默认为通过：多 epoch 的 label-wave 局部极小/单调 fallback、Stage 1 selected epoch 与加载权重的端到端比对、clean-label sentinel 隔离、prediction-time BMM fallback 审计、component collapse 阈值、checkpoint resume、实际 DINOv2 GPU/AMP、IDO D1 和 27-task scheduler。

## F. 三种噪声协议的 observed-class 样本规模

统计只读取已有 V1 `labels.npy`，并用现有固定 validation manifest 的 `training_pool` 分区筛出 IDO 将实际训练的索引；没有重新生成标签，也没有将 clean labels 输送给算法决策。路径分别是：

* `transfer_to_new_server/v1_inputs/v1_cub_asym_r0p4_s42/labels.npy` + `outputs_major_revision/pgdf_ref_cub40/validation_manifest.csv`
* `transfer_to_new_server/v1_inputs/v1_cars_asym_r0p4_s42/labels.npy` + `outputs_major_revision/pgdf_ref_cars40/validation_manifest.csv`
* `transfer_to_new_server/v1_inputs/v1_aircraft_asym_r0p4_s42/labels.npy` + `outputs_major_revision/pgdf_ref_aircraft40/validation_manifest.csv`

| 本地可用协议 | 数据集 | training-pool N | min | P5 | median | P95 | max | empty class |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Cyclic-40/V1 `asym_r0p4_s42` | CUB | 5,400 | 24 | 25.0 | 27.0 | 29.0 | 30 | 0 |
| 同上 | Cars | 7,410 | 27 | 32.0 | 38.0 | 42.0 | 50 | 0 |
| 同上 | Aircraft | 6,067 | 57 | 58.0 | 61.0 | 63.0 | 65 | 0 |

作为交叉核对，未分 validation 的原始 V1 最小样本数为 CUB 29、Cars 31、Aircraft 66。IDN 和 GCCN：工作树没有相应 V1 `labels.npy/paths.txt` 训练资产；`rg --files --no-ignore` 也未发现已落地 IDN/GCCN 运行输入。故三数据集的 IDN/GCCN 统计均为 **UNVERIFIED**，不得据此编造 BMM 稳定性结论。

## G. 批量运行能力结论

| 目标 | READY? | 原因 |
|---|---|---|
| cyclic 9 次（3 数据集×3 seed） | **NO — NOT IMPLEMENTED** | 无 `tools/run_ido_cyclic_all.py`、任务计划、汇总或 resume 语义 |
| IDN/GCCN 18 次 | **NO — NOT IMPLEMENTED** | 无 `tools/run_ido_noise_generalization_all.py`，且本地 V1 IDN/GCCN 资产不可核对 |
| 完整 27 次 | **NO — NOT IMPLEMENTED** | 无 `tools/run_ido_all.py`；还受协议批准、D1 和前述数值裁决阻塞 |
| 单数据集/单 seed 受控适配 | **CONDITIONALLY READY** | 仅在维护者批准 10+20、fallback、WELoss 表述后，先做非正式最小 GPU smoke；不等同于正式实验 READY |
| 服务器 GPU 验收 | **REQUIRED LATER** | CPU fake model 不能证明真实 DINOv2-B/14 模块匹配、CUDA 内存、AMP 或计量 |

## H. 最终阻塞清单（按优先级）

### 在最小 GPU 测试前必须解决

1. **[BLOCKER/协议]** 获得并冻结 30 epoch `10+20` 的人工批准；禁止以 official-test 结果重新选择比例。
2. **[HIGH/算法]** 对 `clean_safe_ce1_mse1_entropy0` 作出明确接受、替换或停止运行的预注册决定，并将其披露为算法适配。
3. **[HIGH/方法忠实度]** 决定官方 WELoss 的 `[B,B]` entropy 广播是否保留；若保留，所有文档均使用 `official-source WELoss adaptation`，不宣称未证实的论文逐式一致。
4. **[HIGH/数值]** 规定 BMM 参数近边界/weight collapse 的识别和处理；当前有限不等于可靠。

### 在 27 次正式训练前必须解决

1. **[BLOCKER]** 实现并验收 `tools/run_ido_cyclic_all.py`、`tools/run_ido_noise_generalization_all.py`、`tools/run_ido_all.py` 及其 dry-run/preflight/resume/summary/SHA/output-isolation 语义，且保证 126 次原调度器不受影响。
2. **[MEDIUM]** 接入并验收 IDO D1：Stage 1/2/BMM/总耗时、optimizer steps、阶段样本数、额外 forward、peak allocated/reserved。
3. **[MEDIUM]** 设计 checkpoint/resume 产物，至少保存可重启所需 Stage 1 base、wrong-event、BMM、optimizer/scheduler/scaler 状态，或明确禁用并在计划器中安全重启。
4. **[MEDIUM]** 增补单测：多 epoch label-wave、Stage 1 state→Stage 2 加载、component collapse、prediction-time fallback 记录、oracle isolation、scheduler 27-task 生成与 completed/incomplete handling。
5. **[UNVERIFIED]** 在服务器可读到 IDN/GCCN V1 资产后，重新生成三数据集×两协议的 observed-class size 统计；不改变噪声资产。

### 可通过文档/限制说明解决（但不得伪称 PASS）

* 将名称固定为 `IDO-style DINOv2-B/14+LoRA controlled adaptation`，不称“IDO 原版等价复现”。
* 明确 Stage 1 deep-copy 是针对官方 `state_dict` 引用风险的受控修正。
* 明确 fixed clean-validation 是既有统一协议；clean labels 不进入训练/BMM，official test 不用于 checkpoint 选择。
* 若暂不支持恢复、D1 或 IDN/GCCN，文档必须显示 NOT IMPLEMENTED/UNVERIFIED，不能用单次入口存在替代这些能力。

## I. 外部复核交付建议

建议优先上传真实源码/原始 diff，而不是本地绝对路径或改写伪代码：

1. 本报告 `audit_reports/IDO_IMPLEMENTATION_ACCEPTANCE_REPORT.md`；
2. `gcdd/ido.py`；
3. `tests/test_ido.py`；
4. `scripts/run_lora_checkpoint_validation.py`，或包含它、`gcdd/ido.py`、`requirements.txt`、`train.md` 的完整 IDO 相关 diff；
5. `gcdd/lora_training.py` 与 `gcdd/checkpoint_validation.py`（复核 stable ID、LoRA/freeze、validation 数据流）；
6. 若未来实现，三个实际 IDO 调度脚本及其配置/共享调度模块。目前它们不存在，不能以空占位文件替代。

供官方来源比对的公开材料：IDO 官方仓库、`stage_1.py`、`stage_2_class.py`、`utils/lnl_methods.py`、`utils/bmm.py`。外部审查者应使用上传的源码版本逐行复核，而不是依赖此报告的摘要。

## 第四轮外部复核问题的定向修复（2026-10-17）

> 本节追加而非替换前述历史验收记录。它只处理第三轮独立复核报告中的 `IDO3-001` 至 `IDO3-004`，没有修改 WELoss、class-wise BMM 数学定义、30-epoch 的 10+20 候选预算、噪声资产、validation manifest 或既有 126 次调度器。

### 当前结论：PASS WITH CONDITIONS（27 次正式训练仍未获批准）

| 外部问题 | 本轮状态 | 实际修复与 CPU 证据 | 仍然不能据此声称 |
|---|---|---|---|
| IDO3-001：`--new-attempt` 重跑已完成 seed | PASS ON CPU | `gcdd/ido_batch.py::find_reusable_completed_task` 仅复用当前 immutable experiment signature 下唯一完整的早期 task；`main_for_protocols` 对该 seed 打印 `SKIP REUSE COMPLETE`，只启动未完成 seed；已有多个同签名完成副本则 fail-closed。`tests/test_ido_batch.py::test_new_attempt_reuses_only_one_prior_completed_seed_with_matching_signature` 覆盖 1/42/88 跨 attempt 的正常汇总。 | 不提供 epoch-level resume；不完整 task 仍必须新 attempt。 |
| IDO3-002：noise index 与 V1 observed labels 未逐样本对齐 | PASS ON CPU | `inspect_input_contract` 现调用 `validate_observed_label_alignment`。cyclic 以 path key 对齐后逐项比较 `web_label` 与 `labels.npy`；IDN/GCCN 复用 `validate_noise_generalization_training_input`，额外要求 `source_train_index` 和 `web_label == noisy_label`。测试故意写入错位路径、错误 observed label、重复 source index，均被 preflight 拒绝。该检查不读取 `clean_label`，不进入 BMM/loss。 | 本地仍没有服务器 9 组真实资产；必须在服务器 preflight 验证。 |
| IDO3-003：可反序列化但无关的 state dict 被接受 | PASS ON CPU / GPU UNVERIFIED | `validate_ido_trainable_state_dict` 要求仅有 `backbone.*.lora_*` 与 `head.*` 的有限浮点 tensor、LoRA a/b 对、分类 head 的输出行数与 observed class 数一致，且 stage1/best/last 的 key/shape/dtype layout 一致。随机 key 的负例在 `test_completion_rejects_semantically_invalid_trainable_state_dict` 被拒绝。 | CPU 结构审计不能替代真实 DINOv2-B/14+LoRA 的 `load_state_dict`；后者仍是无 official-test GPU pilot 的验收项目。 |
| IDO3-004：复核 ZIP 的本地依赖闭包不完整 | PENDING REPACKAGE | 下一份独立复核包会按 `test_ido.py`、`test_ido_batch.py` 和训练入口的递归本地 import 闭包纳入 `gcdd`/`revision_experiments` 支撑模块，并在解压副本原样执行两套 IDO CPU 测试；不以 `missing_requested_files.txt=none` 代替实际导入验证。 | 这不会使 GPU、服务器资产或 27 次正式运行自动通过。 |

### 本轮新增/更新的 CPU 测试

- `python -m unittest discover -s tests -p test_ido.py -v`：14 PASS，0 fail/error/skip；未调用 GPU。
- `python -m unittest discover -s tests -p test_ido_batch.py -v`：9 PASS，0 fail/error/skip；包括三项第三轮报告的负例：无关 state dict、错位/错误 noise index、跨 attempt 部分完成后的安全复用。

完整仓库回归、真实配置 dry-run、服务器 preflight 与新 ZIP 解压复跑必须在本节之后再次记录；在它们完成前，本报告仍保留 **PASS WITH CONDITIONS**，不改变“10+20 待批准、服务器资产未核验、BMM 真训练稳定性未验证、GPU pilot 未运行”的结论。

### 第四轮复核包（IDO3-004 的交付证据）

已生成 `audit_reports/IDO_FOURTH_REVIEW_BUNDLE.zip`，不覆盖第二、三轮历史包。包内纳入 IDO 核心、批量调度、训练入口、两套 IDO 测试，以及从这些入口递归得到的本地 `gcdd` / `revision_experiments` 支撑模块（包含此前遗漏的 `baselines.py`、`graph.py`、`scoring.py`、`selection_utils.py`、`noise_realization.py`、`data.py` 与 `offline_diagnostics.py`）。

独立解压到临时目录后实际运行：

```bash
PYTHONPATH=. python -m unittest discover -s tests -p "test_ido*.py" -v
```

结果为 **23 PASS，0 fail/error/skip**（`test_ido.py` 14 项 + `test_ido_batch.py` 9 项）；ZIP CRC、49 项归档 manifest SHA256、关键文件存在性和相对路径安全检查均通过。该包仍不含真实图像、`.npy` 标签、noise index、权重、checkpoint、实验输出、`.git` 或凭据。上述结果只证明打包内的 IDO CPU 测试可重跑，不构成真实数据/GPU/27-run 验收。

## 第五轮外部复核问题的定向修复（2026-10-18）

> 本节追加记录第四轮独立审查的 `F4-001` 至 `F4-004`。修复仅收紧批量完成性审计、任务签名和服务器 preflight 证据；没有修改 `gcdd/ido.py` 的 WELoss、class-wise BMM 数学、Stage 1/2 预算、噪声资产、validation 划分或既有 126 次调度器。

### 当前结论：PASS WITH CONDITIONS；正式 27 次训练仍不获批准

| 外部问题 | 本轮状态 | 实际修复与 CPU 证据 | 仍不能据此声称 |
|---|---|---|---|
| F4-001：LoRA A/B 可不配对或内部形状不相容 | **PASS ON CPU / GPU UNVERIFIED** | `gcdd/ido_batch.py::validate_ido_trainable_state_dict` 现要求每个 `backbone.<module>` 恰有一对 A/B；两矩阵均为 2-D，A 的 rank 与 B 的内秩相等且等于冻结 V1 config 的 `lora.rank`；`qkv/proj/fc1/fc2` 还按 DINOv2-B/14 维度规则校验，`head.weight` 必须为 `C×768`。`tests/test_ido_batch.py::test_trainable_state_audit_rejects_unpaired_or_rank_incompatible_lora` 覆盖跨模块错配与秩不一致负例。 | CPU 结构审计替代真实 DINOv2 hub 模型的 `load_state_dict`；后者仍必须由无 official-test GPU pilot 验收。 |
| F4-002：Stage-1/BMM CSV 只要存在即被接受 | **PASS ON CPU** | `is_complete` 现调用 `validate_stage1_wrong_event_artifact`：要求训练池的稳定 index 顺序、path、observed label、整数非负且不超过 Stage-1 epoch 数，并与 `stage1_base.pt` 的完整 `wrong_event` 向量逐项交叉验证。`validate_bmm_fit_artifact` 要求 initial + 所有非末轮 Stage-2 epoch 的每类别记录（总计 `C×stage2_epochs`），并验证支持数、class/epoch/round、迭代数、数值有限性、权重、Beta 均值语义及 initial fit 与 Stage-1 统计一致。负例覆盖空 Stage-1 CSV、重复 index 和缺失 BMM 记录。 | 合成 CPU 夹具不证明实际 BMM 在 CUB 等小类别上不会 fail-closed。 |
| F4-003：experiment signature 未覆盖完整运行依赖 | **PASS ON CPU** | `IDO_RUNTIME_CODE_PATHS` 是显式运行/审计依赖清单，`build_task_contract` 对入口、IDO、LoRA、validation、noise、选择、D1/审计支撑模块逐一 SHA256。`test_contract_covers_runtime_dependency_closure_and_changes_signature` 模拟 `gcdd/lora_dynamic.py` hash 改变，确认 experiment signature 必变。 | 它固定的是当前受控运行闭包；外部 Python/PyTorch/DINO hub 版本仍需在服务器 pilot 记录。 |
| F4-004：IDO 与既有 126 次 IDN/GCCN 验证集一致性缺少实物证据 | **PASS ON CPU / SERVER UNVERIFIED** | IDN/GCCN preflight 现除核对 noise-index SHA256 外，还要求 `reference_126_training_root` 中的 `validation_manifest.csv/json` 实际存在并与 IDO 输入目录逐字节 SHA256 一致。`test_preflight_rejects_idn_reference_validation_manifest_mismatch` 覆盖反例。比较仅读取/哈希既有验证工件，不把 clean labels 传给 IDO 的 BMM 或训练损失。 | 本地缺少九组真实 V1/IDN/GCCN 资产；必须在隔离服务器目录再次 preflight。 |

### 本轮实际 CPU 与 dry-run 证据

| 命令 | 结果 | 边界 |
|---|---|---|
| `python -m unittest discover -s tests -p test_ido_batch.py -v` | **13 PASS**, 0 fail/error/skip | 含本轮 4 项新增负例；不调用 GPU。 |
| `python -m unittest discover -s tests -p test_ido.py -v` | **14 PASS**, 0 fail/error/skip | CPU fake-backbone smoke；不证明真实 DINOv2/CUDA。 |
| `python -m unittest discover -s tests -v` | **85 PASS**, 0 fail/error/skip | 全仓 CPU 回归；仅有既有 `torch.cuda.amp` FutureWarning 与 joblib CPU-core warning。 |
| `python tools/run_ido_all.py --config configs/ido_batch_all.yaml --dry-run` | `planned=27, blocked=9`，未启动子进程 | 九项均因本地 V1 输入资产缺失而 BLOCKED；因此本机没有、也不能伪称已有 IDN/GCCN/126-run validation parity 证据。 |

### 仍需人工/服务器验收的门槛

1. 在看到任何正式 official-test 结果前，人工预注册并冻结 30 epoch 的 `10+20` 候选协议；`protocol_frozen` 仍保持 `false`。
2. 在独立服务器 worktree/输出根目录运行上述 dry-run，并获得 `planned=27, blocked=0`；IDN/GCCN 必须通过新增的 reference-126 validation manifest SHA256 比对。
3. 使用 `--ido-pilot-no-official-test` 做单 seed、独立输出目录的 GPU pilot；记录实际代码/配置/输入 hash、真实 DINOv2-LoRA state 加载、BMM 退化记录、D1/CUDA 指标。若 BMM fail-closed，只记录并停止，不引入新的回退策略。
4. 仅在 GPU pilot 稳定完成后，才可重新评审是否批准 27 次正式运行；本报告不构成该批准。

## 第六轮定向修复：第五轮 F5-001 checkpoint 结构完整性（2026-10-18）

> 本节保留第五轮外部复核发现，并记录其获维护者授权后的最小修复。修改范围仅限 checkpoint 结构审计、训练期结构溯源元数据和 CPU 负例；未修改 WELoss、BMM、wrong-event、Stage 1/2 机制、10+20 候选协议、噪声/validation 资产或既有 126 次调度器。

### F5-001 修复状态：PASS ON CPU / GPU REAL-LOAD UNVERIFIED

第五轮复核正确指出，旧版 `validate_ido_trainable_state_dict()` 仅检查任意 LoRA A/B 对的后缀和矩阵尺寸，因此 `backbone.blocks.999.*` 仍可能被接受。现在的审计不再从待审 checkpoint 推导“允许的模型结构”：

1. `gcdd/lora_training.py::build_dinov2_vitb14_lora_structure_contract()` 固定受控 `dinov2_vitb14` 的 12 个 Transformer block（`blocks.0`—`blocks.11`）线性模块目录，并与 `inject_lora()` 共用 `lora_module_matches_target()` 的 target-pattern 语义。
2. `gcdd/ido_batch.py::validate_checkpoints()` 从实际运行目录的 `resolved_config.yaml` 读取冻结后的 `lora.rank`、`target_modules` 和 feature backend，生成独立的 expected contract；不再仅依赖原 V1 配置的默认值。
3. `validate_ido_trainable_state_dict()` 要求 checkpoint 的 key 集合与该 expected contract **完全相等**，并逐一验证 head、A/B 矩阵的 rank/shape、浮点 dtype 和有限性。未知 block、缺失 block、额外 block 和不在冻结 target 中的模块均 fail-closed。
4. `gcdd/ido.py::build_ido_lora_structure_metadata()` 在模型实际完成 `inject_lora()` 后记录 live trainable-key/shape/dtype 结构及 SHA256；同时向 Stage-1、best 和 last checkpoint 写入独立 frozen-contract hash、DINOv2 预训练来源描述和 Python/PyTorch/Torchvision 环境版本。批量完成审计要求 frozen contract 与 live-structure 元数据均可验证。

该 CPU 审计防止错误 checkpoint 被跳过或计入汇总；它**不替代**后续 GPU pilot 中真实 DINOv2-B/14 实例的 `load_state_dict` 检验，也不证明 hub 权重或 CUDA 行为。

### 新增 CPU 负例与实际结果

`tests/test_ido_batch.py::test_f5_checkpoint_structure_contract_rejects_every_listed_invalid_lora_layout` 和 `::test_f5_is_complete_fail_closed_for_every_invalid_lora_layout` 覆盖以下每一项，并同时验证直接结构审计与 `is_complete()`：

* `blocks.999` 不存在层；
* 仅保存 `blocks.0`；
* 缺失 `blocks.11`；
* 额外 block；
* 错误 target module（`qkv` 替换为 `proj`）；
* LoRA A/B 不配对；
* 单独的 rank/shape 不一致负例。

所有上述非法状态均被拒绝；包含 12 个 `qkv` block、rank 与 head 维度均与有效 config 一致的合成状态通过。`tests/test_ido.py` 的 CPU fake-backbone smoke 还验证 checkpoint 确实写入 frozen/live 结构 hash、预训练来源描述和环境元数据。

| 命令 | 结果 | 边界 |
|---|---|---|
| `python -m unittest discover -s tests -p test_ido_batch.py -v` | **15 PASS**, 0 fail/error/skip | 含 F5-001 直接及 `is_complete()` 负例；CPU 合成状态。 |
| `python -m unittest discover -s tests -p test_ido.py -v` | **14 PASS**, 0 fail/error/skip | CPU fake-backbone smoke；不加载真实 DINO hub 模型。 |
| `python -m unittest discover -s tests -v` | **87 PASS**, 0 fail/error/skip | 全仓 CPU 回归；仅有既有/依赖的 warning。 |
| `python tools/run_ido_all.py --config configs/ido_batch_all.yaml --dry-run` | `planned=27, blocked=9`，未启动子进程 | 本地仍缺 9 组真实输入资产；不构成服务器 preflight 通过。 |

### 剩余正式运行门槛

1. 人工在任何正式结果前冻结 `10+20`，`protocol_frozen` 继续保持 `false` 直至批准。
2. 在不影响 126 次任务的独立服务器 worktree 中，确认 27-task preflight 为 `planned=27, blocked=0`，并保留 IDN/GCCN reference-126 validation manifest 的 SHA256 证据。
3. 仅以独立输出目录执行单 seed `--ido-pilot-no-official-test` GPU pilot；核验真实 DINOv2-B/14 LoRA key 集合和完整加载、预训练权重来源、环境版本、BMM fail-closed 诊断与 D1/CUDA 指标。
4. GPU pilot 或服务器 preflight 失败时，只记录并停止；不得自行引入 global-BMM、pseudo-count、clean-safe 或其他未经批准的算法回退。
