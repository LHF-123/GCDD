# JAL-CE-DINOv2+LoRA Full-Noisy Baseline

JAL-CE 作为独立 full-noisy robust-loss baseline 运行：

```text
DINOv2+LoRA + full noisy training set + JAL-CE
```

该入口会拒绝 JAL-CE 与 Centroid、GCDD、Dynamic Loss 或 PGDF 的组合。配置优先级为：

```text
CLI > JAL method YAML > V1 resolved_config.yaml > 代码默认值
```

## WebFG-496 448 命令

```bash
python scripts/run_lora_web_bird.py --input-dir outputs/Web-Bird/v1_web_bird_448 --config configs/web_bird_448_lora_jal_ce.yaml --output-dir outputs/Web-Bird/v1_web_bird_448/jal_ce_seed42 --methods all --seeds 42

python scripts/run_lora_web_bird.py --input-dir outputs/Web-Car/v1_web_car_0.9_448 --config configs/web_car_448_lora_jal_ce.yaml --output-dir outputs/Web-Car/v1_web_car_0.9_448/jal_ce_seed42 --methods all --seeds 42

python scripts/run_lora_web_bird.py --input-dir outputs/Web-Aircraft/v1_web_aircraft_0.9_448 --config configs/web_aircraft_448_lora_jal_ce.yaml --output-dir outputs/Web-Aircraft/v1_web_aircraft_0.9_448/jal_ce_seed42 --methods all --seeds 42
```

## Synthetic Asym40 448 命令

```bash
python scripts/run_lora_web_bird.py --input-dir outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42 --config configs/cub_asym40_lora_jal_ce.yaml --output-dir outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42/jal_ce_seed42 --methods all --seeds 42

python scripts/run_lora_web_bird.py --input-dir outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42 --config configs/stanford_cars_asym40_lora_jal_ce.yaml --output-dir outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42/jal_ce_seed42 --methods all --seeds 42

python scripts/run_lora_web_bird.py --input-dir outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42 --config configs/fgvc_aircraft_asym40_lora_jal_ce.yaml --output-dir outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42/jal_ce_seed42 --methods all --seeds 42
```

CUB 复用现有 all-noisy CE 的 `batch_size=48 / eval_batch_size=96`；Stanford Cars 和 FGVC-Aircraft 复用 `80 / 160`。其余 LoRA、优化器、448 输入和 30-epoch 设置保持一致。

## CLI 覆盖示例

```bash
python scripts/run_lora_web_bird.py \
  --input-dir outputs/Web-Bird/v1_web_bird_448 \
  --config configs/web_bird_448_lora_jal_ce.yaml \
  --output-dir outputs/Web-Bird/v1_web_bird_448/jal_ce_seed42 \
  --methods all \
  --seeds 42 \
  --jal-alpha 1.0 \
  --jal-beta 1.0 \
  --jal-a 30.0 \
  --jal-eps 1e-8
```

正式运行必须保留 `--methods all`。运行输出中的 `resolved_config.yaml`、`train_log.csv`、`lora_results.csv`、`lora_summary.json` 和 `run_summary.md` 会记录 JAL 参数及 `selection_mode=full_noisy`。
