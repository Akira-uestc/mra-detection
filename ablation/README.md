# Ablation Experiments

该目录在不修改原始 `mra.py` 的前提下，独立实现 MRA 模型的消融实验。

## Experiments

- `A00_full`: 完整基线
- `A01_no_gcn_expert`: 去掉 GCN 专家
- `A02_no_freq_expert`: 去掉频域专家
- `A03_mean_fusion`: 双专家均值融合
- `A04_no_gate_rate`: 门控层不使用采样率嵌入
- `A05_no_detector_rate`: 检测器不使用采样率嵌入
- `A06_no_phase`: 检测器不使用 phase 特征
- `A07_no_rate_aware`: 同时去掉 gate 与 detector 的采样率感知
- `A08_no_ewaf`: 异常检测方式消融 D4，去掉 EWAF 平滑

## Usage

列出实验：

```bash
python ablation/run_ablation.py --list-experiments
```

运行全部实验：

```bash
python ablation/run_ablation.py
```

仅运行指定实验：

```bash
python ablation/run_ablation.py --experiments A00_full A08_no_ewaf
```

结果默认输出到 `ablation/outputs/`，每个实验会保存：

- `metrics.json`
- `test_predictions.csv`
- `anomaly_scores.png`
- `training_curve.png`
- `train_gate_weights.csv`
- `test_gate_weights.csv`
- `ablation_summary.csv`（汇总表）
