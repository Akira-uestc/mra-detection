# Sensitivity Experiments

该目录在不修改原始 `mra.py` 的前提下，独立实现 MRA 参数敏感性实验。

## 设计说明

- `training` 组参数会重新训练模型。
- `postprocess` 组参数会复用 baseline 训练得到的分数上下文，只重新执行后处理，避免重复训练。
- 每个参数都维护一组候选值。
- 可以通过接口选择要跑哪些参数。
- 可以通过接口控制每个参数保留几个候选值。
- 也可以直接显式指定某个参数要测试的具体取值。

## 用法

列出可用参数及候选值：

```bash
python sensitivity/run_sensitivity.py --list-params
```

运行全部参数的敏感性实验，每个参数默认保留 3 个值：

```bash
python sensitivity/run_sensitivity.py
```

只运行指定参数：

```bash
python sensitivity/run_sensitivity.py --params seq_len lr ewaf_alpha
```

全局控制每个参数保留几个值：

```bash
python sensitivity/run_sensitivity.py --num-values 4
```

对不同参数分别指定保留值个数：

```bash
python sensitivity/run_sensitivity.py \
  --params seq_len lr ewaf_alpha \
  --param-counts seq_len=5 lr=4 ewaf_alpha=3
```

显式指定某些参数的测试值：

```bash
python sensitivity/run_sensitivity.py \
  --params seq_len threshold_quantile \
  --param-values seq_len=30,50,80 threshold_quantile=0.9,0.95,0.99
```

只运行后处理参数：

```bash
python sensitivity/run_sensitivity.py --group postprocess
```

断点续跑，跳过已存在且配置一致的实验：

```bash
python sensitivity/run_sensitivity.py --skip-existing
```

## 输出

默认输出到 `sensitivity/outputs/`：

- `baseline/`: baseline 实验输出
- `training/<param>/<value>/`: 训练参数扫描结果
- `postprocess/<param>/<value>/`: 后处理参数扫描结果
- `selection_plan.json`: 本次实验的参数与取值计划
- `sensitivity_summary.csv`: 全部实验汇总
- `sensitivity_ranking.csv`: 参数敏感性排序
- `plots/*.png`: 每个参数的 `F1` / `FDR` 曲线

每个实验目录至少包含：

- `metrics.json`
- `test_predictions.csv`
- `anomaly_scores.png`

重新训练的实验额外保存：

- `training_curve.png`
- `score_context.npz`
