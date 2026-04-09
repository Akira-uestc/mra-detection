#!/usr/bin/env python3
from __future__ import annotations

from single_expert_transformer_base import build_parser, run_experiment


def parse_args():
    parser = build_parser(
        description="只使用频域模块做插补，不保留融合和额外 trick，最后直接使用 Transformer 做异常检测。",
        default_output_dir="test/outputs/freq_only_transformer",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(
        args=parse_args(),
        expert_type="freq",
        experiment_name="freq_only_transformer",
        description="频域专家插补 + 简单 Transformer 异常检测",
    )
