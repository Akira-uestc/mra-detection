#!/usr/bin/env python3
from __future__ import annotations

from single_expert_direct_detection_base import build_parser, run_experiment


def parse_args():
    parser = build_parser(
        description="只使用图模块做插补与直接检测，不保留门控融合和 Transformer。",
        default_output_dir="test/outputs/graph_only_direct_detection",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(
        args=parse_args(),
        expert_type="graph",
        experiment_name="graph_only_direct_detection",
        description="图专家重构 + 直接重构误差异常检测",
    )
