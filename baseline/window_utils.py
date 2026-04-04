from __future__ import annotations

from _project_root import PROJECT_ROOT as _PROJECT_ROOT
from utils.methods.postprocess import apply_ewaf, apply_ewaf_by_segments
from utils.methods.windowing import (
    TEST_SEGMENT_LENGTH,
    TEST_WINDOW_COUNT,
    build_prompt_test_windows as build_prompt_test_windows_with_mask,
    build_prompt_test_windows_values as build_prompt_test_windows,
)

__all__ = [
    "TEST_SEGMENT_LENGTH",
    "TEST_WINDOW_COUNT",
    "apply_ewaf",
    "apply_ewaf_by_segments",
    "build_prompt_test_windows",
    "build_prompt_test_windows_with_mask",
]
