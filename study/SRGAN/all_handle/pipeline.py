from __future__ import annotations

try:
    from .pipeline_cli import apply_cli_overrides, build_arg_parser, main
    from .pipeline_core import AllHandlePipeline, GroupContext, SampleBundle
except ImportError:
    # 允许用户在 SRGAN 根目录直接执行：python all_handle/pipeline.py
    from pipeline_cli import apply_cli_overrides, build_arg_parser, main
    from pipeline_core import AllHandlePipeline, GroupContext, SampleBundle


__all__ = [
    "AllHandlePipeline",
    "GroupContext",
    "SampleBundle",
    "apply_cli_overrides",
    "build_arg_parser",
    "main",
]


if __name__ == "__main__":
    main()
