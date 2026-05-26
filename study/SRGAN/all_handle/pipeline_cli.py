from __future__ import annotations

import argparse

try:
    from .global_class import global_data
    from .pipeline_core import AllHandlePipeline
except ImportError:
    # 允许用户在 SRGAN 根目录直接执行：python all_handle/pipeline.py
    from global_class import global_data
    from pipeline_core import AllHandlePipeline


def build_arg_parser() -> argparse.ArgumentParser:
    """创建 all_handle 后处理命令行参数；pipeline.py 只负责调用这里的 main。"""

    parser = argparse.ArgumentParser(
        description="Generate all_handle comparison figures from existing npy/png experiment outputs."
    )
    parser.add_argument("--class-name", action="append", help="Only process the given class name, e.g. class_1.")
    parser.add_argument("--split", action="append", help="Only process predict_all or test_all.")
    parser.add_argument("--category", action="append", help="Only process the given category name.")
    parser.add_argument("--sample", action="append", help="Only process the given sample name.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit composite sample count per category.")
    parser.add_argument("--list-only", action="store_true", help="Only list discovered groups without drawing figures.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress printing for this run.")
    return parser


def apply_cli_overrides(cfg: type[global_data.all_handle], args: argparse.Namespace) -> None:
    """命令行参数只覆盖本次运行，不改变 global_class.py 文件内容。"""

    if args.class_name:
        cfg.CLASS_NAMES = tuple(args.class_name)
    if args.split:
        cfg.SPLIT_NAMES = tuple(args.split)
    if args.category:
        cfg.CATEGORY_FILTER = tuple(args.category)
    if args.sample:
        cfg.SAMPLE_FILTER = tuple(args.sample)
    if args.max_samples is not None:
        cfg.MAX_SAMPLE_COMPOSITES_PER_CATEGORY = args.max_samples
    if args.no_progress:
        cfg.PROGRESS_ENABLED = False


def main() -> None:
    """all_handle 统一入口：解析参数、创建 pipeline，并按 list/run 模式执行。"""

    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = global_data.all_handle
    apply_cli_overrides(cfg, args)
    pipeline = AllHandlePipeline(cfg, enable_plotting=not args.list_only)
    groups = pipeline.discover_groups()
    if args.list_only:
        for group in groups:
            print(
                f"{group.comparison_name}/{group.class_name}/{group.split_name}/{group.category_name}: "
                f"{', '.join(group.experiment_dirs.keys())}"
            )
        pipeline.write_summary(groups)
        return
    pipeline.run_all()


__all__ = ["apply_cli_overrides", "build_arg_parser", "main"]
