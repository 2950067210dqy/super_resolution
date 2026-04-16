"""
FAMO 权重记录与可视化工具。

这个文件只负责一件事：
把 train_step 返回的 FAMO 权重保存成独立 CSV，并根据 CSV 画出权重变化折线图。

为什么不直接塞进原来的训练 loss CSV：
1. 原来的 loss CSV 依赖 global_class.loss_label 和 metric.add(...) 的固定顺序。
2. FAMO 权重是额外诊断信息，不应该破坏已有训练曲线、验证指标和历史 CSV 格式。
3. 独立 CSV 更方便后续单独分析 FAMO 是否把权重压到某一项、是否发生振荡。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

# 训练通常跑在服务器/无显示器环境，Agg 后端可以直接保存 png，不需要 GUI。
matplotlib.use("Agg")
from matplotlib import pyplot as plt


# 统一 CSV 表头。
# ESRuRAFT_PIV 的 Generator FAMO 有 epe 任务，因此会写入 famo_generator_epe_weight。
# PIV_esrgan_RAFT 没有 Generator 侧 epe 任务，因此这一列会留空，保证两套模型 CSV 结构一致、方便横向对比。
FAMO_WEIGHT_CSV_COLUMNS = [
    "epoch",
    "batch_index",
    "global_step",
    "generator_content_weight",
    "generator_adversarial_weight",
    "generator_pixel_weight",
    "generator_consistency_weight",
    "generator_epe_weight",
    "joint_sr_weight",
    "joint_raft_weight",
]


def _to_csv_value(value: Any) -> str:
    """
    把 Python 数值安全转换成 CSV 字符串。

    参数：
        value: 可能是 float、int、None 或其他对象。

    返回：
        str: 用于写入 CSV 的文本；None 会写为空字符串。
    """
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _ensure_famo_weight_csv(csv_path: str | Path) -> Path:
    """
    确保 FAMO 权重 CSV 文件存在，且第一次创建时写入表头。

    参数：
        csv_path: CSV 文件路径。

    返回：
        Path: 标准化后的 Path 对象。
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FAMO_WEIGHT_CSV_COLUMNS)
            writer.writeheader()
    return path


def append_famo_weight_row(
    csv_path: str | Path,
    *,
    epoch: int,
    batch_index: int,
    global_step: int,
    loss_dict: dict[str, Any],
) -> bool:
    """
    从 train_step 返回的 loss_dict 中提取 FAMO 权重，并追加写入 CSV。

    参数：
        csv_path: FAMO 权重 CSV 保存路径。
        epoch: 当前 epoch，按训练循环中的 0-based epoch 保存，便于和原代码一致。
        batch_index: 当前 epoch 内的 batch 下标。
        global_step: 全局 step，通常为 epoch * len(train_loader) + batch_index。
        loss_dict: train_step 返回的日志字典，里面包含 famo_*_weight 字段。

    返回：
        bool: True 表示本 batch 成功写入；False 表示当前没有 FAMO 权重字段，跳过写入。
    """
    # 如果 USE_FAMO=False，loss_dict 不会包含这些字段；此时直接跳过，不影响原训练逻辑。
    if "famo_generator_content_weight" not in loss_dict:
        return False

    path = _ensure_famo_weight_csv(csv_path)
    row = {
        "epoch": epoch,
        "batch_index": batch_index,
        "global_step": global_step,
        "generator_content_weight": loss_dict.get("famo_generator_content_weight"),
        "generator_adversarial_weight": loss_dict.get("famo_generator_adversarial_weight"),
        "generator_pixel_weight": loss_dict.get("famo_generator_pixel_weight"),
        "generator_consistency_weight": loss_dict.get("famo_generator_consistency_weight"),
        "generator_epe_weight": loss_dict.get("famo_generator_epe_weight"),
        "joint_sr_weight": loss_dict.get("famo_joint_sr_weight"),
        "joint_raft_weight": loss_dict.get("famo_joint_raft_weight"),
    }

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FAMO_WEIGHT_CSV_COLUMNS)
        writer.writerow({key: _to_csv_value(row.get(key)) for key in FAMO_WEIGHT_CSV_COLUMNS})
    return True


def _read_numeric_famo_csv(csv_path: str | Path) -> tuple[list[int], dict[str, list[float]]]:
    """
    读取 FAMO 权重 CSV，并把可用数值列转换成折线图需要的 x/y 序列。

    参数：
        csv_path: FAMO 权重 CSV 路径。

    返回：
        steps: global_step 序列。
        series: 每个权重列对应的 y 序列；空列会自动跳过。
    """
    path = Path(csv_path)
    if not path.exists():
        return [], {}

    steps: list[int] = []
    series: dict[str, list[float]] = {
        "generator_content_weight": [],
        "generator_adversarial_weight": [],
        "generator_pixel_weight": [],
        "generator_consistency_weight": [],
        "generator_epe_weight": [],
        "joint_sr_weight": [],
        "joint_raft_weight": [],
    }

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                step = int(float(row["global_step"]))
            except (KeyError, TypeError, ValueError):
                continue

            row_values: dict[str, float | None] = {}
            has_any_value = False
            for key in series:
                text = row.get(key, "")
                if text == "":
                    row_values[key] = None
                    continue
                try:
                    row_values[key] = float(text)
                    has_any_value = True
                except ValueError:
                    row_values[key] = None

            if not has_any_value:
                continue

            steps.append(step)
            for key in series:
                value = row_values[key]
                # matplotlib 可以用 NaN 断开曲线；这里用 float("nan") 保持所有序列长度一致。
                series[key].append(float("nan") if value is None else value)

    # 如果某一列全是 NaN，说明当前模型没有这个任务，例如 PIV_esrgan_RAFT 没有 epe；
    # 这种列不画，避免图例里出现一条没有意义的空曲线。
    cleaned_series = {
        key: values
        for key, values in series.items()
        if any(value == value for value in values)
    }
    return steps, cleaned_series


def save_famo_weight_plot(csv_path: str | Path, png_path: str | Path, title: str) -> bool:
    """
    根据 FAMO 权重 CSV 保存折线图。

    参数：
        csv_path: FAMO 权重 CSV 路径。
        png_path: 输出 png 路径。
        title: 图标题，通常包含模型名、类别、数据类型和尺度。

    返回：
        bool: True 表示成功保存；False 表示 CSV 不存在或暂无有效数据。
    """
    steps, series = _read_numeric_famo_csv(csv_path)
    if not steps or not series:
        return False

    out_path = Path(png_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(9.5, 5.2), dpi=160)
    for key, values in series.items():
        ax.plot(steps, values, linewidth=1.6, label=key)

    ax.set_title(title)
    ax.set_xlabel("global step")
    ax.set_ylabel("FAMO weight")
    ax.grid(True, alpha=0.28, linestyle="--", linewidth=0.8)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True
