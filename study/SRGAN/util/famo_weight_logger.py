from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch


def _extract_generator_famo_weights(loss_dict: dict[str, Any]) -> dict[str, float]:
    """
    从 train_step 返回的 loss_dict 中提取 Generator FAMO 权重。

    PIV_A_Esrgan / ESRuRAFT_PIV 的模型里，FAMO 权重日志字段统一命名为：
        generator_famo_{task_name}_weight

    训练管线会在每个 batch 后无条件调用 append_famo_weight_row(...)。
    当 USE_FAMO=False 时，loss_dict 里不会有这些字段，本函数返回空字典，
    后续写 CSV / 画图函数也会直接跳过，保证不开 FAMO 时训练行为不被影响。
    """
    weights: dict[str, float] = {}
    prefix = "generator_famo_"
    suffix = "_weight"
    for key, value in loss_dict.items():
        if not (key.startswith(prefix) and key.endswith(suffix)):
            continue
        # CSV 列名里保留 task_name，避免前缀太长导致后续画图图例臃肿。
        task_name = key[len(prefix):-len(suffix)]
        if not task_name:
            continue
        if torch.is_tensor(value):
            value = value.detach().cpu().item()
        weights[f"famo_{task_name}_weight"] = float(value)
    return weights


def append_famo_weight_row(
    csv_path: str | Path,
    epoch: int,
    batch_index: int,
    global_step: int,
    loss_dict: dict[str, Any],
) -> None:
    """
    逐 batch 追加 Generator FAMO 权重。

    为什么需要这个函数：
        save_famo_weight_snapshot(...) 是按 epoch 保存一次快照；
        但 FAMO 权重通常按 step 更新，只看 epoch 快照会错过 batch 内的变化。

    兼容策略：
        - USE_FAMO=False：loss_dict 没有 generator_famo_*_weight 字段，直接 return；
        - USE_FAMO=True：写入 epoch、batch_index、global_step 和每个任务权重；
        - 如果 matplotlib / 画图失败，不影响这里的 CSV 记录。
    """
    weights = _extract_generator_famo_weights(loss_dict)
    if not weights:
        return

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "epoch": int(epoch) + 1,  # 外部训练循环 epoch 是 0-based；CSV 里保存人眼更直观的 1-based epoch。
        "batch_index": int(batch_index),
        "global_step": int(global_step),
        **weights,
    }
    header = list(row.keys())

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    if not write_header:
        # 正常情况下同一次实验的 FAMO 任务列不会变化；
        # 这里仍读取旧表头，避免追加时因为列顺序不同导致 CSV 混乱。
        with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames:
                header = list(reader.fieldnames)

    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in header})


def save_famo_weight_plot(
    csv_path: str | Path,
    png_path: str | Path,
    title: str = "Generator FAMO Weights",
) -> None:
    """
    根据逐 batch FAMO 权重 CSV 绘制折线图。

    这个函数可以被训练循环每个 epoch 结束时无条件调用：
        - CSV 不存在：说明 USE_FAMO=False 或还没有记录，直接 return；
        - CSV 存在但没有权重列：直接 return；
        - matplotlib 后端不可用：吞掉异常，不中断训练。
    """
    csv_path = Path(csv_path)
    png_path = Path(png_path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return

    try:
        with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            rows = list(reader)
            fieldnames = reader.fieldnames or []
        if not rows:
            return

        x_key = "global_step" if "global_step" in fieldnames else "epoch"
        weight_keys = [
            name for name in fieldnames
            if name.startswith("famo_") and name.endswith("_weight")
        ]
        if not weight_keys:
            return

        import matplotlib.pyplot as plt

        x_values = [int(float(row[x_key])) for row in rows]
        plt.figure(figsize=(11, 6))
        for key in weight_keys:
            y_values = [float(row[key]) for row in rows if row.get(key, "") != ""]
            if len(y_values) != len(x_values):
                # 如果历史 CSV 中某列有缺失，就跳过该列，避免画出错位曲线。
                continue
            label = key.removeprefix("famo_").removesuffix("_weight")
            plt.plot(x_values, y_values, linewidth=1.5, label=label)
        plt.xlabel(x_key)
        plt.ylabel("FAMO weight")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        png_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(png_path, dpi=160)
        plt.close()
    except Exception:
        # 画图是辅助产物，失败时不应该中断训练；CSV 已经保留了可复现数据。
        return


def save_famo_weight_snapshot(
    model: Any,
    epoch: int,
    output_dir: str | Path,
    file_prefix: str = "famo_weights",
    logger: Any | None = None,
) -> None:
    """
    保存 Generator FAMO 权重的 CSV 和折线图。

    如果当前模型没有启用 FAMO（model.generator_famo 为 None），函数会直接返回；
    因此训练管线可以无条件调用它，不会影响手动权重训练。
    """
    famo = getattr(model, "generator_famo", None)
    task_names = getattr(model, "generator_famo_task_names", None)
    if famo is None or not task_names:
        return

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{file_prefix}.csv"
    png_path = out_dir / f"{file_prefix}.png"

    with torch.no_grad():
        weights = famo.weights.detach().cpu().tolist()

    header = ["epoch"] + [f"famo_{name}_weight" for name in task_names]
    row = [int(epoch)] + [float(weight) for weight in weights]

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

    try:
        import matplotlib.pyplot as plt

        epochs = []
        series = {name: [] for name in task_names}
        with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for item in reader:
                epochs.append(int(float(item["epoch"])))
                for name in task_names:
                    series[name].append(float(item[f"famo_{name}_weight"]))

        plt.figure(figsize=(10, 6))
        for name in task_names:
            plt.plot(epochs, series[name], marker="o", linewidth=1.5, markersize=3, label=name)
        plt.xlabel("epoch")
        plt.ylabel("FAMO weight")
        plt.title("Generator FAMO Weight Schedule")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(png_path, dpi=160)
        plt.close()
    except Exception as exc:
        # CSV 是主要记录；画图失败通常是 matplotlib 后端或环境问题，不应该中断训练。
        if logger is not None:
            logger.warning(f"FAMO weight csv saved, but plot failed: {exc}")
