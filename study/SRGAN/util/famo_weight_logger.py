from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch


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
