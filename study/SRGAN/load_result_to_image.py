"""
结果指标读取与折线图绘制脚本。

功能说明：
1. 从 Linux 路径 `/train_datas/` 下读取指定实验目录的结果。
2. 自动扫描实验目录下各类别的 `image_pair` 和 `flo` 子目录。
3. 读取每个 `predict_all/metrics_all.csv` 中的逐样本指标。
4. 为 ESRGAN 与 SRGAN 在相同类别、相同数据类型、相同 scale 下绘制对比折线图。
5. 将图像与聚合后的均值 CSV 输出到 `/train_datas/result_plots/`。

当前默认读取的实验目录：
- /train_datas/esrganv1
- /train_datas/srganv1
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

# 服务器环境通常没有桌面显示，使用非交互式后端避免报错。
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 默认输入根目录。用户已明确说明结果保存在 Linux 路径 /train_datas/ 下。
DEFAULT_RESULT_ROOT = Path("/train_datas")

# 默认需要对比的实验名。
DEFAULT_EXPERIMENTS = ("esrganv1", "srganv1")

# 绘图输出目录。
DEFAULT_OUTPUT_DIR = DEFAULT_RESULT_ROOT / "result_plots"

# metrics_all.csv 中的数值指标列。
METRIC_COLUMNS = (
    "mse",
    "psnr",
    "energy_spectrum_mse",
    "r2",
    "ssim",
    "tke_acc",
    "nrmse",
)


def _safe_float(value: str) -> float | None:
    """
    将字符串安全转换为浮点数。

    参数：
        value: CSV 中读取到的字符串。

    返回：
        - 转换成功时返回 float
        - 空字符串或非法值时返回 None
    """
    try:
        if value is None:
            return None
        value = str(value).strip()
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None



def _sanitize_filename(name: str) -> str:
    """
    将文本清洗为适合文件名的格式。

    参数：
        name: 原始文件名片段。

    返回：
        仅保留字母、数字、下划线、横线和点号的安全文件名。
    """
    return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in name)



def _extract_scale_name(scale_dir: Path) -> str:
    """
    从 scale 目录提取显示名称。

    参数：
        scale_dir: 类似 `scale_4` 的目录路径。

    返回：
        用于标题和输出路径的 scale 名称，例如 `scale_4`。
    """
    return scale_dir.name



def find_metrics_csv_files(
    result_root: Path,
    experiments: Iterable[str],
) -> list[dict]:
    """
    扫描指定实验目录下的 metrics_all.csv 文件。

    目录结构约定：
        /train_datas/<experiment>/<class_name>/<data_type>/scale_x/predict_all/metrics_all.csv

    参数：
        result_root: 结果根目录，默认 `/train_datas`。
        experiments: 需要扫描的实验名列表。

    返回：
        每个元素是一条文件元数据记录，包含：
        - experiment
        - class_name
        - data_type
        - scale_name
        - csv_path
    """
    discovered_files: list[dict] = []

    for experiment in experiments:
        experiment_dir = result_root / experiment
        if not experiment_dir.exists():
            continue

        # 遍历类别目录，例如 cylinder、backstep 等。
        for class_dir in sorted(p for p in experiment_dir.iterdir() if p.is_dir()):
            # 遍历数据类型目录，仅处理 image_pair 和 flo。
            for data_type_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
                if data_type_dir.name not in {"image_pair", "flo"}:
                    continue

                # 遍历 scale 目录，例如 scale_4。
                for scale_dir in sorted(p for p in data_type_dir.iterdir() if p.is_dir()):
                    csv_path = scale_dir / "predict_all" / "metrics_all.csv"
                    if not csv_path.exists():
                        continue

                    discovered_files.append(
                        {
                            "experiment": experiment,
                            "class_name": class_dir.name,
                            "data_type": data_type_dir.name,
                            "scale_name": _extract_scale_name(scale_dir),
                            "csv_path": csv_path,
                        }
                    )

    return discovered_files



def load_metrics_rows(csv_path: Path) -> list[dict]:
    """
    读取单个 metrics_all.csv 并解析为统一结构。

    参数：
        csv_path: metrics_all.csv 的完整路径。

    返回：
        逐样本记录列表。

    说明：
        - 会自动跳过 `sample_id == "MEAN"` 的聚合行，避免影响折线图。
        - 会保留 pair_type，便于 image_pair 的 previous / next 分组。
    """
    rows: list[dict] = []

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            # MEAN 是整份 CSV 的聚合统计行，不适合作为逐样本折线图数据。
            if str(row.get("sample_id", "")).strip().upper() == "MEAN":
                continue

            parsed_row = {
                "sample_id": str(row.get("sample_id", "")).strip(),
                "pair_type": str(row.get("pair_type", "")).strip(),
            }

            # 逐个解析数值指标列。
            for metric_name in METRIC_COLUMNS:
                parsed_row[metric_name] = _safe_float(row.get(metric_name, ""))

            rows.append(parsed_row)

    return rows



def _build_x_positions(rows: list[dict]) -> list[int]:
    """
    为折线图生成横轴位置。

    参数：
        rows: 逐样本记录列表。

    返回：
        形如 [0, 1, 2, ...] 的整数序列。
    """
    return list(range(len(rows)))



def plot_metric_lines(
    output_path: Path,
    title: str,
    ylabel: str,
    model_to_rows: dict[str, list[dict]],
    metric_name: str,
) -> None:
    """
    绘制单个指标的多模型对比折线图。

    参数：
        output_path: 图像保存路径。
        title: 图标题。
        ylabel: 纵轴标签。
        model_to_rows: 不同实验名到逐样本记录的映射。
        metric_name: 需要绘制的指标列名。
    """
    plt.figure(figsize=(14, 6))

    # 为不同实验固定颜色与样式，方便比较。
    style_map = {
        "esrganv1": {"color": "#d1495b", "linewidth": 1.4},
        "srganv1": {"color": "#00798c", "linewidth": 1.4},
    }

    plotted = False

    for model_name, rows in sorted(model_to_rows.items()):
        y_values = [row.get(metric_name) for row in rows]
        x_values = _build_x_positions(rows)

        # 过滤非法数值，避免 matplotlib 绘图警告。
        valid_pairs = [(x, y) for x, y in zip(x_values, y_values) if y is not None]
        if not valid_pairs:
            continue

        x_plot = [item[0] for item in valid_pairs]
        y_plot = [item[1] for item in valid_pairs]
        style = style_map.get(model_name, {"linewidth": 1.2})

        plt.plot(
            x_plot,
            y_plot,
            label=model_name,
            marker="",
            **style,
        )
        plotted = True

    if not plotted:
        plt.close()
        return

    plt.title(title)
    plt.xlabel("sample index")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()



def write_mean_summary_csv(
    output_path: Path,
    grouped_rows: dict[tuple[str, str, str], dict[str, list[dict]]],
) -> None:
    """
    将每组图对应的均值统计输出到汇总 CSV。

    参数：
        output_path: 汇总 CSV 保存路径。
        grouped_rows:
            key 为 `(class_name, data_type, scale_name)`，
            value 为 `experiment -> rows` 的映射。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["class_name", "data_type", "scale_name", "experiment", *METRIC_COLUMNS]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for (class_name, data_type, scale_name), model_to_rows in sorted(grouped_rows.items()):
            for experiment, rows in sorted(model_to_rows.items()):
                row_dict = {
                    "class_name": class_name,
                    "data_type": data_type,
                    "scale_name": scale_name,
                    "experiment": experiment,
                }

                for metric_name in METRIC_COLUMNS:
                    metric_values = [item[metric_name] for item in rows if item.get(metric_name) is not None]
                    row_dict[metric_name] = (
                        sum(metric_values) / len(metric_values) if metric_values else ""
                    )

                writer.writerow(row_dict)



def load_result_to_image(
    result_root: str | Path = DEFAULT_RESULT_ROOT,
    experiments: Iterable[str] = DEFAULT_EXPERIMENTS,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> None:
    """
    读取 ESRGAN / SRGAN 的 metrics_all.csv，并绘制逐样本折线图。

    参数：
        result_root: 训练结果根目录，默认 `/train_datas`。
        experiments: 需要读取的实验目录名，默认读取 `esrganv1` 与 `srganv1`。
        output_dir: 图像输出目录，默认 `/train_datas/result_plots`。

    输出内容：
        - 每个类别、数据类型、scale、pair_type 下，各项指标的对比折线图 PNG
        - 一份均值汇总 CSV，便于后续快速查看不同实验整体表现
    """
    result_root = Path(result_root)
    output_dir = Path(output_dir)

    discovered_files = find_metrics_csv_files(result_root=result_root, experiments=experiments)
    if not discovered_files:
        raise FileNotFoundError(
            f"在 {result_root} 下没有找到实验 {tuple(experiments)} 对应的 metrics_all.csv 文件。"
        )

    # 第一级分组：按类别 + 数据类型 + scale 组织。
    grouped_rows: dict[tuple[str, str, str], dict[str, list[dict]]] = defaultdict(dict)

    for file_info in discovered_files:
        rows = load_metrics_rows(file_info["csv_path"])
        group_key = (
            file_info["class_name"],
            file_info["data_type"],
            file_info["scale_name"],
        )
        grouped_rows[group_key][file_info["experiment"]] = rows

    # 将 image_pair 再按 previous / next 拆开，flo 则单独成组。
    expanded_groups: dict[tuple[str, str, str, str], dict[str, list[dict]]] = defaultdict(dict)

    for (class_name, data_type, scale_name), model_to_rows in grouped_rows.items():
        if data_type == "image_pair":
            for pair_type in ("previous", "next"):
                for experiment, rows in model_to_rows.items():
                    filtered_rows = [row for row in rows if row.get("pair_type") == pair_type]
                    if filtered_rows:
                        expanded_groups[(class_name, data_type, scale_name, pair_type)][experiment] = filtered_rows
        else:
            for experiment, rows in model_to_rows.items():
                filtered_rows = [row for row in rows if row.get("pair_type") == "flo" or row.get("pair_type") == ""]
                if filtered_rows:
                    expanded_groups[(class_name, data_type, scale_name, "flo")][experiment] = filtered_rows

    # 为每个分组绘制各项指标折线图。
    for (class_name, data_type, scale_name, pair_type), model_to_rows in sorted(expanded_groups.items()):
        safe_pair_type = _sanitize_filename(pair_type)
        safe_class_name = _sanitize_filename(class_name)

        for metric_name in METRIC_COLUMNS:
            title = (
                f"{class_name} | {data_type} | {scale_name} | {pair_type} | {metric_name}"
            )
            output_path = (
                output_dir
                / safe_class_name
                / data_type
                / scale_name
                / safe_pair_type
                / f"{metric_name}_line.png"
            )
            plot_metric_lines(
                output_path=output_path,
                title=title,
                ylabel=metric_name,
                model_to_rows=model_to_rows,
                metric_name=metric_name,
            )

    # 额外输出一份均值汇总表，便于后续查看整体表现。
    write_mean_summary_csv(
        output_path=output_dir / "metrics_mean_summary.csv",
        grouped_rows=grouped_rows,
    )

    print(f"Plot images saved to: {output_dir}")
    print(f"Mean summary CSV saved to: {output_dir / 'metrics_mean_summary.csv'}")



def main() -> None:
    """
    脚本入口函数。

    直接运行本文件时，将按默认配置：
    - 从 `/train_datas/esrganv1`
    - 从 `/train_datas/srganv1`
    读取结果并绘图。
    """
    load_result_to_image()


if __name__ == "__main__":
    main()
