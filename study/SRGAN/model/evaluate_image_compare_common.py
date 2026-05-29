from pathlib import Path
import csv
import math
import base64
import shutil

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import save_image

from study.SRGAN.util.image_util import add_vertical_separator, build_triplet_row


def save_svg_sidecar_for_png(png_path) -> Path | None:
    """
    为已有 PNG 生成同名 SVG sidecar。

    说明：
        很多 evaluate_all 图片是 tensor / PIL 拼出的栅格图，不能无损重建成真正的矢量曲线；
        这里生成的是 SVG 容器并嵌入原 PNG。这样后处理/论文排版软件能按矢量图文件管理，
        同时不改变、不删除原 PNG 的像素内容和已有可视化流程。
        因此开启轻量落图模式后，最终会同时保留 xxx.png 和 xxx.svg 两份文件。
    """
    png_path = Path(png_path)
    if not png_path.exists() or png_path.suffix.lower() != ".png":
        return None
    svg_path = png_path.with_suffix(".svg")
    try:
        with Image.open(png_path) as image:
            width, height = image.size
        encoded = base64.b64encode(png_path.read_bytes()).decode("ascii")
        svg_text = (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
            f'  <image width="{width}" height="{height}" '
            f'href="data:image/png;base64,{encoded}"/>\n'
            f'</svg>\n'
        )
        svg_path.write_text(svg_text, encoding="utf-8")
        return svg_path
    except Exception:
        return None


def save_svg_sidecars_for_png_tree(root_dir) -> int:
    """
    递归为目录下所有 PNG 生成 SVG sidecar，返回成功生成数量。

    evaluate_all 的“每类只保留自然顺序前 10 个样本”模式会在删除未保留样本后调用这里，
    因此最终保留下来的样本图、类别汇总图、总体汇总图都会有对应 SVG。
    """
    root_dir = Path(root_dir)
    if not root_dir.exists():
        return 0
    saved_count = 0
    for png_path in root_dir.rglob("*.png"):
        if save_svg_sidecar_for_png(png_path) is not None:
            saved_count += 1
    return saved_count


def _safe_metric_float(row: dict, key: str) -> float:
    """从一行指标里安全读取 float；失败时返回 NaN。"""
    try:
        value = float(row.get(key, float("nan")))
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _best_only_row_score(rows_for_sample: list[dict]) -> float:
    """
    兼容旧 summary 的样本分数读取函数。

    当前 EVALUATE_ALL_SAVE_BEST_ONLY=True 已改为“每个类别按自然遍历顺序保留前 10 个
    sample”，不再按指标排序挑最佳 sample。这个函数只用于 summary CSV 中记录一个参考分数，
    方便回看被保留样本的 VAL_AEE / VAL_C_AEE / energy_spectrum_mse 情况，不参与保留顺序。
    """
    for key in ("VAL_AEE", "VAL_C_AEE", "energy_spectrum_mse"):
        values = [_safe_metric_float(row, key) for row in rows_for_sample]
        values = [value for value in values if np.isfinite(value)]
        if values:
            return float(np.mean(values))
    return float("inf")


def prune_evaluate_all_to_best_sample_dirs(
        output_root,
        rows,
        logger=None,
        *,
        save_svg_sidecars: bool = True,
) -> dict:
    """
    evaluate_all 轻量落图保留器：每个类别只保留自然遍历顺序前 10 个样本目录。

    重要约束：
        - 只处理本次 rows 中出现过的 sample_id 目录；
        - 不删除 metrics.csv、ALL_CLASS*.CSV、类别/总体汇总图等非样本目录文件；
        - 指标 rows/CSV/均值已经在外部按全量样本计算，本函数只影响磁盘上的样本图像和 NPY；
        - 删除前 10 之外的样本目录后，最终保留目录里的 PNG 原图会继续存在；
        - 同时为最终保留的 PNG 生成同名 SVG sidecar，因此保留样本会同时有 PNG、SVG 和 NPY。
    """
    keep_top_k = 10
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if not output_root.exists():
        return {}

    sample_rows: dict[tuple[str, str], list[dict]] = {}
    sample_order_by_class: dict[str, list[str]] = {}
    for row in rows or []:
        class_name = str(row.get("class_name", "")).strip()
        sample_id = str(row.get("sample_id", "")).strip()
        if not class_name or not sample_id or sample_id in {"MEAN", "CLASS_MEAN"}:
            continue
        sample_key = (class_name, sample_id)
        if sample_key not in sample_rows:
            # Python dict 保序，这里显式保存每个类别内 sample 第一次出现在 rows 中的顺序；
            # EVALUATE_ALL_SAVE_BEST_ONLY=True 时只保留这个自然顺序的前 keep_top_k 个。
            sample_order_by_class.setdefault(class_name, []).append(sample_id)
        sample_rows.setdefault(sample_key, []).append(row)

    root_resolved = output_root.resolve()
    kept_summary = {
        class_name: [
            {
                "rank": rank + 1,
                "sample_id": sample_id,
                "score": _best_only_row_score(sample_rows.get((class_name, sample_id), [])),
            }
            for rank, sample_id in enumerate(sample_ids[:keep_top_k])
        ]
        for class_name, sample_ids in sample_order_by_class.items()
    }

    deleted_count = 0
    for class_name, infos in kept_summary.items():
        keep_sample_ids = {info["sample_id"] for info in infos}
        final_class_dir = output_root / class_name
        if not final_class_dir.exists() or not final_class_dir.is_dir():
            continue
        # best-only 保存完成后，正式类别目录里只允许保留该类别自然顺序前 keep_top_k 个 sample 子目录；
        # 类别级 PNG/CSV 都是文件，不会被这里删除。
        for sample_dir in final_class_dir.iterdir():
            if not sample_dir.is_dir() or sample_dir.name in keep_sample_ids:
                continue
            try:
                # 只允许删除 output_root 内部的样本目录，避免路径异常时误删外部文件。
                if root_resolved not in sample_dir.resolve().parents:
                    continue
                shutil.rmtree(sample_dir)
                deleted_count += 1
            except Exception as exc:
                if logger is not None:
                    logger.warning(f"[evaluate_all best-only] failed to remove {sample_dir}: {exc}")

    summary_path = output_root / "evaluate_all_best_samples.csv"
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "natural_rank", "sample_id", "reference_score"])
        for class_name, infos in sorted(kept_summary.items()):
            for info in infos:
                writer.writerow([class_name, info["rank"], info["sample_id"], info["score"]])

    svg_count = save_svg_sidecars_for_png_tree(output_root) if save_svg_sidecars else 0
    if logger is not None:
        kept_sample_dirs = sum(len(infos) for infos in kept_summary.values())
        logger.info(
            f"[evaluate_all best-only] kept_classes={len(kept_summary)}, "
            f"kept_sample_dirs={kept_sample_dirs}, top_k_per_class={keep_top_k}, "
            f"removed_sample_dirs={deleted_count}, svg_sidecars={svg_count}, summary={summary_path}"
        )
    return kept_summary


def _read_optional_float_from_global(global_data, attr_name: str, default=None):
    """
    从 global_data.esrgan 中读取可选浮点超参数。

    None / 空字符串 / 非有限值都回退到 default。这样全局变量可以把某个上限设置为 None，
    表示“下限固定、上限按当前图的数据自动补齐”，既能固定坐标起点，又不破坏旧实验的显示范围。
    """
    if global_data is None or not hasattr(global_data, "esrgan"):
        return default
    value = getattr(global_data.esrgan, attr_name, default)
    if value is None or value == "":
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _nice_tick_interval(span: float, target_ticks: int = 6) -> float:
    """
    根据坐标轴范围给出稳定的“好看”刻度间隔。

    当用户没有显式设置 *_TICK_INTERVAL 时使用，避免不同图因为 Matplotlib 自动 tick
    策略不同而出现横纵坐标间隔不一致。
    """
    span = float(abs(span))
    if not np.isfinite(span) or span <= 0:
        return 1.0
    raw = span / max(int(target_ticks), 1)
    exponent = math.floor(math.log10(raw))
    fraction = raw / (10 ** exponent)
    if fraction <= 1:
        nice_fraction = 1.0
    elif fraction <= 2:
        nice_fraction = 2.0
    elif fraction <= 5:
        nice_fraction = 5.0
    else:
        nice_fraction = 10.0
    return nice_fraction * (10 ** exponent)


def _set_fixed_tick_interval(ax, axis_name: str, axis_min: float, axis_max: float, interval: float) -> None:
    """
    给 x/y 轴设置固定 tick 间隔，tick 范围严格跟随固定的坐标起止范围。
    """
    if not (np.isfinite(axis_min) and np.isfinite(axis_max) and np.isfinite(interval)):
        return
    if interval <= 0 or axis_max <= axis_min:
        return
    ticks = np.arange(axis_min, axis_max + interval * 0.5, interval, dtype=np.float64)
    ticks = ticks[(ticks >= axis_min - 1e-9) & (ticks <= axis_max + 1e-9)]
    if ticks.size == 0:
        ticks = np.asarray([axis_min, axis_max], dtype=np.float64)
    if axis_name == "x":
        ax.set_xticks(ticks)
    else:
        ax.set_yticks(ticks)


def _finite_axis_values(values) -> np.ndarray:
    """把坐标候选值整理成有限 float 数组；空数组由调用方按默认范围兜底。"""
    if values is None:
        return np.asarray([], dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def is_tbl_category(category_name) -> bool:
    """
    判断当前类别是否是 TBL。

    evaluate_all 传入的是 sample bucket，test_all 传入的是 dataset_name；
    这里统一做小写和 `-` 到 `_` 的归一化，避免每个调用点重复判断。
    """
    text = "" if category_name is None else str(category_name).strip().lower()
    text = text.replace("-", "_")
    return text == "tbl" or text.endswith("_tbl")


def is_twcf_category(category_name) -> bool:
    """
    判断当前类别是否是 TWCF。

    额外兼容历史里偶尔写成 `twcl` 的拼写，只影响坐标轴配置选择，不改变 dataset 名称。
    """
    text = "" if category_name is None else str(category_name).strip().lower()
    text = text.replace("-", "_")
    return text in {"twcf", "twcl"} or text.endswith("_twcf") or text.endswith("_twcl")


def is_tbl_twcf_category(category_name) -> bool:
    """
    判断当前误差直方图是否属于 TBL/TWCF 大图数据集。

    evaluate_all 里传入的一般是 class bucket，test_all 里传入的是 dataset_name；
    这里统一做小写与常见拼写兼容，避免各分支重复写判断逻辑。
    """
    return is_tbl_category(category_name) or is_twcf_category(category_name)


def select_error_hist_axis_prefix(base_prefix: str, category_name=None) -> str:
    """
    按类别选择误差直方图坐标轴配置前缀。

    三套配置规则：
    - 普通类别：继续使用原来的 `{BASE_PREFIX}_*`，例如 `FLOW_ERROR_HIST_X_MIN`；
    - TBL：使用 `{BASE_PREFIX}_TBL_*`，例如 `FLOW_ERROR_HIST_TBL_X_MIN`；
    - TWCF：使用 `{BASE_PREFIX}_TWCF_*`，例如 `FLOW_ERROR_HIST_TWCF_X_MIN`。

    这样 evaluate_all/test_all 的直方图绘图代码不用关心具体类别，只要把 class/dataset
    名字传进来即可自动切换坐标范围；原有非 TBL/TWCF 行为保持不变。
    """
    base_prefix = str(base_prefix).upper()
    if is_tbl_category(category_name):
        return f"{base_prefix}_TBL"
    if is_twcf_category(category_name):
        return f"{base_prefix}_TWCF"
    return base_prefix


def _auto_axis_bound(values, *, fallback: float, use_max: bool, pad_fraction: float = 0.05) -> float:
    """
    根据数据自动补齐坐标范围端点。

    只有当对应全局变量为 None 时才会走到这里；也就是说，每一种图都已经有自己的
    `*_X_MIN/MAX/Y_MIN/MAX` 超参数入口，用户需要完全固定时直接把 None 改成数值即可。
    """
    arr = _finite_axis_values(values)
    if arr.size == 0:
        return float(fallback)
    value = float(np.max(arr) if use_max else np.min(arr))
    if not np.isfinite(value):
        return float(fallback)
    if use_max:
        # 上限自动时留一点空白，避免曲线/柱子顶到图框；下限通常直接使用固定起点。
        return value * (1.0 + pad_fraction) if value > 0 else value + max(abs(value), 1.0) * pad_fraction
    return value


def apply_plot_axis_config(
    ax,
    global_data,
    prefix: str,
    *,
    x_values=None,
    y_values=None,
    x_is_numeric: bool = True,
    y_is_numeric: bool = True,
    default_x_min=None,
    default_x_max=None,
    default_y_min=None,
    default_y_max=None,
    x_tick_target: int = 6,
    y_tick_target: int = 6,
    log_x: bool = False,
    log_y: bool = False,
) -> None:
    """
    按“图类型前缀”统一设置坐标范围和 tick 间隔。

    命名规则：
        `{PREFIX}_X_MIN`, `{PREFIX}_X_MAX`, `{PREFIX}_Y_MIN`, `{PREFIX}_Y_MAX`,
        `{PREFIX}_X_TICK_INTERVAL`, `{PREFIX}_Y_TICK_INTERVAL`

    例如：
        - ENERGY_SPECTRUM_* 控制能量谱曲线图；
        - ENERGY_SPECTRUM_MSE_* 控制 ESMSE 指标对比图；
        - FLOW_ERROR_HIST_* 控制普通类别 Δu/Δv/Δw 误差直方图；
        - FLOW_ERROR_HIST_TBL_* / FLOW_ERROR_HIST_TWCF_* 分别控制 TBL/TWCF 的 Δu/Δv/Δw 误差直方图；
        - EPE_HIST_* / EPE_HIST_TBL_* / EPE_HIST_TWCF_* 分别控制普通类别、TBL、TWCF 的 EPE 直方图；
        - PARTICLE_ERROR_HIST_* / PARTICLE_ERROR_HIST_TBL_* / PARTICLE_ERROR_HIST_TWCF_* 控制颗粒图像误差直方图；
        - TBL_PROFILE_* 控制 TBL 剖面图。

    这样每一种带坐标轴的图都有独立的全局变量，不会互相串用范围。
    `*_MAX=None` 时只自动补齐对应上限；改成具体数值后即为完全固定范围。
    """
    prefix = str(prefix).upper()

    if x_is_numeric:
        x_min = _read_optional_float_from_global(global_data, f"{prefix}_X_MIN", default_x_min)
        x_max = _read_optional_float_from_global(global_data, f"{prefix}_X_MAX", default_x_max)
        if x_min is None:
            x_min = _auto_axis_bound(x_values, fallback=0.0, use_max=False)
        if x_max is None:
            x_max = _auto_axis_bound(x_values, fallback=float(x_min) + 1.0, use_max=True)
        if log_x:
            # log 坐标必须为正数；若用户误填 0/负数，这里只修正显示轴，不改任何原始数据。
            x_min = max(float(x_min), 1e-12)
            x_max = max(float(x_max), x_min * 10.0)
        elif x_max <= x_min:
            x_max = x_min + 1.0
        ax.set_xlim(x_min, x_max)

        x_interval = _read_optional_float_from_global(global_data, f"{prefix}_X_TICK_INTERVAL", None)
        if x_interval is None and not log_x:
            x_interval = _nice_tick_interval(x_max - x_min, target_ticks=x_tick_target)
        if x_interval is not None:
            _set_fixed_tick_interval(ax, "x", x_min, x_max, x_interval)

    if y_is_numeric:
        y_min = _read_optional_float_from_global(global_data, f"{prefix}_Y_MIN", default_y_min)
        y_max = _read_optional_float_from_global(global_data, f"{prefix}_Y_MAX", default_y_max)
        if y_min is None:
            y_min = _auto_axis_bound(y_values, fallback=0.0, use_max=False)
        if y_max is None:
            y_max = _auto_axis_bound(y_values, fallback=float(y_min) + 1.0, use_max=True)
        if log_y:
            y_min = max(float(y_min), 1e-12)
            y_max = max(float(y_max), y_min * 10.0)
        elif y_max <= y_min:
            y_max = y_min + 1.0
        ax.set_ylim(y_min, y_max)

        y_interval = _read_optional_float_from_global(global_data, f"{prefix}_Y_TICK_INTERVAL", None)
        if y_interval is None and not log_y:
            y_interval = _nice_tick_interval(y_max - y_min, target_ticks=y_tick_target)
        if y_interval is not None:
            _set_fixed_tick_interval(ax, "y", y_min, y_max, y_interval)


def apply_energy_spectrum_mse_axis_config(ax, x_values, y_values, global_data=None, *, x_is_numeric: bool = True) -> None:
    """
    统一设置 energy_spectrum_mse 对比图的坐标范围和刻度间隔。

    全局变量约定：
    - ENERGY_SPECTRUM_MSE_X_MIN / X_MAX: 横轴显示范围；X_MAX=None 时自动取当前图最大 x。
    - ENERGY_SPECTRUM_MSE_Y_MIN / Y_MAX: 纵轴显示范围；Y_MAX=None 时自动取当前图最大 ESMSE 并留 5% 空白。
    - ENERGY_SPECTRUM_MSE_X_TICK_INTERVAL / Y_TICK_INTERVAL: 固定 tick 间隔；None 时按固定范围计算稳定间隔。

    对 dataset 名这种类别横轴，横坐标标签不是连续数值，所以只固定纵轴；数值 sample_index 横轴会同时固定 x/y。
    """
    apply_plot_axis_config(
        ax,
        global_data,
        "ENERGY_SPECTRUM_MSE",
        x_values=x_values,
        y_values=y_values,
        x_is_numeric=x_is_numeric,
        y_is_numeric=True,
        default_x_min=0.0,
        default_x_max=None,
        default_y_min=0.0,
        default_y_max=None,
    )


def save_energy_spectrum_curve_plot(pred_curve, gt_curve, out_png, title, *, global_data=None) -> None:
    """
    保存能量谱曲线图，并使用 ENERGY_SPECTRUM_* 独立全局变量控制坐标轴。

    该函数同时服务 evaluate_all 与 test_all；和 ESMSE 指标折线图分开配置，
    避免“频谱曲线的 log-log 坐标”和“频谱 MSE 的普通折线坐标”共用同一套范围。
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    pred_curve = np.asarray(pred_curve, dtype=np.float64)
    gt_curve = np.asarray(gt_curve, dtype=np.float64)
    k = np.arange(1, len(pred_curve) + 1, dtype=np.float64)
    pred_plot = np.maximum(pred_curve, 1e-12)
    gt_plot = np.maximum(gt_curve, 1e-12)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=160)
    ax.loglog(k, gt_plot, label="GT", linewidth=2)
    ax.loglog(k, pred_plot, label="Pred", linewidth=2, linestyle="--")
    apply_plot_axis_config(
        ax,
        global_data,
        "ENERGY_SPECTRUM",
        x_values=k,
        y_values=np.concatenate([pred_plot.reshape(-1), gt_plot.reshape(-1)]),
        default_x_min=1.0,
        default_x_max=None,
        default_y_min=1e-12,
        default_y_max=None,
        log_x=True,
        log_y=True,
    )
    ax.set_xlabel("Wavenumber k")
    ax.set_ylabel("E(k)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _to_single_image_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    将输入统一成 `[1, C, H, W]`。

    evaluate_all 里传进来的 LR/Fake/HR 本来就是 batch 中的单张图，
    但保持这个兜底可以兼容后续有人直接传 `[C,H,W]` 的情况。
    """
    if tensor.dim() == 3:
        return tensor.unsqueeze(0)
    if tensor.dim() != 4:
        raise ValueError(f"Expected image tensor shape [1,C,H,W] or [C,H,W], got {tuple(tensor.shape)}")
    return tensor


def _ensure_three_channel_for_panel(tensor: torch.Tensor) -> torch.Tensor:
    """
    统一可视化通道数。

    - 单通道颗粒图复制为 RGB 灰度图；
    - 三通道及以上只取前三个通道；
    - 两通道这种非常规图像先取通道均值，再复制为灰度 RGB。
    这样后面 Error 伪彩面板能和 LR/Fake/HR 在同一个 3 通道拼图里直接拼接。
    """
    tensor = _to_single_image_tensor(tensor).detach().clamp(0, 1)
    if tensor.shape[1] == 1:
        return tensor.repeat(1, 3, 1, 1)
    if tensor.shape[1] >= 3:
        return tensor[:, :3]
    return tensor.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)


def _pad_tensor_to_canvas(tensor: torch.Tensor, height: int, width: int, value: float = 1.0) -> torch.Tensor:
    """
    把图像居中放到固定画布，不做插值。

    这里特意不把 LR 放大到 HR 大小；它只是在同一行同一列宽中居中展示，
    保留真实低分辨率外观，同时仍然能和 Fake/HR/Error 对齐比较。
    """
    tensor = _to_single_image_tensor(tensor)
    b, c, h, w = tensor.shape
    if h > height or w > width:
        raise ValueError(f"tensor shape {(h, w)} exceeds target canvas {(height, width)}")
    canvas = torch.full((b, c, height, width), value, device=tensor.device, dtype=tensor.dtype)
    top = max((height - h) // 2, 0)
    left = max((width - w) // 2, 0)
    canvas[:, :, top:top + h, left:left + w] = tensor
    return canvas


def _tensor_to_rgb_pil(tensor: torch.Tensor) -> Image.Image:
    """将 `[1,3,H,W]` 图像转为 PIL RGB，用于在顶部写列标题。"""
    tensor = _to_single_image_tensor(tensor)[0].detach().cpu().clamp(0, 1)
    if tensor.shape[0] == 1:
        arr = (tensor[0].numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(arr, mode="L").convert("RGB")
    arr = (tensor[:3].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _pil_rgb_to_tensor01(image: Image.Image, device, dtype) -> torch.Tensor:
    """将 PIL RGB 图像转回 `[1,3,H,W]`、范围 `[0,1]` 的 torch 张量。"""
    arr = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)


def _add_headers_to_panel(panel: torch.Tensor, headers, column_widths, separator_widths, header_height: int = 22):
    """
    在拼图顶部添加列标题。

    column_widths 和 separator_widths 用真实拼接宽度计算，避免 LR 居中显示后标题偏位。
    """
    base = _tensor_to_rgb_pil(panel)
    canvas = Image.new("RGB", (base.width, base.height + header_height), color=(255, 255, 255))
    canvas.paste(base, (0, header_height))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    x = 0
    for idx, (title, width) in enumerate(zip(headers, column_widths)):
        bbox = draw.textbbox((0, 0), str(title), font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        tx = int(x + max((width - text_w) * 0.5, 0))
        ty = int(max((header_height - text_h) * 0.5, 0))
        draw.text((tx, ty), str(title), fill=(0, 0, 0), font=font)
        x += int(width)
        if idx < len(separator_widths):
            x += int(separator_widths[idx])
    return _pil_rgb_to_tensor01(canvas, panel.device, panel.dtype)


def _particle_error_tensor_for_panel(
        pred_chw,
        gt_chw,
        device,
        dtype,
        height: int,
        width: int,
        limit: float = 1.0,
) -> torch.Tensor:
    """
    将颗粒图像误差 `SR-HR` 转成 bwr 伪彩图面板。

    颗粒图像进入 evaluate_all 保存前已经被限制在 [0, 1]：
    - class_1 目录图像通过 transforms.ToTensor() 变为 [0, 1]；
    - class_2 TFRecord 图像在 data_load.py 中显式 /256.0；
    - evaluate_all 保存前还会 clamp(0, 1)。

    因此按 SR-HR 的理论误差范围，默认使用 [-1, 1] 的对称色条；
    实际 limit 会由各模型的 global_data.esrgan.PARTICLE_ERROR_COLORBAR_LIMIT 传入：
    - 蓝色表示 SR-HR < 0；
    - 白色表示 SR-HR = 0；
    - 红色表示 SR-HR > 0。

    固定色条牺牲了单张图的局部对比度，但能保证不同 sample / epoch 之间的颜色可直接比较。
    NaN/Inf 会被临时替换到有限范围内，只影响显示，不改变后续 sr_error.npy 的原始误差保存。
    """
    limit = float(limit)
    if limit <= 0:
        raise ValueError(f"particle image error colorbar limit must be positive, got {limit}")

    pred = np.asarray(pred_chw, dtype=np.float32)
    gt = np.asarray(gt_chw, dtype=np.float32)
    error = pred - gt
    if error.ndim == 3:
        if error.shape[0] == 1:
            error_2d = error[0]
        else:
            error_2d = np.nanmean(error, axis=0)
    else:
        error_2d = np.squeeze(error)

    error_2d = np.nan_to_num(error_2d, nan=0.0, posinf=limit, neginf=-limit)
    normalized = np.clip((error_2d + limit) / (2.0 * limit), 0.0, 1.0)
    rgb = plt.get_cmap("bwr")(normalized)[..., :3].astype(np.float32)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    return _pad_tensor_to_canvas(tensor, height, width, value=1.0)


def _particle_error_colorbar_tensor(device, dtype, height: int, limit: float = 1.0, width: int = 58) -> torch.Tensor:
    """
    为 evaluate_all 的颗粒误差面板生成固定范围色条。

    色条和 `_particle_error_tensor_for_panel` 使用同一个 bwr colormap、同一个 [-limit, limit] 范围：
    - 顶部红色对应 `SR-HR = +limit`；
    - 中间白色对应 `SR-HR = 0`；
    - 底部蓝色对应 `SR-HR = -limit`。
    这里直接画成 RGB 小图再拼到四联图右侧，避免改动原有 save_image 拼图链路。
    """
    height = max(int(height), 1)
    width = max(int(width), 44)
    limit = float(limit)
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    # 色条主体留出少量上下边距，右侧保留 tick 文本空间。
    bar_left = 6
    bar_right = 18
    bar_top = 6
    bar_bottom = max(height - 7, bar_top + 1)
    bar_h = max(bar_bottom - bar_top + 1, 1)
    cmap = plt.get_cmap("bwr")
    for offset, norm_value in enumerate(np.linspace(1.0, 0.0, bar_h)):
        rgb = tuple(int(round(v * 255.0)) for v in cmap(norm_value)[:3])
        y = bar_top + offset
        draw.line((bar_left, y, bar_right, y), fill=rgb)
    draw.rectangle((bar_left, bar_top, bar_right, bar_bottom), outline=(0, 0, 0), width=1)

    # 只标注 -limit / 0 / +limit 三个关键刻度，足够表达固定色条范围且不挤占图像主体。
    for tick_value in (limit, 0.0, -limit):
        norm = (float(tick_value) + limit) / (2.0 * limit)
        y = int(round(bar_top + (1.0 - norm) * (bar_h - 1)))
        draw.line((bar_right + 1, y, bar_right + 5, y), fill=(0, 0, 0))
        text = f"{tick_value:g}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_h = bbox[3] - bbox[1]
        draw.text((bar_right + 7, max(0, y - text_h // 2)), text, fill=(0, 0, 0), font=font)

    # 竖排标签用简单字符栈表示，兼容 PIL 默认字体；说明这是 SR-HR 的颗粒强度误差。
    label = "SR-HR"
    label_x = max(width - 10, bar_right + 25)
    label_y = max((height - len(label) * 7) // 2, 0)
    for idx, ch in enumerate(label):
        draw.text((label_x, label_y + idx * 7), ch, fill=(0, 0, 0), font=font)

    return _pil_rgb_to_tensor01(image, device, dtype)


def _match_common_chw(pred_chw, gt_chw) -> tuple[np.ndarray, np.ndarray]:
    """
    将 SR/HR 图像整理成公共 `[C,H,W]` 区域。

    evaluate_all 里大多数颗粒图是单通道，但为了兼容三通道生成器或偶发尺寸差异，
    这里只取两者共同通道和共同空间区域，不改变原图保存，只服务 ESMSE 诊断数值。
    """
    pred = np.asarray(pred_chw, dtype=np.float32)
    gt = np.asarray(gt_chw, dtype=np.float32)
    if pred.ndim == 2:
        pred = pred[None, ...]
    if gt.ndim == 2:
        gt = gt[None, ...]
    if pred.ndim != 3 or gt.ndim != 3:
        raise ValueError(f"Expected image arrays [C,H,W] or [H,W], got pred={pred.shape}, gt={gt.shape}")

    common_c = min(int(pred.shape[0]), int(gt.shape[0]))
    common_h = min(int(pred.shape[-2]), int(gt.shape[-2]))
    common_w = min(int(pred.shape[-1]), int(gt.shape[-1]))
    if common_c <= 0 or common_h <= 0 or common_w <= 0:
        raise ValueError(f"No common image region for metrics: pred={pred.shape}, gt={gt.shape}")
    return pred[:common_c, :common_h, :common_w], gt[:common_c, :common_h, :common_w]


def _radial_spectrum(ch2d: np.ndarray) -> np.ndarray:
    """
    计算单通道二维图像的径向平均能量谱。

    NaN/Inf 只在频谱副本中置 0，避免无效像素让 FFT 结果全变 NaN。
    """
    field = np.nan_to_num(np.asarray(ch2d, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    spectrum = np.fft.fftshift(np.fft.fft2(field))
    power = np.abs(spectrum) ** 2
    h, w = power.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)
    summed_power = np.bincount(radius.ravel(), power.ravel())
    counts = np.bincount(radius.ravel())
    return summed_power / np.maximum(counts, 1)


def compute_particle_image_esmse(pred_chw, gt_chw) -> float:
    """
    计算颗粒图像的 ESMSE（Energy Spectrum MSE）。

    公式与 evaluate_all/test_all 的 `energy_spectrum_mse` 保持一致：
    先计算 SR/HR 的径向平均能量谱，再对 `log1p` 后的谱差做 MSE。
    """
    pred, gt = _match_common_chw(pred_chw, gt_chw)
    pred_specs, gt_specs = [], []
    min_len = None
    for channel_idx in range(pred.shape[0]):
        pred_spec = _radial_spectrum(pred[channel_idx])
        gt_spec = _radial_spectrum(gt[channel_idx])
        n = min(len(pred_spec), len(gt_spec))
        min_len = n if min_len is None else min(min_len, n)
        pred_specs.append(pred_spec[:n])
        gt_specs.append(gt_spec[:n])
    if min_len is None or min_len <= 0:
        return float("nan")
    pred_curve = np.mean(np.stack([x[:min_len] for x in pred_specs], axis=0), axis=0)
    gt_curve = np.mean(np.stack([x[:min_len] for x in gt_specs], axis=0), axis=0)
    return float(np.mean((np.log1p(pred_curve) - np.log1p(gt_curve)) ** 2))


def _annotate_metric_on_panel(panel: torch.Tensor, text: str, x: int, y: int) -> torch.Tensor:
    """
    在拼接后的 RGB tensor 上写指标文本。

    这里用于 Error 面板左上角的 ESMSE，采用白底黑字，风格与 PIV error 图里的 AEE 标注一致。
    """
    base = _tensor_to_rgb_pil(panel)
    draw = ImageDraw.Draw(base)
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad = 4
    draw.rectangle(
        [x, y, x + text_w + pad * 2, y + text_h + pad * 2],
        fill=(255, 255, 255),
    )
    draw.text((x + pad, y + pad), text, fill=(0, 0, 0), font=font)
    return _pil_rgb_to_tensor01(base, panel.device, panel.dtype)


def compute_particle_image_error(pred_chw, gt_chw) -> np.ndarray:
    """
    计算颗粒图像的有符号误差图：`Generated SR - Original HR`。

    设计说明：
    1. 颗粒图像可能是 `[C,H,W]` 或已经 squeeze 后的 `[H,W]`，这里统一转成 2D；
    2. 多通道图像会先在公共通道上计算误差，再对通道取有效均值，避免某个通道 NaN 污染整点；
    3. 返回值保留 NaN/Inf，由后续 hist 函数过滤有限值，便于无效区域不参与误差分布统计。
    """
    pred = np.asarray(pred_chw, dtype=np.float32)
    gt = np.asarray(gt_chw, dtype=np.float32)

    if pred.ndim == 3 and gt.ndim == 3:
        common_channels = min(int(pred.shape[0]), int(gt.shape[0]))
        common_h = min(int(pred.shape[-2]), int(gt.shape[-2]))
        common_w = min(int(pred.shape[-1]), int(gt.shape[-1]))
        if common_channels <= 0 or common_h <= 0 or common_w <= 0:
            return np.full((1, 1), np.nan, dtype=np.float32)

        # 只在公共区域/公共通道上做误差统计；这不改变原图保存，只保护 hist 计算不被 shape 差异打断。
        error_chw = pred[:common_channels, :common_h, :common_w] - gt[:common_channels, :common_h, :common_w]
        if common_channels == 1:
            return error_chw[0].astype(np.float32, copy=False)

        finite_count = np.sum(np.isfinite(error_chw), axis=0)
        error_sum = np.nansum(error_chw, axis=0)
        error_2d = np.full((common_h, common_w), np.nan, dtype=np.float32)
        valid = finite_count > 0
        error_2d[valid] = (error_sum[valid] / finite_count[valid]).astype(np.float32, copy=False)
        return error_2d

    pred_2d = np.squeeze(pred).astype(np.float32, copy=False)
    gt_2d = np.squeeze(gt).astype(np.float32, copy=False)
    common_h = min(int(pred_2d.shape[-2]), int(gt_2d.shape[-2]))
    common_w = min(int(pred_2d.shape[-1]), int(gt_2d.shape[-1]))
    if common_h <= 0 or common_w <= 0:
        return np.full((1, 1), np.nan, dtype=np.float32)
    return (pred_2d[:common_h, :common_w] - gt_2d[:common_h, :common_w]).astype(np.float32, copy=False)


def _particle_to_gray_2d(image) -> np.ndarray:
    """
    将原图/超分辨率图统一成二维灰度图。

    设计口径：
    - evaluate_all/test_all 里常见输入是 `[C,H,W]`，单通道直接取第 0 通道；
    - 若生成器输出三通道或其他多通道结果，则对通道做有限值均值，得到一张灰度图；
    - 若外部偶然传入 `[H,W,C]`，这里也兼容最后一维为 1/3/4 的图像；
    - NaN/Inf 暂时保留，后续阈值、二值化和统计都会显式过滤无效点。
    """
    arr = np.asarray(image, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr.astype(np.float32, copy=False)
    if arr.ndim != 3:
        raise ValueError(f"Expected image array [H,W], [C,H,W] or [H,W,C], got shape={arr.shape}")

    # 优先识别 CHW，这是本项目 evaluate_all/test_all 的主格式。
    if arr.shape[0] <= 4 and arr.shape[0] <= min(arr.shape[-2], arr.shape[-1]):
        chw = arr
        if chw.shape[0] == 1:
            return chw[0].astype(np.float32, copy=False)
        finite_count = np.sum(np.isfinite(chw), axis=0)
        finite_sum = np.nansum(chw, axis=0)
        gray = np.full(chw.shape[1:], np.nan, dtype=np.float32)
        valid = finite_count > 0
        gray[valid] = (finite_sum[valid] / finite_count[valid]).astype(np.float32, copy=False)
        return gray

    # 兼容 HWC 图像，避免后续有人把 PIL/numpy 读图结果直接传进来时统计失败。
    if arr.shape[-1] <= 4:
        hwc = arr
        if hwc.shape[-1] == 1:
            return hwc[..., 0].astype(np.float32, copy=False)
        finite_count = np.sum(np.isfinite(hwc), axis=-1)
        finite_sum = np.nansum(hwc, axis=-1)
        gray = np.full(hwc.shape[:2], np.nan, dtype=np.float32)
        valid = finite_count > 0
        gray[valid] = (finite_sum[valid] / finite_count[valid]).astype(np.float32, copy=False)
        return gray

    raise ValueError(f"Cannot infer gray image layout from shape={arr.shape}")


def _match_common_gray_hw(pred_gray, gt_gray) -> tuple[np.ndarray, np.ndarray]:
    """
    将 SR 灰度图和 HR 灰度图裁到共同空间区域。

    这里只服务颗粒阈值统计，不改变原图保存；当模型输出尺寸和 HR 有一两个像素差异时，
    取公共区域可以保证统计流程不中断，同时和已有 ESMSE/误差图的公共区域口径一致。
    """
    pred = np.asarray(pred_gray, dtype=np.float32)
    gt = np.asarray(gt_gray, dtype=np.float32)
    common_h = min(int(pred.shape[-2]), int(gt.shape[-2]))
    common_w = min(int(pred.shape[-1]), int(gt.shape[-1]))
    if common_h <= 0 or common_w <= 0:
        raise ValueError(f"No common gray image region: pred={pred.shape}, gt={gt.shape}")
    return pred[:common_h, :common_w], gt[:common_h, :common_w]


def _otsu_threshold_from_gray(gt_gray: np.ndarray, bins: int = 256) -> tuple[float, np.ndarray, np.ndarray]:
    """
    只根据原图 HR 灰度直方图计算 Otsu 阈值 T。

    重要：阈值不使用超分图，避免 SR 结果自身分布改变导致“自适应占便宜”。
    计算得到的同一个 T 会同时用于 HR 和 SR 的二值化，保证二者比较口径完全一致。
    """
    finite = np.asarray(gt_gray, dtype=np.float32).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)

    min_value = float(np.min(finite))
    max_value = float(np.max(finite))
    if not np.isfinite(min_value) or not np.isfinite(max_value) or max_value <= min_value:
        # 全常数图没有可分峰谷，阈值取该常数；后续严格 >/< 二值化会得到空前景。
        centers = np.asarray([min_value], dtype=np.float32)
        counts = np.asarray([finite.size], dtype=np.float32)
        return min_value, centers, counts

    counts, edges = np.histogram(finite, bins=int(bins), range=(min_value, max_value))
    counts = counts.astype(np.float64)
    centers = ((edges[:-1] + edges[1:]) * 0.5).astype(np.float64)
    total = float(np.sum(counts))
    if total <= 0:
        return float("nan"), centers.astype(np.float32), counts.astype(np.float32)

    probability = counts / total
    omega = np.cumsum(probability)
    mu = np.cumsum(probability * centers)
    mu_total = float(mu[-1])
    denominator = omega * (1.0 - omega)
    between_class_var = np.full_like(denominator, -np.inf, dtype=np.float64)
    valid = denominator > 1e-12
    between_class_var[valid] = ((mu_total * omega[valid] - mu[valid]) ** 2) / denominator[valid]
    best_idx = int(np.nanargmax(between_class_var))
    return float(centers[best_idx]), centers.astype(np.float32), counts.astype(np.float32)


def _choose_particle_foreground_rule(gt_gray: np.ndarray, threshold: float) -> str:
    """
    根据 HR 二值化后较小的一侧自动判断颗粒是亮点还是暗点。

    本项目颗粒图通常是黑底白点，前景应为 `gray > T`；但为了兼容反相颗粒图，
    如果阈值以下的像素更稀少，就用 `gray < T` 作为颗粒前景。这个判断只看 HR。
    """
    finite = np.isfinite(gt_gray)
    if not np.any(finite) or not np.isfinite(threshold):
        return "greater"
    high_ratio = float(np.mean(gt_gray[finite] > threshold))
    low_ratio = float(np.mean(gt_gray[finite] < threshold))
    return "greater" if high_ratio <= low_ratio else "less"


def _binary_from_threshold(gray: np.ndarray, threshold: float, foreground_rule: str) -> np.ndarray:
    """
    使用同一个 HR 阈值 T 将 HR/SR 灰度图二值化。

    使用严格 `>` / `<`，而不是 `>=` / `<=`，是为了全常数图像时不会把整张图误判成颗粒。
    无效像素永远为 False，避免 NaN/Inf 进入颗粒统计。
    """
    gray = np.asarray(gray, dtype=np.float32)
    finite = np.isfinite(gray)
    if not np.isfinite(threshold):
        return np.zeros(gray.shape, dtype=bool)
    if str(foreground_rule).lower() == "less":
        return (gray < threshold) & finite
    return (gray > threshold) & finite


def _connected_component_areas(binary: np.ndarray) -> np.ndarray:
    """
    提取二值图中的颗粒连通域面积。

    优先使用 OpenCV / SciPy 的连通域实现；如果环境没有这些包，则回退到纯 numpy/Python
    的 8 邻域搜索。回退路径只影响速度，不改变输出含义。
    """
    mask = np.asarray(binary, dtype=bool)
    if mask.size == 0 or not np.any(mask):
        return np.asarray([], dtype=np.float32)

    try:
        import cv2  # type: ignore

        num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if num_labels <= 1:
            return np.asarray([], dtype=np.float32)
        return stats[1:, cv2.CC_STAT_AREA].astype(np.float32, copy=False)
    except Exception:
        pass

    try:
        from scipy import ndimage as ndi  # type: ignore

        structure = np.ones((3, 3), dtype=np.uint8)
        labels, num_labels = ndi.label(mask, structure=structure)
        if num_labels <= 0:
            return np.asarray([], dtype=np.float32)
        areas = np.bincount(labels.reshape(-1))[1:]
        return areas.astype(np.float32, copy=False)
    except Exception:
        pass

    # 最后兜底：纯 Python 8 邻域 flood fill。大图会比 SciPy 慢，但能保证功能可用。
    visited = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    areas = []
    ys, xs = np.nonzero(mask)
    neighbors = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if visited[start_y, start_x]:
            continue
        stack = [(start_y, start_x)]
        visited[start_y, start_x] = True
        area = 0
        while stack:
            y, x = stack.pop()
            area += 1
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        areas.append(area)
    return np.asarray(areas, dtype=np.float32)


def _particle_component_summary(binary: np.ndarray) -> dict:
    """
    从二值图中提取颗粒统计量。

    统计量说明：
    - particle_count：连通颗粒数量；
    - particle_pixels / particle_density：前景像素数量及占比；
    - area_*：单个连通颗粒的面积分布，用于判断 SR 是否把颗粒粘连或打碎。
    """
    mask = np.asarray(binary, dtype=bool)
    areas = _connected_component_areas(mask)
    foreground_pixels = int(np.count_nonzero(mask))
    total_pixels = int(mask.size)
    return {
        "particle_count": int(areas.size),
        "particle_pixels": foreground_pixels,
        "particle_density": float(foreground_pixels / max(total_pixels, 1)),
        "area_mean": float(np.mean(areas)) if areas.size else 0.0,
        "area_median": float(np.median(areas)) if areas.size else 0.0,
        "area_std": float(np.std(areas)) if areas.size else 0.0,
        "area_min": float(np.min(areas)) if areas.size else 0.0,
        "area_max": float(np.max(areas)) if areas.size else 0.0,
    }


def compute_particle_binary_stats(pred_chw, gt_chw, *, bins: int = 256) -> dict:
    """
    计算 HR/SR 颗粒二值化统计结果。

    流程严格对应用户要求：
    1. HR 与 SR 都转成灰度；
    2. 只用 HR 灰度直方图计算 Otsu 阈值 T；
    3. HR/SR 使用同一个 T 和同一个前景方向二值化；
    4. 对二值图提取颗粒数量、面积、密度；
    5. 计算 SR 相对 HR 的颗粒统计差异、前景 IoU/Precision/Recall/F1。
    """
    pred_gray_raw = _particle_to_gray_2d(pred_chw)
    gt_gray_raw = _particle_to_gray_2d(gt_chw)
    pred_gray, gt_gray = _match_common_gray_hw(pred_gray_raw, gt_gray_raw)
    threshold, hist_centers, hist_counts = _otsu_threshold_from_gray(gt_gray, bins=bins)
    foreground_rule = _choose_particle_foreground_rule(gt_gray, threshold)
    gt_binary = _binary_from_threshold(gt_gray, threshold, foreground_rule)
    pred_binary = _binary_from_threshold(pred_gray, threshold, foreground_rule)

    gt_stats = _particle_component_summary(gt_binary)
    pred_stats = _particle_component_summary(pred_binary)
    intersection = int(np.count_nonzero(gt_binary & pred_binary))
    union = int(np.count_nonzero(gt_binary | pred_binary))
    pred_pixels = int(pred_stats["particle_pixels"])
    gt_pixels = int(gt_stats["particle_pixels"])
    precision = float(intersection / pred_pixels) if pred_pixels > 0 else float("nan")
    recall = float(intersection / gt_pixels) if gt_pixels > 0 else float("nan")
    f1 = float((2.0 * precision * recall) / (precision + recall)) if np.isfinite(precision + recall) and (precision + recall) > 0 else float("nan")

    flat_stats = {
        "threshold_method": "otsu_from_hr_only",
        "threshold": float(threshold),
        "foreground_rule": foreground_rule,
        "hist_bins": int(bins),
        "height": int(gt_gray.shape[0]),
        "width": int(gt_gray.shape[1]),
        "gt_particle_count": gt_stats["particle_count"],
        "pred_particle_count": pred_stats["particle_count"],
        "delta_particle_count": int(pred_stats["particle_count"] - gt_stats["particle_count"]),
        "count_ratio_pred_over_gt": float(pred_stats["particle_count"] / gt_stats["particle_count"]) if gt_stats["particle_count"] else float("nan"),
        "gt_particle_pixels": gt_stats["particle_pixels"],
        "pred_particle_pixels": pred_stats["particle_pixels"],
        "delta_particle_pixels": int(pred_stats["particle_pixels"] - gt_stats["particle_pixels"]),
        "gt_particle_density": gt_stats["particle_density"],
        "pred_particle_density": pred_stats["particle_density"],
        "delta_particle_density": float(pred_stats["particle_density"] - gt_stats["particle_density"]),
        "gt_area_mean": gt_stats["area_mean"],
        "pred_area_mean": pred_stats["area_mean"],
        "delta_area_mean": float(pred_stats["area_mean"] - gt_stats["area_mean"]),
        "gt_area_median": gt_stats["area_median"],
        "pred_area_median": pred_stats["area_median"],
        "gt_area_std": gt_stats["area_std"],
        "pred_area_std": pred_stats["area_std"],
        "gt_area_min": gt_stats["area_min"],
        "pred_area_min": pred_stats["area_min"],
        "gt_area_max": gt_stats["area_max"],
        "pred_area_max": pred_stats["area_max"],
        "binary_iou": float(intersection / union) if union > 0 else float("nan"),
        "binary_precision": precision,
        "binary_recall": recall,
        "binary_f1": f1,
        "binary_intersection_pixels": intersection,
        "binary_union_pixels": union,
    }
    return {
        "stats": flat_stats,
        "pred_gray": pred_gray.astype(np.float32, copy=False),
        "gt_gray": gt_gray.astype(np.float32, copy=False),
        "pred_binary": pred_binary.astype(np.uint8),
        "gt_binary": gt_binary.astype(np.uint8),
        "hist_centers": hist_centers.astype(np.float32, copy=False),
        "hist_counts": hist_counts.astype(np.float32, copy=False),
    }


def _write_particle_binary_stats_csv(csv_path: Path, stats: dict) -> None:
    """保存颗粒统计结果为两列 CSV，便于直接用 Excel 或 pandas 检查。"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in stats.items():
            writer.writerow([key, value])


def _save_particle_binary_stats_plot(
    out_png: Path,
    payload: dict,
    *,
    title: str,
    global_data=None,
) -> None:
    """
    保存颗粒统计对比图。

    图中包含：
    - HR/SR 灰度图；
    - 使用同一阈值 T 得到的 HR/SR 二值图；
    - 只来自 HR 的灰度直方图，并用红线标注阈值 T；
    - 颗粒数量、密度、平均面积、IoU、F1 的统计对比。
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    stats = payload["stats"]
    gt_gray = payload["gt_gray"]
    pred_gray = payload["pred_gray"]
    gt_binary = payload["gt_binary"]
    pred_binary = payload["pred_binary"]
    hist_centers = payload["hist_centers"]
    hist_counts = payload["hist_counts"]
    threshold = float(stats.get("threshold", float("nan")))

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.8), dpi=160)
    fig.suptitle(title, fontsize=12)
    gray_values = np.concatenate([
        gt_gray[np.isfinite(gt_gray)].reshape(-1),
        pred_gray[np.isfinite(pred_gray)].reshape(-1),
    ])
    if gray_values.size > 0:
        gray_vmin, gray_vmax = float(np.min(gray_values)), float(np.max(gray_values))
        if gray_vmax <= gray_vmin:
            gray_vmax = gray_vmin + 1.0
    else:
        gray_vmin, gray_vmax = 0.0, 1.0

    axes[0, 0].imshow(gt_gray, cmap="gray", vmin=gray_vmin, vmax=gray_vmax)
    axes[0, 0].set_title("HR gray")
    axes[0, 1].imshow(pred_gray, cmap="gray", vmin=gray_vmin, vmax=gray_vmax)
    axes[0, 1].set_title("SR gray")
    axes[0, 2].imshow(gt_binary, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title(f"HR binary (T={threshold:.4g})")
    axes[1, 0].imshow(pred_binary, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("SR binary (same T)")
    for ax in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]):
        ax.set_xticks([])
        ax.set_yticks([])

    ax_hist = axes[1, 1]
    if hist_centers.size > 0:
        width = float(np.median(np.diff(hist_centers))) if hist_centers.size > 1 else 1.0
        if not np.isfinite(width) or width <= 0:
            width = 1.0
        ax_hist.bar(hist_centers, hist_counts, width=width, color="#4C72B0", alpha=0.72, edgecolor="none")
        ax_hist.axvline(threshold, color="red", linestyle="-", linewidth=1.7, label=f"T={threshold:.4g}")
        apply_plot_axis_config(
            ax_hist,
            global_data,
            "PARTICLE_BINARY_HIST",
            x_values=hist_centers,
            y_values=hist_counts,
            default_x_min=float(np.min(hist_centers)),
            default_x_max=float(np.max(hist_centers)),
            default_y_min=0.0,
            default_y_max=None,
        )
        ax_hist.legend(fontsize=8)
    ax_hist.set_title("HR gray histogram")
    ax_hist.set_xlabel("gray value")
    ax_hist.set_ylabel("count")

    ax_bar = axes[1, 2]
    compare_labels = ["count", "density", "mean_area"]
    gt_values = [
        float(stats.get("gt_particle_count", 0.0)),
        float(stats.get("gt_particle_density", 0.0)),
        float(stats.get("gt_area_mean", 0.0)),
    ]
    pred_values = [
        float(stats.get("pred_particle_count", 0.0)),
        float(stats.get("pred_particle_density", 0.0)),
        float(stats.get("pred_area_mean", 0.0)),
    ]
    x = np.arange(len(compare_labels))
    hr_bars = ax_bar.bar(x - 0.18, gt_values, width=0.36, label="HR", color="#555555")
    sr_bars = ax_bar.bar(x + 0.18, pred_values, width=0.36, label="SR", color="#DD8452")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(compare_labels, rotation=18)
    ax_bar.set_title(
        f"IoU={float(stats.get('binary_iou', float('nan'))):.4f}, "
        f"F1={float(stats.get('binary_f1', float('nan'))):.4f}"
    )
    ax_bar.legend(fontsize=8)
    ax_bar.grid(True, axis="y", alpha=0.25)

    max_bar_value = max([0.0] + gt_values + pred_values)
    if np.isfinite(max_bar_value) and max_bar_value > 0:
        ax_bar.set_ylim(top=max_bar_value * 1.18)

    def _format_particle_bar_value(metric_name: str, value: float) -> str:
        """
        给颗粒统计柱状图显示数值。

        density 是前景像素占比，通常只有 0.x；和 count / mean_area 放在同一个 y 轴时柱子会显得像“空的”。
        因此这里直接在柱子顶部写出数值，避免读图时误以为 density 没有计算出来。
        """
        if not np.isfinite(value):
            return "nan"
        if metric_name == "density":
            return f"{value:.4f}"
        if abs(value) >= 100:
            return f"{value:.0f}"
        return f"{value:.2f}"

    for bars, values in ((hr_bars, gt_values), (sr_bars, pred_values)):
        for bar, metric_name, value in zip(bars, compare_labels, values):
            text_y = bar.get_height()
            ax_bar.annotate(
                _format_particle_bar_value(metric_name, value),
                xy=(bar.get_x() + bar.get_width() / 2.0, text_y),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90 if metric_name == "density" else 0,
            )

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _save_particle_binary_threshold_plot(
    out_png: Path,
    payload: dict,
    *,
    title: str,
    global_data=None,
) -> None:
    """
    保存拆分后的颗粒阈值图。

    这张图只展示阈值相关内容：HR/SR 灰度图、同一阈值得到的 HR/SR 二值图、
    以及只由 HR 计算得到的灰度直方图和阈值 T。统计柱状图单独输出到
    `{prefix}_metrics_bar.png`，避免原来的 compare 图信息过密。
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    stats = payload["stats"]
    gt_gray = payload["gt_gray"]
    pred_gray = payload["pred_gray"]
    gt_binary = payload["gt_binary"]
    pred_binary = payload["pred_binary"]
    hist_centers = payload["hist_centers"]
    hist_counts = payload["hist_counts"]
    threshold = float(stats.get("threshold", float("nan")))

    gray_values = np.concatenate([
        gt_gray[np.isfinite(gt_gray)].reshape(-1),
        pred_gray[np.isfinite(pred_gray)].reshape(-1),
    ])
    if gray_values.size > 0:
        gray_vmin, gray_vmax = float(np.min(gray_values)), float(np.max(gray_values))
        if gray_vmax <= gray_vmin:
            gray_vmax = gray_vmin + 1.0
    else:
        gray_vmin, gray_vmax = 0.0, 1.0

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.2), dpi=160)
    fig.suptitle(title, fontsize=12)
    panels = (
        (axes[0, 0], gt_gray, "HR gray", "gray", gray_vmin, gray_vmax),
        (axes[0, 1], pred_gray, "SR gray", "gray", gray_vmin, gray_vmax),
        (axes[1, 0], gt_binary, f"HR binary (T={threshold:.4g})", "gray", 0, 1),
        (axes[1, 1], pred_binary, "SR binary (same T)", "gray", 0, 1),
    )
    for ax, arr, panel_title, cmap, vmin, vmax in panels:
        ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(panel_title)
        ax.set_xticks([])
        ax.set_yticks([])

    ax_hist = axes[0, 2]
    if hist_centers.size > 0:
        width = float(np.median(np.diff(hist_centers))) if hist_centers.size > 1 else 1.0
        if not np.isfinite(width) or width <= 0:
            width = 1.0
        ax_hist.bar(hist_centers, hist_counts, width=width, color="#4C72B0", alpha=0.72, edgecolor="none")
        ax_hist.axvline(threshold, color="red", linestyle="-", linewidth=1.7, label=f"T={threshold:.4g}")
        apply_plot_axis_config(
            ax_hist,
            global_data,
            "PARTICLE_BINARY_HIST",
            x_values=hist_centers,
            y_values=hist_counts,
            default_x_min=float(np.min(hist_centers)),
            default_x_max=float(np.max(hist_centers)),
            default_y_min=0.0,
            default_y_max=None,
        )
        ax_hist.legend(fontsize=8)
    ax_hist.set_title("HR gray histogram")
    ax_hist.set_xlabel("gray value")
    ax_hist.set_ylabel("count")

    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.02,
        0.96,
        "\n".join([
            f"threshold = {threshold:.6g}",
            f"method = {stats.get('threshold_method')}",
            f"foreground = {stats.get('foreground_rule')}",
            f"size = {stats.get('width')} x {stats.get('height')}",
        ]),
        transform=axes[1, 2].transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _save_particle_binary_metrics_bar_plot(out_png: Path, payload: dict, *, title: str) -> None:
    """
    保存拆分后的颗粒条形统计图。

    三个指标分别单独占一个子图，避免 particle_pixels 数值远大于 count / area_mean 时
    把其它柱子压得看不见。指标顺序按用户要求：particle pixels / count / area_mean。
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    stats = payload["stats"]
    metric_specs = (
        ("particle pixels", "gt_particle_pixels", "pred_particle_pixels"),
        ("count", "gt_particle_count", "pred_particle_count"),
        ("area_mean", "gt_area_mean", "pred_area_mean"),
    )

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), dpi=160)
    fig.suptitle(title, fontsize=12)
    for ax, (metric_label, gt_key, pred_key) in zip(axes, metric_specs):
        values = [
            float(stats.get(gt_key, float("nan"))),
            float(stats.get(pred_key, float("nan"))),
        ]
        finite_values = [value for value in values if np.isfinite(value)]
        x = np.arange(2)
        bars = ax.bar(
            x,
            values,
            width=0.55,
            color=("#555555", "#DD8452"),
            edgecolor="#222222",
            linewidth=0.6,
        )
        ax.set_title(metric_label)
        ax.set_xticks(x)
        ax.set_xticklabels(("HR", "SR"))
        ax.grid(True, axis="y", alpha=0.25)
        max_value = max([0.0] + finite_values)
        if max_value > 0:
            ax.set_ylim(top=max_value * 1.18)
        for bar, value in zip(bars, values):
            if not np.isfinite(value):
                text = "nan"
            elif abs(value) >= 100:
                text = f"{value:.0f}"
            else:
                text = f"{value:.2f}"
            ax.annotate(
                text,
                xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height() if np.isfinite(value) else 0.0),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def save_particle_binary_stats_artifacts(
    out_dir,
    pred_chw,
    gt_chw,
    *,
    file_prefix: str = "particle_binary_stats",
    title: str = "Particle Binary Statistics",
    global_data=None,
) -> dict:
    """
    计算并保存 HR/SR 颗粒二值统计的全部产物。

    保存文件：
    - `{prefix}_threshold.txt`：阈值 T、阈值算法、前景方向；
    - `{prefix}_stats.csv`：颗粒数量、面积、密度、IoU、Precision/Recall/F1 等统计；
    - `{prefix}_stats.npy`：与 CSV 同内容的 key/value 数组，便于脚本读取；
    - `{prefix}_hist.npy`：只由 HR 计算得到的灰度直方图 `[gray_center, count]`；
    - `{prefix}_gt_binary.npy` / `{prefix}_pred_binary.npy`：同阈值二值化结果；
    - `{prefix}_compare.png`：拆分后的颗粒阈值图，保留旧文件名便于已有路径继续打开；
    - `{prefix}_threshold_compare.png`：同上，使用更明确的文件名；
    - `{prefix}_metrics_bar.png`：particle pixels / count / area_mean 条形统计图。

    注意：该函数的 NPY 是“统计结果与阈值复现实验”的必要产物，不走 IS_SAVE_NPY，
    因为用户明确要求保存统计结果、对比图和阈值。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = compute_particle_binary_stats(pred_chw, gt_chw)
    stats = payload["stats"]

    threshold_path = out_dir / f"{file_prefix}_threshold.txt"
    threshold_path.write_text(
        "\n".join([
            f"threshold_method={stats.get('threshold_method')}",
            f"threshold={stats.get('threshold')}",
            f"foreground_rule={stats.get('foreground_rule')}",
            f"hist_bins={stats.get('hist_bins')}",
            f"height={stats.get('height')}",
            f"width={stats.get('width')}",
        ]) + "\n",
        encoding="utf-8",
    )
    _write_particle_binary_stats_csv(out_dir / f"{file_prefix}_stats.csv", stats)
    np.save(
        out_dir / f"{file_prefix}_stats.npy",
        np.asarray([[key, str(value)] for key, value in stats.items()], dtype=object),
        allow_pickle=True,
    )
    hist_matrix = np.stack([payload["hist_centers"], payload["hist_counts"]], axis=1) if payload["hist_centers"].size else np.empty((0, 2), dtype=np.float32)
    np.save(out_dir / f"{file_prefix}_hist.npy", hist_matrix.astype(np.float32, copy=False))
    np.save(out_dir / f"{file_prefix}_gt_binary.npy", payload["gt_binary"].astype(np.uint8, copy=False))
    np.save(out_dir / f"{file_prefix}_pred_binary.npy", payload["pred_binary"].astype(np.uint8, copy=False))
    _save_particle_binary_threshold_plot(
        out_dir / f"{file_prefix}_compare.png",
        payload,
        title=title,
        global_data=global_data,
    )
    _save_particle_binary_threshold_plot(
        out_dir / f"{file_prefix}_threshold_compare.png",
        payload,
        title=title,
        global_data=global_data,
    )
    _save_particle_binary_metrics_bar_plot(
        out_dir / f"{file_prefix}_metrics_bar.png",
        payload,
        title=title,
    )
    return stats


def particle_error_histogram_matrix(error_values, bins: int = 201) -> np.ndarray:
    """
    将颗粒图像误差值统计成两列 hist 矩阵 `[bin_center, count]`。

    口径与光流 Δu/Δv/Δw hist 保持一致：
    - 只统计有限值，自动跳过 NaN/Inf 无效区域；
    - 使用以 0 为中心的对称范围，正负误差分布可以直接对比；
    - 输出 `.npy` 采用固定两列格式，便于后处理脚本统一读取。
    """
    finite_values = np.asarray(error_values, dtype=np.float32).reshape(-1)
    finite_values = finite_values[np.isfinite(finite_values)]
    max_abs = float(np.max(np.abs(finite_values))) + 1e-12 if finite_values.size > 0 else 1e-12
    edges = np.linspace(-max_abs, max_abs, int(bins) + 1, dtype=np.float32)
    counts, edges = np.histogram(finite_values, bins=edges)
    centers = ((edges[:-1] + edges[1:]) * 0.5).astype(np.float32)
    return np.stack([centers, counts.astype(np.float32)], axis=1)


def _save_particle_error_histogram_plot(
    hist_matrix,
    out_png,
    title,
    xlabel,
    color="#AA3377",
    *,
    global_data=None,
    axis_prefix: str = "PARTICLE_ERROR_HIST",
) -> None:
    """把颗粒图像误差 hist 矩阵保存成 png，和 npy 文件一一对应。"""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    hist = np.asarray(hist_matrix, dtype=np.float32)
    if hist.ndim != 2 or hist.shape[1] != 2 or hist.shape[0] == 0:
        return

    centers = hist[:, 0]
    counts = hist[:, 1]
    if centers.size > 1:
        width = float(np.median(np.diff(centers)))
        width = width if np.isfinite(width) and width > 0 else 1.0
    else:
        width = 1.0

    fig, ax = plt.subplots(1, 1, figsize=(5.2, 3.8), dpi=150)
    ax.bar(centers, counts, width=width, color=color, alpha=0.72, edgecolor="none")
    # 颗粒/涡度等误差直方图统一用红色竖线标出 x=0 的位置；
    # 这只是 PNG 视觉标注，不改变保存的 histogram npy 数据。
    ax.axvline(0.0, color="red", linewidth=1.6, linestyle="-", alpha=0.95, zorder=5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    # 颗粒误差 hist 和涡度误差 hist 使用不同 prefix，因此可以分别固定横纵轴范围与 tick。
    apply_plot_axis_config(
        ax,
        global_data,
        axis_prefix,
        x_values=centers,
        y_values=counts,
        default_x_min=float(np.min(centers)) if centers.size else -1.0,
        default_x_max=float(np.max(centers)) if centers.size else 1.0,
        default_y_min=0.0,
        default_y_max=None,
    )
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _save_hist_npy(path, hist_matrix, save_npy_fn, save_npy: bool, force_npy: bool) -> None:
    """
    兼容 evaluate_all/test_all 两套 NPY 开关函数。

    evaluate_all 使用 `_save_evaluate_npy`，test_all 使用 `_save_optional_npy`；
    两者参数形式相同，因此通过回调可以保证 hist NPY 都遵循“强制保存诊断结果”的规则。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if save_npy_fn is None:
        np.save(path, np.asarray(hist_matrix, dtype=np.float32))
    else:
        save_npy_fn(path, np.asarray(hist_matrix, dtype=np.float32), save_npy, force=force_npy)


def save_particle_error_histogram(
    out_dir,
    error_values,
    *,
    file_prefix: str = "sr_error",
    npy_name: str | None = None,
    png_name: str | None = None,
    title: str = "Particle Image Error Distribution",
    xlabel: str = "SR-HR image intensity error",
    color: str = "#AA3377",
    save_npy_fn=None,
    save_npy: bool = True,
    force_npy: bool = True,
    global_data=None,
    axis_prefix: str | None = None,
    category_name=None,
) -> np.ndarray:
    """
    保存单个样本或单个集合的颗粒图像误差直方图。

    默认输出：
    - `{file_prefix}_hist.npy`
    - `{file_prefix}_hist.png`

    `force_npy=True` 与光流 hist 规则一致：hist 是诊断结果，即使 `IS_SAVE_NPY=False`
    也会保存，避免关闭普通中间 NPY 后失去误差分布数据。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hist = particle_error_histogram_matrix(error_values)
    npy_name = npy_name or f"{file_prefix}_hist.npy"
    png_name = png_name or f"{file_prefix}_hist.png"
    if axis_prefix is None:
        # 颗粒图像误差和涡量误差都是通过这个公共函数保存，但它们的物理量不同，
        # 因此坐标轴全局变量也要拆开：sr_error -> PARTICLE_ERROR_HIST，delta_vorticity -> VORTICITY_ERROR_HIST。
        axis_prefix = "VORTICITY_ERROR_HIST" if str(file_prefix).startswith("delta_vorticity") else "PARTICLE_ERROR_HIST"
    # 普通类别和 TBL/TWCF 大图类别使用两套误差直方图坐标轴配置。
    # 这里只切换 PNG 坐标轴全局变量前缀，不改变 hist npy 里的 bin/count 数据。
    axis_prefix = select_error_hist_axis_prefix(axis_prefix, category_name)
    _save_hist_npy(out_dir / npy_name, hist, save_npy_fn, save_npy, force_npy)
    _save_particle_error_histogram_plot(
        hist,
        out_dir / png_name,
        title=title,
        xlabel=xlabel,
        color=color,
        global_data=global_data,
        axis_prefix=axis_prefix,
    )
    return hist


def save_particle_error_histogram_bundle(
    out_dir,
    error_values_list,
    *,
    file_prefix: str = "sr_error",
    title: str = "Particle Image Error Distribution",
    xlabel: str = "SR-HR image intensity error",
    color: str = "#AA3377",
    save_npy_fn=None,
    save_npy: bool = True,
    force_npy: bool = True,
    global_data=None,
    axis_prefix: str | None = None,
    category_name=None,
) -> np.ndarray | None:
    """
    保存类别级/总体级颗粒图像误差 hist。

    `error_values_list` 通常由多张 previous/next 的误差图 flatten 后组成；
    合并后输出命名为 `{file_prefix}_hist_all.npy/png`，与光流的
    `delta_u_hist_all.npy/png` 命名保持同一风格。
    """
    if not error_values_list:
        return None
    merged = np.concatenate(
        [np.asarray(values, dtype=np.float32).reshape(-1) for values in error_values_list],
        axis=0,
    )
    return save_particle_error_histogram(
        out_dir,
        merged,
        file_prefix=file_prefix,
        npy_name=f"{file_prefix}_hist_all.npy",
        png_name=f"{file_prefix}_hist_all.png",
        title=title,
        xlabel=xlabel,
        color=color,
        save_npy_fn=save_npy_fn,
        save_npy=save_npy,
        force_npy=force_npy,
        global_data=global_data,
        axis_prefix=axis_prefix,
        category_name=category_name,
    )


def save_energy_spectrum_mse_compare_npy(
    rows,
    out_dir,
    file_prefix: str,
    value_key: str = "energy_spectrum_mse",
    *,
    global_data=None,
    title: str | None = None,
) -> np.ndarray | None:
    """
    保存 energy_spectrum_mse 对比图的源数据 NPY，并同步输出同名 PNG。

    这个文件不是普通中间大数组，而是指标图的点数据：
    - evaluate_all/test_all 后续可以直接用它复现能量谱均方误差曲线；
    - 文件体积很小，且属于评价图对应的实验记录；
    - 按用户要求不受 IS_SAVE_NPY 控制，因此这里直接 np.save，不走外部 save_npy 开关。
    - PNG 坐标轴范围和 tick 间隔读取 ENERGY_SPECTRUM_MSE_* 全局变量，和 test_all 保持一致。
    """
    records = []
    for row_index, row in enumerate(rows or []):
        try:
            value = float(row.get(value_key, float("nan")))
        except (TypeError, ValueError):
            value = float("nan")
        if not np.isfinite(value):
            continue
        records.append(
            (
                int(row_index),
                str(row.get("class_name", "")),
                str(row.get("data_type", "")),
                str(row.get("sample_id", row.get("sample_index", ""))),
                str(row.get("pair_type", "")),
                float(value),
            )
        )

    if not records:
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    values = np.asarray(
        records,
        dtype=[
            ("row_index", "i4"),
            ("class_name", "U128"),
            ("data_type", "U64"),
            ("sample_id", "U128"),
            ("pair_type", "U64"),
            ("energy_spectrum_mse", "f4"),
        ],
    )
    npy_path = out_dir / f"{file_prefix}_energy_spectrum_mse_compare.npy"
    np.save(npy_path, values)

    # evaluate_all 之前只保存 NPY；这里补同名 PNG，便于和 test_all 的图直接比较。
    x_values = values["row_index"].astype(np.float32)
    y_values = values["energy_spectrum_mse"].astype(np.float32)
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.2), dpi=160)
    labels = sorted(set(str(v) for v in values["pair_type"]))
    for label in labels:
        mask = values["pair_type"] == label
        ax.plot(
            x_values[mask],
            y_values[mask],
            marker="o",
            linewidth=1.6,
            markersize=3.5,
            label=label or "metric",
        )
    if y_values.size > 0:
        mean_value = float(np.mean(y_values))
        ax.axhline(mean_value, color="k", linestyle=":", linewidth=1.1, label=f"mean={mean_value:.4g}")
    apply_energy_spectrum_mse_axis_config(ax, x_values, y_values, global_data, x_is_numeric=True)
    ax.set_xlabel("row index")
    ax.set_ylabel("Energy spectrum MSE")
    ax.set_title(title or f"{file_prefix} Energy Spectrum MSE")
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{file_prefix}_energy_spectrum_mse_compare.png", bbox_inches="tight")
    plt.close(fig)
    return values


def save_image_triplet_with_error(
    lr_tensor,
    fake_tensor,
    hr_tensor,
    pred_chw,
    gt_chw,
    out_path,
    particle_error_colorbar_limit: float = 1.0,
):
    """
    保存 evaluate_all 的颗粒图像四联图：`LR | Fake | HR | Error`。

    Error 面板与 test_all 里的颗粒误差图保持一致，表示 `Generated SR - Original HR`。
    particle_error_colorbar_limit 来自各模型的 global_data.esrgan.PARTICLE_ERROR_COLORBAR_LIMIT；
    只控制 Error 面板和右侧色条的显示范围，不改变 sr_error.npy 或 ESMSE 等指标。
    Error 面板左上角额外标注 ESMSE，便于直接从图片判断频域颗粒重建误差。
    LR 面板只居中补白，不做插值放大，避免误导真实低分辨率细节。
    """
    lr_vis = _ensure_three_channel_for_panel(lr_tensor)
    fake_vis = _ensure_three_channel_for_panel(fake_tensor)
    hr_vis = _ensure_three_channel_for_panel(hr_tensor)

    canvas_h = max(int(lr_vis.shape[-2]), int(fake_vis.shape[-2]), int(hr_vis.shape[-2]))
    canvas_w = max(int(lr_vis.shape[-1]), int(fake_vis.shape[-1]), int(hr_vis.shape[-1]))
    triplet = build_triplet_row(lr_vis, fake_vis, hr_vis, sep_width=6)
    error_panel = _particle_error_tensor_for_panel(
        pred_chw,
        gt_chw,
        device=triplet.device,
        dtype=triplet.dtype,
        height=canvas_h,
        width=canvas_w,
        limit=particle_error_colorbar_limit,
    )
    sep = add_vertical_separator(triplet, sep_width=6, value=1.0)
    colorbar_width = 58
    colorbar_sep = add_vertical_separator(triplet, sep_width=4, value=1.0)
    error_colorbar = _particle_error_colorbar_tensor(
        triplet.device,
        triplet.dtype,
        height=canvas_h,
        limit=particle_error_colorbar_limit,
        width=colorbar_width,
    )
    # Error 面板右侧补一条固定范围色条，便于直接判断颗粒误差的数值大小。
    panel = torch.cat([triplet, sep, error_panel, colorbar_sep, error_colorbar], dim=3)
    panel = _add_headers_to_panel(
        panel,
        headers=("LR", "Fake", "HR", "Error", ""),
        column_widths=(canvas_w, canvas_w, canvas_w, canvas_w, colorbar_width),
        separator_widths=(6, 6, 6, 4),
    )
    esmse_value = compute_particle_image_esmse(pred_chw, gt_chw)
    esmse_text = f"ESMSE = {esmse_value:.4f}" if np.isfinite(esmse_value) else "ESMSE = nan"
    # Error 列起点 = 前三列宽度 + 三条分隔线宽度；header 高度为 _add_headers_to_panel 默认 22。
    panel = _annotate_metric_on_panel(panel, esmse_text, x=3 * canvas_w + 3 * 6 + 8, y=22 + 8)
    save_image(panel.clamp(0, 1), str(out_path), normalize=False)
