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
        因此开启最佳样本保存模式后，最终会同时保留 xxx.png 和 xxx.svg 两份文件。
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

    evaluate_all 的“每类只保留最佳样本”模式会在删除非最佳样本后调用这里，
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
    计算 evaluate_all 最佳样本排序分数，越小越好。

    优先级：
        1. VAL_AEE：优先按 RAFT/PIV 端点误差选图，越小表示流场预测越接近 GT；
        2. VAL_C_AEE：当样本没有有效 VAL_AEE 时，退回综合指标；
        3. energy_spectrum_mse：SR-only 或没有有效 C-AEE 时，再退回频谱误差。

    这样不改变任何指标计算，只用已有 rows 做“保留哪一个样本目录”的选择。
    如果某个样本同一指标有多行有效值，例如 previous / next 图像对，会取均值后比较。
    """
    for key in ("VAL_AEE", "VAL_C_AEE", "energy_spectrum_mse"):
        values = [_safe_metric_float(row, key) for row in rows_for_sample]
        values = [value for value in values if np.isfinite(value)]
        if values:
            return float(np.mean(values))
    return float("inf")


def prune_evaluate_all_to_best_sample_dirs(output_root, rows, logger=None, *, save_svg_sidecars: bool = True) -> dict:
    """
    evaluate_all 最佳样本保留器：每个类别只保留指标最好的一个样本目录。

    重要约束：
        - 只处理本次 rows 中出现过的 sample_id 目录；
        - 不删除 metrics.csv、ALL_CLASS*.CSV、类别/总体汇总图等非样本目录文件；
        - 指标 rows/CSV/均值已经在外部按全量样本计算，本函数只影响磁盘上的样本图像和 NPY；
        - 删除非最佳样本目录后，最终保留目录里的 PNG 原图会继续存在；
        - 同时为最终保留的 PNG 生成同名 SVG sidecar，因此最佳样本会同时有 PNG、SVG 和 NPY。
    """
    output_root = Path(output_root)
    if not output_root.exists():
        return {}

    sample_rows: dict[tuple[str, str], list[dict]] = {}
    for row in rows or []:
        class_name = str(row.get("class_name", "")).strip()
        sample_id = str(row.get("sample_id", "")).strip()
        if not class_name or not sample_id or sample_id in {"MEAN", "CLASS_MEAN"}:
            continue
        sample_rows.setdefault((class_name, sample_id), []).append(row)

    best_by_class: dict[str, tuple[str, float]] = {}
    for (class_name, sample_id), rows_for_sample in sample_rows.items():
        score = _best_only_row_score(rows_for_sample)
        current = best_by_class.get(class_name)
        if current is None or score < current[1]:
            best_by_class[class_name] = (sample_id, score)

    deleted_count = 0
    kept_summary = {}
    root_resolved = output_root.resolve()
    for (class_name, sample_id) in sample_rows.keys():
        best_sample_id, best_score = best_by_class.get(class_name, ("", float("inf")))
        sample_dir = output_root / class_name / sample_id
        if sample_id == best_sample_id:
            kept_summary[class_name] = {"sample_id": sample_id, "score": best_score}
            continue
        if not sample_dir.exists() or not sample_dir.is_dir():
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
        writer.writerow(["class_name", "sample_id", "best_score"])
        for class_name, info in sorted(kept_summary.items()):
            writer.writerow([class_name, info["sample_id"], info["score"]])

    svg_count = save_svg_sidecars_for_png_tree(output_root) if save_svg_sidecars else 0
    if logger is not None:
        logger.info(
            f"[evaluate_all best-only] kept_classes={len(kept_summary)}, "
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
        - FLOW_ERROR_HIST_* 控制 Δu/Δv/Δw 误差直方图；
        - EPE_HIST_* 控制 EPE 直方图；
        - PARTICLE_ERROR_HIST_* 控制颗粒图像误差直方图；
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
