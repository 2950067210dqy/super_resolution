from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import save_image

from study.SRGAN.util.image_util import add_vertical_separator, build_triplet_row


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


def _particle_error_tensor_for_panel(pred_chw, gt_chw, device, dtype, height: int, width: int) -> torch.Tensor:
    """
    将颗粒图像误差 `SR-HR` 转成 bwr 伪彩图面板。

    误差范围使用 99.5% 分位数做对称色条，能避免少数极端点把整张误差图压暗；
    NaN/Inf 会被安全替换为 0，避免保存图像时出现非法像素。
    """
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

    finite_abs = np.abs(error_2d[np.isfinite(error_2d)])
    if finite_abs.size == 0:
        limit = 0.05
    else:
        limit = float(np.quantile(finite_abs, 0.995))
        limit = max(limit, 0.05)

    error_2d = np.nan_to_num(error_2d, nan=0.0, posinf=limit, neginf=-limit)
    normalized = np.clip((error_2d + limit) / (2.0 * limit), 0.0, 1.0)
    rgb = plt.get_cmap("bwr")(normalized)[..., :3].astype(np.float32)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    return _pad_tensor_to_canvas(tensor, height, width, value=1.0)


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


def _save_particle_error_histogram_plot(hist_matrix, out_png, title, xlabel, color="#AA3377") -> None:
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
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
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
    _save_hist_npy(out_dir / npy_name, hist, save_npy_fn, save_npy, force_npy)
    _save_particle_error_histogram_plot(hist, out_dir / png_name, title=title, xlabel=xlabel, color=color)
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
    )


def save_image_triplet_with_error(lr_tensor, fake_tensor, hr_tensor, pred_chw, gt_chw, out_path):
    """
    保存 evaluate_all 的颗粒图像四联图：`LR | Fake | HR | Error`。

    Error 面板与 test_all 里的颗粒误差图保持一致，表示 `Generated SR - Original HR`。
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
    )
    sep = add_vertical_separator(triplet, sep_width=6, value=1.0)
    panel = torch.cat([triplet, sep, error_panel], dim=3)
    panel = _add_headers_to_panel(
        panel,
        headers=("LR", "Fake", "HR", "Error"),
        column_widths=(canvas_w, canvas_w, canvas_w, canvas_w),
        separator_widths=(6, 6, 6),
    )
    esmse_value = compute_particle_image_esmse(pred_chw, gt_chw)
    esmse_text = f"ESMSE = {esmse_value:.4f}" if np.isfinite(esmse_value) else "ESMSE = nan"
    # Error 列起点 = 前三列宽度 + 三条分隔线宽度；header 高度为 _add_headers_to_panel 默认 22。
    panel = _annotate_metric_on_panel(panel, esmse_text, x=3 * canvas_w + 3 * 6 + 8, y=22 + 8)
    save_image(panel.clamp(0, 1), str(out_path), normalize=False)
