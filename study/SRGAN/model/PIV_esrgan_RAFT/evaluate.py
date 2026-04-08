from loguru import logger
import os
import time
from datetime import datetime
from pathlib import Path
import csv
import math
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
from types import SimpleNamespace
from PIL import Image, ImageDraw, ImageFont

from study.SRGAN.model.PIV_esrgan_RAFT.Module.loss import pixel_loss
from study.SRGAN.model.PIV_esrgan_RAFT.global_class import global_data
from study.SRGAN.model.PIV_esrgan_RAFT.judge_delicators import _to_np_chw, _mse, _psnr_from_mse, _energy_spectrum_mse, \
    _r2_score, _ssim_score, _tke_reconstruction_accuracy, _nrmse, _energy_spectrum_curves


from study.SRGAN.model.PIV_esrgan_RAFT.visual_plot_init import build_flo_uvw_pred_gt_panel, _omega_star_from_uv
from study.SRGAN.model.PIV_esrgan_RAFT.visual_plot_save import save_vorticity_quiver_compare, _save_triplet, _save_pair, \
    _save_energy_spectrum_plot
from study.SRGAN.util.image_util import flow_to_color_tensor, build_triplet_row, add_vertical_separator, \
    add_horizontal_separator, build_pair_row, _select_metric_or_save_channels

"""
验证函数 start
"""


def validate_and_save(result_dir, model, val_dataloader, device, epoch, data_type, SAVE_AS_GRAY=None):
    """
    每轮验证时保存主对比图，并在 flo 模态下额外保存 U/V/S 与涡量矢量图。 只验证保存loader的第一个batch的图
    flo:
        LR | Fake | HR

    image_pair:
        (previous: LR|Fake|HR) || (next: LR|Fake|HR)


    """

    if SAVE_AS_GRAY is None:
        SAVE_AS_GRAY = global_data.esrgan.SAVE_AS_GRAY

    model.eval()
    generator = model.piv_esrgan_generator
    os.makedirs(result_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            # RAFT 联合评估固定同时读取图像对与流场真值，不再按 data_type 做 image_pair/flo 分叉。
            lr_prev = batch["image_pair"]["previous"]["lr_data"].to(device)
            hr_prev = batch["image_pair"]["previous"]["gr_data"].to(device)
            lr_next = batch["image_pair"]["next"]["lr_data"].to(device)
            hr_next = batch["image_pair"]["next"]["gr_data"].to(device)
            hr_images = batch["flo"]["gr_data"].to(device)
            # RAFT 监督只吃 uv 两个通道；评估对比仍保留三通道真值 [u, v, magnitude]。
            hr_images_uv = hr_images[:, :2, :, :]

            # 联合模型 forward 当前返回 4 个值：
            # pred_prev, pred_next, flow_predictions, outputs_dict。
            # 这里显式解包，避免继续沿用旧版“只返回字典”的调用方式。
            _, _, _, outputs = model.forward(
                input_lr_prev=lr_prev,
                input_lr_next=lr_next,
                input_gr_prev=hr_prev,
                input_gr_next=hr_next,
                # 这里显式只把 uv 两个通道送给 RAFT，避免 magnitude 参与监督。
                flowl0=hr_images_uv,
            )

            fake_prev = outputs["sr_prev"]
            fake_next = outputs["sr_next"]
            fake_images = outputs["flow_predictions"][-1]
            # RAFT 预测本身只有 uv 两个通道；为了和三通道真值对齐，
            # 这里补一个 magnitude 通道，供评估指标和可视化使用。
            fake_images_uvw = _flow_uv_to_uvw(fake_images)

            resize_lr_prev = F.interpolate(
                lr_prev,
                size=hr_prev.shape[2:],
                mode="nearest",
            )
            resize_lr_next = F.interpolate(
                lr_next,
                size=hr_next.shape[2:],
                mode="nearest",
            )
            if SAVE_AS_GRAY:
                if hr_prev.shape[1] < 1:
                    logger.error(f'Unsupported previous channel count: {hr_prev.shape[1]}')
                    raise ValueError(f"Unsupported previous channel count: {hr_prev.shape[1]}")
                if hr_next.shape[1] < 1:
                    logger.error(f'Unsupported next channel count: {hr_next.shape[1]}')
                    raise ValueError(f"Unsupported next channel count: {hr_next.shape[1]}")
            else:
                if hr_prev.shape[1] != 3:
                    logger.error(f'Unsupported previous channel count: {hr_prev.shape[1]}')
                    raise ValueError(f"Unsupported previous channel count: {hr_prev.shape[1]}")
                if hr_next.shape[1] != 3:
                    logger.error(f'Unsupported next channel count: {hr_next.shape[1]}')
                    raise ValueError(f"Unsupported next channel count: {hr_next.shape[1]}")

            if hr_images.shape[1] < 2 or fake_images.shape[1] < 2:
                logger.error('RAFT 联合可视化至少需要前两通道(u,v)')
                raise ValueError("RAFT 联合可视化至少需要前两通道(u,v)")

            # 统一颜色尺度：用 HR 的 uv 计算 ref_max_rad。
            ref_max_rad = _compute_flow_ref_max_rad(hr_images)

            # RAFT 分支统一用 uv 转彩色可视化（不直接 save 原3通道）。
            # 这里不再使用 flo 的 LR 图像，只保留 Pred/HR 两列。
            fake_color = _flow_to_color_preview(fake_images[:, :2], ref_max_rad=ref_max_rad)
            hr_color = _flow_to_color_preview(hr_images[:, :2], ref_max_rad=ref_max_rad)

            sample_rows = []
            for i in range(lr_prev.size(0)):
                # 上半部分：previous / next 的超分三联图。
                single_lr_prev = _prepare_image_pair_tensor_for_save(
                    resize_lr_prev[i:i + 1], SAVE_AS_GRAY
                )
                single_fake_prev = _prepare_image_pair_tensor_for_save(
                    fake_prev[i:i + 1], SAVE_AS_GRAY
                )
                single_hr_prev = _prepare_image_pair_tensor_for_save(
                    hr_prev[i:i + 1], SAVE_AS_GRAY
                )

                single_lr_next = _prepare_image_pair_tensor_for_save(
                    resize_lr_next[i:i + 1], SAVE_AS_GRAY
                )
                single_fake_next = _prepare_image_pair_tensor_for_save(
                    fake_next[i:i + 1], SAVE_AS_GRAY
                )
                single_hr_next = _prepare_image_pair_tensor_for_save(
                    hr_next[i:i + 1], SAVE_AS_GRAY
                )

                left_group = build_triplet_row(single_lr_prev, single_fake_prev, single_hr_prev, sep_width=6)
                right_group = build_triplet_row(single_lr_next, single_fake_next, single_hr_next, sep_width=6)
                group_sep = add_vertical_separator(left_group, sep_width=16, value=1.0)
                image_row = torch.cat([left_group, group_sep, right_group], dim=3)
                # 在图像对拼图顶部标注每列含义，便于直接区分 LR / Fake / HR。
                image_w = single_lr_prev.shape[3]
                image_row = _add_headers_to_panel(
                    image_row,
                    headers=["Prev-LR", "Prev-Fake", "Prev-HR", "Next-LR", "Next-Fake", "Next-HR"],
                    column_widths=[image_w, image_w, image_w, image_w, image_w, image_w],
                    separator_widths=[6, 6, 16, 6, 6],
                )

                # 下半部分：流场双联图 Pred/HR，不再包含 flo 的 LR 图像。
                flow_row = build_pair_row(
                    fake_color[i:i + 1],
                    hr_color[i:i + 1],
                    sep_width=6,
                )
                flow_w = fake_color.shape[3]
                flow_row = _add_headers_to_panel(
                    flow_row,
                    headers=["Flow-Pred", "Flow-HR"],
                    column_widths=[flow_w, flow_w],
                    separator_widths=[6],
                )

                # 图像对分支在灰度保存模式下仍应保持“灰度外观”，
                # 而 flow_row 需要继续保持 3 通道彩色。
                # 这里采用的做法是：
                # 仅在拼接这一刻，把单通道灰度 image_row 复制成 3 个相同通道。
                # 这样视觉上仍然是灰度图像，但张量通道数能和 flow_row 对齐。
                if image_row.shape[1] != flow_row.shape[1]:
                    if image_row.shape[1] == 1 and flow_row.shape[1] == 3:
                        image_row = image_row.repeat(1, 3, 1, 1)
                    elif image_row.shape[1] == 3 and flow_row.shape[1] == 1:
                        flow_row = flow_row.repeat(1, 3, 1, 1)
                    else:
                        raise ValueError(
                            f"Cannot align channel counts for preview concat: "
                            f"image_row={tuple(image_row.shape)}, flow_row={tuple(flow_row.shape)}"
                        )

                # 同一样本把图像对结果和流场结果拼成一整行，便于一次性总览。
                h_sep = add_horizontal_separator(
                    width=max(image_row.shape[3], flow_row.shape[3]),
                    channels=image_row.shape[1],
                    sep_height=10,
                    value=1.0,
                    device=image_row.device,
                    dtype=image_row.dtype,
                )

                if image_row.shape[3] < h_sep.shape[3]:
                    pad = h_sep.shape[3] - image_row.shape[3]
                    image_row = F.pad(image_row, (0, pad, 0, 0), value=1.0)
                if flow_row.shape[3] < h_sep.shape[3]:
                    pad = h_sep.shape[3] - flow_row.shape[3]
                    flow_row = F.pad(flow_row, (0, pad, 0, 0), value=1.0)

                row = torch.cat([image_row, h_sep, flow_row], dim=2)
                sample_rows.append(row)

            uvs_compare_panel = build_flo_uvw_pred_gt_panel(fake_images_uvw, hr_images)
            uvs_compare_panel = _add_headers_to_panel(
                uvs_compare_panel,
                headers=["Flow-Pred", "Flow-HR"],
                column_widths=[fake_images_uvw.shape[-1], hr_images.shape[-1]],
                separator_widths=[6],
            )

            batch_combined = sample_rows[0]
            for row in sample_rows[1:]:
                h_sep = add_horizontal_separator(
                    width=batch_combined.shape[3],
                    channels=batch_combined.shape[1],
                    sep_height=10,
                    value=1.0,
                    device=batch_combined.device,
                    dtype=batch_combined.dtype,
                )
                batch_combined = torch.cat([batch_combined, h_sep, row], dim=2)

            save_path = os.path.join(
                result_dir,
                f"epoch_{epoch + 1}_batch_{batch_idx}_results.png"
            )
            # 在额外保存一张 u / v / s 对比图。
            save_image(
                uvs_compare_panel,
                os.path.join(result_dir, f"epoch_{epoch + 1}_batch_{batch_idx}_results_uvs.png"),
                normalize=False
            )
            # 额外保存瞬时涡流速度场图。
            save_vorticity_quiver_compare(
                fake_images_uvw, hr_images,
                os.path.join(result_dir, f"epoch_{epoch + 1}_batch_{batch_idx}_vorticity_quiver.png"),
                stride=6
            )
            save_image(batch_combined.clamp(0, 1), save_path, normalize=False)
            logger.info(f"Saved validation image: {save_path}")
            break
# 计算 PSNR 函数
def calculate_psnr(fake_image, hr_image):
    """计算单张图 PSNR。"""

    mse = torch.mean((fake_image - hr_image) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(1.0 / math.sqrt(mse))
    return psnr


def _build_raft_eval_args():
    """
    为 RAFT 评估阶段构造一个最小参数对象。
    """
    return SimpleNamespace(
        amp=bool(getattr(global_data.esrgan, "AMP", False)),
        iters=int(getattr(global_data.esrgan, "GRU_ITERS", 12)),
    )


def _compute_aee_from_flow_tensors(pred_bchw: torch.Tensor, gt_bchw: torch.Tensor) -> float:
    """
    计算 batch 流场的 Average Endpoint Error（AEE）。
    AEE 在这里等价于像素级 EPE 的全局平均值。
    """
    if pred_bchw.shape[1] < 2 or gt_bchw.shape[1] < 2:
        return float("nan")
    epe_map = torch.sqrt(torch.sum((pred_bchw[:, :2] - gt_bchw[:, :2]) ** 2, dim=1))
    return float(epe_map.mean().item())


def _flow_uv_to_uvw(flow_bchw: torch.Tensor) -> torch.Tensor:
    """
    将 2 通道流场 [u, v] 扩展成 3 通道 [u, v, magnitude]。

    说明：
        训练中的 RAFT 预测只有两个通道 uv，
        但评估和可视化里，flo 真值是三通道 [u, v, magnitude]，
        因此这里显式补出第三通道 magnitude = sqrt(u^2 + v^2)。
    """
    if flow_bchw.dim() != 4:
        raise ValueError(f"Expected flow to be 4D [B, C, H, W], got shape={tuple(flow_bchw.shape)}")
    if flow_bchw.shape[1] < 2:
        raise ValueError(f"Expected flow to have at least 2 channels, got shape={tuple(flow_bchw.shape)}")

    u = flow_bchw[:, 0:1]
    v = flow_bchw[:, 1:2]
    mag = torch.sqrt(u * u + v * v + 1e-12)
    return torch.cat([u, v, mag], dim=1)


def _compute_aee_from_chw(pred_chw: np.ndarray, gt_chw: np.ndarray) -> float:
    """
    计算单样本 CHW 流场的 AEE。
    """
    if pred_chw.shape[0] < 2 or gt_chw.shape[0] < 2:
        return float("nan")
    du = pred_chw[0] - gt_chw[0]
    dv = pred_chw[1] - gt_chw[1]
    epe = np.sqrt(du * du + dv * dv)
    return float(np.mean(epe))


def _prepare_image_pair_tensor_for_save(
    tensor_bchw: torch.Tensor,
    save_as_gray: bool,
) -> torch.Tensor:
    """
    统一 image_pair 分支的保存前处理：
    1. 先按 SAVE_AS_GRAY 选择可视化通道；
    2. 再统一裁剪到 [0, 1]。
    """
    return _select_metric_or_save_channels(
        tensor_bchw, "image_pair", save_as_gray
    ).clamp(0, 1)


def _compute_flow_ref_max_rad(flow_gt_bchw: torch.Tensor) -> float:
    """
    用 GT 的 uv 通道统一计算彩色光流图的参考半径，
    避免不同位置各自计算导致同批样本颜色尺度不一致。
    """
    if flow_gt_bchw.dim() != 4 or flow_gt_bchw.shape[1] < 2:
        raise ValueError(f"Expected GT flow with shape [B, >=2, H, W], got {tuple(flow_gt_bchw.shape)}")
    gt_u = flow_gt_bchw[:, 0]
    gt_v = flow_gt_bchw[:, 1]
    gt_mag_uv = torch.sqrt(gt_u * gt_u + gt_v * gt_v)
    return max(torch.quantile(gt_mag_uv.flatten(), 0.99).item(), 1e-6)


def _flow_to_color_preview(flow_uv_bchw: torch.Tensor, ref_max_rad: float) -> torch.Tensor:
    """
    统一预测/真值光流彩图的生成方式，避免不同位置混用不同的 clamp/维度写法。
    """
    color, _ = flow_to_color_tensor(flow_uv_bchw[:, :2], ref_max_rad=ref_max_rad)
    return color.clamp(0, 1)


def _tensor_to_rgb_pil(tensor: torch.Tensor) -> Image.Image:
    """
    将 [1,C,H,W] 或 [C,H,W] 的张量转成 PIL RGB 图像，便于在顶部写标题。
    单通道输入会复制成 3 通道，因此视觉上仍然保持灰度外观。
    """
    if tensor.dim() == 4:
        tensor = tensor[0]
    tensor = tensor.detach().cpu().clamp(0, 1)
    if tensor.shape[0] == 1:
        arr = (tensor[0].numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(arr, mode="L").convert("RGB")
    arr = (tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _pil_rgb_to_tensor01(image: Image.Image, device, dtype) -> torch.Tensor:
    """
    将 PIL RGB 图像转回 [1,3,H,W] 且范围为 [0,1] 的张量。
    """
    arr = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)


def _add_headers_to_panel(
    panel: torch.Tensor,
    headers: list[str],
    column_widths: list[int],
    separator_widths: list[int],
    header_height: int = 22,
) -> torch.Tensor:
    """
    在拼图顶部加列标题。

    参数：
        panel:            [1,C,H,W] 的可视化图
        headers:          每列标题
        column_widths:    每列图像本体宽度
        separator_widths: 列间分隔宽度，长度应为 len(headers)-1
    """
    if len(headers) != len(column_widths):
        raise ValueError("headers and column_widths must have the same length")
    if len(separator_widths) != max(0, len(headers) - 1):
        raise ValueError("separator_widths length must be len(headers)-1")

    base = _tensor_to_rgb_pil(panel)
    canvas = Image.new("RGB", (base.width, base.height + header_height), color=(255, 255, 255))
    canvas.paste(base, (0, header_height))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    x = 0
    for idx, (title, width) in enumerate(zip(headers, column_widths)):
        bbox = draw.textbbox((0, 0), title, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        tx = int(x + max((width - text_w) * 0.5, 0))
        ty = int(max((header_height - text_h) * 0.5, 0))
        draw.text((tx, ty), title, fill=(0, 0, 0), font=font)
        x += width
        if idx < len(separator_widths):
            x += separator_widths[idx]

    return _pil_rgb_to_tensor01(canvas, panel.device, panel.dtype)


def _save_heatmap(arr_2d: np.ndarray, out_png: Path, title: str, cmap: str = "viridis", symmetric: bool = False) -> None:
    """
    保存二维热力图。
    symmetric=True 时，颜色范围关于 0 对称，更适合误差正负分布图。
    """
    plt.figure(figsize=(4.8, 4.0), dpi=160)
    if symmetric:
        vmax = float(np.max(np.abs(arr_2d))) + 1e-12
        vmin = -vmax
    else:
        vmin = float(np.min(arr_2d))
        vmax = float(np.max(arr_2d))
    plt.imshow(arr_2d, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def _compute_flow_error_maps(pred_bchw: torch.Tensor, gt_bchw: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    统一计算水平位移误差、垂直位移误差和端点误差图。
    """
    pred_np = _to_np_chw(pred_bchw[0])
    gt_np = _to_np_chw(gt_bchw[0])
    delta_u = pred_np[0] - gt_np[0]
    delta_v = pred_np[1] - gt_np[1]
    epe = np.sqrt(delta_u * delta_u + delta_v * delta_v)
    return delta_u, delta_v, epe


def _save_signed_error_map(
    arr_2d: np.ndarray,
    out_png: Path,
    title_text: str,
    colorbar_label: str,
    vmin: float = -0.5,
    vmax: float = 0.5,
    cmap: str = "bwr",
    stat_text: str | None = None,
) -> None:
    """
    保存带固定对称色标的误差图。
    这里固定到 [-0.5, 0.5]，并采用蓝-白-红渐变，以对齐参考图风格。
    """
    fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.9), dpi=160)
    im = ax.imshow(arr_2d, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    if stat_text:
        ax.text(
            0.02, 0.98, stat_text,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=18, color="black",
        )
    else:
        ax.set_title(title_text)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(colorbar_label, rotation=270, labelpad=18)
    cb.set_ticks(np.linspace(vmin, vmax, 5))
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _save_error_histogram(
    values_1d: np.ndarray,
    out_png: Path,
    title_text: str,
    xlabel: str,
    bins: int = 121,
    color: str = "#F4A142",
) -> None:
    """
    保存单组误差直方图，风格对齐参考图的中心对称分布展示。
    """
    values = values_1d.reshape(-1).astype(np.float32)
    max_abs = float(np.quantile(np.abs(values), 0.995))
    max_abs = max(max_abs, 0.05)
    fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.9), dpi=160)
    ax.hist(values, bins=bins, range=(-max_abs, max_abs), color=color, alpha=0.65, edgecolor="none")
    ax.set_xlim(-max_abs, max_abs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title_text)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _save_flow_error_visuals(pred_bchw: torch.Tensor, gt_bchw: torch.Tensor, out_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    保存预测光流彩色图、AEE 误差图、Δu/Δv 误差分布图、涡度误差图。

    返回：
        delta_u_2d: 水平位移误差图
        delta_v_2d: 垂直位移误差图
        epe_2d:     端点误差图
    """
    pred_np = _to_np_chw(pred_bchw[0])
    gt_np = _to_np_chw(gt_bchw[0])

    du, dv, epe = _compute_flow_error_maps(pred_bchw, gt_bchw)
    aee = float(np.mean(epe))

    ref_max_rad = _compute_flow_ref_max_rad(gt_bchw)
    pred_color = _flow_to_color_preview(pred_bchw[:, :2], ref_max_rad=ref_max_rad)
    gt_color = _flow_to_color_preview(gt_bchw[:, :2], ref_max_rad=ref_max_rad)
    # 统一命名为 pred_flow / gt_flow，避免和别处的 pred_flow.png 重名但语义重复。
    save_image(pred_color, str(out_dir / "pred_flow.png"), normalize=False)
    save_image(gt_color, str(out_dir / "gt_flow.png"), normalize=False)

    # AEE 仍按标准 EPE 公式计算：sqrt((u_pred-u_gt)^2 + (v_pred-v_gt)^2)，
    # 这里只把绘图色标固定到 [-0.5, 0.5]，以对齐参考图的视觉风格。
    _save_signed_error_map(
        epe,
        out_dir / "aee_error.png",
        title_text="AEE Error",
        colorbar_label="Abs.error [px]",
        stat_text=rf"$AEE={aee:.4f}$",
    )
    _save_error_histogram(
        du,
        out_dir / "delta_u_error.png",
        title_text=r"$\Delta u$ Error Distribution",
        xlabel=r"$\Delta u$ displacement [px]",
        color="#F4A142",
    )
    _save_error_histogram(
        dv,
        out_dir / "delta_v_error.png",
        title_text=r"$\Delta v$ Error Distribution",
        xlabel=r"$\Delta v$ displacement [px]",
        color="#4C9F70",
    )

    pred_omega = _omega_star_from_uv(pred_np[0], pred_np[1])
    gt_omega = _omega_star_from_uv(gt_np[0], gt_np[1])
    vorticity_error = pred_omega - gt_omega

    np.save(out_dir / "pred_vorticity.npy", pred_omega.astype(np.float32))
    np.save(out_dir / "gt_vorticity.npy", gt_omega.astype(np.float32))
    np.save(out_dir / "vorticity_error.npy", vorticity_error.astype(np.float32))
    _save_heatmap(vorticity_error, out_dir / "vorticity_error.png", title="Vorticity Error", cmap="RdBu_r", symmetric=True)

    return du.astype(np.float32), dv.astype(np.float32), epe.astype(np.float32)


def _histogram_matrix(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """
    将直方图统计整理成两列矩阵：[bin_center, count]。
    这样保存成 .npy 后，后处理读起来更直接。
    """
    counts, edges = np.histogram(values.reshape(-1), bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return np.stack([centers.astype(np.float32), counts.astype(np.float32)], axis=1)


def _delta_u_histogram_matrix(delta_u_2d: np.ndarray, bins: int = 201) -> np.ndarray:
    """
    统计 Δu 的像素计数分布。
    分布区间以 0 为中心，用来观察误差是否高度集中在零附近。
    """
    max_abs = float(np.max(np.abs(delta_u_2d))) + 1e-12
    edges = np.linspace(-max_abs, max_abs, bins + 1, dtype=np.float32)
    return _histogram_matrix(delta_u_2d, edges)


def _delta_v_histogram_matrix(delta_v_2d: np.ndarray, bins: int = 201) -> np.ndarray:
    """
    统计 Δv 的像素计数分布。
    分布区间同样以 0 为中心，便于和 Δu 采用一致的误差分析口径。
    """
    max_abs = float(np.max(np.abs(delta_v_2d))) + 1e-12
    edges = np.linspace(-max_abs, max_abs, bins + 1, dtype=np.float32)
    return _histogram_matrix(delta_v_2d, edges)


def _epe_histogram_matrix(epe_2d: np.ndarray, bins: int = 201) -> np.ndarray:
    """
    统计端点误差 EPE 的像素计数分布。
    """
    max_val = float(np.max(epe_2d)) + 1e-12
    edges = np.linspace(0.0, max_val, bins + 1, dtype=np.float32)
    return _histogram_matrix(epe_2d, edges)


def _ensure_csv_columns(csvOperator, required_columns: list[str]) -> None:
    """
    运行时确保损失 CSV 包含新增列。
    为了不改动外部初始化逻辑，这里在 evaluate 阶段按需扩列并保留已有内容。
    """
    if csvOperator is None or not hasattr(csvOperator, "columns"):
        return
    missing = [col for col in required_columns if col not in csvOperator.columns]
    if not missing:
        return
    old_rows = csvOperator.read()
    columns = list(csvOperator.columns)
    if "time" in columns:
        time_idx = columns.index("time")
        columns = columns[:time_idx] + missing + columns[time_idx:]
    else:
        columns = columns + missing
    csvOperator.columns = columns
    csvOperator._write_all(old_rows)

# 验证函数  RAFT 联合验证
def validate_raft(model, dataloader, device):
    """
    在 RAFT 联合验证集上计算平均像素损失、平均 PSNR、平均能量谱 MSE 与平均 AEE。
    这里不再区分 image_pair / flo，而是固定走联合模型的一条评估路径。
    """
    model.eval()
    generator = model.piv_esrgan_generator
    total_val_ssim_loss = 0.0
    total_val_mse_loss = 0.0

    total_psnr = 0.0
    total_energy_spectrum_mse = 0.0
    total_aee = 0.0
    loss_count = 0
    num_images = 0

    with torch.no_grad():
        for batch in dataloader:
            lr_prev = batch["image_pair"]["previous"]["lr_data"].to(device)
            hr_prev = batch["image_pair"]["previous"]["gr_data"].to(device)
            lr_next = batch["image_pair"]["next"]["lr_data"].to(device)
            hr_next = batch["image_pair"]["next"]["gr_data"].to(device)
            flow_gt = batch["flo"]["gr_data"].to(device)
            # RAFT 监督只使用 uv 两个通道；这里保留三通道原始真值用于其他评估项。
            flow_gt_uv = flow_gt[:, :2, :, :]

            if hasattr(generator, "forward_pair"):
                fake_prev, fake_next = generator.forward_pair(lr_prev, lr_next)
            else:
                fake_prev = generator(lr_prev)
                fake_next = generator(lr_next)

            for fake_images, hr_images in ((fake_prev, hr_prev), (fake_next, hr_next)):
                _, _, mse_total, ssim_total, _ = pixel_loss(fake_images, hr_images, global_data.esrgan.SAVE_AS_GRAY)
                total_val_ssim_loss += ssim_total.item()
                total_val_mse_loss += mse_total.item()
                loss_count += 1

                fake_images_for_metric = _select_metric_or_save_channels(
                    fake_images, "image_pair", global_data.esrgan.SAVE_AS_GRAY
                )
                hr_images_for_metric = _select_metric_or_save_channels(
                    hr_images, "image_pair", global_data.esrgan.SAVE_AS_GRAY
                )

                for fake_image, hr_image in zip(fake_images_for_metric, hr_images_for_metric):
                    total_psnr += calculate_psnr(fake_image, hr_image)
                    total_energy_spectrum_mse += _energy_spectrum_mse(_to_np_chw(fake_image), _to_np_chw(hr_image))
                    num_images += 1

            # 联合模型下直接顺带评估 RAFT 的 AEE。
            # 联合模型 forward 当前返回 4 个值；这里取最后一个 outputs 字典读取 RAFT 指标。
            _, _, _, outputs = model.forward(
                input_lr_prev=lr_prev,
                input_lr_next=lr_next,
                input_gr_prev=hr_prev,
                input_gr_next=hr_next,
                # 这里显式只把 uv 两个通道送给 RAFT。
                flowl0=flow_gt_uv,
            )
            total_aee += float(outputs["raft_metrics"]["epe"])
            #一个batch 就行了 因为训练中的验证只需要1次batch验证
            break
    avg_val_ssim_loss = total_val_ssim_loss / max(loss_count, 1)
    avg_val_mse_loss = total_val_mse_loss / max(loss_count, 1)
    avg_psnr = total_psnr / max(num_images, 1)
    avg_energy_spectrum_mse = total_energy_spectrum_mse / max(num_images, 1)
    avg_aee = total_aee / max(loss_count, 1)
    return avg_val_ssim_loss, avg_val_mse_loss, avg_psnr, avg_energy_spectrum_mse, avg_aee

"""
验证函数 end
"""
def evaluate(epoch,class_name,data_type,device,
             animator=None,
             validate_loader=None,loss_label=None,validate_label=None,SCALE=1,csvOperator=None,metric=None,train_loader_lens=1,
             model=None):
    """
   每轮结束后执行验证、记录日志、保存模型与损失曲线。
    :param epoch: 轮次
    :param class_name:类别
    :param data_type: 数据类型，当前联合评估路径固定使用 "RAFT"
    :param device: cuda或者cpu
    :param model: 模型
    :param animator: 图表动画
    :param validate_loader: 验证集数据加载器
    :param loss_label: 损失函数描述label
    :param validate_label:  验证参数的label
    :param SCALE:上采样因子 具体放大平方倍
    :param csvOperator:loss等数据 存储csv
    :param metric:累加器
    :param train_loader_lens:训练数据长度
    :return:
    """
    # 每轮训练结束后进行验证

    avg_val_ssim_loss, avg_val_mse_loss, avg_psnr, avg_val_energy_spectrum_mse, avg_val_aee = validate_raft(
        model, validate_loader, device
    )

    # # 训练阶段损失 CSV 需要新增验证能量谱误差和 RAFT 的 AEE。
    # required_csv_columns = ["VAL_energy_spectrum_mse", "VAL_AEE"]
    # _ensure_csv_columns(csvOperator, required_csv_columns)

    wandb.log({
        "classname": class_name,
        "data_type": data_type,
        "VAL_AVG_MSE_LOSS": avg_val_mse_loss ,
        "VAL_AVG_SSIM_LOSS": avg_val_ssim_loss ,
        "VAL_energy_spectrum_mse": avg_val_energy_spectrum_mse,
        "VAL_AEE": avg_val_aee,
        "avg_psnr": avg_psnr,
        "Epoch": epoch,
        **{
            loss_label[index]: metric[index] / (train_loader_lens)
            for index in range(len(loss_label))
        }
    })
    current_time = time.time()
    logger.info(
        f"Epoch [{epoch + 1}/{global_data.esrgan.EPOCH_NUMS}] |{class_name} {data_type} |running time:{int(current_time - global_data.esrgan.START_TIME )}s | "
        f"VAL_AVG_MSE_LOSS: {avg_val_mse_loss} | VAL_AVG_SSIM_LOSS: {avg_val_ssim_loss} | "
        f"VAL_energy_spectrum_mse: {avg_val_energy_spectrum_mse} | VAL_AEE: {avg_val_aee} | Avg PSNR: {avg_psnr:.2f}"
    )
    loss_str = "".join([loss_label[index] + ':' + str(metric[index] / train_loader_lens) + "," for index in
                        range(len(loss_label))])
    logger.info(loss_str)

    # 每轮训练结束后进行验证，并保存最后一批图像
    validate_and_save(f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.PREDICT_DIR}", model,
                      validate_loader, device, epoch, data_type=data_type)
    # 保存模型
    model_save_path = f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.MODEL_DIR}/PIV_ESRGAN_RAFT_model_{global_data.esrgan.name}.pth"
    torch.save(model.state_dict(), model_save_path)
    logger.info(
        f"{class_name} {data_type} |Models saved: v -> {model_save_path}")

    # 保存每一epoch的损失
    all_loss_and_val_Datas = [metric[index] / (train_loader_lens)  for index in range(len(loss_label))] + [
        avg_val_mse_loss,
        avg_val_ssim_loss,
        avg_psnr,
        avg_val_energy_spectrum_mse,
        avg_val_aee,
    ]
    animator.add(epoch + 1,all_loss_and_val_Datas )
    # 保存到csv文件中
    csv_row = {"EPOCH": epoch + 1}
    for index in range(len(loss_label)):
        csv_row[loss_label[index]] = metric[index] / train_loader_lens
    csv_row["VAL_MSE_LOSS"] = avg_val_mse_loss
    csv_row["VAL_SSIM_Loss"] = avg_val_ssim_loss
    csv_row["Avg_PSNR"] = avg_psnr
    csv_row["VAL_energy_spectrum_mse"] = avg_val_energy_spectrum_mse
    csv_row["VAL_AEE"] = avg_val_aee
    csv_row["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csvOperator.create(csv_row)
    animator.save_png(
        f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.LOSS_DIR}/train_loss_epoch_{epoch + 1}_{global_data.esrgan.name}.png",
        fixed_groups=[
            ["g_loss", "d_loss","raft_loss"],
            ["g_perceptual_loss", "g_content_loss", "g_adversarial_loss"],
            ["g_loss_pixel", "g_loss_pixel_l1", "g_loss_pixel_mse", "g_loss_ssim", "g_loss_fft"],
            ["g_pair_temporal_loss", "g_pair_delta_loss", "g_pair_gradient_loss"],
            ["d_loss", "d_real_loss", "d_fake_loss"],
            [validate_label[0], validate_label[1], ],
            [validate_label[2]],
            [validate_label[3]],
            [validate_label[4]]
        ])
    pass

def evaluate_all(
    data_loader=None,
    class_name: str = "",
    data_type: str = "",
    SCALE: float = 1.0,
    output_root: str | Path = "",
    metrics_csv_path: str | Path | None = None,
    stride: int = 6,
    model=None,
):
    """
    验证整个 data_loader，计算并保存每个样本指标与可视化结果。
    evaluate_all(
        generator=generator,
        data_loader=validate_loader,
        class_name=class_name,
        data_type=data_type,
        SCALE=SCALE,
        output_root=f"{OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE*SCALE)}/{PREDICT_ALL_DIR}",
        metrics_csv_path=f"{OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE*SCALE)}/metrics_all.csv",
        stride=6,
    )
    指标:
    - MSE
    - PSNR
    - Energy Spectrum MSE
    - R2
    - SSIM
    - TKE 重建精度
    - NRMSE

    保存内容:
    - 每个样本独立文件夹
    - image_pair: 保存 LR/Fake/HR 自身图 + 差异图 + 三联图 + 能量谱曲线(.npy/.png)
    - flo: 保存 flo 数组(.npy) + 颜色流场三联图 + U/V/S 对比图 + 涡量矢量图 + 能量谱曲线(.npy/.png)
    - CSV: 每样本指标 + 均值行
    - 全局均值能量谱曲线(.npy/.png)
    """
    # evaluate_all 做的是“全量验证集/测试集统计”，不是训练中的单 batch 快速验证。
    # 所以这里除了保存每个样本结果，还会：
    # 1. 按类别写子目录
    # 2. 按类别写 metrics.csv
    # 3. 额外汇总 전체 metrics_all.csv
    generator = model.piv_esrgan_generator
    device = next(generator.parameters()).device
    generator.eval()
    model.eval()

    logger.info(f"[evaluate_all] SAVE_AS_GRAY={global_data.esrgan.SAVE_AS_GRAY}")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if metrics_csv_path is None:
        metrics_csv_path = output_root / f"metrics_{class_name}_{data_type}_x{int(SCALE * SCALE)}.csv"
    else:
        metrics_csv_path = Path(metrics_csv_path)
        metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)

    csv_fields = [
        "class_name", "data_type", "scale", "sample_id", "pair_type",
        "mse", "psnr", "energy_spectrum_mse", "VAL_AEE", "r2", "ssim", "tke_acc", "nrmse"
    ]

    # 这些属性是在 load_data 里给 validate/test dataset 动态挂上的。
    # mixed 模式下它们用于把样本重新按真实类别归档。
    dataset = getattr(data_loader, "dataset", None)
    known_class_names = list(getattr(dataset, "known_class_names", []))
    other_name = getattr(dataset, "other_class_name", "other")

    def bucket_class_name(sample_class_name: str | None) -> str:
        # 优先用已知类别；没有命中就归到 other，避免评估阶段因脏类别名直接丢样本。
        if sample_class_name in known_class_names:
            return str(sample_class_name)
        if sample_class_name is None or str(sample_class_name).strip() == "":
            return other_name
        return str(sample_class_name) if not known_class_names else other_name

    rows = []
    rows_by_class: dict[str, list[dict]] = {}
    curves_by_class: dict[str, dict[str, list[np.ndarray]]] = {}
    delta_u_hist_by_class: dict[str, list[np.ndarray]] = {}
    delta_v_hist_by_class: dict[str, list[np.ndarray]] = {}
    epe_hist_by_class: dict[str, list[np.ndarray]] = {}
    all_pred_curves = []
    all_gt_curves = []

    def register_curve(bucket_name: str, pred_curve: np.ndarray, gt_curve: np.ndarray) -> None:
        # 频谱曲线既要进全局统计，也要进分类别统计，所以这里同时登记两份。
        all_pred_curves.append(pred_curve)
        all_gt_curves.append(gt_curve)
        if bucket_name not in curves_by_class:
            curves_by_class[bucket_name] = {"pred": [], "gt": []}
        curves_by_class[bucket_name]["pred"].append(pred_curve)
        curves_by_class[bucket_name]["gt"].append(gt_curve)

    def append_row(row: dict) -> None:
        # 同一条样本记录：
        # - rows 用于总表
        # - rows_by_class 用于类别子表
        rows.append(row)
        bucket = row["class_name"]
        rows_by_class.setdefault(bucket, []).append(row)
        delta_u_hist_by_class.setdefault(bucket, [])
        delta_v_hist_by_class.setdefault(bucket, [])
        epe_hist_by_class.setdefault(bucket, [])

    def save_mean_spectrum(pred_curves: list[np.ndarray], gt_curves: list[np.ndarray], out_dir: Path, title: str) -> None:
        if not pred_curves or not gt_curves:
            return
        # 不同样本曲线长度可能略有不同，所以先截到共同最短长度再平均。
        min_len = min(min(len(x) for x in pred_curves), min(len(x) for x in gt_curves))
        pred_mean = np.mean(np.stack([x[:min_len] for x in pred_curves], axis=0), axis=0)
        gt_mean = np.mean(np.stack([x[:min_len] for x in gt_curves], axis=0), axis=0)
        np.save(out_dir / "energy_spectrum_pred_mean.npy", pred_mean.astype(np.float32))
        np.save(out_dir / "energy_spectrum_gt_mean.npy", gt_mean.astype(np.float32))
        _save_energy_spectrum_plot(pred_mean, gt_mean, out_dir / "energy_spectrum_mean_compare.png", title=title)

    def build_mean_row(target_rows: list[dict], bucket_name: str) -> dict:
        def _mean_of(key: str) -> float:
            vals = [float(r[key]) for r in target_rows if np.isfinite(float(r[key]))]
            return float(np.mean(vals)) if vals else float("nan")

        return {
            "class_name": bucket_name,
            "data_type": data_type,
            "scale": int(SCALE * SCALE),
            "sample_id": "MEAN",
            "pair_type": "all",
            "mse": _mean_of("mse"),
            "psnr": _mean_of("psnr"),
            "energy_spectrum_mse": _mean_of("energy_spectrum_mse"),
            "VAL_AEE": _mean_of("VAL_AEE"),
            "r2": _mean_of("r2"),
            "ssim": _mean_of("ssim"),
            "tke_acc": _mean_of("tke_acc"),
            "nrmse": _mean_of("nrmse"),
        }

    with torch.no_grad():
        pbar = tqdm(
            data_loader,
            desc=f"{class_name} {data_type} scale_{int(SCALE * SCALE)} Validating(all)",
            unit="batch", dynamic_ncols=True,
            ascii=True,
            leave=True,
        )

        for batch_idx, batch in enumerate(pbar):
            batch_class_names = batch.get("class_name", [])
            # batch['class_name'] 来自 data_load 的 collate_fn，是当前 batch 每个样本的真实类别名列表。

            # RAFT 联合评估固定同时读取图像对与流场，不再按 image_pair / flo 分路。
            lr_prev = batch["image_pair"]["previous"]["lr_data"].to(device)
            hr_prev = batch["image_pair"]["previous"]["gr_data"].to(device)
            lr_next = batch["image_pair"]["next"]["lr_data"].to(device)
            hr_next = batch["image_pair"]["next"]["gr_data"].to(device)
            # lr = batch["flo"]["lr_data"].to(device)
            hr = batch["flo"]["gr_data"].to(device)
            # RAFT 监督只使用 uv 两个通道；这里保留三通道原始真值用于评估和可视化。
            hr_uv = hr[:, :2, :, :]

            # 联合模型 forward 当前返回 4 个值；这里显式解包，后面统一从 outputs 字典取结果。
            _, _, _, outputs = model.forward(
                input_lr_prev=lr_prev,
                input_lr_next=lr_next,
                input_gr_prev=hr_prev,
                input_gr_next=hr_next,
                # 这里显式只把 uv 两个通道送给 RAFT。
                flowl0=hr_uv,
            )

            fake_prev = outputs["sr_prev"]
            fake_next = outputs["sr_next"]
            fake = outputs["flow_predictions"][-1]
            # RAFT 预测只有 uv 两个通道；这里补出 magnitude，
            # 这样后面和三通道真值 [u, v, magnitude] 的评估/可视化才能一一对应。
            fake_uvw = _flow_uv_to_uvw(fake)

            lr_prev_up = F.interpolate(lr_prev, size=hr_prev.shape[2:], mode="nearest")
            lr_next_up = F.interpolate(lr_next, size=hr_next.shape[2:], mode="nearest")
            # lr_up = F.interpolate(lr, size=hr.shape[2:], mode="nearest")

            B = hr.shape[0]
            for i in range(B):
                sample_bucket = bucket_class_name(batch_class_names[i] if i < len(batch_class_names) else None)
                class_root = output_root / sample_bucket
                class_root.mkdir(parents=True, exist_ok=True)

                sid = f"batch_{batch_idx}_idx_{i}_fid_{batch_idx}"
                one_dir = class_root / sid
                one_dir.mkdir(parents=True, exist_ok=True)

                # 保存 previous / next 两个超分结果。
                for pair_type, lr1_up, fk1, hr1 in [
                    ("previous", lr_prev_up[i:i + 1], fake_prev[i:i + 1], hr_prev[i:i + 1]),
                    ("next", lr_next_up[i:i + 1], fake_next[i:i + 1], hr_next[i:i + 1]),
                ]:
                    pair_dir = one_dir / pair_type
                    pair_dir.mkdir(parents=True, exist_ok=True)

                    lr_save = _prepare_image_pair_tensor_for_save(lr1_up, global_data.esrgan.SAVE_AS_GRAY)
                    fk_save = _prepare_image_pair_tensor_for_save(fk1, global_data.esrgan.SAVE_AS_GRAY)
                    hr_save = _prepare_image_pair_tensor_for_save(hr1, global_data.esrgan.SAVE_AS_GRAY)

                    # 频谱等数值指标继续复用统一通道选择后的结果，避免保存和计算口径不一致。
                    lr_eval = lr_save
                    fk_eval = fk_save
                    hr_eval = hr_save

                    save_image(lr_save, str(pair_dir / "lr.png"), normalize=False)
                    save_image(fk_save, str(pair_dir / "fake.png"), normalize=False)
                    save_image(hr_save, str(pair_dir / "hr.png"), normalize=False)

                    diff = (fk_save - hr_save).abs()
                    diff_gray = diff if diff.shape[1] == 1 else diff.mean(dim=1, keepdim=True)
                    save_image(diff, str(pair_dir / "diff_abs.png"), normalize=False)
                    save_image(diff_gray, str(pair_dir / "diff_abs_gray.png"), normalize=False)
                    save_image(diff_gray / (diff_gray.max() + 1e-8), str(pair_dir / "diff_abs_gray_norm.png"), normalize=False)
                    _save_triplet(lr_save, fk_save, hr_save, pair_dir / "image_triplet.png")

                    p_img = _to_np_chw(fk_eval[0])
                    g_img = _to_np_chw(hr_eval[0])
                    pred_curve, gt_curve = _energy_spectrum_curves(p_img, g_img)
                    np.save(pair_dir / "energy_spectrum_pred.npy", pred_curve.astype(np.float32))
                    np.save(pair_dir / "energy_spectrum_gt.npy", gt_curve.astype(np.float32))
                    _save_energy_spectrum_plot(pred_curve, gt_curve, pair_dir / "energy_spectrum_compare.png", title=f"{sid}-{pair_type} Energy Spectrum")

                # 保存流场预测及误差分析结果。
                # lr1 = lr[i:i + 1]
                # lr_up1 = lr_up[i:i + 1]
                fk1 = fake[i:i + 1]
                fk1_uvw = fake_uvw[i:i + 1]
                hr1 = hr[i:i + 1]

                # np.save(one_dir / "lr_flo.npy", _to_np_chw(lr1[0]).transpose(1, 2, 0))
                # 保存三通道预测流场 [u, v, magnitude]，便于和三通道真值直接对比。
                np.save(one_dir / "fake_flo.npy", _to_np_chw(fk1_uvw[0]).transpose(1, 2, 0))
                np.save(one_dir / "hr_flo.npy", _to_np_chw(hr1[0]).transpose(1, 2, 0))

                ref_max_rad = _compute_flow_ref_max_rad(hr1)

                # lr_color, _ = flow_to_color_tensor(lr_up1[:, :2], ref_max_rad=ref_max_rad)
                fk_color = _flow_to_color_preview(fk1[:, :2], ref_max_rad=ref_max_rad)
                hr_color = _flow_to_color_preview(hr1[:, :2], ref_max_rad=ref_max_rad)
                # 流场样本级拼图不再包含 flo 的 LR 图像，只保留 Pred/HR 两列。
                _save_pair(fk_color, hr_color, one_dir / "flow_triplet.png")
                # 单独的预测流场彩图已经在 _save_flow_error_visuals 中按统一命名保存为 pred_flow.png，
                # 这里不再重复落盘，避免同目录下出现语义重复的文件。

                # U/V/S 面板改为 Pred/HR 双列对比，不再包含 flo 的 LR 图像。
                uvs_panel = build_flo_uvw_pred_gt_panel(fk1_uvw, hr1)
                save_image(uvs_panel.clamp(0, 1), str(one_dir / "uvs_compare.png"), normalize=False)
                # 涡度图内部实际只依赖前两个通道 uv，这里传三通道预测和真值保持接口一致。
                save_vorticity_quiver_compare(fk1_uvw, hr1, str(one_dir / "vorticity_quiver.png"), stride=stride)
                # 额外保存 AEE 误差图、涡度误差图，并返回像素级误差用于统计分布。
                delta_u_map, delta_v_map, epe_map = _save_flow_error_visuals(fk1, hr1, one_dir)

                # 逐样本保存 Δu / Δv / EPE 分布。
                # Δu、Δv 直方图都以 0 为中心，便于横向比较水平/垂直位移误差。
                sample_delta_u_hist = _delta_u_histogram_matrix(delta_u_map)
                sample_delta_v_hist = _delta_v_histogram_matrix(delta_v_map)
                sample_epe_hist = _epe_histogram_matrix(epe_map)
                np.save(one_dir / "delta_u_hist.npy", sample_delta_u_hist)
                np.save(one_dir / "delta_v_hist.npy", sample_delta_v_hist)
                np.save(one_dir / "epe_hist.npy", sample_epe_hist)

                # 同时把每个 sample 的误差统计登记到所属类别，便于类别级汇总。
                # 这里要先确保类别桶已经创建，再去 append；
                # 否则像 DNS_turbulence 这类首次出现的类别会在 append 时直接 KeyError。
                delta_u_hist_by_class.setdefault(sample_bucket, [])
                delta_v_hist_by_class.setdefault(sample_bucket, [])
                epe_hist_by_class.setdefault(sample_bucket, [])
                delta_u_hist_by_class[sample_bucket].append(delta_u_map.reshape(-1))
                delta_v_hist_by_class[sample_bucket].append(delta_v_map.reshape(-1))
                epe_hist_by_class[sample_bucket].append(epe_map.reshape(-1))

                # 数值指标对比统一使用三通道 [u, v, magnitude]。
                p = _to_np_chw(fk1_uvw[0])
                g = _to_np_chw(hr1[0])

                mse = _mse(p, g)
                psnr = _psnr_from_mse(mse)
                es_mse = _energy_spectrum_mse(p, g)
                aee = _compute_aee_from_chw(p, g)
                r2 = _r2_score(p, g)
                ssim = _ssim_score(p, g)
                tke = _tke_reconstruction_accuracy(p, g)
                nrmse = _nrmse(p, g)

                pred_curve, gt_curve = _energy_spectrum_curves(p, g)
                np.save(one_dir / "energy_spectrum_pred.npy", pred_curve.astype(np.float32))
                np.save(one_dir / "energy_spectrum_gt.npy", gt_curve.astype(np.float32))
                _save_energy_spectrum_plot(pred_curve, gt_curve, one_dir / "energy_spectrum_compare.png", title=f"{sid} Energy Spectrum")
                register_curve(sample_bucket, pred_curve, gt_curve)

                append_row({
                    "class_name": sample_bucket,
                    "data_type": data_type,
                    "scale": int(SCALE * SCALE),
                    "sample_id": sid,
                    "pair_type": "RAFT",
                    "mse": mse,
                    "psnr": psnr,
                    "energy_spectrum_mse": es_mse,
                    "VAL_AEE": aee,
                    "r2": r2,
                    "ssim": ssim,
                    "tke_acc": tke,
                    "nrmse": nrmse,
                })

    mean_row = build_mean_row(rows, class_name)
    all_rows_with_mean = rows + [mean_row]

    # 根目录下的均值频谱图表示“整个 validate/test loader”的总体表现。
    save_mean_spectrum(
        all_pred_curves,
        all_gt_curves,
        output_root,
        title=f"{class_name}-{data_type}-x{int(SCALE*SCALE)} Mean Energy Spectrum",
    )

    # 根目录额外保存整套验证集的 Δu / EPE 统计结果，便于做全局误差分布分析。
    all_delta_u_values = [v for values in delta_u_hist_by_class.values() for v in values]
    all_delta_v_values = [v for values in delta_v_hist_by_class.values() for v in values]
    all_epe_values = [v for values in epe_hist_by_class.values() for v in values]
    if all_delta_u_values:
        np.save(output_root / "delta_u_hist_all.npy", _delta_u_histogram_matrix(np.concatenate(all_delta_u_values, axis=0)))
    if all_delta_v_values:
        np.save(output_root / "delta_v_hist_all.npy", _delta_v_histogram_matrix(np.concatenate(all_delta_v_values, axis=0)))
    if all_epe_values:
        np.save(output_root / "epe_hist_all.npy", _epe_histogram_matrix(np.concatenate(all_epe_values, axis=0)))

    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(all_rows_with_mean)

    for bucket_name, class_rows in rows_by_class.items():
        class_root = output_root / bucket_name
        class_root.mkdir(parents=True, exist_ok=True)
        class_mean_row = build_mean_row(class_rows, bucket_name)
        # 每个类别目录都有自己的 metrics.csv，末尾附一行该类别的均值结果。
        with open(class_root / "metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(class_rows + [class_mean_row])
        bucket_curves = curves_by_class.get(bucket_name, {"pred": [], "gt": []})
        # 每个类别也各自输出一张平均能量谱对比图，方便直接做类间比较。
        save_mean_spectrum(
            bucket_curves["pred"],
            bucket_curves["gt"],
            class_root,
            title=f"{bucket_name}-{data_type}-x{int(SCALE*SCALE)} Mean Energy Spectrum",
        )

        # 类别级 Δu / EPE 统计：把该类别所有 sample 的像素误差拼起来，再统一做直方图。
        if delta_u_hist_by_class.get(bucket_name):
            class_delta_u_values = np.concatenate(delta_u_hist_by_class[bucket_name], axis=0)
            np.save(class_root / "delta_u_hist_all.npy", _delta_u_histogram_matrix(class_delta_u_values))
        if delta_v_hist_by_class.get(bucket_name):
            class_delta_v_values = np.concatenate(delta_v_hist_by_class[bucket_name], axis=0)
            np.save(class_root / "delta_v_hist_all.npy", _delta_v_histogram_matrix(class_delta_v_values))
        if epe_hist_by_class.get(bucket_name):
            class_epe_values = np.concatenate(epe_hist_by_class[bucket_name], axis=0)
            np.save(class_root / "epe_hist_all.npy", _epe_histogram_matrix(class_epe_values))

    logger.info(f"[evaluate_all] metrics csv: {metrics_csv_path}")
    logger.info(f"[evaluate_all] sample outputs: {output_root}")
    return mean_row
