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

from study.SRGAN.model.ESRuRAFT_PIV.Module.loss import pixel_loss
from study.SRGAN.model.ESRuRAFT_PIV.global_class import global_data
from study.SRGAN.model.ESRuRAFT_PIV.judge_delicators import _to_np_chw, _mse, _psnr_from_mse, _energy_spectrum_mse, \
    _r2_score, _ssim_score, _tke_reconstruction_accuracy, _nrmse, _energy_spectrum_curves


from study.SRGAN.model.ESRuRAFT_PIV.visual_plot_init import build_flo_uvw_pred_gt_panel, _omega_star_from_uv
from study.SRGAN.model.ESRuRAFT_PIV.visual_plot_save import save_vorticity_quiver_compare, _save_triplet, _save_pair, \
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
    use_adversarial = epoch >= global_data.esrgan.PRE_TRIAN_G_EPOCH - 1

    model.eval()
    generator = model.piv_esrgan_generator
    os.makedirs(result_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            # RAFT 联合评估固定同时读取图像对与流场真值，不再按 data_type 做 image_pair/flo 分叉。
            lr_prev = batch["image_pair"]["previous"]["lr_data"].to(device, non_blocking=True)
            hr_prev = batch["image_pair"]["previous"]["gr_data"].to(device, non_blocking=True)
            lr_next = batch["image_pair"]["next"]["lr_data"].to(device, non_blocking=True)
            hr_next = batch["image_pair"]["next"]["gr_data"].to(device, non_blocking=True)
            hr_images = batch["flo"]["gr_data"].to(device, non_blocking=True)
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
                is_adversarial=use_adversarial,
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


def _mean_sum_per_100_pixels(values_1d: np.ndarray, group_size: int = 100) -> float:
    """
    将一维误差序列按每 100 个像素分组求和，再对所有满 100 像素分组和取平均。
    最后一组不足 100 个像素时直接丢弃，不参与统计。
    """
    values = values_1d.reshape(-1).astype(np.float32)
    if values.size < group_size:
        return float("nan")
    usable_count = (values.size // group_size) * group_size
    if usable_count <= 0:
        return float("nan")
    values = values[:usable_count].reshape(-1, group_size)
    group_sums = np.sum(values, axis=1, dtype=np.float32)
    return float(np.mean(group_sums, dtype=np.float32)) if group_sums.size > 0 else float("nan")


def _compute_norm_aee_per100_from_flow_tensors(pred_bchw: torch.Tensor, gt_bchw: torch.Tensor) -> float:
    """
    用 batch 流场直接计算“每 100 个像素 EPE 累加值的平均”。
    """
    if pred_bchw.shape[1] < 2 or gt_bchw.shape[1] < 2:
        return float("nan")
    epe_map = torch.sqrt(torch.sum((pred_bchw[:, :2] - gt_bchw[:, :2]) ** 2, dim=1))
    epe_flat = epe_map.detach().cpu().numpy().reshape(-1)
    return _mean_sum_per_100_pixels(epe_flat, group_size=100)


def _compute_norm_aee_per100_from_chw(pred_chw: np.ndarray, gt_chw: np.ndarray) -> float:
    """
    用单样本 CHW 流场计算“每 100 个像素 EPE 累加值的平均”。
    """
    if pred_chw.shape[0] < 2 or gt_chw.shape[0] < 2:
        return float("nan")
    du = pred_chw[0] - gt_chw[0]
    dv = pred_chw[1] - gt_chw[1]
    epe = np.sqrt(du * du + dv * dv)
    return _mean_sum_per_100_pixels(epe, group_size=100)


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


def _batch_to_np_chw(tensor_bchw: torch.Tensor) -> np.ndarray:
    """
    将整个 batch 一次性转成 CPU numpy，形状保持为 [B,C,H,W]。

    evaluate_all 会为每个样本同时保存 png/npy 并计算多个 numpy 指标。
    旧写法在样本循环里多次调用 tensor.cpu().numpy()，每次都会触发一次
    GPU/CPU 同步；batch 级转换会多占一些内存，但可以显著减少同步次数。
    """
    return tensor_bchw.detach().float().cpu().numpy()


def _energy_spectrum_mse_from_curves(pred_curve: np.ndarray, gt_curve: np.ndarray) -> float:
    """
    复用已经算出的能量谱曲线计算 log1p 谱差 MSE。

    原逻辑先调用 _energy_spectrum_mse，再调用 _energy_spectrum_curves 保存曲线，
    等于同一张图重复做两次 FFT。这里保持公式完全一致，只把第二次 FFT 省掉。
    """
    return float(np.mean((np.log1p(pred_curve) - np.log1p(gt_curve)) ** 2))


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


def _compute_flow_ref_max_rad_batch(flow_gt_bchw: torch.Tensor) -> list[float]:
    """
    batch 级计算每个样本的光流彩图参考半径。

    每个样本仍然使用自己 GT 的 0.99 分位数，输出颜色尺度与旧逻辑一致；
    区别只是把 B 次 quantile/sync 合并为一次 batch 运算，更适合内存充足的全量评估。
    """
    if flow_gt_bchw.dim() != 4 or flow_gt_bchw.shape[1] < 2:
        raise ValueError(f"Expected GT flow with shape [B, >=2, H, W], got {tuple(flow_gt_bchw.shape)}")
    gt_u = flow_gt_bchw[:, 0]
    gt_v = flow_gt_bchw[:, 1]
    gt_mag_uv = torch.sqrt(gt_u * gt_u + gt_v * gt_v)
    ref_max = torch.quantile(gt_mag_uv.flatten(1), 0.99, dim=1).clamp_min(1e-6)
    return [float(v) for v in ref_max.detach().cpu().tolist()]


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
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype, non_blocking=True)


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


def _add_row_and_column_headers_to_panel(
    panel: torch.Tensor,
    row_labels: list[str],
    row_heights: list[int],
    row_separator_heights: list[int],
    column_headers: list[str],
    column_widths: list[int],
    column_separator_widths: list[int],
    header_height: int = 22,
    left_label_width: int = 34,
) -> torch.Tensor:
    """
    同时给拼图补顶部列标题和左侧行标题。
    这里主要用于 u/v/w 对比面板，保持最右侧色条追加前的主体布局清晰可读。
    """
    if len(row_labels) != len(row_heights):
        raise ValueError("row_labels and row_heights must have the same length")
    if len(row_separator_heights) != max(0, len(row_labels) - 1):
        raise ValueError("row_separator_heights length must be len(row_labels)-1")
    if len(column_headers) != len(column_widths):
        raise ValueError("column_headers and column_widths must have the same length")
    if len(column_separator_widths) != max(0, len(column_headers) - 1):
        raise ValueError("column_separator_widths length must be len(column_headers)-1")

    base = _tensor_to_rgb_pil(panel)
    canvas = Image.new(
        "RGB",
        (base.width + left_label_width, base.height + header_height),
        color=(255, 255, 255),
    )
    canvas.paste(base, (left_label_width, header_height))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    x = left_label_width
    for idx_col, (title, width) in enumerate(zip(column_headers, column_widths)):
        bbox = draw.textbbox((0, 0), title, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        tx = int(x + max((width - text_w) * 0.5, 0))
        ty = int(max((header_height - text_h) * 0.5, 0))
        draw.text((tx, ty), title, fill=(0, 0, 0), font=font)
        x += width
        if idx_col < len(column_separator_widths):
            x += column_separator_widths[idx_col]

    y = header_height
    for idx_row, (label, height) in enumerate(zip(row_labels, row_heights)):
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        tx = int(max((left_label_width - text_w) * 0.5, 0))
        ty = int(y + max((height - text_h) * 0.5, 0))
        draw.text((tx, ty), label, fill=(0, 0, 0), font=font)
        y += height
        if idx_row < len(row_separator_heights):
            y += row_separator_heights[idx_row]

    return _pil_rgb_to_tensor01(canvas, panel.device, panel.dtype)


def _make_vertical_colorbar_image(
    height: int,
    vmin: float,
    vmax: float,
    cmap_name: str = "jet",
    width: int = 26,
    tick_count: int = 5,
) -> Image.Image:
    """
    生成单个竖直颜色映射条。
    色条放在最右侧，顶部对应 vmax，底部对应 vmin。
    """
    canvas_w = width + 42
    canvas = Image.new("RGB", (canvas_w, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    # 色条上下各留一点边距，避免顶端/底端刻度文字被裁切，
    # 同时让色条主体比对应图像略矮一点，更利于完整显示文字。
    inner_pad = min(max(height // 18, 6), 12)
    bar_h = max(height - 2 * inner_pad, 1)
    grad = np.linspace(1.0, 0.0, bar_h, dtype=np.float32).reshape(bar_h, 1)
    rgba = plt.get_cmap(cmap_name)(grad)
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    rgb = np.repeat(rgb, width, axis=1)
    bar = Image.fromarray(rgb, mode="RGB")
    canvas.paste(bar, (0, inner_pad))

    for i in range(tick_count):
        y = inner_pad if tick_count == 1 else int(round(inner_pad + i * (bar_h - 1) / max(tick_count - 1, 1)))
        value = vmax - (vmax - vmin) * (i / max(tick_count - 1, 1))
        draw.line([(width, y), (width + 5, y)], fill=(0, 0, 0), width=1)
        text = f"{value:.2f}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_h = bbox[3] - bbox[1]
        text_y = min(max(y - text_h // 2, 0), max(height - text_h, 0))
        draw.text((width + 8, text_y), text, fill=(0, 0, 0), font=font)
    return canvas


def _append_colorbar_sections_to_panel(
    panel: torch.Tensor,
    sections: list[dict],
    gap: int = 8,
    label_width: int = 68,
    top_margin: int = 0,
    section_heights: list[int] | None = None,
    section_gaps: list[int] | None = None,
) -> torch.Tensor:
    """
    在现有拼图最右侧追加一个或多个竖直颜色条。

    sections 中每个元素包含：
    - vmin
    - vmax
    - cmap
    - label

    top_margin / section_heights / section_gaps 用来让色条只覆盖对应图像主体区域，
    不把顶部标题带和行间空白也算进色条高度。
    """
    base = _tensor_to_rgb_pil(panel)
    total_h = base.height
    canvas = Image.new("RGB", (base.width + gap + 26 + label_width, total_h), color=(255, 255, 255))
    canvas.paste(base, (0, 0))

    if not sections:
        return _pil_rgb_to_tensor01(canvas, panel.device, panel.dtype)

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    x0 = base.width + gap
    y_cursor = int(max(top_margin, 0))

    if section_heights is None:
        available_h = max(total_h - y_cursor, 1)
        base_h = available_h // len(sections)
        section_heights = [base_h] * len(sections)
        section_heights[-1] += available_h - base_h * len(sections)
    else:
        section_heights = [max(int(h), 1) for h in section_heights]

    if section_gaps is None:
        section_gaps = [0] * max(len(sections) - 1, 0)
    else:
        section_gaps = [max(int(g), 0) for g in section_gaps]
        if len(section_gaps) < max(len(sections) - 1, 0):
            section_gaps = section_gaps + [0] * (len(sections) - 1 - len(section_gaps))

    for idx, section in enumerate(sections):
        section_h = section_heights[idx] if idx < len(section_heights) else 1
        y1 = min(y_cursor, total_h)
        y2 = min(y1 + section_h, total_h)
        bar_img = _make_vertical_colorbar_image(
            height=max(y2 - y1, 1),
            vmin=float(section["vmin"]),
            vmax=float(section["vmax"]),
            cmap_name=str(section.get("cmap", "jet")),
        )
        canvas.paste(bar_img, (x0, y1))
        label = str(section.get("label", "")).strip()
        if label:
            bbox = draw.textbbox((0, 0), label, font=font)
            text_h = bbox[3] - bbox[1]
            draw.text((x0, y1 + max(((max(y2 - y1, 1)) - text_h) // 2, 0)), label, fill=(0, 0, 0), font=font)
        y_cursor = y2
        if idx < len(sections) - 1:
            y_cursor += section_gaps[idx]

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


def _compute_flow_error_maps(
    pred_bchw: torch.Tensor,
    gt_bchw: torch.Tensor,
    pred_np_chw: np.ndarray | None = None,
    gt_np_chw: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    统一计算水平位移误差、垂直位移误差、幅值(w)误差和端点误差图。
    """
    if pred_np_chw is None or gt_np_chw is None:
        # 兼容旧调用：没有传入 batch 级 numpy 缓存时，仍在函数内部完成转换。
        # evaluate_all 的新路径会传入缓存，避免同一个样本在指标、npy、误差图里反复 cpu().numpy()。
        pred_flow = pred_bchw if pred_bchw.shape[1] >= 3 else _flow_uv_to_uvw(pred_bchw)
        gt_flow = gt_bchw if gt_bchw.shape[1] >= 3 else _flow_uv_to_uvw(gt_bchw)
        pred_np = _to_np_chw(pred_flow[0])
        gt_np = _to_np_chw(gt_flow[0])
    else:
        pred_np = pred_np_chw
        gt_np = gt_np_chw
    delta_u = pred_np[0] - gt_np[0]
    delta_v = pred_np[1] - gt_np[1]
    delta_w = pred_np[2] - gt_np[2]
    epe = np.sqrt(delta_u * delta_u + delta_v * delta_v)
    return delta_u, delta_v, delta_w, epe


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
            0.03, 0.97, stat_text,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=18, color="black",
            bbox=dict(boxstyle="square,pad=0.18", facecolor="white", edgecolor="none", alpha=0.58),
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


def _save_flow_error_visuals(
    pred_bchw: torch.Tensor,
    gt_bchw: torch.Tensor,
    out_dir: Path,
    pred_np_chw: np.ndarray | None = None,
    gt_np_chw: np.ndarray | None = None,
    ref_max_rad: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    保存预测光流彩色图、u/v/w 三张 AEE 风格误差图、Δu/Δv 误差分布图、涡度误差图。

    返回：
        delta_u_2d: 水平位移误差图
        delta_v_2d: 垂直位移误差图
        delta_w_2d: 深度/幅值位移误差图
        epe_2d:     端点误差图
    """
    pred_flow = pred_bchw if pred_bchw.shape[1] >= 3 else _flow_uv_to_uvw(pred_bchw)
    gt_flow = gt_bchw if gt_bchw.shape[1] >= 3 else _flow_uv_to_uvw(gt_bchw)
    if pred_np_chw is None or gt_np_chw is None:
        # 兼容旧调用，同时保证涡度 npy 与误差图使用同一份 CHW 数据。
        pred_np = _to_np_chw(pred_flow[0])
        gt_np = _to_np_chw(gt_flow[0])
    else:
        # evaluate_all 已经在 batch 维度做过 CPU 缓存，这里直接复用。
        pred_np = pred_np_chw
        gt_np = gt_np_chw

    du, dv, dw, epe = _compute_flow_error_maps(pred_bchw, gt_bchw, pred_np, gt_np)
    aee = float(np.mean(epe))

    if ref_max_rad is None:
        ref_max_rad = _compute_flow_ref_max_rad(gt_flow)
    pred_color = _flow_to_color_preview(pred_flow[:, :2], ref_max_rad=ref_max_rad)
    gt_color = _flow_to_color_preview(gt_flow[:, :2], ref_max_rad=ref_max_rad)
    pred_color = _append_colorbar_sections_to_panel(
        pred_color,
        [{"vmin": 0.0, "vmax": ref_max_rad, "cmap": "jet", "label": "|V|"}],
    )
    gt_color = _append_colorbar_sections_to_panel(
        gt_color,
        [{"vmin": 0.0, "vmax": ref_max_rad, "cmap": "jet", "label": "|V|"}],
    )
    # 统一命名为 pred_flow / gt_flow，避免和别处的 pred_flow.png 重名但语义重复。
    save_image(pred_color, str(out_dir / "pred_flow.png"), normalize=False)
    save_image(gt_color, str(out_dir / "gt_flow.png"), normalize=False)

    # 这里按照“有符号分量误差图 + 左上角显示整体 AEE 数值”的方式保存 u/v/w 三张图。
    aee_text = rf"$AEE={aee:.4f}$"
    _save_signed_error_map(
        du,
        out_dir / "aee_u_error.png",
        title_text="AEE-u Error",
        colorbar_label="u error [px]",
        stat_text=aee_text,
    )
    _save_signed_error_map(
        dv,
        out_dir / "aee_v_error.png",
        title_text="AEE-v Error",
        colorbar_label="v error [px]",
        stat_text=aee_text,
    )
    _save_signed_error_map(
        dw,
        out_dir / "aee_w_error.png",
        title_text="AEE-w Error",
        colorbar_label="w error [px]",
        stat_text=aee_text,
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
    _save_error_histogram(
        dw,
        out_dir / "delta_w_error.png",
        title_text=r"$\Delta w$ Error Distribution",
        xlabel=r"$\Delta w$ displacement [px]",
        color="#8E6BBE",
    )

    pred_omega = _omega_star_from_uv(pred_np[0], pred_np[1])
    gt_omega = _omega_star_from_uv(gt_np[0], gt_np[1])
    np.save(out_dir / "pred_vorticity.npy", pred_omega.astype(np.float32))
    np.save(out_dir / "gt_vorticity.npy", gt_omega.astype(np.float32))

    return du.astype(np.float32), dv.astype(np.float32), dw.astype(np.float32), epe.astype(np.float32)


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


def _delta_w_histogram_matrix(delta_w_2d: np.ndarray, bins: int = 201) -> np.ndarray:
    """
    统计 Δw 的像素计数分布。
    分布区间同样以 0 为中心，便于和 Δu / Δv 采用一致的误差分析口径。
    """
    max_abs = float(np.max(np.abs(delta_w_2d))) + 1e-12
    edges = np.linspace(-max_abs, max_abs, bins + 1, dtype=np.float32)
    return _histogram_matrix(delta_w_2d, edges)


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
def validate_raft(model, dataloader, device, epoch):
    """
    在 RAFT 联合验证集上计算平均像素损失、平均 PSNR、平均能量谱 MSE、平均 AEE，
    以及“每 100 个像素 EPE 累加值的平均”。
    这里不再区分 image_pair / flo，而是固定走联合模型的一条评估路径。
    """
    model.eval()
    generator = model.piv_esrgan_generator
    use_adversarial = epoch >= global_data.esrgan.PRE_TRIAN_G_EPOCH - 1
    total_val_ssim_loss = 0.0
    total_val_mse_loss = 0.0

    total_psnr = 0.0
    total_energy_spectrum_mse = 0.0
    total_aee = 0.0
    total_norm_aee_per100 = 0.0
    loss_count = 0
    num_images = 0

    with torch.no_grad():
        for batch in dataloader:
            lr_prev = batch["image_pair"]["previous"]["lr_data"].to(device, non_blocking=True)
            hr_prev = batch["image_pair"]["previous"]["gr_data"].to(device, non_blocking=True)
            lr_next = batch["image_pair"]["next"]["lr_data"].to(device, non_blocking=True)
            hr_next = batch["image_pair"]["next"]["gr_data"].to(device, non_blocking=True)
            flow_gt = batch["flo"]["gr_data"].to(device, non_blocking=True)
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
                is_adversarial=use_adversarial,
            )
            total_aee += float(outputs["raft_metrics"]["epe"])
            total_norm_aee_per100 += _compute_norm_aee_per100_from_flow_tensors(
                outputs["flow_predictions"][-1],
                flow_gt_uv,
            )
            #一个batch 就行了 因为训练中的验证只需要1次batch验证
            break
    avg_val_ssim_loss = total_val_ssim_loss / max(loss_count, 1)
    avg_val_mse_loss = total_val_mse_loss / max(loss_count, 1)
    avg_psnr = total_psnr / max(num_images, 1)
    avg_energy_spectrum_mse = total_energy_spectrum_mse / max(num_images, 1)
    # outputs["raft_metrics"]["epe"] 本身已经是当前验证 batch 的平均 EPE/AEE，
    # 这里不能再按 image_pair 分支累计出来的 loss_count 再除一次，否则会把 AEE 额外缩小。
    avg_aee = total_aee
    avg_norm_aee_per100 = total_norm_aee_per100
    return avg_val_ssim_loss, avg_val_mse_loss, avg_psnr, avg_energy_spectrum_mse, avg_aee, avg_norm_aee_per100

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

    avg_val_ssim_loss, avg_val_mse_loss, avg_psnr, avg_val_energy_spectrum_mse, avg_val_aee, avg_val_norm_aee_per100 = validate_raft(
        model, validate_loader, device, epoch
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
        "VAL_NORM_AEE_PER100PIXEL": avg_val_norm_aee_per100,
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
        f"VAL_energy_spectrum_mse: {avg_val_energy_spectrum_mse} | VAL_AEE: {avg_val_aee} | "
        f"VAL_NORM_AEE_PER100PIXEL: {avg_val_norm_aee_per100} | Avg PSNR: {avg_psnr:.2f}"
    )
    loss_str = "".join([loss_label[index] + ':' + str(metric[index] / train_loader_lens) + "," for index in
                        range(len(loss_label))])
    logger.info(loss_str)

    # 每轮训练结束后进行验证，并保存最后一批图像
    validate_and_save(f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.PREDICT_DIR}", model,
                      validate_loader, device, epoch, data_type=data_type)

    # 保存每一epoch的损失
    all_loss_and_val_Datas = [metric[index] / (train_loader_lens)  for index in range(len(loss_label))] + [
        avg_val_mse_loss,
        avg_val_ssim_loss,
        avg_psnr,
        avg_val_energy_spectrum_mse,
        avg_val_aee,
        avg_val_norm_aee_per100,
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
    csv_row["VAL_NORM_AEE_PER100PIXEL"] = avg_val_norm_aee_per100
    csv_row["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csvOperator.create(csv_row)
    animator.save_png(
        f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.LOSS_DIR}/train_loss_epoch_{epoch + 1}_{global_data.esrgan.name}.png",
        fixed_groups=[
            ["g_loss", "d_loss","raft_loss"],
            ["g_perceptual_loss", "g_content_loss", "g_adversarial_loss"],
            ["g_loss_pixel", "g_loss_pixel_l1", "g_loss_pixel_mse", "g_loss_ssim", "g_loss_fft"],
            ["g_flow_warp_consistency_loss", "g_flow_warp_consistency_weighted_loss"],
            ["d_loss", "d_real_loss", "d_fake_loss"],
            [validate_label[0], validate_label[1], ],
            [validate_label[2]],
            [validate_label[3]],
            [validate_label[4]],
            [validate_label[5]]
        ])
    return avg_val_mse_loss,avg_val_ssim_loss,avg_psnr,avg_val_energy_spectrum_mse,avg_val_aee,avg_val_norm_aee_per100

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
    # evaluate_all 是训练完成后的全量评估路径，没有 epoch 参数；
    # 这里默认进入“允许对抗项”的正式评估口径。
    use_adversarial = True

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
        "mse", "psnr", "energy_spectrum_mse", "VAL_AEE", "VAL_NORM_AEE_PER100PIXEL", "r2", "ssim", "tke_acc", "nrmse"
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
    image_pair_curves_by_class: dict[str, dict[str, list[np.ndarray]]] = {}
    flow_curves_by_class: dict[str, dict[str, list[np.ndarray]]] = {}
    delta_u_hist_by_class: dict[str, list[np.ndarray]] = {}
    delta_v_hist_by_class: dict[str, list[np.ndarray]] = {}
    delta_w_hist_by_class: dict[str, list[np.ndarray]] = {}
    epe_hist_by_class: dict[str, list[np.ndarray]] = {}
    all_image_pair_pred_curves = []
    all_image_pair_gt_curves = []
    all_flow_pred_curves = []
    all_flow_gt_curves = []

    def register_curve(bucket_name: str, pred_curve: np.ndarray, gt_curve: np.ndarray, curve_group: str) -> None:
        # 频谱曲线按 image_pair / flow 两条链分别登记，
        # 这样全局和分类别输出时可以同时保留两套均值能量谱图与 .npy。
        if curve_group == "image_pair":
            all_image_pair_pred_curves.append(pred_curve)
            all_image_pair_gt_curves.append(gt_curve)
            curves_by_group = image_pair_curves_by_class
        else:
            all_flow_pred_curves.append(pred_curve)
            all_flow_gt_curves.append(gt_curve)
            curves_by_group = flow_curves_by_class
        if bucket_name not in curves_by_group:
            curves_by_group[bucket_name] = {"pred": [], "gt": []}
        curves_by_group[bucket_name]["pred"].append(pred_curve)
        curves_by_group[bucket_name]["gt"].append(gt_curve)

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

    def save_mean_spectrum(
        pred_curves: list[np.ndarray],
        gt_curves: list[np.ndarray],
        out_dir: Path,
        title: str,
        file_prefix: str,
        also_save_legacy_names: bool = False,
    ) -> None:
        if not pred_curves or not gt_curves:
            return
        # 不同样本曲线长度可能略有不同，所以先截到共同最短长度再平均。
        min_len = min(min(len(x) for x in pred_curves), min(len(x) for x in gt_curves))
        pred_mean = np.mean(np.stack([x[:min_len] for x in pred_curves], axis=0), axis=0)
        gt_mean = np.mean(np.stack([x[:min_len] for x in gt_curves], axis=0), axis=0)
        np.save(out_dir / f"{file_prefix}_energy_spectrum_pred_mean.npy", pred_mean.astype(np.float32))
        np.save(out_dir / f"{file_prefix}_energy_spectrum_gt_mean.npy", gt_mean.astype(np.float32))
        _save_energy_spectrum_plot(
            pred_mean,
            gt_mean,
            out_dir / f"{file_prefix}_energy_spectrum_mean_compare.png",
            title=title,
        )
        if also_save_legacy_names:
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
            "VAL_NORM_AEE_PER100PIXEL": _mean_of("VAL_NORM_AEE_PER100PIXEL"),
            "r2": _mean_of("r2"),
            "ssim": _mean_of("ssim"),
            "tke_acc": _mean_of("tke_acc"),
            "nrmse": _mean_of("nrmse"),
        }

    def write_rows_with_mean(csv_path: Path, target_rows: list[dict], bucket_name: str) -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        mean_row = build_mean_row(target_rows, bucket_name)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(target_rows + [mean_row])

    def write_all_class_mean_csv(csv_path: Path, mean_rows: list[dict]) -> None:
        """
        保存跨类别均值表：每一行对应一个类别的预测结果平均指标。
        """
        if not mean_rows:
            return
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(mean_rows)

    with torch.inference_mode():
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
            lr_prev = batch["image_pair"]["previous"]["lr_data"].to(device, non_blocking=True)
            hr_prev = batch["image_pair"]["previous"]["gr_data"].to(device, non_blocking=True)
            lr_next = batch["image_pair"]["next"]["lr_data"].to(device, non_blocking=True)
            hr_next = batch["image_pair"]["next"]["gr_data"].to(device, non_blocking=True)
            # lr = batch["flo"]["lr_data"].to(device, non_blocking=True)
            hr = batch["flo"]["gr_data"].to(device, non_blocking=True)
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
                is_adversarial=use_adversarial,
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

            # evaluate_all 的耗时大头不只在网络 forward，还在每个样本反复做保存和 numpy 指标。
            # 这里把 image_pair 分支的保存张量一次性裁剪/转 CPU，并把指标数组一次性缓存。
            # 这样会占用更多内存，但用户机器内存充足时，可以减少大量逐样本 GPU->CPU 同步。
            save_as_gray = global_data.esrgan.SAVE_AS_GRAY
            image_pair_batches = {
                "previous": {
                    "lr": _prepare_image_pair_tensor_for_save(lr_prev_up, save_as_gray).detach().cpu(),
                    "fake": _prepare_image_pair_tensor_for_save(fake_prev, save_as_gray).detach().cpu(),
                    "hr": _prepare_image_pair_tensor_for_save(hr_prev, save_as_gray).detach().cpu(),
                },
                "next": {
                    "lr": _prepare_image_pair_tensor_for_save(lr_next_up, save_as_gray).detach().cpu(),
                    "fake": _prepare_image_pair_tensor_for_save(fake_next, save_as_gray).detach().cpu(),
                    "hr": _prepare_image_pair_tensor_for_save(hr_next, save_as_gray).detach().cpu(),
                },
            }
            for pair_cache in image_pair_batches.values():
                pair_cache["fake_np"] = _batch_to_np_chw(pair_cache["fake"])
                pair_cache["hr_np"] = _batch_to_np_chw(pair_cache["hr"])

            # flow 分支同样把每个 batch 的预测/真值缓存到 CPU：
            # 1. fake_cpu / hr_cpu 继续用于原有 png 可视化函数；
            # 2. fake_uvw_np_batch / hr_np_batch 直接用于 npy 保存和指标计算；
            # 3. flow_ref_max_rads 提前按 batch 计算，避免 flow_triplet 和 pred_flow/gt_flow 重复 quantile。
            flow_ref_max_rads = _compute_flow_ref_max_rad_batch(hr)
            fake_cpu = fake.detach().cpu()
            fake_uvw_cpu = fake_uvw.detach().cpu()
            hr_cpu = hr.detach().cpu()
            fake_uvw_np_batch = _batch_to_np_chw(fake_uvw_cpu)
            hr_np_batch = _batch_to_np_chw(hr_cpu)

            B = hr.shape[0]
            for i in range(B):
                sample_bucket = bucket_class_name(batch_class_names[i] if i < len(batch_class_names) else None)
                class_root = output_root / sample_bucket
                class_root.mkdir(parents=True, exist_ok=True)

                sid = f"batch_{batch_idx}_idx_{i}_fid_{batch_idx}"
                one_dir = class_root / sid
                one_dir.mkdir(parents=True, exist_ok=True)

                # 保存 previous / next 两个超分结果。
                for pair_type, pair_cache in image_pair_batches.items():
                    pair_dir = one_dir / pair_type
                    pair_dir.mkdir(parents=True, exist_ok=True)

                    lr_save = pair_cache["lr"][i:i + 1]
                    fk_save = pair_cache["fake"][i:i + 1]
                    hr_save = pair_cache["hr"][i:i + 1]

                    save_image(lr_save, str(pair_dir / "lr.png"), normalize=False)
                    save_image(fk_save, str(pair_dir / "fake.png"), normalize=False)
                    save_image(hr_save, str(pair_dir / "hr.png"), normalize=False)

                    _save_triplet(lr_save, fk_save, hr_save, pair_dir / "image_triplet.png")

                    # 指标直接使用 batch 级 numpy 缓存，避免每张 previous/next 图再次触发 CPU 拷贝。
                    p_img = pair_cache["fake_np"][i]
                    g_img = pair_cache["hr_np"][i]
                    mse_img = _mse(p_img, g_img)
                    psnr_img = _psnr_from_mse(mse_img)
                    r2_img = _r2_score(p_img, g_img)
                    ssim_img = _ssim_score(p_img, g_img)
                    tke_img = _tke_reconstruction_accuracy(p_img, g_img)
                    nrmse_img = _nrmse(p_img, g_img)
                    pred_curve, gt_curve = _energy_spectrum_curves(p_img, g_img)
                    es_mse_img = _energy_spectrum_mse_from_curves(pred_curve, gt_curve)
                    np.save(pair_dir / "energy_spectrum_pred.npy", pred_curve.astype(np.float32))
                    np.save(pair_dir / "energy_spectrum_gt.npy", gt_curve.astype(np.float32))
                    _save_energy_spectrum_plot(pred_curve, gt_curve, pair_dir / "energy_spectrum_compare.png", title=f"{sid}-{pair_type} Energy Spectrum")
                    register_curve(sample_bucket, pred_curve, gt_curve, curve_group="image_pair")

                    append_row({
                        "class_name": sample_bucket,
                        "data_type": data_type,
                        "scale": int(SCALE * SCALE),
                        "sample_id": sid,
                        "pair_type": pair_type,
                        "mse": mse_img,
                        "psnr": psnr_img,
                        "energy_spectrum_mse": es_mse_img,
                        "VAL_AEE": float("nan"),
                        "VAL_NORM_AEE_PER100PIXEL": float("nan"),
                        "r2": r2_img,
                        "ssim": ssim_img,
                        "tke_acc": tke_img,
                        "nrmse": nrmse_img,
                    })

                # 保存流场预测及误差分析结果。
                # lr1 = lr[i:i + 1]
                # lr_up1 = lr_up[i:i + 1]
                fk1 = fake_cpu[i:i + 1]
                fk1_uvw = fake_uvw_cpu[i:i + 1]
                hr1 = hr_cpu[i:i + 1]
                p = fake_uvw_np_batch[i]
                g = hr_np_batch[i]

                # np.save(one_dir / "lr_flo.npy", _to_np_chw(lr1[0]).transpose(1, 2, 0))
                # 保存三通道预测流场 [u, v, magnitude]，便于和三通道真值直接对比。
                np.save(one_dir / "fake_flo.npy", p.transpose(1, 2, 0))
                np.save(one_dir / "hr_flo.npy", g.transpose(1, 2, 0))

                ref_max_rad = flow_ref_max_rads[i]

                # lr_color, _ = flow_to_color_tensor(lr_up1[:, :2], ref_max_rad=ref_max_rad)
                fk_color = _flow_to_color_preview(fk1[:, :2], ref_max_rad=ref_max_rad)
                hr_color = _flow_to_color_preview(hr1[:, :2], ref_max_rad=ref_max_rad)
                flow_pair_panel = build_pair_row(fk_color, hr_color, sep_width=6)
                flow_pair_panel = _add_headers_to_panel(
                    flow_pair_panel,
                    headers=["Pred", "HR"],
                    column_widths=[fk_color.shape[-1], hr_color.shape[-1]],
                    separator_widths=[6],
                )
                flow_pair_panel = _append_colorbar_sections_to_panel(
                    flow_pair_panel,
                    [{"vmin": 0.0, "vmax": ref_max_rad, "cmap": "jet", "label": "|V|"}],
                    top_margin=22,
                    section_heights=[fk_color.shape[-2]],
                )
                # 流场样本级拼图不再包含 flo 的 LR 图像，只保留 Pred/HR 两列。
                save_image(flow_pair_panel.clamp(0, 1), str(one_dir / "flow_triplet.png"), normalize=False)
                # 单独的预测流场彩图已经在 _save_flow_error_visuals 中按统一命名保存为 pred_flow.png，
                # 这里不再重复落盘，避免同目录下出现语义重复的文件。

                # U/V/S 面板改为 Pred/HR 双列对比，并补左侧行标题、顶部列标题，最右侧追加色条。
                uvs_panel = build_flo_uvw_pred_gt_panel(fk1_uvw, hr1)
                uvs_panel = _add_row_and_column_headers_to_panel(
                    uvs_panel,
                    row_labels=["U", "V", "W"],
                    row_heights=[fk1_uvw.shape[-2], fk1_uvw.shape[-2], fk1_uvw.shape[-2]],
                    row_separator_heights=[8, 8],
                    column_headers=["Pred", "HR"],
                    column_widths=[fk1_uvw.shape[-1], hr1.shape[-1]],
                    column_separator_widths=[6],
                )
                hr_min = g[:3].min(axis=(1, 2))
                hr_max = g[:3].max(axis=(1, 2))
                uvs_panel = _append_colorbar_sections_to_panel(
                    uvs_panel,
                    [
                        {"vmin": float(hr_min[0]), "vmax": float(hr_max[0]), "cmap": "jet", "label": "U"},
                        {"vmin": float(hr_min[1]), "vmax": float(hr_max[1]), "cmap": "jet", "label": "V"},
                        {"vmin": float(hr_min[2]), "vmax": float(hr_max[2]), "cmap": "jet", "label": "W"},
                    ],
                    top_margin=22,
                    section_heights=[fk1_uvw.shape[-2], fk1_uvw.shape[-2], fk1_uvw.shape[-2]],
                    section_gaps=[8, 8],
                )
                save_image(uvs_panel.clamp(0, 1), str(one_dir / "uvs_compare.png"), normalize=False)
                # 涡度图内部实际只依赖前两个通道 uv，这里传三通道预测和真值保持接口一致。
                save_vorticity_quiver_compare(fk1_uvw, hr1, str(one_dir / "vorticity_quiver.png"), stride=stride)
                # 额外保存 AEE 误差图、涡度误差图，并返回像素级误差用于统计分布。
                delta_u_map, delta_v_map, delta_w_map, epe_map = _save_flow_error_visuals(
                    fk1,
                    hr1,
                    one_dir,
                    pred_np_chw=p,
                    gt_np_chw=g,
                    ref_max_rad=ref_max_rad,
                )

                # 逐样本保存 Δu / Δv / EPE 分布。
                # Δu、Δv 直方图都以 0 为中心，便于横向比较水平/垂直位移误差。
                sample_delta_u_hist = _delta_u_histogram_matrix(delta_u_map)
                sample_delta_v_hist = _delta_v_histogram_matrix(delta_v_map)
                sample_delta_w_hist = _delta_w_histogram_matrix(delta_w_map)
                sample_epe_hist = _epe_histogram_matrix(epe_map)
                np.save(one_dir / "delta_u_hist.npy", sample_delta_u_hist)
                np.save(one_dir / "delta_v_hist.npy", sample_delta_v_hist)
                np.save(one_dir / "delta_w_hist.npy", sample_delta_w_hist)
                np.save(one_dir / "epe_hist.npy", sample_epe_hist)

                # 同时把每个 sample 的误差统计登记到所属类别，便于类别级汇总。
                # 这里要先确保类别桶已经创建，再去 append；
                # 否则像 DNS_turbulence 这类首次出现的类别会在 append 时直接 KeyError。
                delta_u_hist_by_class.setdefault(sample_bucket, [])
                delta_v_hist_by_class.setdefault(sample_bucket, [])
                delta_w_hist_by_class.setdefault(sample_bucket, [])
                epe_hist_by_class.setdefault(sample_bucket, [])
                delta_u_hist_by_class[sample_bucket].append(delta_u_map.reshape(-1))
                delta_v_hist_by_class[sample_bucket].append(delta_v_map.reshape(-1))
                delta_w_hist_by_class[sample_bucket].append(delta_w_map.reshape(-1))
                epe_hist_by_class[sample_bucket].append(epe_map.reshape(-1))

                # 数值指标对比统一使用三通道 [u, v, magnitude]。
                # p/g 已经来自 batch 级 numpy 缓存；AEE 和每 100 像素 EPE 直接复用上面生成的 epe_map。
                mse = _mse(p, g)
                psnr = _psnr_from_mse(mse)
                aee = float(np.mean(epe_map))
                norm_aee_per100 = _mean_sum_per_100_pixels(epe_map, group_size=100)
                r2 = _r2_score(p, g)
                ssim = _ssim_score(p, g)
                tke = _tke_reconstruction_accuracy(p, g)
                nrmse = _nrmse(p, g)

                pred_curve, gt_curve = _energy_spectrum_curves(p, g)
                es_mse = _energy_spectrum_mse_from_curves(pred_curve, gt_curve)
                np.save(one_dir / "energy_spectrum_pred.npy", pred_curve.astype(np.float32))
                np.save(one_dir / "energy_spectrum_gt.npy", gt_curve.astype(np.float32))
                np.save(one_dir / "flow_energy_spectrum_pred.npy", pred_curve.astype(np.float32))
                np.save(one_dir / "flow_energy_spectrum_gt.npy", gt_curve.astype(np.float32))
                _save_energy_spectrum_plot(pred_curve, gt_curve, one_dir / "energy_spectrum_compare.png", title=f"{sid} Energy Spectrum")
                _save_energy_spectrum_plot(pred_curve, gt_curve, one_dir / "flow_energy_spectrum_compare.png", title=f"{sid} Flow Energy Spectrum")
                register_curve(sample_bucket, pred_curve, gt_curve, curve_group="flow")

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
                    "VAL_NORM_AEE_PER100PIXEL": norm_aee_per100,
                    "r2": r2,
                    "ssim": ssim,
                    "tke_acc": tke,
                    "nrmse": nrmse,
                })

    image_pair_rows = [row for row in rows if row.get("pair_type") in {"previous", "next"}]
    raft_rows = [row for row in rows if row.get("pair_type") == "RAFT"]

    # 根目录下分别输出 image_pair / flow 两套均值频谱图，避免图像对和光流频谱混在一起。
    save_mean_spectrum(
        all_image_pair_pred_curves,
        all_image_pair_gt_curves,
        output_root,
        title=f"{class_name}-{data_type}-x{int(SCALE*SCALE)} Image Pair Mean Energy Spectrum",
        file_prefix="image_pair",
    )
    save_mean_spectrum(
        all_flow_pred_curves,
        all_flow_gt_curves,
        output_root,
        title=f"{class_name}-{data_type}-x{int(SCALE*SCALE)} Flow Mean Energy Spectrum",
        file_prefix="flow",
    )

    # 根目录额外保存整套验证集的 Δu / EPE 统计结果，便于做全局误差分布分析。
    all_delta_u_values = [v for values in delta_u_hist_by_class.values() for v in values]
    all_delta_v_values = [v for values in delta_v_hist_by_class.values() for v in values]
    all_delta_w_values = [v for values in delta_w_hist_by_class.values() for v in values]
    all_epe_values = [v for values in epe_hist_by_class.values() for v in values]
    if all_delta_u_values:
        np.save(output_root / "delta_u_hist_all.npy", _delta_u_histogram_matrix(np.concatenate(all_delta_u_values, axis=0)))
    if all_delta_v_values:
        np.save(output_root / "delta_v_hist_all.npy", _delta_v_histogram_matrix(np.concatenate(all_delta_v_values, axis=0)))
    if all_delta_w_values:
        np.save(output_root / "delta_w_hist_all.npy", _delta_w_histogram_matrix(np.concatenate(all_delta_w_values, axis=0)))
    if all_epe_values:
        np.save(output_root / "epe_hist_all.npy", _epe_histogram_matrix(np.concatenate(all_epe_values, axis=0)))

    metrics_csv_path = Path(metrics_csv_path)
    metrics_image_pair_csv_path = metrics_csv_path.with_name(f"{metrics_csv_path.stem}_image_pair{metrics_csv_path.suffix}")
    metrics_raft_csv_path = metrics_csv_path.with_name(f"{metrics_csv_path.stem}_raft{metrics_csv_path.suffix}")
    all_class_image_pair_csv_path = output_root / "ALL_CLASS_IMAGE_PAIR.CSV"
    all_class_flow_csv_path = output_root / "ALL_CLASS_flow.CSV"
    if image_pair_rows:
        write_rows_with_mean(metrics_image_pair_csv_path, image_pair_rows, class_name)
    if raft_rows:
        write_rows_with_mean(metrics_raft_csv_path, raft_rows, class_name)

    all_class_image_pair_mean_rows: list[dict] = []
    all_class_flow_mean_rows: list[dict] = []

    for bucket_name, class_rows in rows_by_class.items():
        class_root = output_root / bucket_name
        class_root.mkdir(parents=True, exist_ok=True)
        class_image_pair_rows = [row for row in class_rows if row.get("pair_type") in {"previous", "next"}]
        class_raft_rows = [row for row in class_rows if row.get("pair_type") == "RAFT"]
        # 每个类别目录下也拆成 image_pair / RAFT 两张表，避免图像超分指标和流场指标混在一起。
        if class_image_pair_rows:
            write_rows_with_mean(class_root / "metrics_image_pair.csv", class_image_pair_rows, bucket_name)
            image_pair_mean_row = build_mean_row(class_image_pair_rows, bucket_name)
            image_pair_mean_row["sample_id"] = "CLASS_MEAN"
            image_pair_mean_row["pair_type"] = "image_pair"
            all_class_image_pair_mean_rows.append(image_pair_mean_row)
        if class_raft_rows:
            write_rows_with_mean(class_root / "metrics_raft.csv", class_raft_rows, bucket_name)
            flow_mean_row = build_mean_row(class_raft_rows, bucket_name)
            flow_mean_row["sample_id"] = "CLASS_MEAN"
            flow_mean_row["pair_type"] = "flow"
            all_class_flow_mean_rows.append(flow_mean_row)
        bucket_image_pair_curves = image_pair_curves_by_class.get(bucket_name, {"pred": [], "gt": []})
        bucket_flow_curves = flow_curves_by_class.get(bucket_name, {"pred": [], "gt": []})
        # 每个类别也分别输出 image_pair / flow 两套平均能量谱对比图，方便直接做类间比较。
        save_mean_spectrum(
            bucket_image_pair_curves["pred"],
            bucket_image_pair_curves["gt"],
            class_root,
            title=f"{bucket_name}-{data_type}-x{int(SCALE*SCALE)} Image Pair Mean Energy Spectrum",
            file_prefix="image_pair",
        )
        save_mean_spectrum(
            bucket_flow_curves["pred"],
            bucket_flow_curves["gt"],
            class_root,
            title=f"{bucket_name}-{data_type}-x{int(SCALE*SCALE)} Flow Mean Energy Spectrum",
            file_prefix="flow",
        )

        # 类别级 Δu / EPE 统计：把该类别所有 sample 的像素误差拼起来，再统一做直方图。
        if delta_u_hist_by_class.get(bucket_name):
            class_delta_u_values = np.concatenate(delta_u_hist_by_class[bucket_name], axis=0)
            np.save(class_root / "delta_u_hist_all.npy", _delta_u_histogram_matrix(class_delta_u_values))
        if delta_v_hist_by_class.get(bucket_name):
            class_delta_v_values = np.concatenate(delta_v_hist_by_class[bucket_name], axis=0)
            np.save(class_root / "delta_v_hist_all.npy", _delta_v_histogram_matrix(class_delta_v_values))
        if delta_w_hist_by_class.get(bucket_name):
            class_delta_w_values = np.concatenate(delta_w_hist_by_class[bucket_name], axis=0)
            np.save(class_root / "delta_w_hist_all.npy", _delta_w_histogram_matrix(class_delta_w_values))
        if epe_hist_by_class.get(bucket_name):
            class_epe_values = np.concatenate(epe_hist_by_class[bucket_name], axis=0)
            np.save(class_root / "epe_hist_all.npy", _epe_histogram_matrix(class_epe_values))

    write_all_class_mean_csv(all_class_image_pair_csv_path, all_class_image_pair_mean_rows)
    write_all_class_mean_csv(all_class_flow_csv_path, all_class_flow_mean_rows)

    if image_pair_rows:
        logger.info(f"[evaluate_all] image pair metrics csv: {metrics_image_pair_csv_path}")
    if raft_rows:
        logger.info(f"[evaluate_all] raft metrics csv: {metrics_raft_csv_path}")
    if all_class_image_pair_mean_rows:
        logger.info(f"[evaluate_all] all class image pair metrics csv: {all_class_image_pair_csv_path}")
    if all_class_flow_mean_rows:
        logger.info(f"[evaluate_all] all class flow metrics csv: {all_class_flow_csv_path}")
    logger.info(f"[evaluate_all] sample outputs: {output_root}")

    if raft_rows:
        return build_mean_row(raft_rows, class_name)
    if image_pair_rows:
        return build_mean_row(image_pair_rows, class_name)
    return build_mean_row(rows, class_name) if rows else {
        "class_name": class_name,
        "data_type": data_type,
        "scale": int(SCALE * SCALE),
        "sample_id": "MEAN",
        "pair_type": "all",
        "mse": float("nan"),
        "psnr": float("nan"),
        "energy_spectrum_mse": float("nan"),
        "VAL_AEE": float("nan"),
        "VAL_NORM_AEE_PER100PIXEL": float("nan"),
        "r2": float("nan"),
        "ssim": float("nan"),
        "tke_acc": float("nan"),
        "nrmse": float("nan"),
    }






