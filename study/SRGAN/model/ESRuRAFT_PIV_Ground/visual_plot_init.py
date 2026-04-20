"""
可视化 生成 start
"""
from loguru import logger
import numpy as np
import torch

from study.SRGAN.util.image_util import scalar_to_jet, build_triplet_row, build_pair_row, add_horizontal_separator

def build_flo_uvw_fake_panel(fake_bchw, col_sep=8):
    """
    仅对 fake 的 U/V/S 三通道做伪彩展示并横向拼接。
    只显示 fake，按三列排列：
    U* | V* | S*
    """
    if fake_bchw.ndim != 4 or fake_bchw.shape[1] < 3:
        logger.error(f'Need [B,>=3,H,W], got {tuple(fake_bchw.shape)}')
        raise ValueError(f"Need [B,>=3,H,W], got {tuple(fake_bchw.shape)}")

    cmin = fake_bchw[:, :3].amin(dim=(0, 2, 3), keepdim=True)
    cmax = fake_bchw[:, :3].amax(dim=(0, 2, 3), keepdim=True)
    den = (cmax - cmin).clamp_min(1e-8)
    x = (fake_bchw[:, :3] - cmin) / den  # [B,3,H,W]

    col_u = scalar_to_jet(x[:, 0:1])# [B,3,H,W]
    col_v = scalar_to_jet(x[:, 1:2])
    col_s = scalar_to_jet(x[:, 2:3])

    B, C, H, _ = col_u.shape
    v_sep = torch.full((B, C, H, col_sep), 1.0, device=x.device, dtype=x.dtype)  # 竖向分隔条

    # 关键：dim=3 左右拼接 -> 三列
    out = torch.cat([col_u, v_sep, col_v, v_sep, col_s], dim=3)
    return out.clamp(0, 1)


def build_flo_uvw_compare_panel(lr_bchw, fake_bchw, hr_bchw, sep_width=6, row_sep=8, sample_sep=10):
    """
    对 LR/Fake/HR 的 U/V/S 三通道做对比拼图。
    validate用：每个样本三列
    U*: LR|Fake|HR
    V*: LR|Fake|HR
    S*: LR|Fake|HR
    """
    for t in (lr_bchw, fake_bchw, hr_bchw):
        if t.shape[1] < 3:
            logger.error('Need 3 channels (U,V,S).')
            raise ValueError("Need 3 channels (U,V,S).")

    # 统一用 HR 做每通道 min-max，保证可比
    cmin = hr_bchw[:, :3].amin(dim=(0, 2, 3), keepdim=True)
    cmax = hr_bchw[:, :3].amax(dim=(0, 2, 3), keepdim=True)
    den = (cmax - cmin).clamp_min(1e-8)

    lr_n = (lr_bchw[:, :3] - cmin) / den
    fk_n = (fake_bchw[:, :3] - cmin) / den
    hr_n = (hr_bchw[:, :3] - cmin) / den

    sample_rows = []
    for i in range(lr_n.size(0)):
        ch_rows = []
        for ch in range(3):  # U,V,S
            lr_ch = scalar_to_jet(lr_n[i:i+1, ch:ch+1])
            fk_ch = scalar_to_jet(fk_n[i:i+1, ch:ch+1])
            hr_ch = scalar_to_jet(hr_n[i:i+1, ch:ch+1])
            # lr_ch = lr_n[i:i+1, ch:ch+1].repeat(1, 3, 1, 1)
            # fk_ch = fk_n[i:i+1, ch:ch+1].repeat(1, 3, 1, 1)
            # hr_ch = hr_n[i:i+1, ch:ch+1].repeat(1, 3, 1, 1)
            ch_rows.append(build_triplet_row(lr_ch, fk_ch, hr_ch, sep_width=sep_width))

        one = ch_rows[0]
        for r in ch_rows[1:]:
            h_sep = add_horizontal_separator(
                width=one.shape[3], channels=one.shape[1], sep_height=row_sep,
                value=1.0, device=one.device, dtype=one.dtype
            )
            one = torch.cat([one, h_sep, r], dim=2)
        sample_rows.append(one)

    out = sample_rows[0]
    for r in sample_rows[1:]:
        h_sep = add_horizontal_separator(
            width=out.shape[3], channels=out.shape[1], sep_height=sample_sep,
            value=1.0, device=out.device, dtype=out.dtype
        )
        out = torch.cat([out, h_sep, r], dim=2)
    return out.clamp(0, 1)


def build_flo_uvw_pred_gt_panel(pred_bchw, hr_bchw, sep_width=6, row_sep=8, sample_sep=10):
    """
    对 Pred/HR 的 U/V/S 三通道做对比拼图。
    每个样本两列：
    U*: Pred|HR
    V*: Pred|HR
    S*: Pred|HR
    """
    for t in (pred_bchw, hr_bchw):
        if t.shape[1] < 3:
            logger.error('Need 3 channels (U,V,S).')
            raise ValueError("Need 3 channels (U,V,S).")

    cmin = hr_bchw[:, :3].amin(dim=(0, 2, 3), keepdim=True)
    cmax = hr_bchw[:, :3].amax(dim=(0, 2, 3), keepdim=True)
    den = (cmax - cmin).clamp_min(1e-8)

    pred_n = (pred_bchw[:, :3] - cmin) / den
    hr_n = (hr_bchw[:, :3] - cmin) / den

    sample_rows = []
    for i in range(pred_n.size(0)):
        ch_rows = []
        for ch in range(3):
            pred_ch = scalar_to_jet(pred_n[i:i+1, ch:ch+1])
            hr_ch = scalar_to_jet(hr_n[i:i+1, ch:ch+1])
            ch_rows.append(build_pair_row(pred_ch, hr_ch, sep_width=sep_width))

        one = ch_rows[0]
        for r in ch_rows[1:]:
            h_sep = add_horizontal_separator(
                width=one.shape[3], channels=one.shape[1], sep_height=row_sep,
                value=1.0, device=one.device, dtype=one.dtype
            )
            one = torch.cat([one, h_sep, r], dim=2)
        sample_rows.append(one)

    out = sample_rows[0]
    for r in sample_rows[1:]:
        h_sep = add_horizontal_separator(
            width=out.shape[3], channels=out.shape[1], sep_height=sample_sep,
            value=1.0, device=out.device, dtype=out.dtype
        )
        out = torch.cat([out, h_sep, r], dim=2)
    return out.clamp(0, 1)


def _omega_star_from_uv(u: np.ndarray, v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """由 u,v 计算涡量并归一化到 [-2,2] 得到 omega*。"""
    # omega = dv/dx - du/dy
    dv_dy, dv_dx = np.gradient(v)
    du_dy, du_dx = np.gradient(u)
    omega = dv_dx - du_dy

    omin = omega.min()
    omax = omega.max()
    omega01 = (omega - omin) / (omax - omin + eps)   # [0,1]
    omega_star = omega01 * 4.0 - 2.0                 # [-2,2]
    return omega_star
"""
可视化 生成 end
"""
