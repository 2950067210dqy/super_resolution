import csv
import math
import time
from pathlib import Path
from subprocess import call

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

from study.SRGAN.model.c_aee_metric_common import attach_c_aee_to_raft_rows
from study.SRGAN.model.evaluate_image_compare_common import (
    save_particle_error_histogram,
    save_particle_error_histogram_bundle,
)
from study.SRGAN.model.metric_outlier_filter import robust_metric_mean
from study.SRGAN.data_downscal import DOWNSAMPLE_METHOD, INTERPOLATION_MODE, downsample_tif

try:
    autocast = torch.cuda.amp.autocast
except Exception:
    # 兼容旧版 PyTorch：如果没有 AMP，就保留 with autocast(enabled=False) 的调用形式。
    class autocast:
        def __init__(self, enabled=False):
            self.enabled = enabled

        def __enter__(self):
            return None

        def __exit__(self, *args):
            return False


_CACHED_2D_WINDOWS = {}


def _save_optional_npy(path, array, save_npy, *, force=False):
    """
    test_all 的 NPY 统一保存入口。

    IS_SAVE_NPY=False 时会跳过普通中间结果，减少大图测试时的磁盘占用；
    force=True 用于用户要求必须保留的文件，例如 TBL 三位置 profile、hist 直方图和误差 NPY。
    """
    if force or bool(save_npy):
        np.save(path, array)


def _match_common_channels(pred_chw, gt_chw):
    """
    对齐预测/真值的公共通道数，避免不同分支的 SR 输出通道数不一致时直接报 shape 错误。

    说明：
        test_all 的 HR 输入大多是单通道颗粒图，但个别生成器可能输出多通道结果。
        evaluate_all 的指标本质上都只依赖“可对齐的公共通道”，因此这里统一截到两者共同拥有的最小通道数。
    """
    pred = np.asarray(pred_chw, dtype=np.float32)
    gt = np.asarray(gt_chw, dtype=np.float32)
    if pred.ndim != 3 or gt.ndim != 3:
        raise ValueError(f"Expected CHW arrays, got pred={pred.shape}, gt={gt.shape}")
    cnum = min(int(pred.shape[0]), int(gt.shape[0]))
    if cnum <= 0:
        raise ValueError(f"No common channels for metrics: pred={pred.shape}, gt={gt.shape}")
    return pred[:cnum], gt[:cnum]


def _finite_pair_mask(pred, gt):
    """
    返回预测/真值同时为有限值的位置掩码。

    TBL/TWCF 的 full-frame 数据里会包含边界外、无效区或历史 fold 边界留下的 NaN/Inf。
    这些点不能参与指标计算，否则任意一个 NaN 都会把整张图的 mean/max/fft/EPE 变成 NaN。
    """
    return np.isfinite(pred) & np.isfinite(gt)


def _finite_pair_values(pred, gt):
    """
    拉平成一维并只保留预测和真值都有效的位置。

    MSE、R2、NRMSE 这类逐像素统计应该严格忽略无效像素，而不是先把 NaN 填 0；
    否则会把无效区域误当作真实误差，尤其会影响 TBL/TWCF 这类大幅面数据。
    """
    valid = _finite_pair_mask(pred, gt)
    if not np.any(valid):
        empty = np.asarray([], dtype=np.float32)
        return empty, empty
    return pred[valid].astype(np.float32, copy=False), gt[valid].astype(np.float32, copy=False)


def _dense_metric_pair(pred_2d, gt_2d, fill_value=0.0):
    """
    为 SSIM/FFT 这类需要完整二维矩阵的指标准备无 NaN 输入。

    逐像素误差可以直接用掩码过滤；但 SSIM 和频谱需要完整的 2D 场。这里先找出
    pred/gt 同时有效的位置，如果存在有效数据，就用各自有效区域均值填补无效点。
    这样无效区域不会制造极端 0 值，也不会让 FFT/SSIM 因 NaN 直接失效。
    """
    pred = np.asarray(pred_2d, dtype=np.float32).copy()
    gt = np.asarray(gt_2d, dtype=np.float32).copy()
    valid = _finite_pair_mask(pred, gt)
    if not np.any(valid):
        return None, None

    pred_fill = float(np.mean(pred[valid], dtype=np.float64)) if np.any(np.isfinite(pred[valid])) else float(fill_value)
    gt_fill = float(np.mean(gt[valid], dtype=np.float64)) if np.any(np.isfinite(gt[valid])) else float(fill_value)
    pred[~valid] = pred_fill
    gt[~valid] = gt_fill
    pred[~np.isfinite(pred)] = pred_fill
    gt[~np.isfinite(gt)] = gt_fill
    return pred, gt


def _nanmean_or_nan(values):
    """只对有限值求均值；没有有效值时返回 NaN。"""
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr, dtype=np.float64)) if arr.size > 0 else float("nan")


def _mse(pred_chw, gt_chw):
    """计算均方误差 MSE，只统计 pred/gt 都有限的像素。"""
    pred, gt = _match_common_channels(pred_chw, gt_chw)
    pred_valid, gt_valid = _finite_pair_values(pred, gt)
    if pred_valid.size == 0:
        return float("nan")
    return float(np.mean((pred_valid - gt_valid) ** 2, dtype=np.float64))


def _psnr_from_mse(mse):
    """由 MSE 计算 PSNR，假设图像范围已经在 [0,1]。"""
    if not np.isfinite(mse):
        return float("nan")
    return float("inf") if mse == 0 else 20.0 * math.log10(1.0 / math.sqrt(mse))


def _r2_score(pred_chw, gt_chw, eps=1e-12):
    """计算决定系数 R^2，只使用有效像素，避免无效边界把结果变成 NaN。"""
    pred, gt = _match_common_channels(pred_chw, gt_chw)
    y_pred, y_true = _finite_pair_values(pred, gt)
    if y_true.size == 0:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / (ss_tot + eps)


def _nrmse(pred_chw, gt_chw, eps=1e-12):
    """按真值范围归一化的 RMSE，只统计有效像素。"""
    pred, gt = _match_common_channels(pred_chw, gt_chw)
    pred_valid, gt_valid = _finite_pair_values(pred, gt)
    if pred_valid.size == 0:
        return float("nan")
    rmse = math.sqrt(float(np.mean((pred_valid - gt_valid) ** 2, dtype=np.float64)))
    den = float(np.max(gt_valid) - np.min(gt_valid))
    return rmse / (den + eps)


def _ssim_score(pred_chw, gt_chw):
    """
    计算 SSIM，按公共通道逐通道求值后取平均。

    优先使用 skimage；如果环境里没有 skimage，则回退到一个简化版 SSIM 公式，
    保证 test_all 在最小依赖环境里也能把指标跑完。
    """
    pred, gt = _match_common_channels(pred_chw, gt_chw)
    try:
        from skimage.metrics import structural_similarity as sk_ssim

        vals = []
        for c in range(pred.shape[0]):
            # skimage 的 SSIM 不能直接处理 NaN/Inf；先按有效区域均值补齐无效点，
            # 补齐仅用于 dense image metric，不会改变 results.npy 或可视化输出。
            pair = _dense_metric_pair(pred[c], gt[c])
            if pair[0] is None:
                continue
            p, g = pair
            dr = float(np.max(g) - np.min(g))
            dr = dr if dr > 1e-12 else 1.0
            vals.append(float(sk_ssim(g, p, data_range=dr)))
        return float(np.mean(vals)) if vals else float("nan")
    except Exception:
        vals = []
        C1, C2 = 0.01**2, 0.03**2
        for c in range(pred.shape[0]):
            pair = _dense_metric_pair(pred[c], gt[c])
            if pair[0] is None:
                continue
            x, y = pair
            mx, my = float(np.mean(x)), float(np.mean(y))
            vx, vy = float(np.var(x)), float(np.var(y))
            cov = float(np.mean((x - mx) * (y - my)))
            num = (2 * mx * my + C1) * (2 * cov + C2)
            den = (mx * mx + my * my + C1) * (vx + vy + C2)
            vals.append(num / den if den != 0 else 0.0)
        return float(np.mean(vals)) if vals else float("nan")


def _tke_reconstruction_accuracy(pred_chw, gt_chw, eps=1e-12):
    """
    计算 TKE 重建精度。

    对单通道 SR 图像，这个指标没有物理意义，因此会返回 nan；
    这与 evaluate_all 的行为保持一致。
    """
    pred, gt = _match_common_channels(pred_chw, gt_chw)
    if pred.shape[0] < 2 or gt.shape[0] < 2:
        return float("nan")
    up, vp = pred[0], pred[1]
    ug, vg = gt[0], gt[1]
    valid = np.isfinite(up) & np.isfinite(vp) & np.isfinite(ug) & np.isfinite(vg)
    if not np.any(valid):
        return float("nan")
    up, vp, ug, vg = up[valid], vp[valid], ug[valid], vg[valid]
    up_p = up - np.mean(up)
    vp_p = vp - np.mean(vp)
    ug_p = ug - np.mean(ug)
    vg_p = vg - np.mean(vg)
    tke_p = 0.5 * float(np.mean(up_p ** 2 + vp_p ** 2))
    tke_g = 0.5 * float(np.mean(ug_p ** 2 + vg_p ** 2))
    return 1.0 - abs(tke_p - tke_g) / (abs(tke_g) + eps)


def _radial_spectrum(ch2d):
    """计算单通道二维场的径向平均能量谱。"""
    f = np.fft.fftshift(np.fft.fft2(ch2d))
    p = np.abs(f) ** 2
    h, w = p.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)
    tbin = np.bincount(r.ravel(), p.ravel())
    nr = np.bincount(r.ravel())
    return tbin / np.maximum(nr, 1)


def _energy_spectrum_curves(pred_chw, gt_chw):
    """计算公共通道上的平均径向能量谱曲线，FFT 前会补齐无效像素。"""
    pred, gt = _match_common_channels(pred_chw, gt_chw)
    pred_specs = []
    gt_specs = []
    min_len = None
    for c in range(pred.shape[0]):
        pair = _dense_metric_pair(pred[c], gt[c])
        if pair[0] is None:
            continue
        pred_2d, gt_2d = pair
        sp = _radial_spectrum(pred_2d)
        sg = _radial_spectrum(gt_2d)
        n = min(len(sp), len(sg))
        min_len = n if min_len is None else min(min_len, n)
        pred_specs.append(sp[:n])
        gt_specs.append(sg[:n])
    if not pred_specs or min_len is None:
        empty = np.asarray([np.nan], dtype=np.float32)
        return empty, empty
    pred_curve = np.mean(np.stack([x[:min_len] for x in pred_specs], axis=0), axis=0)
    gt_curve = np.mean(np.stack([x[:min_len] for x in gt_specs], axis=0), axis=0)
    return pred_curve, gt_curve


def _energy_spectrum_mse_from_curves(pred_curve, gt_curve):
    """对已算好的谱曲线计算 log1p 频谱差 MSE，只统计有限频点。"""
    pred_log = np.log1p(pred_curve)
    gt_log = np.log1p(gt_curve)
    valid = _finite_pair_mask(pred_log, gt_log)
    if not np.any(valid):
        return float("nan")
    return float(np.mean((pred_log[valid] - gt_log[valid]) ** 2, dtype=np.float64))


def _compute_epe_values_from_chw(pred_chw, gt_chw):
    """
    计算单样本 CHW 流场的有效像素 EPE 序列。

    TBL/TWCF 的 GT 或预测可能在无效区域包含 NaN/Inf。这里要求 U/V 两个通道的
    pred 和 gt 都是有限值才参与 EPE，保证 AEE、NORM_AEE_PER100PIXEL 和 C-AEE
    不再被单个无效像素拖成 NaN。
    """
    pred, gt = _match_common_channels(pred_chw, gt_chw)
    if pred.shape[0] < 2 or gt.shape[0] < 2:
        return np.asarray([], dtype=np.float32)
    valid = (
        np.isfinite(pred[0])
        & np.isfinite(pred[1])
        & np.isfinite(gt[0])
        & np.isfinite(gt[1])
    )
    if not np.any(valid):
        return np.asarray([], dtype=np.float32)
    du = pred[0] - gt[0]
    dv = pred[1] - gt[1]
    epe = np.sqrt(du * du + dv * dv)
    epe = epe[valid]
    return epe[np.isfinite(epe)].astype(np.float32, copy=False)


def _compute_aee_from_chw(pred_chw, gt_chw):
    """
    计算单样本 CHW 流场的 AEE。

    这里和 evaluate_all 保持同口径：AEE 等价于像素级 EPE 的平均值。
    """
    epe = _compute_epe_values_from_chw(pred_chw, gt_chw)
    if epe.size == 0:
        return float("nan")
    return float(np.mean(epe, dtype=np.float64))


def _format_metric_annotation(metric_name, value, digits=4):
    """把图上要显示的指标格式化成统一文本，NaN/Inf 时明确显示为 NaN。"""
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = float("nan")
    if not np.isfinite(value):
        return f"{metric_name} = NaN"
    return f"{metric_name} = {value:.{int(digits)}f}"


def _annotate_metric_box(ax, text, fontsize=13):
    """
    在误差图左上角添加白底指标框。

    风格参考用户示例图：白色半透明背景 + 黑色斜体文本；
    使用 axes 坐标而不是数据坐标，保证不同分辨率/裁剪尺寸下位置一致。
    """
    if not text:
        return
    ax.text(
        0.035,
        0.93,
        str(text),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        fontstyle="italic",
        color="black",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.82,
            "pad": 3.0,
        },
    )


def _mean_sum_per_100_pixels(values_1d, group_size=100):
    """
    将一维误差序列按每 100 个像素分组求和，再对所有满 100 像素分组和取平均。

    这就是 evaluate_all 里 NORM_AEE_PER100PIXEL 的统计口径。
    最后一组不足 100 个像素时直接丢弃，不参与统计。
    """
    values = np.asarray(values_1d, dtype=np.float32).reshape(-1)
    # 无效像素已经不属于真实评估区域，先过滤再按每 100 个有效像素分组。
    values = values[np.isfinite(values)]
    if values.size < group_size:
        return float("nan")
    usable_count = (values.size // group_size) * group_size
    if usable_count <= 0:
        return float("nan")
    values = values[:usable_count].reshape(-1, group_size)
    group_sums = np.sum(values, axis=1, dtype=np.float32)
    return float(np.mean(group_sums, dtype=np.float32)) if group_sums.size > 0 else float("nan")


def _compute_norm_aee_per100_from_chw(pred_chw, gt_chw):
    """
    用单样本 CHW 流场计算“每 100 个像素 EPE 累加值的平均”。
    """
    epe = _compute_epe_values_from_chw(pred_chw, gt_chw)
    if epe.size == 0:
        return float("nan")
    return _mean_sum_per_100_pixels(epe, group_size=100)


def _torch_finite_epe_per_sample(predicted_flows, flows):
    """
    在 GPU tensor 上计算每个样本的有限像素平均 EPE。

    旧逻辑直接对整张图 `.mean()`，TBL/TWCF 只要某个无效区域含 NaN/Inf，
    整个 sample 甚至整批 batch 的 EPE 就会变成 NaN。这里和 CSV 明细指标保持一致：
    只有 pred(U/V) 与 gt(U/V) 全部为有限值的像素才参与平均。
    """
    if predicted_flows.size(1) < 2 or flows.size(1) < 2:
        return torch.full(
            (int(predicted_flows.size(0)),),
            float("nan"),
            device=predicted_flows.device,
            dtype=predicted_flows.dtype,
        )

    pred_uv = predicted_flows[:, :2, :, :]
    gt_uv = flows[:, :2, :, :]
    valid = (
        torch.isfinite(pred_uv[:, 0])
        & torch.isfinite(pred_uv[:, 1])
        & torch.isfinite(gt_uv[:, 0])
        & torch.isfinite(gt_uv[:, 1])
    )

    diff = pred_uv - gt_uv
    epe = torch.sqrt(torch.sum(diff * diff, dim=1))
    epe = torch.where(valid & torch.isfinite(epe), epe, torch.zeros_like(epe))

    valid_flat = valid.flatten(1)
    epe_flat = epe.flatten(1)
    counts = valid_flat.sum(dim=1)
    sums = epe_flat.sum(dim=1)
    means = sums / counts.clamp_min(1).to(dtype=sums.dtype)
    return torch.where(counts > 0, means, torch.full_like(means, float("nan")))


def _compute_image_metric_row(dataset_name, sample_index, pair_type, pred_chw, gt_chw):
    """
    计算单个 previous/next SR 图像对的指标行。

    字段对齐 evaluate_all：mse / psnr / energy_spectrum_mse / r2 / ssim / tke_acc / nrmse。
    """
    pred_curve, gt_curve = _energy_spectrum_curves(pred_chw, gt_chw)
    mse = _mse(pred_chw, gt_chw)
    return {
        "dataset": dataset_name,
        "sample_index": sample_index,
        "pair_type": pair_type,
        "mse": mse,
        "psnr": _psnr_from_mse(mse),
        "energy_spectrum_mse": _energy_spectrum_mse_from_curves(pred_curve, gt_curve),
        "r2": _r2_score(pred_chw, gt_chw),
        "ssim": _ssim_score(pred_chw, gt_chw),
        "tke_acc": _tke_reconstruction_accuracy(pred_chw, gt_chw),
        "nrmse": _nrmse(pred_chw, gt_chw),
    }


def _compute_flow_metric_row(dataset_name, sample_index, pred_chw, gt_chw):
    """
    计算单个样本的流场指标行。

    按用户当前需求，test_all 的 RAFT 指标包含：
        1. EPE
        2. NORM_AEE_PER100PIXEL
        3. energy_spectrum_mse
        4. C_AEE

    其中 C_AEE 需要把同一个 sample 的 SR 能量谱误差、SSIM 和 RAFT EPE 配对，
    所以这里先放占位值，等整套 dataset 全部跑完后再统一回填真实结果。
    """
    return {
        "dataset": dataset_name,
        "sample_index": sample_index,
        "pair_type": "RAFT",
        "epe": _compute_aee_from_chw(pred_chw, gt_chw),
        "NORM_AEE_PER100PIXEL": _compute_norm_aee_per100_from_chw(pred_chw, gt_chw),
        # C-AEE 需要同 sample 的 previous/next 图像 ESMSE/SSIM 与 RAFT EPE 配对；
        # 三项会先按固定参考尺度放缩为无量纲量，不再依赖当前 dataset 的 min/max。
        # flow 自身的 energy_spectrum_mse 会在调用处复用已算出的频谱曲线回填，
        # 避免这里重复做一次频域统计。
        "energy_spectrum_mse": float("nan"),
        "C_AEE": float("nan"),
    }


def _flow_uv_to_uvw_np(flow_chw):
    """
    将 test_all 的 2 通道 [u, v] 光流扩展成 [u, v, magnitude]。

    evaluate_all 的 flow 汇总里包含 Δw，其中 w 表示位移幅值；test_all 的 TFRecord
    真值和 RAFT 输出通常只有 uv 两个通道，所以这里在统计直方图/能量谱前补出第三通道，
    保证 test_all 的类别级和总体级 npy/图像输出与 evaluate_all 口径一致。
    """
    flow = np.asarray(flow_chw, dtype=np.float32)
    if flow.shape[0] >= 3:
        return flow[:3]
    if flow.shape[0] < 2:
        raise ValueError(f"Expected flow with at least 2 channels, got shape={flow.shape}")
    u = flow[0:1]
    v = flow[1:2]
    mag = np.sqrt(u * u + v * v + 1e-12).astype(np.float32, copy=False)
    return np.concatenate([u, v, mag], axis=0)


def _compute_flow_error_maps_np(pred_chw, gt_chw):
    """
    计算 evaluate_all 同款的 Δu / Δv / Δw / EPE 误差图。

    返回的数组保留 NaN/Inf，后续直方图函数会只统计有限值；这样不会把无效区域误当作 0 误差。
    """
    pred = _flow_uv_to_uvw_np(pred_chw)
    gt = _flow_uv_to_uvw_np(gt_chw)
    delta_u = pred[0] - gt[0]
    delta_v = pred[1] - gt[1]
    delta_w = pred[2] - gt[2]
    epe = np.sqrt(delta_u * delta_u + delta_v * delta_v)
    return delta_u.astype(np.float32), delta_v.astype(np.float32), delta_w.astype(np.float32), epe.astype(np.float32)


def _finite_display_field(field_2d, fill_value=0.0):
    """
    将单通道场转换成可绘图数组。

    NPY 文件会保存原始 NaN/Inf，便于后处理继续识别无效区域；但 matplotlib 的 colorbar、
    quiver 和 tight_layout 对 NaN/Inf 很敏感，因此只在绘图副本中用有限区域均值做填补。
    """
    field = np.asarray(field_2d, dtype=np.float32).copy()
    finite = np.isfinite(field)
    if np.any(finite):
        fill = float(np.mean(field[finite], dtype=np.float64))
    else:
        fill = float(fill_value)
    field[~finite] = fill
    return field


def _finite_color_limits_from_arrays(arrays, symmetric=False, fallback=(-1.0, 1.0)):
    """
    从一组数组的有限值中求色条范围。

    symmetric=True 用于误差图或涡度差图，使色条以 0 为中心，正负误差更容易比较。
    """
    chunks = []
    for arr in arrays:
        values = np.asarray(arr, dtype=np.float32).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size > 0:
            chunks.append(values)
    if not chunks:
        return float(fallback[0]), float(fallback[1])
    values = np.concatenate(chunks, axis=0)
    if symmetric:
        max_abs = float(np.max(np.abs(values)))
        max_abs = max(max_abs, 1e-6)
        return -max_abs, max_abs
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return float(fallback[0]), float(fallback[1])
    if abs(vmax - vmin) < 1e-6:
        pad = max(abs(vmax), 1.0) * 0.05
        return vmin - pad, vmax + pad
    return vmin, vmax


def _flow_component_limits_like_overview(dataset_name, component_index):
    """
    返回与原始 test_all 流场总览图一致的 U/V/S 色条范围。

    之前 `uvs_compare.png` 会按每个 sample 的有限值自动缩放色条，导致同一张样本在
    `{dataset}_sample_XXXX.png` 和 `flow/sample_XXXX/uvs_compare.png` 里颜色含义不一致。
    这里把 U/V/S compare 的 Pred/GT 色条改成和原总览图同一套固定范围：
    - 普通 256 数据集：沿用 `_plot_regular()` 的 U/V = [-4, 4]；
    - TBL：沿用 `_plot_tbl()` 的 U = [2, 8]、V = [-0.5, 0.5]；
    - TWCF：沿用 `_plot_twcf()` 的 U = [-2, 12]、V = [-1, 1]。

    S 是由 `sqrt(u^2 + v^2)` 得到的位移幅值，原总览图没有单独 S 面板；
    因此 S 使用对应 U/V 范围推导出的非负幅值范围，保证和 U/V 的位移尺度一致。
    """
    dataset_key = str(dataset_name or "").lower()
    if dataset_key == "tbl":
        limits = ((2.0, 8.0), (-0.5, 0.5), (2.0, math.sqrt(8.0 * 8.0 + 0.5 * 0.5)))
    elif dataset_key == "twcf":
        limits = ((-2.0, 12.0), (-1.0, 1.0), (0.0, math.sqrt(12.0 * 12.0 + 1.0 * 1.0)))
    else:
        limits = ((-4.0, 4.0), (-4.0, 4.0), (0.0, math.sqrt(4.0 * 4.0 + 4.0 * 4.0)))
    return limits[int(component_index)]


def _flow_error_limits_like_overview(dataset_name, component_index, display_delta):
    """
    返回 U/V/S compare 误差图的色条范围。

    普通五类数据的原总览图 `_plot_regular()` 明确使用 `[-0.25, 0.25]`，
    因此这里同步该范围。TBL/TWCF 总览图没有误差面板，继续用当前样本的对称范围，
    避免把大尺寸 full-frame 的局部误差过度截断。
    """
    dataset_key = str(dataset_name or "").lower()
    if dataset_key not in {"tbl", "twcf"}:
        return -0.25, 0.25
    return _finite_color_limits_from_arrays(
        [display_delta],
        symmetric=True,
        fallback=(-0.25, 0.25),
    )


def _omega_star_from_uv_np(u_2d, v_2d, eps=1e-8):
    """
    由 U/V 位移场计算 evaluate_all 同款 omega* 涡度场。

    与各分支 visual_plot_init.py 中的 _omega_star_from_uv 保持一致：先算
    omega=dv/dx-du/dy，再 min-max 到 [-2, 2]。这里额外保留无效区域为 NaN，
    避免 TBL/TWCF 的无效边界被误当作真实涡度。
    """
    u = np.asarray(u_2d, dtype=np.float32)
    v = np.asarray(v_2d, dtype=np.float32)
    valid = np.isfinite(u) & np.isfinite(v)
    if not np.any(valid):
        return np.full(u.shape, np.nan, dtype=np.float32)

    u_fill = _finite_display_field(u)
    v_fill = _finite_display_field(v)
    dv_dy, dv_dx = np.gradient(v_fill)
    du_dy, du_dx = np.gradient(u_fill)
    omega = (dv_dx - du_dy).astype(np.float32, copy=False)
    omega[~valid] = np.nan

    finite = np.isfinite(omega)
    if not np.any(finite):
        return omega.astype(np.float32, copy=False)
    omin = float(np.min(omega[finite]))
    omax = float(np.max(omega[finite]))
    omega_star = (omega - omin) / (omax - omin + float(eps))
    omega_star = omega_star * 4.0 - 2.0
    omega_star[~finite] = np.nan
    return omega_star.astype(np.float32, copy=False)


def _repair_vorticity_top_edge_for_display(omega_2d, edge_rows=None, fallback=0.0):
    """
    修复涡度图绘图副本顶部的边界条纹。

    原因：
        vorticity/omega* 是由 U/V 再做空间梯度得到的。即使 U/V/S compare 已经裁掉
        TBL 顶部填充区或修复 TWCF 顶部无效点，`np.gradient` 在图像最高 y 边界仍会使用
        单边差分，容易在最上方生成一条极端值色带，看起来像黑色条。

    处理：
        只对 png 绘图副本生效，把最上方若干行替换为其下方最近的内侧行。
        注意 imshow(origin="lower") 下数组最后几行显示在图片顶部，所以这里修复的是
        `field[-edge_rows:]`。原始 pred_vorticity.npy / gt_vorticity.npy / delta_vorticity.npy
        不会被这个函数改动。
    """
    field = _repair_nonfinite_by_vertical_nearest(omega_2d, fallback=fallback)
    if field.ndim != 2 or field.shape[0] < 3:
        return field.astype(np.float32, copy=False)

    height = int(field.shape[0])
    if edge_rows is None:
        # 自适应修复 1~6 行：小图去掉单边差分行，大图也只修最靠边的窄条。
        edge_rows = max(1, min(6, height // 64))
    edge_rows = int(np.clip(int(edge_rows), 1, max(1, height // 4)))
    if height <= edge_rows:
        return field.astype(np.float32, copy=False)

    repaired = field.astype(np.float32, copy=True)
    source_row = max(height - edge_rows - 1, 0)
    repaired[-edge_rows:, :] = repaired[source_row:source_row + 1, :]
    return repaired


def _prepare_flow_component_for_display(field_2d, dataset_name=None, mask_2d=None, tbl_y_limit=None, fill_value=0.0):
    """
    按 dataset 的可视化习惯准备单个 U/V/S 分量。

    重要：这个函数只服务 png 绘图，不改变保存到 npy 的原始预测/真值。
    - TBL：复用之前解决过的顶部 0 填充问题，只显示有效边界层高度；
    - TWCF：只把底部波浪边界以下填成 0，顶部或孤立无效点用邻近有效值修复；
    - 普通 256 数据集：只修复 NaN/Inf，避免 matplotlib 画图时报错。
    """
    dataset_key = str(dataset_name or "").lower()
    field = np.asarray(field_2d, dtype=np.float32)
    if dataset_key == "tbl":
        repaired = _repair_nonfinite_by_vertical_nearest(field, fallback=fill_value)
        return _crop_tbl_valid_y(repaired, tbl_y_limit)
    if dataset_key == "twcf":
        return _fill_twcf_bottom_invalid_field_for_plot(field, mask_2d, fill_value=fill_value)
    return _finite_display_field(field, fill_value=fill_value)


def _prepare_uvw_display_stacks(pred_uvw, gt_uvw, dataset_name=None, mask_2d=None, tbl_y_limit=None):
    """
    为 U/V/S compare 和涡度速度图准备绘图副本。

    raw pred_uvw / gt_uvw 仍完整保存；这里额外生成 display_pred/display_gt/display_delta，
    让 TBL 顶部填充区和 TWCF 底部波浪区在所有 test_all flow 图中保持同一套视觉口径。
    """
    display_pred = []
    display_gt = []
    for comp_idx in range(3):
        display_pred.append(
            _prepare_flow_component_for_display(
                pred_uvw[comp_idx],
                dataset_name=dataset_name,
                mask_2d=mask_2d,
                tbl_y_limit=tbl_y_limit,
                fill_value=0.0,
            )
        )
        display_gt.append(
            _prepare_flow_component_for_display(
                gt_uvw[comp_idx],
                dataset_name=dataset_name,
                mask_2d=mask_2d,
                tbl_y_limit=tbl_y_limit,
                fill_value=0.0,
            )
        )

    display_pred = np.stack(display_pred, axis=0).astype(np.float32, copy=False)
    display_gt = np.stack(display_gt, axis=0).astype(np.float32, copy=False)
    display_delta = (display_pred - display_gt).astype(np.float32, copy=False)
    return display_pred, display_gt, display_delta


def _save_uvw_compare_artifacts(
    sample_dir,
    pred_chw,
    gt_chw,
    cmap_name="jet",
    save_npy=True,
    dataset_name=None,
    mask_2d=None,
    tbl_y_limit=None,
):
    """
    保存 test_all 的 U/V/S compare 图以及对应 NPY。

    evaluate_all 的样本目录会保存 fake_flo.npy / hr_flo.npy 和 uvs_compare.png；
    test_all 这里补齐同款文件，同时额外保存 pred/gt/delta 的 CHW 版本，方便后续脚本
    直接读取 U、V、S 三个分量做分析。
    """
    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    pred_uvw = _flow_uv_to_uvw_np(pred_chw).astype(np.float32, copy=False)
    gt_uvw = _flow_uv_to_uvw_np(gt_chw).astype(np.float32, copy=False)
    delta_uvw = (pred_uvw - gt_uvw).astype(np.float32, copy=False)

    # evaluate_all 兼容命名和 pred/gt 原始数组属于普通 NPY，由 IS_SAVE_NPY 控制。
    _save_optional_npy(sample_dir / "fake_flo.npy", pred_uvw.transpose(1, 2, 0).astype(np.float32), save_npy)
    _save_optional_npy(sample_dir / "hr_flo.npy", gt_uvw.transpose(1, 2, 0).astype(np.float32), save_npy)
    _save_optional_npy(sample_dir / "pred_uvw.npy", pred_uvw.astype(np.float32), save_npy)
    _save_optional_npy(sample_dir / "gt_uvw.npy", gt_uvw.astype(np.float32), save_npy)
    # delta_* 是误差 NPY，按需求始终保存，方便后处理直接读取误差场。
    _save_optional_npy(sample_dir / "delta_uvw.npy", delta_uvw.astype(np.float32), save_npy, force=True)

    component_names = ("u", "v", "s")
    for idx, name in enumerate(component_names):
        _save_optional_npy(sample_dir / f"pred_{name}.npy", pred_uvw[idx].astype(np.float32), save_npy)
        _save_optional_npy(sample_dir / f"gt_{name}.npy", gt_uvw[idx].astype(np.float32), save_npy)
        _save_optional_npy(sample_dir / f"delta_{name}.npy", delta_uvw[idx].astype(np.float32), save_npy, force=True)

    display_pred_uvw, display_gt_uvw, display_delta_uvw = _prepare_uvw_display_stacks(
        pred_uvw,
        gt_uvw,
        dataset_name=dataset_name,
        mask_2d=mask_2d,
        tbl_y_limit=tbl_y_limit,
    )
    # AEE 是流场误差图的核心标量指标：等价于所有有效像素 EPE 的平均值。
    # 这里使用原始 pred_chw/gt_chw 计算，不受 TBL/TWCF 绘图裁剪和填充逻辑影响。
    sample_aee_text = _format_metric_annotation("AEE", _compute_aee_from_chw(pred_chw, gt_chw))

    fig, axes = plt.subplots(3, 3, figsize=(15, 11), dpi=140, facecolor="w")
    for row_idx, label in enumerate(("U", "V", "S")):
        # Pred/GT 的色条范围不再按当前 sample 自动 min/max 缩放，
        # 而是对齐原始 `{dataset}_sample_XXXX.png` 流场总览图的固定范围。
        # 这样同一个 displacement 数值在两类图中的颜色含义保持一致。
        vmin, vmax = _flow_component_limits_like_overview(dataset_name, row_idx)
        err_min, err_max = _flow_error_limits_like_overview(
            dataset_name,
            row_idx,
            display_delta_uvw[row_idx],
        )
        _plot_field_with_colorbar(
            axes[row_idx, 0],
            display_pred_uvw[row_idx],
            f"Pred {label}",
            cmap_name,
            vmin,
            vmax,
            f"{label} displacement [px]",
        )
        _plot_field_with_colorbar(
            axes[row_idx, 1],
            display_gt_uvw[row_idx],
            f"GT {label}",
            cmap_name,
            vmin,
            vmax,
            f"{label} displacement [px]",
        )
        _plot_field_with_colorbar(
            axes[row_idx, 2],
            display_delta_uvw[row_idx],
            f"Error {label} (Pred-GT)",
            "bwr",
            err_min,
            err_max,
            f"{label} error [px]",
            metric_text=sample_aee_text,
        )

    fig.tight_layout()
    fig.savefig(sample_dir / "uvs_compare.png", bbox_inches="tight")
    plt.close(fig)
    return pred_uvw, gt_uvw, delta_uvw


def _save_vorticity_velocity_artifacts(
    sample_dir,
    pred_uvw,
    gt_uvw,
    stride=None,
    save_npy=True,
    cmap_name="jet",
    dataset_name=None,
    mask_2d=None,
    tbl_y_limit=None,
):
    """
    保存涡度-速度对比图及对应 NPY。

    图中三列分别是：
        1. Pred omega* + 预测速度矢量
        2. GT omega* + 真值速度矢量
        3. Delta omega* + 速度差矢量

    这样既保留 evaluate_all 的 vorticity_quiver.png 命名，又补上用户要求的
    “涡流速度差对比”信息。
    """
    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    pred_omega = _omega_star_from_uv_np(pred_uvw[0], pred_uvw[1])
    gt_omega = _omega_star_from_uv_np(gt_uvw[0], gt_uvw[1])
    delta_omega = (pred_omega - gt_omega).astype(np.float32, copy=False)
    delta_uv = (pred_uvw[:2] - gt_uvw[:2]).astype(np.float32, copy=False)

    # pred/gt 涡度是普通中间结果，由 IS_SAVE_NPY 控制；delta_* 是误差 NPY，始终保存。
    _save_optional_npy(sample_dir / "pred_vorticity.npy", pred_omega.astype(np.float32), save_npy)
    _save_optional_npy(sample_dir / "gt_vorticity.npy", gt_omega.astype(np.float32), save_npy)
    _save_optional_npy(sample_dir / "delta_vorticity.npy", delta_omega.astype(np.float32), save_npy, force=True)
    _save_optional_npy(sample_dir / "delta_velocity_uv.npy", delta_uv.transpose(1, 2, 0).astype(np.float32), save_npy, force=True)
    # vorticity_quiver.png 第三列展示的是 Delta omega*，这里补同源误差直方图。
    # hist 属于诊断结果，force=True 的保存回调会让它不受 IS_SAVE_NPY=False 影响。
    save_particle_error_histogram(
        sample_dir,
        delta_omega,
        file_prefix="delta_vorticity",
        title="Delta Vorticity Error Distribution",
        xlabel="delta omega*",
        color="#4477AA",
        save_npy_fn=_save_optional_npy,
        save_npy=save_npy,
    )

    # 绘图副本复用 U/V/S compare 的 dataset 专用处理，避免 TBL 顶部 0 填充区再次形成黑色条带。
    display_pred_uvw, display_gt_uvw, _ = _prepare_uvw_display_stacks(
        pred_uvw,
        gt_uvw,
        dataset_name=dataset_name,
        mask_2d=mask_2d,
        tbl_y_limit=tbl_y_limit,
    )
    # png 绘图副本额外修复 vorticity 顶部边界条纹：
    # U/V/S 本身已经按 dataset 处理过无效区，但 omega* 是梯度量，最高 y 边界仍可能产生单边差分色带。
    display_pred_omega = _repair_vorticity_top_edge_for_display(
        _omega_star_from_uv_np(display_pred_uvw[0], display_pred_uvw[1])
    )
    display_gt_omega = _repair_vorticity_top_edge_for_display(
        _omega_star_from_uv_np(display_gt_uvw[0], display_gt_uvw[1])
    )
    display_delta_omega = _repair_vorticity_top_edge_for_display(
        (display_pred_omega - display_gt_omega).astype(np.float32, copy=False)
    )
    display_delta_uv = (display_pred_uvw[:2] - display_gt_uvw[:2]).astype(np.float32, copy=False)

    h, w = int(display_pred_uvw.shape[-2]), int(display_pred_uvw.shape[-1])
    if stride is None:
        # 大图自动稀疏采样，避免 TBL/TWCF 的 quiver 过密导致图片巨大或保存过慢。
        stride = max(1, min(h, w) // 32)
    stride = max(1, int(stride))

    yy, xx = np.mgrid[0:h, 0:w]
    panels = [
        ("Pred omega* + velocity", display_pred_omega, display_pred_uvw[0], display_pred_uvw[1], False),
        ("GT omega* + velocity", display_gt_omega, display_gt_uvw[0], display_gt_uvw[1], False),
        ("Delta omega* + velocity error", display_delta_omega, display_delta_uv[0], display_delta_uv[1], True),
    ]
    omega_min, omega_max = _finite_color_limits_from_arrays(
        [display_pred_omega, display_gt_omega],
        symmetric=True,
        fallback=(-2.0, 2.0),
    )
    delta_min, delta_max = _finite_color_limits_from_arrays(
        [display_delta_omega],
        symmetric=True,
        fallback=(-1.0, 1.0),
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=140, facecolor="w")
    for ax, (title, omega, u_field, v_field, is_delta) in zip(axes, panels):
        vmin, vmax = (delta_min, delta_max) if is_delta else (omega_min, omega_max)
        im = ax.imshow(
            _finite_display_field(omega, fill_value=0.0),
            origin="lower",
            # 涡度/速度场图片使用和当前 flow 图一致的色条风格：
            # TBL/TWCF 是 viridis，普通 256 数据集是 jet。
            cmap=cmap_name,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        u_show = _finite_display_field(u_field, fill_value=0.0)
        v_show = _finite_display_field(v_field, fill_value=0.0)
        ax.quiver(
            xx[::stride, ::stride],
            yy[::stride, ::stride],
            u_show[::stride, ::stride],
            v_show[::stride, ::stride],
            color="k",
            pivot="mid",
            angles="xy",
            scale_units="xy",
            scale=0.25,
            width=0.003,
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, w - 1)
        ax.set_ylim(0, h - 1)
        ax.set_xticks([])
        ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.set_ylabel("omega*", fontsize=10)

    fig.tight_layout()
    # evaluate_all 兼容命名。
    fig.savefig(sample_dir / "vorticity_quiver.png", bbox_inches="tight")
    # 显式命名，强调第三列是涡度/速度差。
    fig.savefig(sample_dir / "vorticity_velocity_delta_compare.png", bbox_inches="tight")
    # 兼容用户常用的 vorticity_quiver_delta 命名，内容与上面的 delta compare 图一致。
    fig.savefig(sample_dir / "vorticity_quiver_delta.png", bbox_inches="tight")
    plt.close(fig)


def _save_flow_visual_artifacts(
    sample_dir,
    pred_chw,
    gt_chw,
    cmap_name="jet",
    quiver_stride=None,
    save_npy=True,
    dataset_name=None,
    mask_2d=None,
    tbl_y_limit=None,
):
    """
    汇总保存 test_all 的光流样本级图和 NPY。

    该函数只负责新增的 evaluate_all 同款样本文件，不改动原有
    `{dataset_name}_sample_XXXX.png` 总览图，避免影响现有脚本读取路径。
    """
    pred_uvw, gt_uvw, delta_uvw = _save_uvw_compare_artifacts(
        sample_dir,
        pred_chw,
        gt_chw,
        cmap_name=cmap_name,
        save_npy=save_npy,
        dataset_name=dataset_name,
        mask_2d=mask_2d,
        tbl_y_limit=tbl_y_limit,
    )
    # 补齐 evaluate_all 的样本级误差直方图文件：每个 sample 都输出 Δu/Δv/Δw 的 png 和 npy。
    # 这些属于误差分析结果，和 delta_u.npy 一样始终保存，方便后续直接画统计图。
    _save_sample_flow_error_histograms(sample_dir, delta_uvw, save_npy=save_npy)
    _save_vorticity_velocity_artifacts(
        sample_dir,
        pred_uvw,
        gt_uvw,
        stride=quiver_stride,
        save_npy=save_npy,
        cmap_name=cmap_name,
        dataset_name=dataset_name,
        mask_2d=mask_2d,
        tbl_y_limit=tbl_y_limit,
    )


def _histogram_matrix(values, bins):
    """
    保存成 `[bin_center, count]` 两列矩阵，和 evaluate_all 的直方图 npy 结构保持一致。
    """
    finite_values = np.asarray(values, dtype=np.float32).reshape(-1)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        counts = np.zeros(len(bins) - 1, dtype=np.float32)
        edges = bins
    else:
        counts, edges = np.histogram(finite_values, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return np.stack([centers.astype(np.float32), counts.astype(np.float32)], axis=1)


def _symmetric_histogram_matrix(values, bins=201):
    """
    以 0 为中心统计有符号误差分布，用于 Δu/Δv/Δw。
    """
    finite_values = np.asarray(values, dtype=np.float32).reshape(-1)
    finite_values = finite_values[np.isfinite(finite_values)]
    max_abs = float(np.max(np.abs(finite_values))) + 1e-12 if finite_values.size > 0 else 1e-12
    edges = np.linspace(-max_abs, max_abs, bins + 1, dtype=np.float32)
    return _histogram_matrix(finite_values, edges)


def _epe_histogram_matrix(values, bins=201):
    """统计非负 EPE 分布，输出格式与 evaluate_all 的 epe_hist_all.npy 一致。"""
    finite_values = np.asarray(values, dtype=np.float32).reshape(-1)
    finite_values = finite_values[np.isfinite(finite_values)]
    max_val = float(np.max(finite_values)) + 1e-12 if finite_values.size > 0 else 1e-12
    edges = np.linspace(0.0, max_val, bins + 1, dtype=np.float32)
    return _histogram_matrix(finite_values, edges)


def _save_histogram_plot(hist_matrix, out_png, title, xlabel, color="#4C9F70"):
    """
    将两列直方图矩阵额外画成 png。

    evaluate_all 原本至少保存 npy；test_all 这里在保存同款 npy 的基础上也补一张图，
    方便不用写后处理脚本就能快速检查类别级/总体级误差分布。
    """
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


def _save_sample_flow_error_histograms(sample_dir, delta_uvw, save_npy=True):
    """
    保存单个 sample 的 U/V/W 误差直方图。

    evaluate_all 在每个样本目录下会保存 `delta_u_error.png` 等误差分布图；
    test_all 这里复用同样的命名，并额外保存两列矩阵格式的 hist npy。
    注意：这里的 W 对应位移幅值 S/magnitude，是从 2 通道 uv 光流计算出来的第三个分量。
    """
    sample_dir = Path(sample_dir)
    delta_uvw = np.asarray(delta_uvw, dtype=np.float32)
    specs = [
        ("delta_u", delta_uvw[0], "Delta U Error Distribution", "delta u displacement [px]", "#F4A142"),
        ("delta_v", delta_uvw[1], "Delta V Error Distribution", "delta v displacement [px]", "#4C9F70"),
        ("delta_w", delta_uvw[2], "Delta W Error Distribution", "delta w displacement [px]", "#8E6BBE"),
    ]
    for prefix, values, title, xlabel, color in specs:
        hist = _symmetric_histogram_matrix(values)
        # hist/误差 NPY 属于诊断结果，不受 IS_SAVE_NPY 关闭影响。
        _save_optional_npy(sample_dir / f"{prefix}_hist.npy", hist.astype(np.float32), save_npy, force=True)
        _save_histogram_plot(hist, sample_dir / f"{prefix}_error.png", title=title, xlabel=xlabel, color=color)


def _save_flow_histogram_bundle(out_dir, delta_u_values, delta_v_values, delta_w_values, epe_values):
    """
    保存 evaluate_all 同款的 Δu/Δv/Δw/EPE 汇总直方图 npy，并额外输出 png 快速预览。

    out_dir 可以是单个 dataset 目录，也可以是 test_all 根目录，因此同一函数同时服务
    “分类别”和“总体”两级输出。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("delta_u", delta_u_values, _symmetric_histogram_matrix, "Delta U Error Distribution", "delta u displacement [px]", "#F4A142"),
        ("delta_v", delta_v_values, _symmetric_histogram_matrix, "Delta V Error Distribution", "delta v displacement [px]", "#4C9F70"),
        ("delta_w", delta_w_values, _symmetric_histogram_matrix, "Delta W Error Distribution", "delta w displacement [px]", "#8E6BBE"),
        ("epe", epe_values, _epe_histogram_matrix, "EPE Distribution", "EPE [px]", "#4477AA"),
    ]
    for prefix, values, builder, title, xlabel, color in specs:
        if not values:
            continue
        merged = np.concatenate([np.asarray(v, dtype=np.float32).reshape(-1) for v in values], axis=0)
        hist = builder(merged)
        np.save(out_dir / f"{prefix}_hist_all.npy", hist.astype(np.float32))
        _save_histogram_plot(hist, out_dir / f"{prefix}_hist_all.png", title=title, xlabel=xlabel, color=color)


def _save_energy_spectrum_plot(pred_curve, gt_curve, out_png, title):
    """保存 mean energy spectrum 对比图，文件名和 evaluate_all 的均值频谱图保持同一套口径。"""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    k = np.arange(1, len(pred_curve) + 1)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=160)
    ax.loglog(k, np.maximum(gt_curve, 1e-12), label="GT", linewidth=2)
    ax.loglog(k, np.maximum(pred_curve, 1e-12), label="Pred", linewidth=2, linestyle="--")
    ax.set_xlabel("Wavenumber k")
    ax.set_ylabel("E(k)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _save_energy_spectrum_mse_compare_plot(rows, out_png, title, value_key="energy_spectrum_mse"):
    """
    将 metrics 行里的 energy_spectrum_mse 画成对比图。

    - dataset 级：横轴是 sample_index，用来观察每个 sample 的频谱误差波动；
    - test_all 根目录：横轴是 dataset 名，用来比较不同类别的均值频谱误差。
    这里只保存 png，不新增普通中间 npy，因此不会和 IS_SAVE_NPY 的控制规则冲突。
    """
    out_png = Path(out_png)
    finite_rows = []
    for row in rows or []:
        try:
            value = float(row.get(value_key, float("nan")))
        except (TypeError, ValueError):
            value = float("nan")
        if np.isfinite(value):
            finite_rows.append((row, value))
    if not finite_rows:
        return

    grouped = {}
    for row, value in finite_rows:
        label = str(row.get("pair_type", "metric"))
        grouped.setdefault(label, []).append((row, value))

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.2), dpi=160)
    all_values = []
    all_numeric_x = True
    label_positions = []
    label_names = []

    for group_idx, (label, entries) in enumerate(grouped.items()):
        x_values = []
        y_values = []
        numeric_x = True
        for fallback_idx, (row, value) in enumerate(entries):
            sample_index = row.get("sample_index", fallback_idx)
            try:
                x_value = float(sample_index)
            except (TypeError, ValueError):
                # 根目录的 CLASS_MEAN 行没有数字 sample_index，用 dataset 名作为横轴标签。
                numeric_x = False
                x_value = float(len(label_positions))
                label_positions.append(x_value)
                label_names.append(str(row.get("dataset", sample_index)))
            x_values.append(x_value)
            y_values.append(value)
            all_values.append(value)

        all_numeric_x = all_numeric_x and numeric_x
        ax.plot(x_values, y_values, marker="o", linewidth=1.6, markersize=3.5, label=label)

    finite_values = np.asarray(all_values, dtype=np.float32)
    if finite_values.size > 0:
        mean_value = float(np.mean(finite_values))
        ax.axhline(mean_value, color="k", linestyle=":", linewidth=1.1, label=f"mean={mean_value:.4g}")
        positive = finite_values[finite_values > 0]
        if positive.size > 1 and float(np.max(positive) / max(np.min(positive), 1e-12)) > 100.0:
            ax.set_yscale("log")

    if not all_numeric_x and label_positions:
        ax.set_xticks(label_positions)
        ax.set_xticklabels(label_names, rotation=30, ha="right")

    ax.set_xlabel("sample index" if all_numeric_x else "dataset")
    ax.set_ylabel("Energy spectrum MSE")
    ax.set_title(title)
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _save_mean_spectrum(pred_curves, gt_curves, out_dir, title, file_prefix, also_save_legacy_names=False, save_npy=True):
    """
    保存 evaluate_all 同款的平均能量谱 npy/png。

    test_all 的 sample 图已经有自己的布局，因此这里只补“类别级/总体级”的频谱汇总；
    文件名沿用 evaluate_all：`{prefix}_energy_spectrum_*_mean.npy/png`。
    """
    if not pred_curves or not gt_curves:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    min_len = min(min(len(x) for x in pred_curves), min(len(x) for x in gt_curves))
    pred_mean = np.mean(np.stack([np.asarray(x[:min_len], dtype=np.float32) for x in pred_curves], axis=0), axis=0)
    gt_mean = np.mean(np.stack([np.asarray(x[:min_len], dtype=np.float32) for x in gt_curves], axis=0), axis=0)
    # 均值频谱 NPY 属于普通中间结果，由 IS_SAVE_NPY 控制；PNG 仍按原逻辑保存。
    _save_optional_npy(out_dir / f"{file_prefix}_energy_spectrum_pred_mean.npy", pred_mean.astype(np.float32), save_npy)
    _save_optional_npy(out_dir / f"{file_prefix}_energy_spectrum_gt_mean.npy", gt_mean.astype(np.float32), save_npy)
    _save_energy_spectrum_plot(
        pred_mean,
        gt_mean,
        out_dir / f"{file_prefix}_energy_spectrum_mean_compare.png",
        title=title,
    )
    if also_save_legacy_names:
        _save_optional_npy(out_dir / "energy_spectrum_pred_mean.npy", pred_mean.astype(np.float32), save_npy)
        _save_optional_npy(out_dir / "energy_spectrum_gt_mean.npy", gt_mean.astype(np.float32), save_npy)
        _save_energy_spectrum_plot(pred_mean, gt_mean, out_dir / "energy_spectrum_mean_compare.png", title=title)


def _build_mean_row(rows, fixed_fields, metric_keys, global_data=None):
    """根据给定指标列生成一行均值记录。"""
    mean_row = dict(fixed_fields)
    for key in metric_keys:
        vals = []
        for row in rows:
            value = row.get(key, float("nan"))
            try:
                value = float(value)
            except Exception:
                value = float("nan")
            if np.isfinite(value):
                vals.append(value)
        # test_all 的逐样本 rows 原样保留；只有 MEAN/CLASS_MEAN 等汇总行使用稳健均值，
        # 避免个别异常失败样本把整体平均指标拉得过差。
        mean_row[key] = robust_metric_mean(vals, metric_key=key, global_data=global_data)
    return mean_row


def _write_rows_with_mean(path, rows, fixed_fields, metric_keys, global_data=None):
    """
    写入明细行 + 均值行。

    返回值：
        mean_row: 方便调用方继续把 dataset 级均值拼到 root metrics_all.csv。
    """
    if not rows:
        return None
    mean_row = _build_mean_row(rows, fixed_fields, metric_keys, global_data=global_data)
    _write_csv(path, rows + [mean_row])
    return mean_row


def _find_first_conv2d(module):
    """
    在模块里递归找到第一层 Conv2d。

    test_all 需要知道“这个分支的 SR 生成器训练时期待几通道图像”。
    对当前四个分支来说，最可靠的信息就是生成器第一层卷积的 in_channels。
    """
    if module is None:
        return None
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return child
    return None


def _infer_model_image_channels(model):
    """
    推断当前模型在 SR 分支期望的图像通道数。

    背景：
        - TFRecord 测试集里的 target 存的是 2 通道：[prev_gray, next_gray]；
        - 但这几个分支训练时的 Generator 常常按 inner_chanel=3 初始化，
          即 previous/next 每帧都会被当成 3 通道图像来处理；
        - 如果 test_all 直接喂 1 通道 patch，就会在第一层卷积处报
          “expected input to have 3 channels, but got 1 channels”。

    因此这里统一从 Generator 第一层卷积读取 in_channels，作为测试阶段的适配目标。
    如果模型上没有 Generator，则回退到 1，保持最保守行为。
    """
    generator = getattr(model, "piv_esrgan_generator", None)
    first_conv = _find_first_conv2d(generator)
    if first_conv is None:
        return 1
    return int(first_conv.in_channels)


def _adapt_image_channels_for_model(image_bchw, expected_channels):
    """
    把测试图像适配到模型训练时的通道口径。

    当前 test_all 的原始图片是单通道颗粒图 [B, 1, H, W]。如果模型训练时使用
    inner_chanel=3，那么这里会把单通道复制成 3 通道，再送入 Generator / VGG / GAN
    分支，保证前向逻辑和训练时一致。

    说明：
        - 1 -> 3：直接 repeat，最符合“灰度图复制到 RGB 三通道”的训练习惯；
        - N -> 1：对通道取均值，保持灰度意义；
        - 其他情况：尽量按 repeat + 截断适配，避免因为测试集通道数和实验口径不同而崩溃。
    """
    if image_bchw.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape={tuple(image_bchw.shape)}")

    current_channels = int(image_bchw.shape[1])
    expected_channels = int(expected_channels)
    if current_channels == expected_channels:
        return image_bchw
    if expected_channels <= 0:
        raise ValueError(f"expected_channels must be positive, got {expected_channels}")

    if current_channels == 1 and expected_channels > 1:
        return image_bchw.repeat(1, expected_channels, 1, 1)
    if expected_channels == 1 and current_channels > 1:
        return image_bchw.mean(dim=1, keepdim=True)
    if expected_channels < current_channels:
        return image_bchw[:, :expected_channels, :, :]

    repeat_times = int(math.ceil(expected_channels / current_channels))
    return image_bchw.repeat(1, repeat_times, 1, 1)[:, :expected_channels, :, :]


def _collapse_image_to_single_channel_for_test(image_bchw):
    """
    把模型输出图像压回 test_all 统一使用的单通道口径。

    test_all 的保存图、图像指标、对比图都围绕“单通道 prev / next 颗粒图”设计。
    因此如果某个分支的 Generator 输出为 3 通道，这里统一做通道均值，得到单通道图：
        1. 与 TFRecord 的原始 ground truth 口径一致；
        2. 不改模型内部 RAFT / loss 的真实计算逻辑；
        3. 避免后续可视化代码只取第 0 通道时混入通道偏差。
    """
    if image_bchw.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape={tuple(image_bchw.shape)}")
    if int(image_bchw.shape[1]) == 1:
        return image_bchw
    return image_bchw.mean(dim=1, keepdim=True)


def _load_dali_modules():
    """
    延迟导入 DALI。

    pipeline 导入 test_all 时不一定马上运行 TFRecord 测试；把 DALI 放到运行阶段导入，
    可以避免没有安装 DALI 的普通训练/验证环境在 import 阶段就失败。
    """
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.tfrecord as tfrec
    from nvidia.dali.plugin.pytorch import DALIGenericIterator

    return Pipeline, ops, tfrec, DALIGenericIterator


def _triang(window_size):
    """生成与 scipy.signal.triang 等价的三角窗，SciPy 新旧版本路径不同，因此这里做兼容。"""
    try:
        from scipy.signal import triang

        return triang(window_size)
    except Exception:
        try:
            from scipy.signal.windows import triang

            return triang(window_size)
        except Exception:
            return np.bartlett(window_size)


def _spline_window(window_size, power=2):
    """RAFT256-PIV 测试脚本同款 squared spline 窗，用于重叠 patch 加权融合。"""
    intersection = int(window_size / 4)
    wind_outer = (abs(2 * _triang(window_size)) ** power) / 2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2 * (_triang(window_size) - 1)) ** power) / 2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind


def _window_2d(window_size, power=2):
    """缓存 2D spline window，避免每个 batch 重复计算。"""
    key = f"{window_size}_{power}"
    if key not in _CACHED_2D_WINDOWS:
        wind = _spline_window(window_size, power)
        wind = np.expand_dims(np.expand_dims(wind, -1), -1)
        _CACHED_2D_WINDOWS[key] = wind * wind.transpose(1, 0, 2)
    return _CACHED_2D_WINDOWS[key]


def _build_tfrecord_pipeline_class():
    """
    构造和 RAFT256-PIV_test.py 保持一致的 DALI TFRecord 读取 pipeline。

    TFRecord 中：
    - target: [2, H, W]，第 0 通道是 prev，第 1 通道是 next
    - flow:   [2, H, W]，光流真值
    - label:  保留读取但 test_all 不参与计算，保持原脚本数据结构不变
    """
    Pipeline, ops, tfrec, _ = _load_dali_modules()

    class TFRecordPipeline(Pipeline):
        def __init__(
            self,
            batch_size,
            num_threads,
            device_id,
            tfrecord,
            tfrecord_idx,
            image_shape,
            label_shape,
            is_shuffle=False,
        ):
            super().__init__(
                batch_size,
                num_threads,
                device_id,
                exec_pipelined=False,
                exec_async=False,
            )
            self.input = ops.TFRecordReader(
                path=tfrecord,
                index_path=tfrecord_idx,
                random_shuffle=is_shuffle,
                pad_last_batch=True,
                shard_id=0,
                num_shards=1,
                features={
                    "target": tfrec.FixedLenFeature([], tfrec.string, ""),
                    "label": tfrec.FixedLenFeature([], tfrec.string, ""),
                    "flow": tfrec.FixedLenFeature([], tfrec.string, ""),
                },
            )
            self.decode = ops.PythonFunction(function=self.extract_view, num_outputs=1)
            self.reshape_image = ops.Reshape(shape=image_shape)
            self.reshape_label = ops.Reshape(shape=label_shape)

        def extract_view(self, data):
            return data.view("<f4")

        def define_graph(self):
            inputs = self.input(name="Reader")
            images = self.reshape_image(self.decode(inputs["target"]))
            labels = self.reshape_label(self.decode(inputs["label"]))
            flows = self.reshape_image(self.decode(inputs["flow"]))
            return images, labels, flows

    return TFRecordPipeline


def _as_cuda_device(device):
    """把 global_data 中的 device 配置统一转换成 torch.device 与 DALI 需要的 int gpu id。"""
    if device is None:
        device = torch.device("cuda", 0)
    elif isinstance(device, int):
        device = torch.device("cuda", device)
    elif isinstance(device, str):
        device = torch.device(device)

    if device.type != "cuda":
        raise RuntimeError("test_all 按单 GPU CUDA 流程运行，请将 global_data.esrgan.device 设置为 cuda。")

    if device.index is None:
        device_id = torch.cuda.current_device()
        device = torch.device("cuda", device_id)
    else:
        device_id = device.index

    torch.cuda.set_device(device_id)
    return device, device_id


def _resolve_path(raw_path, global_data):
    """
    解析全局变量中的 TFRecord / idx / TWCF 辅助文件路径。

    优先使用用户配置的原路径；如果是相对路径，再依次尝试当前工作目录、SRGAN 根目录、
    study 根目录和 AUTODL_DATA_PATH。这样既兼容 RAFT256-PIV_test.py 的 ../data 写法，
    也方便在不同启动目录下运行 pipeline。
    """
    path = Path(raw_path)
    if path.is_absolute():
        return path

    this_file = Path(__file__).resolve()
    candidates = [
        Path.cwd() / path,
        this_file.parents[1] / path,  # .../study/SRGAN
        this_file.parents[2] / path,  # .../study
    ]
    autodl_data_path = getattr(global_data.esrgan, "AUTODL_DATA_PATH", None)
    if autodl_data_path:
        candidates.append(Path(autodl_data_path) / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _ensure_tfrecord_idx(tfrecord_path, idx_path, global_data):
    """idx 不存在时复用 RAFT256-PIV_test.py 的 tfrecord2idx 生成方式。"""
    if idx_path.exists():
        return

    script = getattr(global_data.esrgan, "TEST_TFRECORD2IDX_SCRIPT", "tfrecord2idx")
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"[test_all] idx 不存在，开始生成：{idx_path}")
    call([script, str(tfrecord_path), str(idx_path)])


def _build_dali_iterator(dataset_cfg, global_data, device_id):
    """根据单个 dataset 的全局配置创建 DALI iterator。"""
    _, _, _, DALIGenericIterator = _load_dali_modules()
    TFRecordPipeline = _build_tfrecord_pipeline_class()

    tfrecord_path = _resolve_path(dataset_cfg["test_tfrecord"], global_data)
    idx_path = _resolve_path(dataset_cfg["test_tfrecord_idx"], global_data)
    _ensure_tfrecord_idx(tfrecord_path, idx_path, global_data)

    batch_size = int(getattr(global_data.esrgan, "TEST_BATCH_SIZE", 1))
    num_threads = int(getattr(global_data.esrgan, "TEST_NUM_THREADS", 8))
    image_shape = [2, int(dataset_cfg["image_height"]), int(dataset_cfg["image_width"])]
    label_shape = list(dataset_cfg.get("label_shape", [12]))

    pipe = TFRecordPipeline(
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        tfrecord=str(tfrecord_path),
        tfrecord_idx=str(idx_path),
        is_shuffle=False,
        image_shape=image_shape,
        label_shape=label_shape,
    )
    pipe.build()

    size = int(pipe.epoch_size("Reader"))
    iterator = DALIGenericIterator(
        pipe,
        ["target", "label", "flow"],
        size=size,
        last_batch_padded=True,
        fill_last_batch=False,
        auto_reset=True,
    )
    return iterator, size


def _scale_factor_from_scale(scale):
    """训练目录一直使用 scale_{int(SCALE * SCALE)}，测试下采样也保持同一倍率定义。"""
    return max(1, int(scale * scale))


def _make_lr_from_hr_tensor(hr_tensor, factor, device):
    """
    用 data_downscal.py 的 downsample_tif 生成 LR。

    输入是已经归一化后的 HR tensor，形状 [B, 1, H, W]；为了严格复用项目下采样逻辑，
    每个 sample 每个通道转 numpy 后调用 downsample_tif，再转回 CUDA tensor。
    """
    if factor <= 1:
        return hr_tensor.detach().clone()

    lr_images = []
    for sample in hr_tensor.detach().cpu().numpy():
        # sample 形状是 [1, H, W]，downsample_tif 接收单张 2D 图，因此取第 0 通道。
        lr_hw = downsample_tif(
            sample[0],
            factor=factor,
            expand_to_original=False,
            method=DOWNSAMPLE_METHOD,
            interpolation_mode=INTERPOLATION_MODE,
        )
        lr_images.append(lr_hw[None, ...])

    lr = np.stack(lr_images, axis=0).astype(np.float32, copy=False)
    return torch.from_numpy(lr).to(device=device, dtype=hr_tensor.dtype)


def _last_flow_prediction(flow_predictions):
    """兼容 list/tuple/tensor 三种返回形式，统一取最终迭代的 flow。"""
    if isinstance(flow_predictions, (list, tuple)):
        return flow_predictions[-1]
    return flow_predictions


def _predict_patch(model, images_hr, flows_hr, factor, device, flow_init=None):
    """
    对一个 256x256 patch 或普通 256x256 样本做联合模型推理。

    target 的第 0 通道是 prev，第 1 通道是 next；这里先拆 HR，再用 data_downscal.py
    生成 prev_lr/next_lr，最后仍通过 PIV_ESRGAN_RAFT_Model.forward 得到 SR+RAFT 预测。

    额外兼容：
        当前四个分支训练时很多 Generator 是按 inner_chanel=3 初始化的，但 test_all 的
        TFRecord target 是灰度双帧 [prev_gray, next_gray]。因此这里会：
        1. 保留原始单通道 prev/next，供 test_all 的图片保存和指标统计使用；
        2. 自动把 prev/next 的 LR/HR 复制到模型期望的通道数后再送入 forward；
        3. 模型生成出的多通道 SR 图再压回单通道，统一交给 test_all 做后处理。
    """
    # TFRecord 原始双帧是单通道灰度图；这两份 tensor 会原样保留给保存图片/算指标。
    prev_hr = images_hr[:, 0:1, :, :]
    next_hr = images_hr[:, 1:2, :, :]
    prev_lr = _make_lr_from_hr_tensor(prev_hr, factor, device)
    next_lr = _make_lr_from_hr_tensor(next_hr, factor, device)

    # 生成器/VGG/GAN 分支前向必须吃到与训练时一致的通道数，否则会在第一层卷积报错。
    expected_channels = _infer_model_image_channels(model)
    prev_hr_for_model = _adapt_image_channels_for_model(prev_hr, expected_channels)
    next_hr_for_model = _adapt_image_channels_for_model(next_hr, expected_channels)
    prev_lr_for_model = _adapt_image_channels_for_model(prev_lr, expected_channels)
    next_lr_for_model = _adapt_image_channels_for_model(next_lr, expected_channels)

    pred_prev, pred_next, flow_predictions, _ = model(
        input_lr_prev=prev_lr_for_model,
        input_lr_next=next_lr_for_model,
        input_gr_prev=prev_hr_for_model,
        input_gr_next=next_hr_for_model,
        flowl0=flows_hr,
        flow_init=flow_init,
        is_adversarial=False,
    )
    # test_all 的图像输出、对比图和 SR 图像指标都以单通道颗粒图为基准；
    # 这里把多通道 Generator 输出压回单通道，避免后续保存/评估继续带着 3 通道口径。
    pred_prev = _collapse_image_to_single_channel_for_test(pred_prev)
    pred_next = _collapse_image_to_single_channel_for_test(pred_next)
    return {
        "flow": _last_flow_prediction(flow_predictions),  # 最终 RAFT flow 预测
        "prev_lr": prev_lr,  # 用 data_downscal.py 从 HR prev 生成的 LR prev
        "next_lr": next_lr,  # 用 data_downscal.py 从 HR next 生成的 LR next
        "prev_hr": prev_hr,  # TFRecord target 第 0 通道，对应原始 HR prev
        "next_hr": next_hr,  # TFRecord target 第 1 通道，对应原始 HR next
        "pred_prev": pred_prev,  # 联合模型输出的 SR/generated prev
        "pred_next": pred_next,  # 联合模型输出的 SR/generated next
    }


def _fold_weighted_patches(patch_tensor, B, C, H, W, num_y, num_x, offset, shift, window):
    """
    将 patch 预测按 RAFT256-PIV_test.py 的 spline window 加权 fold 回原图尺寸。

    flow 使用 C=2，SR 图像使用 C=1；抽成公共函数后，tbl/twcf 的 flow 和生成图
    都能使用同一套重叠区域融合逻辑，避免图片输出和 flow 输出的拼接方式不一致。
    """
    # squared spline window 的外边界可能正好为 0。full-frame 最外圈像素只被一个 patch 覆盖时，
    # 如果继续使用 0 权重，会在 folded / folding_mask 处产生 0/0，后续绘图再把 NaN 填 0，
    # 就会出现用户看到的 TWCF 顶部紫色/0 displacement 细带。这里仅把 0 权重点抬到极小值，
    # 重叠区域的融合权重不变，外边界则保留该 patch 的真实预测值。
    blend_window = torch.clamp(window, min=1.0e-6)

    weighted_patches = patch_tensor * blend_window
    weighted_patches = weighted_patches.reshape((B, num_y, num_x, C, offset, offset)).permute(0, 3, 1, 2, 4, 5)
    weighted_patches = weighted_patches.contiguous().view(B, C, -1, offset * offset)
    weighted_patches = weighted_patches.permute(0, 1, 3, 2)
    weighted_patches = weighted_patches.contiguous().view(B, C * offset * offset, -1)
    folded = F.fold(weighted_patches, output_size=(H, W), kernel_size=offset, stride=shift)

    mask_source = torch.ones((B, C, H, W), device=patch_tensor.device, dtype=patch_tensor.dtype)
    mask_patches = mask_source.unfold(3, offset, shift).unfold(2, offset, shift)
    mask_patches = mask_patches.contiguous().view(B, C, -1, offset, offset)
    mask_patches = mask_patches * blend_window
    mask_patches = mask_patches.view(B, C, -1, offset * offset)
    mask_patches = mask_patches.permute(0, 1, 3, 2)
    mask_patches = mask_patches.contiguous().view(B, C * offset * offset, -1)
    folding_mask = F.fold(mask_patches, output_size=(H, W), kernel_size=offset, stride=shift)

    # 正常情况下，_predict_full_frame_with_folding 会先 pad 到每个像素都被窗口覆盖；
    # 这里仍保留 clamp 作为最后保护，避免极端尺寸或外部调用产生 0/0。
    return folded / torch.clamp(folding_mask, min=1.0e-6)


def _sliding_full_coverage_size(length, offset, shift):
    """
    计算滑窗能够完整覆盖原始长度所需的 padded 长度。

    PyTorch unfold 只会取完整窗口；例如 TWCF 高度 2160，offset=256，shift=64 时，
    不 padding 会只覆盖到 2112 行，顶部剩余区域没有预测。这里把长度补到 2176，
    让最后一个窗口覆盖原图末端，fold 后再裁回 2160。
    """
    length = int(length)
    offset = int(offset)
    shift = int(shift)
    if length <= offset:
        return offset
    num_windows = int(math.ceil((length - offset) / shift)) + 1
    return (num_windows - 1) * shift + offset


def _pad_full_frame_for_sliding(tensor, padded_h, padded_w):
    """
    将 full-frame tensor 右侧/末端 padding 到滑窗可完整覆盖的尺寸。

    使用 replicate padding 而不是 0 padding，是为了让最边缘 patch 的上下文连续；
    这些 padding 区域只参与边缘 patch 的推理，最终输出会裁回原始 H/W，不会保存到结果里。
    """
    _, _, h, w = tensor.shape
    pad_h = int(padded_h) - int(h)
    pad_w = int(padded_w) - int(w)
    if pad_h <= 0 and pad_w <= 0:
        return tensor
    return F.pad(tensor, (0, max(pad_w, 0), 0, max(pad_h, 0)), mode="replicate")


def _predict_full_frame_with_folding(model, images, flows, factor, device, test_args):
    """
    复刻 RAFT256-PIV_test.py 对 tbl/twcf 的滑窗推理。

    不同点：这里不是直接把 HR 图送 RAFT，而是每个 HR patch 先按当前工程的
    data_downscal.py 逻辑生成 LR patch，再走联合 ESRGAN+RAFT 模型。
    """
    offset = int(test_args["offset"])
    shift = int(test_args["shift"])
    split_size = int(test_args["split_size"])

    B, C, original_h, original_w = images.size()
    padded_h = _sliding_full_coverage_size(original_h, offset, shift)
    padded_w = _sliding_full_coverage_size(original_w, offset, shift)

    # 保存原始 full-frame，后续 LR/HR 图像指标和最终输出都必须回到原图尺寸。
    original_images = images
    images = _pad_full_frame_for_sliding(images, padded_h, padded_w)
    flows = _pad_full_frame_for_sliding(flows, padded_h, padded_w)

    _, _, H, W = images.size()
    num_y = (H - offset) // shift + 1
    num_x = (W - offset) // shift + 1

    predicted_flows = torch.zeros_like(flows, device=device)

    patches = images.unfold(3, offset, shift).unfold(2, offset, shift).permute(0, 2, 3, 1, 5, 4)
    patches = patches.reshape((-1, 2, offset, offset))
    flow_patches = flows.unfold(3, offset, shift).unfold(2, offset, shift).permute(0, 2, 3, 1, 5, 4)
    flow_patches = flow_patches.reshape((-1, 2, offset, offset))

    predicted_flow_patches = predicted_flows.unfold(3, offset, shift).unfold(2, offset, shift).permute(0, 2, 3, 1, 5, 4)
    predicted_flow_patches = predicted_flow_patches.reshape((-1, 2, offset, offset))

    patch_flow_outputs = []
    patch_pred_prev_outputs = []
    patch_pred_next_outputs = []
    split_patches = torch.split(patches, split_size, dim=0)
    split_flows = torch.split(flow_patches, split_size, dim=0)
    split_flow_init = torch.split(predicted_flow_patches, split_size, dim=0)
    for patch, flow_patch, flow_init_patch in zip(split_patches, split_flows, split_flow_init):
        patch_result = _predict_patch(model, patch, flow_patch, factor, device, flow_init=flow_init_patch)
        patch_flow_outputs.append(patch_result["flow"])
        patch_pred_prev_outputs.append(patch_result["pred_prev"])
        patch_pred_next_outputs.append(patch_result["pred_next"])

    window = torch.from_numpy(np.squeeze(_window_2d(window_size=offset, power=2))).to(device=device, dtype=images.dtype)
    predicted_flows = _fold_weighted_patches(
        torch.cat(patch_flow_outputs, dim=0), B, C, H, W, num_y, num_x, offset, shift, window
    )
    pred_prev = _fold_weighted_patches(
        torch.cat(patch_pred_prev_outputs, dim=0), B, 1, H, W, num_y, num_x, offset, shift, window
    )
    pred_next = _fold_weighted_patches(
        torch.cat(patch_pred_next_outputs, dim=0), B, 1, H, W, num_y, num_x, offset, shift, window
    )

    # padding 只用于让滑窗覆盖到原图末端；返回前必须裁回原始 H/W。
    predicted_flows = predicted_flows[:, :, :original_h, :original_w]
    pred_prev = pred_prev[:, :, :original_h, :original_w]
    pred_next = pred_next[:, :, :original_h, :original_w]

    return {
        "flow": predicted_flows,
        # tbl/twcf 的 LR 图保存整张 full-frame 下采样结果；SR/generated 图则来自 patch SR 融合。
        "prev_lr": _make_lr_from_hr_tensor(original_images[:, 0:1, :, :], factor, device),
        "next_lr": _make_lr_from_hr_tensor(original_images[:, 1:2, :, :], factor, device),
        "prev_hr": original_images[:, 0:1, :, :],
        "next_hr": original_images[:, 1:2, :, :],
        "pred_prev": pred_prev,
        "pred_next": pred_next,
    }


def _mask_field_for_plot(field_2d, mask_2d=None):
    """
    按 mask 把无效区域转成 masked array，绘图时显示为空白而不是被最低值颜色染满。
    """
    field = np.asarray(field_2d, dtype=np.float32)
    if mask_2d is None:
        return field
    mask = np.asarray(mask_2d)
    if mask.shape != field.shape:
        return field
    return np.ma.masked_where(mask <= 0, field)


def _fill_invalid_field_for_plot(field_2d, mask_2d=None, fill_value=0.0):
    """
    按 mask 把无效区域填成固定 displacement 值后再绘图。

    TWCF 原始 PascalPIV 图在波浪边界下方不是留白，而是用 0 displacement 对应的颜色填充。
    Current 预测图也需要保持同样视觉口径，否则白色空洞会被误读成没有输出。
    """
    field = np.asarray(field_2d, dtype=np.float32).copy()
    field[~np.isfinite(field)] = float(fill_value)
    if mask_2d is None:
        return field
    mask = np.asarray(mask_2d)
    if mask.shape != field.shape:
        return field
    field[mask <= 0] = float(fill_value)
    return field


def _repair_nonfinite_by_vertical_nearest(field_2d, fallback=0.0):
    """
    将绘图字段中的 NaN/Inf 用同一列最近的有效值补齐。

    TWCF 的 Current 图来自 patch fold，历史输出或极端边界仍可能留下少量非有限值。
    这些点不应该被直接填成 0，因为 0 在 U 色条里会显示成明显的低位移颜色；
    用同列最近有效值补齐可以保持边界连续，只影响 png 可视化，不改模型预测文件。
    """
    field = np.asarray(field_2d, dtype=np.float32).copy()
    if field.ndim != 2:
        field[~np.isfinite(field)] = float(fallback)
        return field

    finite_mask = np.isfinite(field)
    if finite_mask.all():
        return field

    y_positions = np.arange(field.shape[0], dtype=np.float32)
    for col_idx in range(field.shape[1]):
        column = field[:, col_idx]
        valid = np.isfinite(column)
        if valid.all():
            continue
        if valid.any():
            # np.interp 在两端会使用最近端点值外推，正好用于修复 top/bottom 边界 NaN。
            field[:, col_idx] = np.interp(
                y_positions,
                y_positions[valid],
                column[valid],
            ).astype(np.float32)
        else:
            # 极少数整列都无效时才退回固定值，避免 matplotlib 处理非有限数组失败。
            field[:, col_idx] = float(fallback)
    return field


def _bottom_connected_invalid_mask(mask_2d, field_shape):
    """
    从 TWCF mask 中只提取“从底部连通上来”的无效区域。

    TWCF 的论文式 PascalPIV 图只把底部波浪边界以下填成 0 displacement。
    如果 mask 在顶部、孤立噪点或其它区域也标成无效，不能一并填 0，
    否则 Current U/V 顶部会出现本不该存在的 0 displacement 色带。
    """
    if mask_2d is None:
        return None

    mask = np.asarray(mask_2d)
    if mask.shape != tuple(field_shape):
        return None

    invalid = mask <= 0
    bottom_invalid = np.zeros_like(invalid, dtype=bool)
    height, width = invalid.shape

    for col_idx in range(width):
        column_invalid = invalid[:, col_idx]
        # imshow(origin="lower") 下 row=0 是图像底部；只有底部起始就是无效时，
        # 才把连续无效段认定为波浪边界下方的填 0 区域。
        if not column_invalid[0]:
            continue
        first_valid = np.flatnonzero(~column_invalid)
        if first_valid.size == 0:
            bottom_invalid[:, col_idx] = True
        else:
            bottom_invalid[: int(first_valid[0]), col_idx] = True
    return bottom_invalid


def _fill_twcf_bottom_invalid_field_for_plot(field_2d, mask_2d=None, fill_value=0.0):
    """
    TWCF 专用绘图填充：修复 fold 边界，再只填底部波浪无效区。

    这保留了 PascalPIV/TWCF 图中“底部无效区域为 0 displacement”的显示习惯，
    同时避免把顶部有效流场或 fold 产生的边界 NaN 误染成 0。
    """
    field = _repair_nonfinite_by_vertical_nearest(field_2d, fallback=fill_value)
    bottom_invalid = _bottom_connected_invalid_mask(mask_2d, field.shape)
    if bottom_invalid is not None:
        field[bottom_invalid] = float(fill_value)
    return field


def _plot_field_with_colorbar(ax, field_2d, title, cmap_name, vmin, vmax, label, metric_text=None):
    """给单个子图统一绘制位移场和色条。"""
    im = ax.imshow(field_2d, origin="lower", cmap=cmap_name, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=14)
    ax.axis("off")
    # 仅误差图会传 metric_text；Pred/GT 面板保持原样。
    _annotate_metric_box(ax, metric_text, fontsize=13)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.set_ylabel(label, fontsize=12)
    return im


def _plot_twcf(out_path, u_pred, v_pred, piv_results, mask_twcf, sample_index, cmap_name="viridis"):
    """
    保存 TWCF 的 PascalPIV 对比图。

    修改点：
        旧版色条使用 Greys，视觉上是黑白图；这里改成与用户给定示例一致的 viridis 风格色条，
        让位移高低在紫-蓝-绿-黄之间连续过渡，更适合看边界层速度梯度。

        TWCF 的波浪边界下方按照原始 PascalPIV 图的口径填 0 displacement，而不是 mask 成白色。
        填 0 只作用于底部连通的波浪无效区，顶部或孤立无效点不会被误染成 0。
    """
    ref_index = min(int(sample_index), int(piv_results.shape[0]) - 1)
    u_pascal = np.asarray(piv_results[ref_index, 0, :, :], dtype=np.float32)
    v_pascal = np.asarray(piv_results[ref_index, 1, :, :], dtype=np.float32)
    mask_2d = np.asarray(mask_twcf, dtype=np.float32) if mask_twcf is not None else None

    fig, axes = plt.subplots(2, 2, figsize=(24, 16), dpi=120, facecolor="w", edgecolor="k")
    _plot_field_with_colorbar(
        axes[0, 0], _fill_twcf_bottom_invalid_field_for_plot(u_pascal, mask_2d, fill_value=0.0), "PascalPIV U",
        cmap_name, vmin=-2, vmax=12, label="displacement [px]"
    )
    _plot_field_with_colorbar(
        axes[0, 1], _fill_twcf_bottom_invalid_field_for_plot(v_pascal, mask_2d, fill_value=0.0), "PascalPIV V",
        cmap_name, vmin=-1, vmax=1, label="displacement [px]"
    )
    _plot_field_with_colorbar(
        axes[1, 0], _fill_twcf_bottom_invalid_field_for_plot(u_pred, mask_2d, fill_value=0.0), "Current U",
        cmap_name, vmin=-2, vmax=12, label="displacement [px]"
    )
    _plot_field_with_colorbar(
        axes[1, 1], _fill_twcf_bottom_invalid_field_for_plot(v_pred, mask_2d, fill_value=0.0), "Current V",
        cmap_name, vmin=-1, vmax=1, label="displacement [px]"
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _crop_tbl_valid_y(field_2d, y_limit=None):
    """
    裁剪 TBL 可视化使用的有效 y 区域。

    TBL TFRecord 的 full-frame 高度通常是 256，但顶部一段在数据里本来就是 0/无效填充。
    这些 0 会在 U 图里显示成最低色，在 V 图里显示成 0 对应的中间色，容易被误读为模型结果。
    因此 TBL 的图片展示默认只取 y=0..TBL_PROFILE_Y_LIMIT 的有效边界层区域；
    这个裁剪只影响 png 可视化，不改变 results.npy 和所有指标计算。
    """
    field = np.squeeze(np.asarray(field_2d, dtype=np.float32))
    if y_limit is None:
        return field
    crop_h = int(np.clip(int(y_limit), 1, int(field.shape[0])))
    return field[:crop_h, :]


def _annotate_profile_columns(ax, columns, field_height, region_names=None, show_labels=False):
    """
    在 TBL sample 图上标出 Laminar/Transition/Turbulent 的剖面采样位置。

    这些列坐标和 profile_analysis 里抽剖面的列完全一致，因此用户看 sample 对比图时，
    可以直接把红色虚线对应到下面的三张 displacement-vs-y-position 剖面曲线。
    """
    if columns is None:
        return

    labels = tuple(region_names or ())
    label_y = max(float(field_height) - 8.0, 0.0)
    for idx, col in enumerate(columns):
        ax.axvline(int(col), color="red", linestyle="--", linewidth=2.0)
        if not show_labels:
            continue
        label = labels[idx] if idx < len(labels) else f"Region {idx + 1}"
        ax.text(
            int(col),
            label_y,
            label,
            color="red",
            fontsize=13,
            ha="center",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.0},
        )


def _plot_tbl(
    out_path,
    u_pred,
    v_pred,
    u_gt,
    v_gt,
    cmap_name="viridis",
    y_limit=None,
    profile_columns=None,
    profile_region_names=None,
    label_profile_columns=False,
):
    """
    保存 TBL 全图测试的预测/真值对比图。

    这里同样把原来的黑白色条改成 viridis，便于直接和论文式位移场热图保持一致。
    TBL 顶部 0 填充区域不是有效流场，因此按 TBL_PROFILE_Y_LIMIT 裁掉，只影响可视化。

    可选的 profile_columns/profile_region_names 用于在 profile_analysis 文件夹里额外保存
    带 Laminar/Transition/Turbulent 采样虚线的 sample 对比图；普通 sample 图不传入时仍保持原样。
    """
    u_pred_plot = _crop_tbl_valid_y(u_pred, y_limit)
    v_pred_plot = _crop_tbl_valid_y(v_pred, y_limit)
    u_gt_plot = _crop_tbl_valid_y(u_gt, y_limit)
    v_gt_plot = _crop_tbl_valid_y(v_gt, y_limit)

    fig, axes = plt.subplots(2, 2, figsize=(24, 16), dpi=120, facecolor="w", edgecolor="k")
    _plot_field_with_colorbar(
        axes[0, 0], u_pred_plot, "Pred U",
        cmap_name, vmin=2, vmax=8, label="displacement [px]"
    )
    _plot_field_with_colorbar(
        axes[0, 1], v_pred_plot, "Pred V",
        cmap_name, vmin=-0.5, vmax=0.5, label="displacement [px]"
    )
    _plot_field_with_colorbar(
        axes[1, 0], u_gt_plot, "GT U",
        cmap_name, vmin=2, vmax=8, label="displacement [px]"
    )
    _plot_field_with_colorbar(
        axes[1, 1], v_gt_plot, "GT V",
        cmap_name, vmin=-0.5, vmax=0.5, label="displacement [px]"
    )
    if profile_columns is not None:
        # 四个子图都画同一组 x 方向采样虚线；标签只放在左上角 Pred U 图，
        # 避免 2x2 sample 对比图被重复文字遮挡。
        for idx, ax in enumerate(axes.reshape(-1)):
            _annotate_profile_columns(
                ax,
                profile_columns,
                u_gt_plot.shape[0],
                region_names=profile_region_names,
                show_labels=label_profile_columns and idx == 0,
            )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _resolve_tbl_profile_crop_segments(width, columns, crop_width=256):
    """
    计算 TBL 三个剖面位置各自的局部裁剪窗口。

    TBL full-frame 宽度为 3296，但这里不再假设它可以按固定数量均分成 sample。
    profile_analysis 里的 Laminar/Transition/Turbulent 位置是 full-frame 全局列号；
    每个位置都以该列为中心截取一段固定宽度的局部窗口：
        - x_start/x_end: 局部窗口在 full-frame 中的左右边界；
        - local_col: 剖面线在局部窗口内的列号。

    这样保存出来的对比图不是整张 3296 宽图，也不依赖错误的“均分 sample”假设。
    """
    width = int(width)
    crop_width = int(np.clip(int(crop_width), 1, width))
    half_width = crop_width // 2

    segment_rows = []
    for region_idx, global_col in enumerate(columns):
        global_col = int(np.clip(int(global_col), 0, width - 1))
        x_start = global_col - half_width
        x_end = x_start + crop_width
        if x_start < 0:
            x_start = 0
            x_end = crop_width
        if x_end > width:
            x_end = width
            x_start = max(0, width - crop_width)
        local_col = int(np.clip(global_col - x_start, 0, x_end - x_start - 1))
        segment_rows.append([region_idx, x_start, x_end, global_col, local_col, crop_width])

    return np.asarray(segment_rows, dtype=np.int32)


def _safe_filename_token(text):
    """把区域名转换成文件名安全的短 token。"""
    token = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(text))
    token = token.strip("_")
    return token or "region"


def _save_tbl_profile_sample_comparisons(
    analysis_dir,
    u_pred,
    u_gt,
    v_pred,
    v_gt,
    columns,
    region_names,
    profile_height,
    cmap_name,
    crop_width=256,
):
    """
    把 Laminar/Transition/Turbulent 三个位置附近的局部对比图保存到 profile_analysis。

    注意这里不假设 TBL 的 sample 是均分得到的；每个区域都以自己的剖面全局 x 位置为中心，
    截取一段局部窗口。每张图都会在 Pred/GT 的 U/V 四个子图中画出对应位置的红色虚线。
    """
    segment_rows = _resolve_tbl_profile_crop_segments(
        width=np.asarray(u_gt).shape[1],
        columns=columns,
        crop_width=crop_width,
    )
    # TBL 三个位置对应的局部窗口索引是后处理复现实验位置的关键 NPY，
    # 按用户要求不受 IS_SAVE_NPY 影响，始终保存。
    np.save(analysis_dir / "profile_crop_segments.npy", segment_rows)

    for row in segment_rows:
        region_idx, x_start, x_end, global_col, local_col, _ = [int(v) for v in row]
        region_name = region_names[region_idx] if region_idx < len(region_names) else f"Region {region_idx + 1}"
        region_token = _safe_filename_token(region_name)
        out_path = analysis_dir / f"tbl_profile_crop_{region_idx:02d}_{region_token}_x{global_col}_compare.png"
        _plot_tbl(
            out_path,
            u_pred[:, x_start:x_end],
            v_pred[:, x_start:x_end],
            u_gt[:, x_start:x_end],
            v_gt[:, x_start:x_end],
            cmap_name=cmap_name,
            y_limit=profile_height,
            profile_columns=[local_col],
            profile_region_names=[region_name],
            label_profile_columns=True,
        )


def _save_tbl_full_frame_profile_comparison(
    analysis_dir,
    u_pred,
    u_gt,
    v_pred,
    v_gt,
    columns,
    region_names,
    profile_height,
    cmap_name,
):
    """
    保存整张 TBL full-frame 的 2x2 对比图，并标出三条剖面位置。

    局部裁剪图便于看 Laminar/Transition/Turbulent 位置附近的细节；这张 3296 宽 full-frame
    合图则保留全局空间关系，方便确认三条红色虚线在整幅 TBL 场中的相对位置。
    """
    _plot_tbl(
        analysis_dir / "tbl_full_frame_compare_with_profile_positions.png",
        u_pred,
        v_pred,
        u_gt,
        v_gt,
        cmap_name=cmap_name,
        y_limit=profile_height,
        profile_columns=columns,
        profile_region_names=region_names,
        label_profile_columns=True,
    )


def _resolve_profile_columns(width, column_ratios):
    """把论文风格图的三个剖面位置从比例转换成像素列坐标。"""
    cols = []
    for ratio in column_ratios:
        ratio = float(ratio)
        col = int(round((width - 1) * ratio))
        cols.append(int(np.clip(col, 0, width - 1)))
    return cols


def _extract_tbl_component_profiles(gt_profile, pred_profile, columns):
    """
    在 Laminar/Transition/Turbulent 三个 x 位置抽取单个流场分量的剖面。

    gt_profile / pred_profile 是已经按 y_limit 裁剪后的二维场，columns 是 full-frame
    对应的全局列坐标。这里同时要求 GT 和当前方法都为有限值，避免无效像素把剖面曲线
    或后续 npy 对比数据污染成 NaN。
    """
    profile_height = int(gt_profile.shape[0])
    profile_gt = np.full((len(columns), profile_height), np.nan, dtype=np.float32)
    profile_pred = np.full((len(columns), profile_height), np.nan, dtype=np.float32)
    for idx, col in enumerate(columns):
        valid = np.isfinite(gt_profile[:, col]) & np.isfinite(pred_profile[:, col])
        profile_gt[idx, valid] = gt_profile[valid, col]
        profile_pred[idx, valid] = pred_profile[valid, col]
    return profile_gt, profile_pred


def _resolve_tbl_profile_xlim(profile_gt, profile_pred, x_zero_left=False):
    """
    计算 TBL 剖面图的 x 轴范围。

    这个逻辑原本写在 `_save_tbl_component_profile_figure()` 内部。现在抽出来，
    是为了让 `tbl_v_profile_compare.png` 可以直接复用 `tbl_u_profile_compare.png`
    的 x 轴范围，避免 U/V 两张图因为各自数据幅值不同而视觉尺度不一致。
    """
    profile_gt = np.asarray(profile_gt, dtype=np.float32)
    profile_pred = np.asarray(profile_pred, dtype=np.float32)
    x_values = np.concatenate(
        [
            profile_gt[np.isfinite(profile_gt)],
            profile_pred[np.isfinite(profile_pred)],
        ],
        axis=0,
    ) if np.isfinite(profile_gt).any() or np.isfinite(profile_pred).any() else np.asarray([0.0, 1.0], dtype=np.float32)
    x_min = float(np.nanmin(x_values))
    x_max = float(np.nanmax(x_values))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
        x_min, x_max = 0.0, 1.0
    x_pad = 0.05 * max(x_max - x_min, 1.0)
    if x_zero_left:
        return 0.0, max(x_max + x_pad, 1.0)

    # 非强制左端为 0 的情况保留旧逻辑：显式包含 0，避免正负位移曲线被裁掉。
    x_left = min(x_min - x_pad, 0.0)
    x_right = max(x_max + x_pad, 0.0)
    if x_left == x_right:
        x_left, x_right = x_left - 0.5, x_right + 0.5
    return x_left, x_right


def _save_tbl_component_profile_figure(
    analysis_dir,
    output_names,
    top_field,
    profile_gt,
    profile_pred,
    columns,
    y_positions,
    profile_height,
    region_names,
    method_label,
    cmap_name,
    top_title,
    x_zero_left=False,
    forced_xlim=None,
):
    """
    保存 TBL 单个分量的论文风格剖面对比图。

    U/V 两个分量的图形结构完全一致：
        1. 上方显示对应 GT 分量场，并标注 Laminar/Transition/Turbulent 三个 x 位置；
        2. 下方分别画三处位置的 GT 与当前方法 displacement-vs-y-position 剖面。

    x_zero_left=True 会把 x 轴左端固定到 0；
    forced_xlim 不为空时优先使用外部传入范围，用于让 U/V 两张剖面图共用同一套 x 轴。
    """
    output_names = [output_names] if isinstance(output_names, str) else list(output_names)
    top_field = np.asarray(top_field, dtype=np.float32)
    valid_values = top_field.reshape(-1)
    valid_values = valid_values[np.isfinite(valid_values)]
    if valid_values.size > 0:
        vmin = float(np.nanpercentile(valid_values, 1))
        vmax = float(np.nanpercentile(valid_values, 99))
    else:
        vmin, vmax = 0.0, 1.0
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0.0, 1.0

    # 横向稍微放宽，避免 Laminar/Transition 两个相邻标签贴在一起，
    # 但三条红色虚线的位置仍严格由 TBL_PROFILE_COLUMN_RATIOS 控制。
    fig = plt.figure(figsize=(16, 14), dpi=160, facecolor="w", edgecolor="k")
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.8], hspace=0.35, wspace=0.25)

    ax_top = fig.add_subplot(grid[0, :])
    # 顶部图显示当前分析分量的 GT 位移场；TBL 不做无效区填 0，
    # TWCF 的波浪边界填 0 只用于 _plot_twcf。
    im = ax_top.imshow(top_field, origin="lower", cmap=cmap_name, vmin=vmin, vmax=vmax, aspect="auto")
    ax_top.set_title(top_title, fontsize=18)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    for idx, col in enumerate(columns):
        ax_top.axvline(col, color="red", linestyle="--", linewidth=2.0)
        label = region_names[idx] if idx < len(region_names) else f"Region {idx + 1}"
        ax_top.text(
            col,
            profile_height - 8,
            label,
            color="red",
            fontsize=16,
            ha="center",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.5},
        )
    cbar = fig.colorbar(im, ax=ax_top, orientation="horizontal", fraction=0.08, pad=0.18)
    cbar.set_label("Displacement[px]", fontsize=15)

    if forced_xlim is not None:
        # forced_xlim 用于把 tbl_v_profile_compare.png 的 x 轴范围对齐到
        # tbl_u_profile_compare.png，便于同一位置的 U/V 剖面在视觉尺度上直接比较。
        x_left, x_right = [float(v) for v in forced_xlim]
        if not np.isfinite(x_left) or not np.isfinite(x_right) or x_left == x_right:
            x_left, x_right = 0.0, 1.0
    else:
        x_left, x_right = _resolve_tbl_profile_xlim(profile_gt, profile_pred, x_zero_left=x_zero_left)

    for idx, col in enumerate(columns):
        ax = fig.add_subplot(grid[1, idx])
        valid_gt = np.isfinite(profile_gt[idx])
        valid_pred = np.isfinite(profile_pred[idx])
        if np.any(valid_gt):
            ax.plot(
                profile_gt[idx][valid_gt],
                y_positions[valid_gt],
                color="black",
                linewidth=2.0,
                linestyle=(0, (6, 3)),
                label="GT",
            )
        if np.any(valid_pred):
            ax.plot(
                profile_pred[idx][valid_pred],
                y_positions[valid_pred],
                color="red",
                linewidth=2.2,
                linestyle="-",
                label=method_label,
            )
        title = region_names[idx] if idx < len(region_names) else f"Region {idx + 1}"
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.set_xlabel("displacement[px]", fontsize=12)
        if idx == 0:
            ax.set_ylabel("y-position[px]", fontsize=12)
            ax.legend(loc="upper left", fontsize=11, frameon=True)
        ax.set_xlim(x_left, x_right)
        # y_limit=200 表示论文图展示 0..200 px 的坐标范围；profile 数组本身有 200 行，
        # 行索引最大是 199。这里把坐标轴上限显式设为 200，并强制加入 200 刻度，
        # 避免 Matplotlib 自动刻度停在 175，看起来像没有画满 0..200。
        y_ticks = list(np.arange(0, profile_height + 1, 25, dtype=np.int32))
        if y_ticks[-1] != profile_height:
            y_ticks.append(profile_height)
        ax.set_ylim(0, profile_height)
        ax.set_yticks(y_ticks)
        ax.grid(alpha=0.15)

    for output_name in output_names:
        fig.savefig(analysis_dir / output_name, bbox_inches="tight")
    plt.close(fig)


def _save_tbl_profile_artifacts(
    dataset_dir,
    sample_index,
    u_pred,
    u_gt,
    v_pred,
    v_gt,
    method_label,
    cmap_name="viridis",
    column_ratios=(0.15, 0.265, 0.83),
    region_names=("Laminar", "Transition", "Turbulent"),
    y_limit=200,
    sample_crop_width=256,
):
    """
    保存 TBL 的论文风格剖面对比图，并把关键数据落成 .npy 便于和其他方法后处理比较。

    图像结构：
        1. 上半部分：GT 水平位移场 + 三条红色虚线采样位置；
        2. 下半部分：三个位置的 horizontal/U displacement 剖面，只画 GT 与当前方法。

    注意：
        profile_analysis 是 TBL 专属分析，不给 TWCF 生成。
        上半部分显示 horizontal/U 位移场，用来标注 Laminar、Transition、Turbulent 三个 x 位置；
        下半部分的 x 轴是对应列上的 U displacement，y 轴是光流场的空间 y-position。

        TBL 的论文图只展示靠近壁面的有效 y 区域，默认取 0..200px。
        这个裁剪只用于 profile_analysis 的画图和剖面保存，不影响 test_all 的流场指标、
        results.npy，也不影响普通 tbl_sample_xxxx.png 的 2x2 全图对比。

        额外保存的局部对比图不是整张 3296 宽 full-frame，也不按 sample 均分。
        它会以 Laminar/Transition/Turbulent 的全局 x 位置为中心各截取一段窗口。
    """
    analysis_dir = dataset_dir / "profile_analysis" / f"sample_{sample_index:04d}"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    u_pred = np.asarray(u_pred, dtype=np.float32)
    u_gt = np.asarray(u_gt, dtype=np.float32)
    v_pred = np.asarray(v_pred, dtype=np.float32)
    v_gt = np.asarray(v_gt, dtype=np.float32)

    # TBL 论文风格图只看 y=0 开始的有效边界层区域。
    # 之前把完整 256px 高度都画进去时，顶部额外区域会把 Transition 标注和参考图视觉位置拉偏；
    # 这里默认裁到 200px，和参考图的 y-position 范围一致。
    if y_limit is None:
        profile_height = int(u_gt.shape[0])
    else:
        profile_height = int(np.clip(int(y_limit), 1, int(u_gt.shape[0])))
    u_gt_profile = u_gt[:profile_height, :]
    u_pred_profile = u_pred[:profile_height, :]
    v_gt_profile = v_gt[:profile_height, :]
    v_pred_profile = v_pred[:profile_height, :]

    columns = _resolve_profile_columns(u_gt_profile.shape[1], column_ratios)
    y_positions = np.arange(profile_height, dtype=np.float32)

    # 旧版本只抽 horizontal/U displacement。这里保留 U 的旧文件名，
    # 同时新增 V displacement 的同结构剖面，方便后续按同样列号做多方法对比。
    profile_gt, profile_pred = _extract_tbl_component_profiles(u_gt_profile, u_pred_profile, columns)
    v_profile_gt, v_profile_pred = _extract_tbl_component_profiles(v_gt_profile, v_pred_profile, columns)

    # 保存原始场和已经抽好的剖面。profile_analysis 是 TBL 专属的三位置分析数据，
    # 需要和其他方法做后处理叠图对比，因此不受 IS_SAVE_NPY 影响，始终保存。
    np.save(analysis_dir / "u_gt.npy", u_gt.astype(np.float32))
    np.save(analysis_dir / "u_pred.npy", u_pred.astype(np.float32))
    np.save(analysis_dir / "v_gt.npy", v_gt.astype(np.float32))
    np.save(analysis_dir / "v_pred.npy", v_pred.astype(np.float32))
    np.save(analysis_dir / "u_gt_profile_view.npy", u_gt_profile.astype(np.float32))
    np.save(analysis_dir / "u_pred_profile_view.npy", u_pred_profile.astype(np.float32))
    np.save(analysis_dir / "v_gt_profile_view.npy", v_gt_profile.astype(np.float32))
    np.save(analysis_dir / "v_pred_profile_view.npy", v_pred_profile.astype(np.float32))
    np.save(analysis_dir / "profile_columns.npy", np.asarray(columns, dtype=np.int32))
    np.save(analysis_dir / "profile_y_positions.npy", y_positions)
    np.save(analysis_dir / "profile_y_limit.npy", np.asarray(profile_height, dtype=np.int32))
    # 兼容旧后处理：profile_gt/profile_pred/profile_component 仍表示 U 分量。
    np.save(analysis_dir / "profile_gt.npy", profile_gt)
    np.save(analysis_dir / "profile_pred.npy", profile_pred)
    np.save(analysis_dir / "profile_component.npy", np.asarray("u"))
    # 新增显式分量文件：U/V 都用 component 前缀保存，后续多方法叠图时不用猜旧文件语义。
    np.save(analysis_dir / "u_profile_gt.npy", profile_gt)
    np.save(analysis_dir / "u_profile_pred.npy", profile_pred)
    np.save(analysis_dir / "v_profile_gt.npy", v_profile_gt)
    np.save(analysis_dir / "v_profile_pred.npy", v_profile_pred)
    np.save(analysis_dir / "profile_components.npy", np.asarray(["u", "v"]))

    # 除了三处局部裁剪图，也额外保存一张 3296 宽 full-frame 2x2 合图。
    # 这张图同样标注 Laminar/Transition/Turbulent 三条虚线，用于从全局上确认剖面位置。
    _save_tbl_full_frame_profile_comparison(
        analysis_dir,
        u_pred,
        u_gt,
        v_pred,
        v_gt,
        columns,
        region_names,
        profile_height,
        cmap_name=cmap_name,
    )

    # 把 Laminar/Transition/Turbulent 三个位置附近的局部窗口也放进 profile_analysis。
    # 这里不再假设 TBL sample 是均分的，而是围绕每个剖面 x 位置截取固定宽度的局部区域，
    # 避免把整张 3296 宽图混在一起看不清剖面位置。
    _save_tbl_profile_sample_comparisons(
        analysis_dir,
        u_pred,
        u_gt,
        v_pred,
        v_gt,
        columns,
        region_names,
        profile_height,
        cmap_name=cmap_name,
        crop_width=sample_crop_width,
    )

    # 保存 U 分量剖面图：tbl_profile_compare.png 是旧文件名，继续保留；
    # tbl_u_profile_compare.png 是新增的显式命名，和 V 分量文件形成一组。
    # U 图仍使用旧的 x_zero_left=True 规则；V 图通过 forced_xlim 复用这个范围，
    # 这样 tbl_u_profile_compare.png 与 tbl_v_profile_compare.png 的三张剖面子图 x 轴完全对齐。
    shared_uv_profile_xlim = _resolve_tbl_profile_xlim(profile_gt, profile_pred, x_zero_left=True)
    _save_tbl_component_profile_figure(
        analysis_dir=analysis_dir,
        output_names=["tbl_profile_compare.png", "tbl_u_profile_compare.png"],
        top_field=u_gt_profile,
        profile_gt=profile_gt,
        profile_pred=profile_pred,
        columns=columns,
        y_positions=y_positions,
        profile_height=profile_height,
        region_names=region_names,
        method_label=method_label,
        cmap_name=cmap_name,
        top_title="Ground truth of horizontal direction",
        x_zero_left=True,
    )
    # 新增 V 分量剖面图：使用同一组 Laminar/Transition/Turbulent 列坐标，
    # 但顶部显示 GT vertical/V 位移场，下方曲线对应 V displacement。
    _save_tbl_component_profile_figure(
        analysis_dir=analysis_dir,
        output_names="tbl_v_profile_compare.png",
        top_field=v_gt_profile,
        profile_gt=v_profile_gt,
        profile_pred=v_profile_pred,
        columns=columns,
        y_positions=y_positions,
        profile_height=profile_height,
        region_names=region_names,
        method_label=method_label,
        cmap_name=cmap_name,
        top_title="Ground truth of vertical direction",
        x_zero_left=False,
        forced_xlim=shared_uv_profile_xlim,
    )


def _plot_regular(out_path, u_pred, v_pred, u_gt, v_gt, cmap_name="jet"):
    """
    保存 256x256 数据集的 u/v 预测、真值和误差图。

    evaluate_all 的光流图使用 jet 风格的彩色位移图，而旧版 test_all 这里仍是 Greys。
    这会让 cylinder/dns_turb/jhtdb/sqg/backstep 的测试图和 evaluate_all 视觉口径不一致；
    因此主位移图统一改成 jet，误差图继续使用 bwr 以保留正负误差方向。
    """
    min_val_u, max_val_u = -4, 4
    min_val_v, max_val_v = -4, 4
    # 原始 256x256 流场总览图里只有 U/V 两张误差图；
    # 这里用同一张 sample 的 U/V 共同计算 AEE，并在两个误差面板上标注同一个值。
    sample_aee_text = _format_metric_annotation(
        "AEE",
        _compute_aee_from_chw(
            np.stack([u_pred, v_pred], axis=0),
            np.stack([u_gt, v_gt], axis=0),
        ),
    )

    plt.figure(num=None, figsize=(24, 16), dpi=120, facecolor="w", edgecolor="k")
    plt.subplot(3, 2, 1)
    plt.pcolor(u_pred, cmap=cmap_name, vmin=min_val_u, vmax=max_val_u)
    plt.title("Current method - U", fontsize=16)
    plt.axis("off")
    plt.colorbar().ax.set_ylabel("displacement [px]", fontsize=14)
    plt.subplot(3, 2, 3)
    plt.pcolor(u_gt, cmap=cmap_name, vmin=min_val_u, vmax=max_val_u)
    plt.title("GT - U", fontsize=16)
    plt.axis("off")
    plt.colorbar().ax.set_ylabel("displacement [px]", fontsize=14)
    plt.subplot(3, 2, 5)
    plt.pcolor(u_pred - u_gt, cmap="bwr", vmin=-0.25, vmax=0.25)
    plt.title("Error - U (Current - GT)", fontsize=16)
    plt.axis("off")
    _annotate_metric_box(plt.gca(), sample_aee_text, fontsize=20)
    plt.colorbar().ax.set_ylabel("error [px]", fontsize=14)
    plt.subplot(3, 2, 2)
    plt.pcolor(v_pred, cmap=cmap_name, vmin=min_val_v, vmax=max_val_v)
    plt.title("Current method - V", fontsize=16)
    plt.axis("off")
    plt.colorbar().ax.set_ylabel("displacement [px]", fontsize=14)
    plt.subplot(3, 2, 4)
    plt.pcolor(v_gt, cmap=cmap_name, vmin=min_val_v, vmax=max_val_v)
    plt.title("GT - V", fontsize=16)
    plt.axis("off")
    plt.colorbar().ax.set_ylabel("displacement [px]", fontsize=14)
    plt.subplot(3, 2, 6)
    plt.pcolor(v_pred - v_gt, cmap="bwr", vmin=-0.25, vmax=0.25)
    plt.title("Error - V (Current - GT)", fontsize=16)
    plt.axis("off")
    _annotate_metric_box(plt.gca(), sample_aee_text, fontsize=20)
    plt.colorbar().ax.set_ylabel("error [px]", fontsize=14)
    plt.savefig(out_path)
    plt.close()


def _as_numpy_batch(tensor):
    """把 [B, C, H, W] tensor 转成 float32 numpy，供图片保存函数复用。"""
    return tensor.detach().cpu().numpy().astype(np.float32, copy=False)


def _clip_image_for_display(arr):
    """图片保存前做有限值清理和 [0, 1] 裁剪，避免异常值把整张灰度图拉黑/拉白。"""
    arr = np.squeeze(arr).astype(np.float32, copy=False)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(arr, 0.0, 1.0)


def _compute_particle_image_error(sr_img, hr_img):
    """
    计算颗粒图像的有符号误差图：Generated SR - Original HR。

    说明：
        1. 与 PIV 的 Error-U/Error-V 图保持一致，误差保留正负号；
        2. NPY 中保留 NaN，用来表示 SR/HR 任一侧无效的位置；
        3. 如果个别分支输出尺寸和 HR 不一致，只在误差计算前把 SR 放到 HR 尺寸画布上，
           不改变原来的 LR/HR/SR 单图保存逻辑，也不影响指标计算。
    """
    hr_2d = np.squeeze(hr_img).astype(np.float32, copy=False)
    sr_2d = np.squeeze(sr_img).astype(np.float32, copy=False)
    if sr_2d.shape[-2:] != hr_2d.shape[-2:]:
        sr_2d = _place_image_on_display_canvas(sr_2d, hr_2d.shape[-2:], fill_value=np.nan)

    valid = np.isfinite(sr_2d) & np.isfinite(hr_2d)
    error = np.full(hr_2d.shape[-2:], np.nan, dtype=np.float32)
    error[valid] = sr_2d[valid] - hr_2d[valid]
    return error


def _compute_particle_image_esmse(sr_img, hr_img):
    """
    计算颗粒图像 SR-HR 的能量谱均方误差 ESMSE。

    ESMSE = Energy Spectrum MSE，与 metrics_image_pair.csv 的 energy_spectrum_mse 同口径：
    先计算 SR/HR 的径向平均能量谱，再对 log1p 谱差做 MSE。
    如果 SR 尺寸和 HR 不一致，只在这个指标计算副本中把 SR 放到 HR 尺寸画布上，
    不改变原始保存图片，也不会影响其它指标。
    """
    hr_2d = np.squeeze(hr_img).astype(np.float32, copy=False)
    sr_2d = np.squeeze(sr_img).astype(np.float32, copy=False)
    if sr_2d.shape[-2:] != hr_2d.shape[-2:]:
        sr_2d = _place_image_on_display_canvas(sr_2d, hr_2d.shape[-2:], fill_value=np.nan)
    pred_curve, gt_curve = _energy_spectrum_curves(sr_2d[None, ...], hr_2d[None, ...])
    return _energy_spectrum_mse_from_curves(pred_curve, gt_curve)


def _error_image_for_display(error_map, limit=0.25):
    """
    将误差图转换成 matplotlib 可显示数组。

    NPY 保存原始 NaN；显示时参考 PIV error-U 的做法用固定红蓝色条，
    因此这里只把非有限值临时填 0，避免 imshow/colorbar 被 NaN 干扰。
    """
    arr = np.squeeze(error_map).astype(np.float32, copy=False)
    return np.nan_to_num(arr, nan=0.0, posinf=float(limit), neginf=-float(limit))


def _plot_particle_error_panel(ax, error_map, title, limit=0.25, metric_text=None):
    """
    在 comparison 图里绘制颗粒图像误差图。

    色条风格仿照 PIV 的 Error-U/Error-V：bwr 表示正负误差，固定范围便于跨样本比较。
    """
    im = ax.imshow(_error_image_for_display(error_map, limit=limit), cmap="bwr", vmin=-limit, vmax=limit)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    # 在颗粒误差图上标注 ESMSE，方便直接从 comparison.png 判断频域重建误差。
    _annotate_metric_box(ax, metric_text, fontsize=12)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("error", fontsize=9)
    return im


def _place_image_on_display_canvas(arr, out_hw, fill_value=1.0):
    """
    将 LR 图放到 HR 尺寸画布中，仅用于合并对比图的视觉对齐。

    单独保存的 *_lr.png 保留真实低分辨率尺寸；comparison.png 里也不再插值放大 LR。
    这里仅用白色画布补齐到 HR 大小，让 LR 仍位于同一行第一列，与 HR/SR 并排比较。
    """
    arr = np.squeeze(arr).astype(np.float32, copy=False)
    if tuple(arr.shape[-2:]) == tuple(out_hw):
        return arr
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    h, w = int(arr.shape[-2]), int(arr.shape[-1])
    canvas = np.full((out_h, out_w), float(fill_value), dtype=np.float32)
    paste_h = min(h, out_h)
    paste_w = min(w, out_w)
    top = max((out_h - paste_h) // 2, 0)
    left = max((out_w - paste_w) // 2, 0)
    canvas[top:top + paste_h, left:left + paste_w] = arr[:paste_h, :paste_w]
    return canvas


def _save_gray_image(path, arr):
    """保存单通道灰度图，路径父目录自动创建。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(str(path), _clip_image_for_display(arr), cmap="gray", vmin=0.0, vmax=1.0)


def _plot_image_comparison(out_path, sample_images):
    """
    保存 prev/next 的 LR、原图 HR、生成 SR、SR-HR 误差合并对比图。

    画布为 2 行 4 列：第一行 previous，第二行 next；
    列依次为 LR、HR、Generated、Error。Error 使用和 PIV error-U 一样的 bwr 色条。
    """
    frames = [
        (
            "prev",
            sample_images["prev_lr"],
            sample_images["prev_hr"],
            sample_images["pred_prev"],
            sample_images["prev_error"],
            sample_images.get("prev_esmse", float("nan")),
        ),
        (
            "next",
            sample_images["next_lr"],
            sample_images["next_hr"],
            sample_images["pred_next"],
            sample_images["next_error"],
            sample_images.get("next_esmse", float("nan")),
        ),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=140, facecolor="w")
    for row_idx, (frame_name, lr_img, hr_img, sr_img, error_map, esmse_value) in enumerate(frames):
        hr_hw = np.squeeze(hr_img).shape[-2:]
        lr_for_compare = _place_image_on_display_canvas(lr_img, hr_hw)
        panels = [
            ("LR", lr_for_compare),
            ("Original HR", hr_img),
            ("Generated SR", sr_img),
        ]
        for col_idx, (title, arr) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.imshow(_clip_image_for_display(arr), cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_title(f"{frame_name} {title}", fontsize=10)
            ax.axis("off")
        _plot_particle_error_panel(
            axes[row_idx, 3],
            error_map,
            f"{frame_name} Error (SR-HR)",
            metric_text=_format_metric_annotation("ESMSE", esmse_value),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _resolve_tbl_comparison_crop_bounds(image_hw, crop_size=256, center_ratio=0.265):
    """
    计算 TBL 长图 comparison 使用的局部裁剪窗口。

    设计说明：
    1. TBL 的 full-frame 颗粒图宽度远大于高度，直接把整张长图塞进 LR/HR/SR 三联图里，细节会非常小。
    2. 这里在 full-frame 原图上先标一个红框，再把红框内的局部区域单独拿出来对比。
    3. 默认横向中心使用 `center_ratio=0.265`，与当前 Transition 位置保持一致；这样用户在看
       comparison.png 和 profile_analysis 时，更容易把两者对应起来。
    4. 高度方向优先截取 256；对当前 TBL 数据而言原图高就是 256，所以实际会覆盖完整高度。
    """
    image_h = int(image_hw[0])
    image_w = int(image_hw[1])
    crop_h = int(np.clip(int(crop_size), 1, image_h))
    crop_w = int(np.clip(int(crop_size), 1, image_w))

    x_center = int(round((image_w - 1) * float(center_ratio)))
    half_w = crop_w // 2
    x_start = x_center - half_w
    x_end = x_start + crop_w
    if x_start < 0:
        x_start = 0
        x_end = crop_w
    if x_end > image_w:
        x_end = image_w
        x_start = max(0, image_w - crop_w)

    # 当前 TBL 高度通常就是 256，因此这里会自然得到 [0, 256)。
    y_start = max((image_h - crop_h) // 2, 0)
    y_end = y_start + crop_h
    if y_end > image_h:
        y_end = image_h
        y_start = max(0, image_h - crop_h)

    return int(y_start), int(y_end), int(x_start), int(x_end)


def _scale_crop_bounds_to_target(bounds, ref_hw, target_hw):
    """
    把 HR full-frame 上的裁剪框映射到目标图像尺寸。

    用途：
    - HR / SR 通常与 full-frame 同尺寸，可直接复用原框；
    - LR 由于是动态下采样得到的，空间尺寸更小，这里按比例把同一物理区域映射过去，
      再裁成 LR 的局部图。这样 comparison 图里三列仍对应同一空间位置。
    """
    y_start, y_end, x_start, x_end = [int(v) for v in bounds]
    ref_h, ref_w = int(ref_hw[0]), int(ref_hw[1])
    target_h, target_w = int(target_hw[0]), int(target_hw[1])

    if ref_h <= 0 or ref_w <= 0 or target_h <= 0 or target_w <= 0:
        raise ValueError(
            f"Invalid shape for crop scaling: ref_hw={ref_hw}, target_hw={target_hw}"
        )

    scaled_y_start = int(np.floor(y_start / ref_h * target_h))
    scaled_y_end = int(np.ceil(y_end / ref_h * target_h))
    scaled_x_start = int(np.floor(x_start / ref_w * target_w))
    scaled_x_end = int(np.ceil(x_end / ref_w * target_w))

    scaled_y_start = int(np.clip(scaled_y_start, 0, target_h - 1))
    scaled_x_start = int(np.clip(scaled_x_start, 0, target_w - 1))
    scaled_y_end = int(np.clip(max(scaled_y_start + 1, scaled_y_end), 1, target_h))
    scaled_x_end = int(np.clip(max(scaled_x_start + 1, scaled_x_end), 1, target_w))
    return scaled_y_start, scaled_y_end, scaled_x_start, scaled_x_end


def _crop_2d_by_bounds(arr, bounds):
    """按 `(y_start, y_end, x_start, x_end)` 从单通道 2D 图像裁出局部区域。"""
    arr = np.squeeze(arr).astype(np.float32, copy=False)
    y_start, y_end, x_start, x_end = [int(v) for v in bounds]
    return arr[y_start:y_end, x_start:x_end]


def _plot_tbl_image_comparison(
    out_path,
    sample_images,
    crop_size=256,
    center_ratio=0.265,
):
    """
    保存 TBL 专用的颗粒图 comparison 图。

    图像布局：
    - 第 1 行：prev 原始 full-frame 长图，并用红框标出局部比较区域；
    - 第 2 行：prev 的 LR / HR / Generated SR / Error 局部对比；
    - 第 3 行：next 原始 full-frame 长图，并用红框标出局部比较区域；
    - 第 4 行：next 的 LR / HR / Generated SR / Error 局部对比。

    这样既保留了用户想要的“在原来很长的图里画红框”的全局上下文，
    又能把 256x256 局部放大到足够清楚的尺寸来对比超分细节和 SR-HR 误差。
    """
    frames = [
        ("prev", sample_images["prev_lr"], sample_images["prev_hr"], sample_images["pred_prev"], sample_images["prev_error"]),
        ("next", sample_images["next_lr"], sample_images["next_hr"], sample_images["pred_next"], sample_images["next_error"]),
    ]

    fig = plt.figure(figsize=(17, 12), dpi=140, facecolor="w")
    grid = fig.add_gridspec(4, 4, height_ratios=[0.7, 1.0, 0.7, 1.0])

    for frame_idx, (frame_name, lr_img, hr_img, sr_img, error_map) in enumerate(frames):
        hr_2d = np.squeeze(hr_img).astype(np.float32, copy=False)
        crop_bounds_hr = _resolve_tbl_comparison_crop_bounds(
            hr_2d.shape[-2:],
            crop_size=crop_size,
            center_ratio=center_ratio,
        )
        hr_crop = _crop_2d_by_bounds(hr_img, crop_bounds_hr)
        sr_crop = _crop_2d_by_bounds(
            sr_img,
            _scale_crop_bounds_to_target(crop_bounds_hr, hr_2d.shape[-2:], np.squeeze(sr_img).shape[-2:]),
        )
        lr_crop = _crop_2d_by_bounds(
            lr_img,
            _scale_crop_bounds_to_target(crop_bounds_hr, hr_2d.shape[-2:], np.squeeze(lr_img).shape[-2:]),
        )
        # Error 图使用完整 full-frame SR-HR 误差图按同一 HR crop 裁剪，保证和 HR/SR crop 是同一物理区域。
        error_crop = _crop_2d_by_bounds(error_map, crop_bounds_hr)
        # TBL comparison 误差面板显示的是 crop，因此 ESMSE 也按同一 crop 区域计算，
        # 避免图上数值和当前展示区域不一致。
        esmse_crop = _compute_particle_image_esmse(sr_crop, hr_crop)
        # 不插值放大 LR，只把它放到和 HR crop 同尺寸的白色画布中，继续保持项目里现有的展示规则。
        lr_crop_for_compare = _place_image_on_display_canvas(lr_crop, hr_crop.shape[-2:])

        overview_ax = fig.add_subplot(grid[frame_idx * 2, :])
        overview_ax.imshow(_clip_image_for_display(hr_img), cmap="gray", vmin=0.0, vmax=1.0, aspect="auto")
        y_start, y_end, x_start, x_end = crop_bounds_hr
        overview_ax.add_patch(
            Rectangle(
                (x_start - 0.5, y_start - 0.5),
                x_end - x_start,
                y_end - y_start,
                fill=False,
                edgecolor="red",
                linewidth=2.5,
            )
        )
        overview_ax.set_title(
            f"{frame_name} Original HR full-frame with 256x256 crop box",
            fontsize=11,
        )
        overview_ax.axis("off")

        panels = [
            ("LR crop", lr_crop_for_compare),
            ("Original HR crop", hr_crop),
            ("Generated SR crop", sr_crop),
        ]
        for col_idx, (title, arr) in enumerate(panels):
            ax = fig.add_subplot(grid[frame_idx * 2 + 1, col_idx])
            ax.imshow(_clip_image_for_display(arr), cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_title(f"{frame_name} {title}", fontsize=10)
            ax.axis("off")
        _plot_particle_error_panel(
            fig.add_subplot(grid[frame_idx * 2 + 1, 3]),
            error_crop,
            f"{frame_name} Error crop (SR-HR)",
            metric_text=_format_metric_annotation("ESMSE", esmse_crop),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_image_outputs(dataset_name, dataset_dir, image_payload, start_index, plot_args=None):
    """
    保存每个 sample 的 LR、原始 HR、生成 SR 图，以及一张合并对比图。

    目录结构：
    dataset_dir/images/sample_0000/
        prev_lr.png / prev_hr.png / prev_sr.png
        next_lr.png / next_hr.png / next_sr.png
        prev_sr_error.npy / next_sr_error.npy
        comparison.png
    """
    plot_args = plot_args or {}
    payload_np = {
        "prev_lr": _as_numpy_batch(image_payload["prev_lr"]),
        "next_lr": _as_numpy_batch(image_payload["next_lr"]),
        "prev_hr": _as_numpy_batch(image_payload["prev_hr"]),
        "next_hr": _as_numpy_batch(image_payload["next_hr"]),
        "pred_prev": _as_numpy_batch(image_payload["pred_prev"]),
        "pred_next": _as_numpy_batch(image_payload["pred_next"]),
    }

    batch_size = payload_np["prev_hr"].shape[0]
    for local_idx in range(batch_size):
        sample_index = start_index + local_idx
        sample_dir = dataset_dir / "images" / f"sample_{sample_index:04d}"
        sample_images = {
            "prev_lr": payload_np["prev_lr"][local_idx, 0],
            "next_lr": payload_np["next_lr"][local_idx, 0],
            "prev_hr": payload_np["prev_hr"][local_idx, 0],
            "next_hr": payload_np["next_hr"][local_idx, 0],
            "pred_prev": payload_np["pred_prev"][local_idx, 0],
            "pred_next": payload_np["pred_next"][local_idx, 0],
        }
        # 颗粒图像误差图：Generated SR - Original HR。
        # prev/next_sr_error*.npy 属于误差 NPY，按需求不受 IS_SAVE_NPY 影响，始终保存；
        # 这样可视化和后处理数据完全一致。
        sample_images["prev_error"] = _compute_particle_image_error(
            sample_images["pred_prev"],
            sample_images["prev_hr"],
        )
        sample_images["next_error"] = _compute_particle_image_error(
            sample_images["pred_next"],
            sample_images["next_hr"],
        )
        # 颗粒图像误差面板上显示 ESMSE（能量谱均方误差）。
        # 这里按完整 previous/next 图分别计算，和 metrics_image_pair.csv 的单帧口径一致。
        sample_images["prev_esmse"] = _compute_particle_image_esmse(
            sample_images["pred_prev"],
            sample_images["prev_hr"],
        )
        sample_images["next_esmse"] = _compute_particle_image_esmse(
            sample_images["pred_next"],
            sample_images["next_hr"],
        )

        sample_dir.mkdir(parents=True, exist_ok=True)
        save_npy = bool(plot_args.get("save_npy", False))
        np.save(sample_dir / "prev_sr_error.npy", sample_images["prev_error"].astype(np.float32))
        np.save(sample_dir / "next_sr_error.npy", sample_images["next_error"].astype(np.float32))
        # 颗粒图像误差 hist：对齐 flow 的 delta_u/delta_v hist 输出习惯。
        # hist 属于诊断结果，因此通过 force=True 的保存回调，即使 IS_SAVE_NPY=False 也会保留。
        save_particle_error_histogram(
            sample_dir,
            sample_images["prev_error"],
            file_prefix="prev_sr_error",
            title="Previous Particle Image Error Distribution",
            save_npy_fn=_save_optional_npy,
            save_npy=save_npy,
        )
        save_particle_error_histogram(
            sample_dir,
            sample_images["next_error"],
            file_prefix="next_sr_error",
            title="Next Particle Image Error Distribution",
            save_npy_fn=_save_optional_npy,
            save_npy=save_npy,
        )
        _save_gray_image(sample_dir / "prev_lr.png", sample_images["prev_lr"])
        _save_gray_image(sample_dir / "prev_hr.png", sample_images["prev_hr"])
        _save_gray_image(sample_dir / "prev_sr.png", sample_images["pred_prev"])
        _save_gray_image(sample_dir / "next_lr.png", sample_images["next_lr"])
        _save_gray_image(sample_dir / "next_hr.png", sample_images["next_hr"])
        _save_gray_image(sample_dir / "next_sr.png", sample_images["pred_next"])
        if dataset_name == "tbl":
            # TBL 的颗粒图是 full-frame 长图；普通三联图里直接显示整张长图时，细节太小不利于比较。
            # 因此这里改成：先在原始长图上画红框，再把框内 256x256 局部拿出来对比。
            tbl_profile_column_ratios = tuple(
                plot_args.get("tbl_profile_column_ratios", (0.15, 0.265, 0.83))
            )
            tbl_crop_center_ratio = (
                float(tbl_profile_column_ratios[1]) if len(tbl_profile_column_ratios) >= 2 else 0.265
            )
            tbl_crop_size = int(plot_args.get("tbl_profile_sample_crop_width", 256))
            # TBL 的 comparison 显示的是红框里的局部误差图，因此额外保存 crop 版 NPY，
            # 方便直接和 comparison.png 里的 Error crop 面板一一对应。
            prev_crop_bounds_hr = _resolve_tbl_comparison_crop_bounds(
                np.squeeze(sample_images["prev_hr"]).shape[-2:],
                crop_size=tbl_crop_size,
                center_ratio=tbl_crop_center_ratio,
            )
            next_crop_bounds_hr = _resolve_tbl_comparison_crop_bounds(
                np.squeeze(sample_images["next_hr"]).shape[-2:],
                crop_size=tbl_crop_size,
                center_ratio=tbl_crop_center_ratio,
            )
            prev_error_crop = _crop_2d_by_bounds(sample_images["prev_error"], prev_crop_bounds_hr).astype(np.float32)
            next_error_crop = _crop_2d_by_bounds(sample_images["next_error"], next_crop_bounds_hr).astype(np.float32)
            np.save(
                sample_dir / "prev_sr_error_crop.npy",
                prev_error_crop,
            )
            np.save(
                sample_dir / "next_sr_error_crop.npy",
                next_error_crop,
            )
            # TBL comparison 图展示的是局部 crop，因此同时保存 crop 对应 hist，
            # 便于后续直接核对红框区域内的颗粒重建误差分布。
            save_particle_error_histogram(
                sample_dir,
                prev_error_crop,
                file_prefix="prev_sr_error_crop",
                title="Previous TBL Crop Particle Image Error Distribution",
                save_npy_fn=_save_optional_npy,
                save_npy=save_npy,
            )
            save_particle_error_histogram(
                sample_dir,
                next_error_crop,
                file_prefix="next_sr_error_crop",
                title="Next TBL Crop Particle Image Error Distribution",
                save_npy_fn=_save_optional_npy,
                save_npy=save_npy,
            )
            _plot_tbl_image_comparison(
                sample_dir / "comparison.png",
                sample_images,
                crop_size=tbl_crop_size,
                center_ratio=tbl_crop_center_ratio,
            )
        else:
            _plot_image_comparison(sample_dir / "comparison.png", sample_images)


def _save_sample_plots(
    dataset_name,
    dataset_dir,
    predicted_np,
    flow_np,
    start_index,
    twcf_payload,
    image_payload=None,
    plot_args=None,
):
    """
    按 dataset 类别分文件夹保存每个 sample 的 flow 可视化图，并可选保存 LR/HR/SR 图片。

    plot_args 用于把 test_all 的一些可视化超参数往下传，例如：
        - displacement_cmap
        - method_label
        - tbl_profile_column_ratios
        - tbl_profile_region_names
        - tbl_profile_y_limit
        - tbl_profile_sample_crop_width
    """
    plot_args = plot_args or {}
    displacement_cmap = str(plot_args.get("displacement_cmap", "viridis"))
    regular_flow_cmap = str(plot_args.get("regular_flow_cmap", "jet"))
    method_label = str(plot_args.get("method_label", "Current method"))
    # save_npy 来自全局 IS_SAVE_NPY：普通 NPY 受它控制，误差/TBL profile/hist 例外仍会保存。
    save_npy = bool(plot_args.get("save_npy", False))
    quiver_stride = plot_args.get("vorticity_quiver_stride", None)
    tbl_profile_column_ratios = tuple(plot_args.get("tbl_profile_column_ratios", (0.15, 0.265, 0.83)))
    tbl_profile_region_names = tuple(plot_args.get("tbl_profile_region_names", ("Laminar", "Transition", "Turbulent")))
    tbl_profile_y_limit = plot_args.get("tbl_profile_y_limit", 200)
    tbl_profile_sample_crop_width = int(plot_args.get("tbl_profile_sample_crop_width", 256))
    if image_payload is not None:
        _save_image_outputs(dataset_name, dataset_dir, image_payload, start_index, plot_args=plot_args)

    for local_idx in range(predicted_np.shape[0]):
        sample_index = start_index + local_idx
        out_path = dataset_dir / f"{dataset_name}_sample_{sample_index:04d}.png"
        u_pred = predicted_np[local_idx, 0, :, :]
        v_pred = predicted_np[local_idx, 1, :, :]
        u_gt = flow_np[local_idx, 0, :, :]
        v_gt = flow_np[local_idx, 1, :, :]
        # 新增 evaluate_all 同款的样本级光流细节输出：
        # - flow/sample_xxxx/uvs_compare.png 以及 U/V/S 对应 npy；
        # - flow/sample_xxxx/vorticity_quiver.png 以及涡度、速度差 npy。
        # 原有 dataset_name_sample_xxxx.png 继续保留，保证旧脚本路径不受影响。
        flow_sample_dir = dataset_dir / "flow" / f"sample_{sample_index:04d}"
        twcf_mask_for_flow_plots = (
            np.asarray(twcf_payload["mask"], dtype=np.float32)
            if dataset_name == "twcf" and twcf_payload is not None and "mask" in twcf_payload
            else None
        )
        _save_flow_visual_artifacts(
            flow_sample_dir,
            predicted_np[local_idx],
            flow_np[local_idx],
            cmap_name=displacement_cmap if dataset_name in {"tbl", "twcf"} else regular_flow_cmap,
            quiver_stride=quiver_stride,
            save_npy=save_npy,
            dataset_name=dataset_name,
            mask_2d=twcf_mask_for_flow_plots,
            tbl_y_limit=tbl_profile_y_limit if dataset_name == "tbl" else None,
        )

        if dataset_name == "twcf":
            _plot_twcf(
                out_path,
                u_pred,
                v_pred,
                twcf_payload["piv_results"],
                twcf_payload["mask"],
                sample_index,
                cmap_name=displacement_cmap,
            )
        elif dataset_name == "tbl":
            _plot_tbl(
                out_path,
                u_pred,
                v_pred,
                u_gt,
                v_gt,
                cmap_name=displacement_cmap,
                y_limit=tbl_profile_y_limit,
            )
            # profile_analysis 是 TBL 专属的论文风格剖面对比图：
            # 现在同时保存 U/V 两个分量。U 图保留旧文件名 tbl_profile_compare.png，
            # V 图新增 tbl_v_profile_compare.png；二者使用同一组 Laminar/Transition/Turbulent 列坐标。
            _save_tbl_profile_artifacts(
                dataset_dir=dataset_dir,
                sample_index=sample_index,
                u_pred=u_pred,
                u_gt=u_gt,
                v_pred=v_pred,
                v_gt=v_gt,
                method_label=method_label,
                cmap_name=displacement_cmap,
                column_ratios=tbl_profile_column_ratios,
                region_names=tbl_profile_region_names,
                y_limit=tbl_profile_y_limit,
                sample_crop_width=tbl_profile_sample_crop_width,
            )
        else:
            _plot_regular(out_path, u_pred, v_pred, u_gt, v_gt, cmap_name=regular_flow_cmap)


def _write_csv(path, rows):
    """写入 dataset 级或 test_all 汇总指标。"""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_test_all(model, global_data, class_name, data_type, SCALE, device=None):
    """
    四个分支共享的 test_all 实现。

    行为：
    - 单 GPU，不启用 DDP/多进程；
    - 默认顺序测试 TEST_DATASETS 中定义的全部 dataset；
    - 当 global_data.esrgan.TEST_TBL=True 时，只测试 tbl 数据集，便于单独调试 TBL 大图与 profile 分析；
    - target 两通道按 prev/next 拆分，并用 data_downscal.py 生成 prev_lr/next_lr；
    - 输出目录为 OUT_PUT_DIR/class_name/data_type/scale_x/TEST_DIR/dataset_name。
    """
    if not getattr(global_data.esrgan, "IS_TEST", False):
        logger.info("[test_all] IS_TEST=False，跳过额外 TFRecord 全数据集测试。")
        return []

    if model is None:
        logger.warning("[test_all] model=None，跳过测试。")
        return []

    if hasattr(model, "module"):
        model = model.module

    device, device_id = _as_cuda_device(device or getattr(global_data.esrgan, "device", None))
    test_datasets = getattr(global_data.esrgan, "TEST_DATASETS", {})
    if not test_datasets:
        logger.warning("[test_all] TEST_DATASETS 为空，跳过测试。")
        return []

    if getattr(global_data.esrgan, "TEST_TBL", False):
        # TEST_TBL 是更高优先级的窄化开关：
        # 只保留 tbl，不再受 is_TEST_CLASS3=False 时“跳过 class3 大图”的旧逻辑影响。
        # 这样用户只想重跑 TBL profile/大图结果时，不需要同时跑 twcf 或五个常规小数据集。
        if "tbl" not in test_datasets:
            logger.warning("[test_all] TEST_TBL=True，但 TEST_DATASETS 中没有 tbl，跳过测试。")
            return []
        test_datasets = {"tbl": test_datasets["tbl"]}
    elif not getattr(global_data.esrgan, "is_TEST_CLASS3", False):
        # class3 对应 RAFT256-PIV_test.py 中的 tbl/twcf 大图数据集。
        # 默认跳过它们，避免 test_all 一打开就触发长时间全图滑窗测试。
        test_datasets = {
            name: cfg
            for name, cfg in test_datasets.items()
            if name not in {"tbl", "twcf"}
        }

    factor = _scale_factor_from_scale(SCALE)
    test_base_dir = Path(
        f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.TEST_DIR}"
    )
    test_base_dir.mkdir(parents=True, exist_ok=True)
    save_npy = bool(getattr(global_data.esrgan, "IS_SAVE_NPY", False))
    logger.info(
        "[test_all] IS_SAVE_NPY={}，普通 NPY 由该开关控制；误差 NPY、hist 直方图 NPY 和 TBL profile NPY 始终保存。",
        save_npy,
    )

    test_args = {
        "split_size": getattr(global_data.esrgan, "TEST_SPLIT_SIZE", 1),
        "offset": getattr(global_data.esrgan, "TEST_OFFSET", 256),
        "shift": getattr(global_data.esrgan, "TEST_SHIFT", 64),
        "amp": getattr(global_data.esrgan, "TEST_AMP", False),
        "plot_results": getattr(global_data.esrgan, "TEST_PLOT_RESULTS", True),
        "save_npy": save_npy,
        # viridis 与用户提供的彩色位移色条最接近；这里用 getattr 预留后续配置入口，
        # 这样不改 global_class 也能工作，未来如果用户想换别的 cmap，可以直接在全局变量补同名字段。
        "displacement_cmap": getattr(global_data.esrgan, "TEST_DISPLACEMENT_CMAP", "viridis"),
        # cylinder/dns_turb/jhtdb/sqg/backstep 五类常规 256x256 数据集按 evaluate_all 的光流图口径显示。
        # evaluate_all 中 U/V 位移图使用 jet，误差图使用 bwr；这里保持同样的主色条。
        "regular_flow_cmap": getattr(global_data.esrgan, "TEST_REGULAR_FLOW_CMAP", "jet"),
        "method_label": getattr(global_data.esrgan, "name", "Current method"),
        # TBL 的 Laminar/Transition/Turbulent 剖面位置；优先读新的 TBL_* 全局变量。
        # 为了兼容前一次临时命名，如果用户还没改 global_class，也会回退读取 TWCF_*。
        "tbl_profile_column_ratios": getattr(
            global_data.esrgan,
            "TBL_PROFILE_COLUMN_RATIOS",
            getattr(global_data.esrgan, "TWCF_PROFILE_COLUMN_RATIOS", (0.15, 0.265, 0.83)),
        ),
        "tbl_profile_region_names": getattr(
            global_data.esrgan,
            "TBL_PROFILE_REGION_NAMES",
            getattr(global_data.esrgan, "TWCF_PROFILE_REGION_NAMES", ("Laminar", "Transition", "Turbulent")),
        ),
        "tbl_profile_y_limit": getattr(global_data.esrgan, "TBL_PROFILE_Y_LIMIT", 200),
        # TBL 不再按 sample 均分；profile_analysis 围绕每个剖面位置截取固定宽度窗口。
        "tbl_profile_sample_crop_width": getattr(global_data.esrgan, "TBL_PROFILE_SAMPLE_CROP_WIDTH", 256),
        # test_all 新增的涡度-速度差 quiver 图可选步长；None 时按图像尺寸自动稀疏采样。
        "vorticity_quiver_stride": getattr(global_data.esrgan, "TEST_VORTICITY_QUIVER_STRIDE", None),
    }

    twcf_payload = None
    if "twcf" in test_datasets:
        piv_path = _resolve_path(getattr(global_data.esrgan, "PIV_RESULTS_TWCF_PATH"), global_data)
        mask_path = _resolve_path(getattr(global_data.esrgan, "MASK_TWCF_PATH"), global_data)
        twcf_payload = {
            "piv_results": np.load(piv_path),
            "mask": np.load(mask_path),
        }

    was_training = model.training
    model.eval()
    summary_rows = []
    # test_all 根目录总体汇总：跨 TEST_DATASETS 全部类别累计，用于生成总体 mean spectrum 和误差直方图。
    all_image_pred_curves = []
    all_image_gt_curves = []
    all_image_error_values = []
    all_flow_pred_curves = []
    all_flow_gt_curves = []
    all_delta_u_values = []
    all_delta_v_values = []
    all_delta_w_values = []
    all_epe_values = []
    all_delta_vorticity_values = []
    all_class_image_pair_mean_rows = []
    all_class_flow_mean_rows = []

    try:
        with torch.no_grad():
            for dataset_name, dataset_cfg in test_datasets.items():
                dataset_dir = test_base_dir / dataset_name
                dataset_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"[test_all] 开始测试 dataset={dataset_name}，输出目录：{dataset_dir}")

                test_iterator, dataset_size = _build_dali_iterator(dataset_cfg, global_data, device_id)
                loader_len = int(math.ceil(dataset_size / int(getattr(global_data.esrgan, "TEST_BATCH_SIZE", 1))))
                progress = tqdm(enumerate(test_iterator), total=loader_len, leave=False)

                height = int(dataset_cfg["image_height"])
                width = int(dataset_cfg["image_width"])
                # full-frame 数据集的最后一批或异常样本可能无法写满全部位置；
                # 用 NaN 初始化能防止未写入位置残留 np.empty 的随机内存值，后续均值会自动跳过这些无效项。
                results = np.full((dataset_size, 4, height, width), np.nan, dtype=np.float32)
                epe_array = np.full((dataset_size,), np.nan, dtype=np.float32)
                norm_aee_per100_array = np.full((dataset_size,), np.nan, dtype=np.float32)
                dataset_image_rows = []
                dataset_raft_rows = []
                # dataset 级汇总缓存：每个 TEST_DATASETS 条目相当于 evaluate_all 的一个类别。
                dataset_image_pred_curves = []
                dataset_image_gt_curves = []
                dataset_image_error_values = []
                dataset_flow_pred_curves = []
                dataset_flow_gt_curves = []
                dataset_delta_u_values = []
                dataset_delta_v_values = []
                dataset_delta_w_values = []
                dataset_epe_values = []
                dataset_delta_vorticity_values = []

                total_samples = 0
                sum_epe = 0.0
                total_epe_samples = 0
                start_time = time.time()

                for i_batch, sample_batched in progress:
                    t0 = time.time()
                    local_dict = sample_batched[0]

                    # target[0]=prev，target[1]=next；按 RAFT256-PIV_test.py 的输入口径除以 256，
                    # 将 TFRecord 中的图像强度缩放到模型测试使用的浮点范围。
                    images = local_dict["target"].type(torch.FloatTensor).cuda(device_id) / 256
                    flows = local_dict["flow"].type(torch.FloatTensor).cuda(device_id)

                    batch_size = int(images.size(0))
                    sample_start = i_batch * batch_size
                    valid_size = min(batch_size, dataset_size - sample_start)
                    if valid_size <= 0:
                        continue

                    images = images[:valid_size]
                    flows = flows[:valid_size]

                    with autocast(enabled=bool(test_args["amp"])):
                        if dataset_name in {"tbl", "twcf"}:
                            prediction = _predict_full_frame_with_folding(
                                model, images, flows, factor, device, test_args
                            )
                        else:
                            prediction = _predict_patch(model, images, flows, factor, device)
                        predicted_flows = prediction["flow"]

                    epe_per_sample = _torch_finite_epe_per_sample(predicted_flows, flows)
                    finite_epe_per_sample = epe_per_sample[torch.isfinite(epe_per_sample)]
                    total_samples += valid_size
                    if int(finite_epe_per_sample.numel()) > 0:
                        # running mean 只统计有效样本，避免某个无有效像素的 sample 把整个日志均值拖成 NaN。
                        sum_epe += float(finite_epe_per_sample.sum().item())
                        total_epe_samples += int(finite_epe_per_sample.numel())

                    predicted_np = predicted_flows.detach().cpu().numpy().astype(np.float32, copy=False)
                    flow_np = flows.detach().cpu().numpy().astype(np.float32, copy=False)
                    # 图像指标和 evaluate_all 保持一致：统一按 [0,1] 范围统计。
                    pred_prev_np = _as_numpy_batch(prediction["pred_prev"].clamp(0, 1))
                    pred_next_np = _as_numpy_batch(prediction["pred_next"].clamp(0, 1))
                    prev_hr_np = _as_numpy_batch(prediction["prev_hr"].clamp(0, 1))
                    next_hr_np = _as_numpy_batch(prediction["next_hr"].clamp(0, 1))
                    sample_end = sample_start + valid_size
                    results[sample_start:sample_end, 0, :, :] = predicted_np[:, 0, :, :]
                    results[sample_start:sample_end, 1, :, :] = predicted_np[:, 1, :, :]
                    results[sample_start:sample_end, 2, :, :] = flow_np[:, 0, :, :]
                    results[sample_start:sample_end, 3, :, :] = flow_np[:, 1, :, :]
                    epe_array[sample_start:sample_end] = epe_per_sample.detach().cpu().numpy().astype(np.float32)
                    # 逐样本补充 test_all 的 SR 图像指标和流场额外指标。
                    for local_idx in range(valid_size):
                        sample_index = sample_start + local_idx
                        # 频谱曲线既用于 image_pair 的 ESMSE 指标，也用于后面的 dataset/root 均值频谱图。
                        # 这里按 previous/next 两张颗粒图分别登记，和 evaluate_all 的 image_pair 口径一致。
                        prev_pred_curve, prev_gt_curve = _energy_spectrum_curves(
                            pred_prev_np[local_idx],
                            prev_hr_np[local_idx],
                        )
                        next_pred_curve, next_gt_curve = _energy_spectrum_curves(
                            pred_next_np[local_idx],
                            next_hr_np[local_idx],
                        )
                        dataset_image_pred_curves.extend([prev_pred_curve, next_pred_curve])
                        dataset_image_gt_curves.extend([prev_gt_curve, next_gt_curve])
                        all_image_pred_curves.extend([prev_pred_curve, next_pred_curve])
                        all_image_gt_curves.extend([prev_gt_curve, next_gt_curve])
                        # 颗粒图像误差 hist 使用 SR-HR 的像素级有符号误差；
                        # 这里即使后面关闭 sample 图片保存，也先累计 dataset/root 级分布，保证汇总 hist 完整。
                        prev_image_error = _compute_particle_image_error(
                            pred_prev_np[local_idx],
                            prev_hr_np[local_idx],
                        )
                        next_image_error = _compute_particle_image_error(
                            pred_next_np[local_idx],
                            next_hr_np[local_idx],
                        )
                        dataset_image_error_values.extend([prev_image_error.reshape(-1), next_image_error.reshape(-1)])
                        all_image_error_values.extend([prev_image_error.reshape(-1), next_image_error.reshape(-1)])
                        dataset_image_rows.append(
                            _compute_image_metric_row(
                                dataset_name,
                                sample_index,
                                "previous",
                                pred_prev_np[local_idx],
                                prev_hr_np[local_idx],
                            )
                        )
                        dataset_image_rows.append(
                            _compute_image_metric_row(
                                dataset_name,
                                sample_index,
                                "next",
                                pred_next_np[local_idx],
                                next_hr_np[local_idx],
                            )
                        )
                        flow_row = _compute_flow_metric_row(
                            dataset_name,
                            sample_index,
                            predicted_np[local_idx],
                            flow_np[local_idx],
                        )
                        norm_aee_per100_array[sample_index] = float(flow_row["NORM_AEE_PER100PIXEL"])
                        flow_pred_uvw = _flow_uv_to_uvw_np(predicted_np[local_idx])
                        flow_gt_uvw = _flow_uv_to_uvw_np(flow_np[local_idx])
                        flow_pred_curve, flow_gt_curve = _energy_spectrum_curves(flow_pred_uvw, flow_gt_uvw)
                        # RAFT/flow 的能量谱 MSE 使用 U/V/S 三个分量的频谱曲线计算；
                        # 和后面的 mean spectrum 图复用同一组曲线，保证 CSV 指标和图像完全同源。
                        flow_row["energy_spectrum_mse"] = _energy_spectrum_mse_from_curves(
                            flow_pred_curve,
                            flow_gt_curve,
                        )
                        dataset_raft_rows.append(flow_row)
                        dataset_flow_pred_curves.append(flow_pred_curve)
                        dataset_flow_gt_curves.append(flow_gt_curve)
                        all_flow_pred_curves.append(flow_pred_curve)
                        all_flow_gt_curves.append(flow_gt_curve)
                        delta_u_map, delta_v_map, delta_w_map, epe_map = _compute_flow_error_maps_np(
                            predicted_np[local_idx],
                            flow_np[local_idx],
                        )
                        # vorticity_quiver.png 的误差来自 omega* 的预测-真值差；
                        # 这里累计像素级 delta omega*，用于 dataset/root 级 hist 汇总。
                        delta_vorticity_map = (
                            _omega_star_from_uv_np(predicted_np[local_idx, 0], predicted_np[local_idx, 1])
                            - _omega_star_from_uv_np(flow_np[local_idx, 0], flow_np[local_idx, 1])
                        ).astype(np.float32, copy=False)
                        dataset_delta_u_values.append(delta_u_map.reshape(-1))
                        dataset_delta_v_values.append(delta_v_map.reshape(-1))
                        dataset_delta_w_values.append(delta_w_map.reshape(-1))
                        dataset_epe_values.append(epe_map.reshape(-1))
                        dataset_delta_vorticity_values.append(delta_vorticity_map.reshape(-1))
                        all_delta_u_values.append(delta_u_map.reshape(-1))
                        all_delta_v_values.append(delta_v_map.reshape(-1))
                        all_delta_w_values.append(delta_w_map.reshape(-1))
                        all_epe_values.append(epe_map.reshape(-1))
                        all_delta_vorticity_values.append(delta_vorticity_map.reshape(-1))

                    if bool(test_args["plot_results"]):
                        _save_sample_plots(
                            dataset_name,
                            dataset_dir,
                            predicted_np,
                            flow_np,
                            sample_start,
                            twcf_payload if dataset_name == "twcf" else None,
                            image_payload=prediction,
                            plot_args=test_args,
                        )

                    logger.info(
                        f"[test_all] dataset={dataset_name}, batch={i_batch}, "
                        f"samples={total_samples}/{dataset_size}, "
                        f"mean_epe={(sum_epe / total_epe_samples) if total_epe_samples else float('nan'):.6f}, "
                        f"time={time.time() - t0:.2f}s"
                    )

                # dataset 级 mean_epe 也按稳健均值统计；逐样本 epe_array 仍完整保存，
                # 这里仅影响 metrics_all.csv 里的平均指标展示。
                mean_epe = (
                    robust_metric_mean(epe_array[:total_samples], metric_key="epe", global_data=global_data)
                    if total_samples
                    else float("nan")
                )
                elapsed = time.time() - start_time
                results_path = dataset_dir / "results.npy"
                epe_path = dataset_dir / "epe_array.npy"
                norm_aee_per100_path = dataset_dir / "norm_aee_per100_array.npy"
                # results.npy 是完整预测结果缓存，默认受 IS_SAVE_NPY 控制；
                # epe/norm_aee_per100 是误差指标数组，按需求始终保存，便于后处理核对误差分布。
                _save_optional_npy(results_path, results, save_npy)
                _save_optional_npy(epe_path, epe_array, save_npy, force=True)
                _save_optional_npy(norm_aee_per100_path, norm_aee_per100_array, save_npy, force=True)
                # dataset 级图/npy：补齐 evaluate_all 在类别目录下会保存的均值频谱和误差直方图。
                _save_mean_spectrum(
                    dataset_image_pred_curves,
                    dataset_image_gt_curves,
                    dataset_dir,
                    title=f"{dataset_name}-{data_type}-x{int(SCALE*SCALE)} Image Pair Mean Energy Spectrum",
                    file_prefix="image_pair",
                    save_npy=save_npy,
                )
                _save_mean_spectrum(
                    dataset_flow_pred_curves,
                    dataset_flow_gt_curves,
                    dataset_dir,
                    title=f"{dataset_name}-{data_type}-x{int(SCALE*SCALE)} Flow Mean Energy Spectrum",
                    file_prefix="flow",
                    save_npy=save_npy,
                )
                _save_flow_histogram_bundle(
                    dataset_dir,
                    dataset_delta_u_values,
                    dataset_delta_v_values,
                    dataset_delta_w_values,
                    dataset_epe_values,
                )
                save_particle_error_histogram_bundle(
                    dataset_dir,
                    dataset_delta_vorticity_values,
                    file_prefix="delta_vorticity",
                    title=f"{dataset_name}-{data_type}-x{int(SCALE*SCALE)} Delta Vorticity Error Distribution",
                    xlabel="delta omega*",
                    color="#4477AA",
                    save_npy_fn=_save_optional_npy,
                    save_npy=save_npy,
                )
                save_particle_error_histogram_bundle(
                    dataset_dir,
                    dataset_image_error_values,
                    file_prefix="sr_error",
                    title=f"{dataset_name}-{data_type}-x{int(SCALE*SCALE)} Particle Image Error Distribution",
                    save_npy_fn=_save_optional_npy,
                    save_npy=save_npy,
                )
                _save_energy_spectrum_mse_compare_plot(
                    dataset_image_rows,
                    dataset_dir / "image_pair_energy_spectrum_mse_compare.png",
                    title=f"{dataset_name}-{data_type}-x{int(SCALE*SCALE)} Image Pair Energy Spectrum MSE",
                )
                _save_energy_spectrum_mse_compare_plot(
                    dataset_raft_rows,
                    dataset_dir / "flow_energy_spectrum_mse_compare.png",
                    title=f"{dataset_name}-{data_type}-x{int(SCALE*SCALE)} Flow Energy Spectrum MSE",
                )

                # C-AEE 需要把 same sample 的 previous/next 图像 ESMSE/SSIM 与该 sample 的 RAFT EPE 配对；
                # 现在使用固定参考尺度把三项放缩为无量纲量，不再使用当前 dataset 的 min/max。
                # 这里仍然放在整个 dataset 收集完成后执行，是为了确保 image_pair 行和 RAFT 行都已经完整。
                attach_c_aee_to_raft_rows(
                    image_rows=dataset_image_rows,
                    raft_rows=dataset_raft_rows,
                    sample_key_fields=("dataset", "sample_index"),
                    ese_key="energy_spectrum_mse",
                    aee_key="epe",
                    ssim_key="ssim",
                    output_key="C_AEE",
                )

                image_metric_keys = ["mse", "psnr", "energy_spectrum_mse", "r2", "ssim", "tke_acc", "nrmse"]
                raft_metric_keys = ["epe", "NORM_AEE_PER100PIXEL", "energy_spectrum_mse", "C_AEE"]
                image_mean_row = _write_rows_with_mean(
                    dataset_dir / "metrics_image_pair.csv",
                    dataset_image_rows,
                    fixed_fields={
                        "dataset": dataset_name,
                        "sample_index": "MEAN",
                        "pair_type": "all",
                    },
                    metric_keys=image_metric_keys,
                    global_data=global_data,
                )
                raft_mean_row = _write_rows_with_mean(
                    dataset_dir / "metrics_raft.csv",
                    dataset_raft_rows,
                    fixed_fields={
                        "dataset": dataset_name,
                        "sample_index": "MEAN",
                        "pair_type": "RAFT",
                    },
                    metric_keys=raft_metric_keys,
                    global_data=global_data,
                )
                # 兼容原有只看 metrics.csv 的脚本：继续保留这个名字，并让它等同于 flow/RAFT 指标表。
                if dataset_raft_rows:
                    _write_rows_with_mean(
                        dataset_dir / "metrics.csv",
                        dataset_raft_rows,
                        fixed_fields={
                            "dataset": dataset_name,
                            "sample_index": "MEAN",
                            "pair_type": "RAFT",
                        },
                        metric_keys=raft_metric_keys,
                        global_data=global_data,
                    )

                if image_mean_row is not None:
                    image_class_row = dict(image_mean_row)
                    image_class_row["sample_index"] = "CLASS_MEAN"
                    image_class_row["pair_type"] = "image_pair"
                    all_class_image_pair_mean_rows.append(image_class_row)
                if raft_mean_row is not None:
                    flow_class_row = dict(raft_mean_row)
                    flow_class_row["sample_index"] = "CLASS_MEAN"
                    flow_class_row["pair_type"] = "flow"
                    all_class_flow_mean_rows.append(flow_class_row)

                summary_rows.append(
                    {
                        "dataset": dataset_name,
                        "samples": total_samples,
                        "mean_epe": mean_epe,
                        "mean_norm_aee_per100pixel": (
                            float(raft_mean_row["NORM_AEE_PER100PIXEL"]) if raft_mean_row is not None else float("nan")
                        ),
                        "mean_c_aee": float(raft_mean_row["C_AEE"]) if raft_mean_row is not None else float("nan"),
                        "image_mse_mean": float(image_mean_row["mse"]) if image_mean_row is not None else float("nan"),
                        "image_psnr_mean": float(image_mean_row["psnr"]) if image_mean_row is not None else float("nan"),
                        "image_energy_spectrum_mse_mean": (
                            float(image_mean_row["energy_spectrum_mse"]) if image_mean_row is not None else float("nan")
                        ),
                        "flow_energy_spectrum_mse_mean": (
                            float(raft_mean_row["energy_spectrum_mse"]) if raft_mean_row is not None else float("nan")
                        ),
                        "image_r2_mean": float(image_mean_row["r2"]) if image_mean_row is not None else float("nan"),
                        "image_ssim_mean": float(image_mean_row["ssim"]) if image_mean_row is not None else float("nan"),
                        "image_tke_acc_mean": float(image_mean_row["tke_acc"]) if image_mean_row is not None else float("nan"),
                        "image_nrmse_mean": float(image_mean_row["nrmse"]) if image_mean_row is not None else float("nan"),
                        "elapsed_seconds": elapsed,
                        "results_npy": str(results_path) if save_npy else "",
                        "epe_npy": str(epe_path),
                        "norm_aee_per100_npy": str(norm_aee_per100_path),
                        "metrics_image_pair_csv": str(dataset_dir / "metrics_image_pair.csv"),
                        "metrics_raft_csv": str(dataset_dir / "metrics_raft.csv"),
                    }
                )
                logger.info(
                    f"[test_all] 完成 dataset={dataset_name}, samples={total_samples}, "
                    f"mean_epe={mean_epe:.6f}, "
                    f"mean_norm_aee_per100={summary_rows[-1]['mean_norm_aee_per100pixel']:.6f}, "
                    f"mean_c_aee={summary_rows[-1]['mean_c_aee']:.6f}, "
                    f"elapsed={elapsed:.2f}s"
                )

        # test_all 根目录总体图/npy：跨所有 dataset 汇总，和 evaluate_all 的 output_root 总体文件对应。
        _save_mean_spectrum(
            all_image_pred_curves,
            all_image_gt_curves,
            test_base_dir,
            title=f"{class_name}-{data_type}-x{int(SCALE*SCALE)} Image Pair Mean Energy Spectrum",
            file_prefix="image_pair",
            save_npy=save_npy,
        )
        _save_mean_spectrum(
            all_flow_pred_curves,
            all_flow_gt_curves,
            test_base_dir,
            title=f"{class_name}-{data_type}-x{int(SCALE*SCALE)} Flow Mean Energy Spectrum",
            file_prefix="flow",
            save_npy=save_npy,
        )
        _save_flow_histogram_bundle(
            test_base_dir,
            all_delta_u_values,
            all_delta_v_values,
            all_delta_w_values,
            all_epe_values,
        )
        save_particle_error_histogram_bundle(
            test_base_dir,
            all_delta_vorticity_values,
            file_prefix="delta_vorticity",
            title=f"{class_name}-{data_type}-x{int(SCALE*SCALE)} All Dataset Delta Vorticity Error Distribution",
            xlabel="delta omega*",
            color="#4477AA",
            save_npy_fn=_save_optional_npy,
            save_npy=save_npy,
        )
        save_particle_error_histogram_bundle(
            test_base_dir,
            all_image_error_values,
            file_prefix="sr_error",
            title=f"{class_name}-{data_type}-x{int(SCALE*SCALE)} All Dataset Particle Image Error Distribution",
            save_npy_fn=_save_optional_npy,
            save_npy=save_npy,
        )
        _save_energy_spectrum_mse_compare_plot(
            all_class_image_pair_mean_rows,
            test_base_dir / "image_pair_energy_spectrum_mse_compare.png",
            title=f"{class_name}-{data_type}-x{int(SCALE*SCALE)} All Dataset Image Pair Energy Spectrum MSE",
        )
        _save_energy_spectrum_mse_compare_plot(
            all_class_flow_mean_rows,
            test_base_dir / "flow_energy_spectrum_mse_compare.png",
            title=f"{class_name}-{data_type}-x{int(SCALE*SCALE)} All Dataset Flow Energy Spectrum MSE",
        )
        _write_csv(test_base_dir / "ALL_CLASS_IMAGE_PAIR.CSV", all_class_image_pair_mean_rows)
        _write_csv(test_base_dir / "ALL_CLASS_flow.CSV", all_class_flow_mean_rows)
        _write_csv(test_base_dir / "metrics_all.csv", summary_rows)
        return summary_rows
    finally:
        if was_training:
            model.train()
