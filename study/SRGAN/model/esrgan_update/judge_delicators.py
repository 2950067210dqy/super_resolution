"""
评价指标 start
"""
import math

import numpy as np
import torch


def _to_np_chw(x: torch.Tensor) -> np.ndarray:
    """将单张张量从 torch 转为 numpy，形状保持为 [C,H,W]。"""
    return x.detach().float().cpu().numpy()


def _mse(pred_chw: np.ndarray, gt_chw: np.ndarray) -> float:
    """计算均方误差 MSE。"""
    return float(np.mean((pred_chw - gt_chw) ** 2))


def _psnr_from_mse(mse: float) -> float:
    """由 MSE 计算 PSNR（假设数据范围 [0,1]）。"""
    return float("inf") if mse == 0 else 20.0 * math.log10(1.0 / math.sqrt(mse))


def _r2_score(pred_chw: np.ndarray, gt_chw: np.ndarray, eps: float = 1e-12) -> float:
    """计算决定系数 R²。"""
    y_true = gt_chw.reshape(-1)
    y_pred = pred_chw.reshape(-1)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / (ss_tot + eps)


def _nrmse(pred_chw: np.ndarray, gt_chw: np.ndarray, eps: float = 1e-12) -> float:
    """计算归一化均方根误差 NRMSE（按真值范围归一化）。"""
    rmse = math.sqrt(float(np.mean((pred_chw - gt_chw) ** 2)))
    den = float(np.max(gt_chw) - np.min(gt_chw))
    return rmse / (den + eps)


def _ssim_score(pred_chw: np.ndarray, gt_chw: np.ndarray) -> float:
    """计算 SSIM（按通道平均；优先使用 skimage，失败则回退简化公式）。"""
    try:
        from skimage.metrics import structural_similarity as sk_ssim
        vals = []
        cnum = min(pred_chw.shape[0], gt_chw.shape[0])
        for c in range(cnum):
            p = pred_chw[c]
            g = gt_chw[c]
            dr = float(np.max(g) - np.min(g))
            dr = dr if dr > 1e-12 else 1.0
            vals.append(float(sk_ssim(g, p, data_range=dr)))
        return float(np.mean(vals))
    except Exception:
        vals = []
        cnum = min(pred_chw.shape[0], gt_chw.shape[0])
        C1, C2 = 0.01**2, 0.03**2
        for c in range(cnum):
            x = pred_chw[c]
            y = gt_chw[c]
            mx, my = float(np.mean(x)), float(np.mean(y))
            vx, vy = float(np.var(x)), float(np.var(y))
            cov = float(np.mean((x - mx) * (y - my)))
            num = (2 * mx * my + C1) * (2 * cov + C2)
            den = (mx * mx + my * my + C1) * (vx + vy + C2)
            vals.append(num / den if den != 0 else 0.0)
        return float(np.mean(vals))


def _tke_reconstruction_accuracy(pred_chw: np.ndarray, gt_chw: np.ndarray, eps: float = 1e-12) -> float:
    """计算 TKE 重建精度（使用前两通道 u,v）。"""
    if pred_chw.shape[0] < 2 or gt_chw.shape[0] < 2:
        return float("nan")
    up, vp = pred_chw[0], pred_chw[1]
    ug, vg = gt_chw[0], gt_chw[1]
    up_p = up - np.mean(up)
    vp_p = vp - np.mean(vp)
    ug_p = ug - np.mean(ug)
    vg_p = vg - np.mean(vg)
    tke_p = 0.5 * float(np.mean(up_p**2 + vp_p**2))
    tke_g = 0.5 * float(np.mean(ug_p**2 + vg_p**2))
    return 1.0 - abs(tke_p - tke_g) / (abs(tke_g) + eps)


def _radial_spectrum(ch2d: np.ndarray) -> np.ndarray:
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


def _energy_spectrum_curves(pred_chw: np.ndarray, gt_chw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """计算多通道平均能量谱曲线，返回 (pred_curve, gt_curve)。"""
    pred_specs, gt_specs = [], []
    cnum = min(pred_chw.shape[0], gt_chw.shape[0])
    min_len = None

    for c in range(cnum):
        sp = _radial_spectrum(pred_chw[c])
        sg = _radial_spectrum(gt_chw[c])
        n = min(len(sp), len(sg))
        min_len = n if min_len is None else min(min_len, n)
        pred_specs.append(sp[:n])
        gt_specs.append(sg[:n])

    pred_curve = np.mean(np.stack([x[:min_len] for x in pred_specs], axis=0), axis=0)
    gt_curve = np.mean(np.stack([x[:min_len] for x in gt_specs], axis=0), axis=0)
    return pred_curve, gt_curve


def _energy_spectrum_mse(pred_chw: np.ndarray, gt_chw: np.ndarray) -> float:
    """计算能量谱误差（log1p 谱差 MSE）。"""
    pred_curve, gt_curve = _energy_spectrum_curves(pred_chw, gt_chw)
    return float(np.mean((np.log1p(pred_curve) - np.log1p(gt_curve)) ** 2))
"""
计算评价指标 end
"""