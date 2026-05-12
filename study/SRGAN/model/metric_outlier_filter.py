from __future__ import annotations

import numpy as np


LOWER_IS_BETTER_HINTS = (
    "loss",
    "mse",
    "rmse",
    "nrmse",
    "epe",
    "aee",
    "error",
)
HIGHER_IS_BETTER_HINTS = (
    "psnr",
    "ssim",
    "r2",
    "tke_acc",
    "1px",
    "3px",
    "5px",
)


def _get_esrgan_config(global_data):
    """兼容传入 global_data 或 global_data.esrgan 两种对象。"""
    return getattr(global_data, "esrgan", global_data)


def _metric_direction(metric_key: str) -> str:
    """
    判断指标方向，用于只剔除“坏方向”的异常点。

    例如：
    - EPE/MSE/AEE/C-AEE 越小越好，只剔除异常偏大的坏点；
    - PSNR/SSIM/R2/TKE_ACC 越大越好，只剔除异常偏小的坏点。
    这样保留每个样本原始记录不变，同时让汇总均值更少被极端失败样本拖偏。
    """
    key = str(metric_key or "").lower()
    if any(hint in key for hint in HIGHER_IS_BETTER_HINTS):
        return "higher"
    if any(hint in key for hint in LOWER_IS_BETTER_HINTS):
        return "lower"
    return "two_sided"


def _filter_bad_side_iqr(values: np.ndarray, metric_key: str, iqr_factor: float) -> np.ndarray:
    """
    使用 IQR 剔除异常值。

    这里默认按指标方向只剔除“坏方向”异常值：
    - lower-better 指标：剔除高于 Q3 + k*IQR 的值；
    - higher-better 指标：剔除低于 Q1 - k*IQR 的值；
    - 未知方向指标：退回双侧 IQR。
    """
    q1, q3 = np.percentile(values, [25.0, 75.0])
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr <= 0:
        return values

    direction = _metric_direction(metric_key)
    lower_bound = q1 - float(iqr_factor) * iqr
    upper_bound = q3 + float(iqr_factor) * iqr
    if direction == "lower":
        mask = values <= upper_bound
    elif direction == "higher":
        mask = values >= lower_bound
    else:
        mask = (values >= lower_bound) & (values <= upper_bound)
    filtered = values[mask]
    return filtered if filtered.size > 0 else values


def robust_metric_mean(values, *, metric_key: str = "", global_data=None) -> float:
    """
    计算用于 evaluate_all/test_all 汇总行的稳健均值。

    注意：该函数只服务 MEAN/CLASS_MEAN/metrics_all 等“平均评价指标”，不会改动逐样本原始指标。
    全局开关：
    - METRIC_OUTLIER_FILTER_ENABLED: 是否启用异常值剔除；
    - METRIC_OUTLIER_FILTER_IQR_FACTOR: IQR 阈值系数，默认 1.5；
    - METRIC_OUTLIER_FILTER_MIN_COUNT: 样本数少于该值时不剔除，避免小样本类别被过度过滤。
    """
    arr = np.asarray(list(values), dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")

    cfg = _get_esrgan_config(global_data)
    enabled = bool(getattr(cfg, "METRIC_OUTLIER_FILTER_ENABLED", True))
    min_count = int(getattr(cfg, "METRIC_OUTLIER_FILTER_MIN_COUNT", 8))
    iqr_factor = float(getattr(cfg, "METRIC_OUTLIER_FILTER_IQR_FACTOR", 1.5))
    if not enabled or arr.size < max(1, min_count):
        return float(np.mean(arr, dtype=np.float64))

    filtered = _filter_bad_side_iqr(arr, metric_key=metric_key, iqr_factor=iqr_factor)
    return float(np.mean(filtered, dtype=np.float64))
