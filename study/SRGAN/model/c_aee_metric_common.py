import numpy as np


# 用户指定的新 C-AEE 组合系数：
#   C-AEE = λ1 * ESMSE_norm + λ2 * EPE_norm + λ3 * SSIM_error_norm
# 其中：
#   ESMSE           = 超分辨率能量谱均方误差；
#   EPE             = RAFT 平均端点误差；
#   SSIM_error      = 1 - SSIM。
# 这些常量放在公共模块，确保 evaluate / evaluate_all / test_all 三条路径完全同口径。
C_AEE_LAMBDA_ESMSE = 0.3
C_AEE_LAMBDA_EPE = 0.3
C_AEE_LAMBDA_SSIM_ERROR = 0.2

# 兼容旧导入名：外部如果还引用 C_AEE_LAMBDA，不会因为本次公式升级直接报错。
# 新代码不再使用它做计算。
C_AEE_LAMBDA = C_AEE_LAMBDA_ESMSE

# C-AEE 绝对归一化参考尺度：
# - ESMSE、EPE、(1-SSIM) 的量纲/数值范围不同，直接相加会让数值大的项主导总分；
# - 这里先除以固定参考尺度，把三项都变成“相对于参考误差的无量纲误差”；
# - 默认 1.0 表示“以 1 个单位误差作为参考”。如果后续你希望 EPE 按 12px 位移量程归一化，
#   可以把 C_AEE_EPE_ABS_SCALE 改成 12.0；ESMSE 或 SSIM error 有新的固定参考值时也只改这里；
# - 不裁剪到 [0, 1]，超过参考尺度时允许归一化值 > 1，避免大误差被压成同一个 1。
C_AEE_ESMSE_ABS_SCALE = 1.0
C_AEE_EPE_ABS_SCALE = 1.0
C_AEE_SSIM_ERROR_ABS_SCALE = 1.0

# 兼容旧常量名：旧代码/调试脚本如果还用 ESE/AEE 命名，仍然得到同一组绝对尺度。
C_AEE_ESE_ABS_SCALE = C_AEE_ESMSE_ABS_SCALE
C_AEE_AEE_ABS_SCALE = C_AEE_EPE_ABS_SCALE


def _safe_float(value) -> float:
    """尽量把输入转成 float；失败时返回 NaN。"""
    try:
        return float(value)
    except Exception:
        return float("nan")


def min_max_normalize(values, eps: float = 1e-12) -> np.ndarray:
    """
    对一维数值序列做 min-max 归一化。

    注意：
        这是历史相对归一化工具，当前 C-AEE 已经改为 absolute_error_normalize。
        保留该函数是为了不破坏外部可能存在的导入或调试脚本。

    规则：
        1. 只对有限值参与 min/max 统计；
        2. 原始 NaN/Inf 位置保持 NaN，便于后续均值时自动跳过；
        3. 若所有有效值都相同，归一化结果统一置 0。

    说明：
        当 max == min 时，常规 min-max 公式会出现除 0。
        这里把常量序列映射为 0，表示“在当前评估集合内部没有相对差异”，
        这样不会凭空放大或制造 C-AEE 的差别。
    """
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    normalized = np.full(arr.shape, np.nan, dtype=np.float32)
    valid_mask = np.isfinite(arr)
    if not np.any(valid_mask):
        return normalized

    valid_values = arr[valid_mask]
    min_value = float(np.min(valid_values))
    max_value = float(np.max(valid_values))
    if max_value - min_value <= eps:
        normalized[valid_mask] = 0.0
        return normalized

    normalized[valid_mask] = (valid_values - min_value) / (max_value - min_value)
    return normalized


def absolute_error_normalize(values, abs_scale: float, eps: float = 1e-12) -> np.ndarray:
    """
    对误差指标做固定参考尺度的绝对归一化。

    公式：
        normalized_error = max(error, 0) / abs_scale

    设计原因：
        1. ESMSE、EPE、(1-SSIM) 都是误差，理论最优值是 0，所以绝对归一化的下界固定为 0；
        2. abs_scale 是人为指定的固定参考尺度，不再从当前 batch / dataset 里取 min/max；
        3. 不裁剪到 [0, 1]，这样当误差超过参考尺度时仍然能保留“超过多少”的信息；
        4. NaN/Inf 保持 NaN，后续均值统计会自动跳过无效样本。
    """
    scale = _safe_float(abs_scale)
    if not np.isfinite(scale) or scale <= eps:
        raise ValueError(f"C-AEE absolute normalization scale must be finite and > 0, got {abs_scale}")

    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    normalized = np.full(arr.shape, np.nan, dtype=np.float32)
    valid_mask = np.isfinite(arr)
    if not np.any(valid_mask):
        return normalized

    # ESMSE/EPE/(1-SSIM) 这类误差偶尔可能因为数值误差出现极小负数；
    # 绝对归一化中把负误差钳到 0，避免产生“比完美更好”的负 C-AEE。
    normalized[valid_mask] = np.maximum(arr[valid_mask], 0.0) / scale
    return normalized


def ssim_to_error(ssim_values) -> np.ndarray:
    """
    将 SSIM 相似性分数转换成误差项 `1 - SSIM`。

    SSIM 越接近 1 表示结构越相似，因此 C-AEE 里使用 `1-SSIM` 才能保持“越小越好”。
    由于实际实现或异常图像可能让 SSIM 略大于 1，这里把负误差钳到 0；
    若 SSIM 为 NaN/Inf，则保持 NaN，后续 C-AEE 会自动跳过该样本。
    """
    ssim_arr = np.asarray(ssim_values, dtype=np.float32).reshape(-1)
    ssim_error = np.full(ssim_arr.shape, np.nan, dtype=np.float32)
    valid_mask = np.isfinite(ssim_arr)
    if not np.any(valid_mask):
        return ssim_error
    ssim_error[valid_mask] = np.maximum(1.0 - ssim_arr[valid_mask], 0.0)
    return ssim_error


def compute_c_aee_array(
    esmse_values,
    epe_values,
    ssim_values=None,
    lambda_esmse: float = C_AEE_LAMBDA_ESMSE,
    lambda_epe: float = C_AEE_LAMBDA_EPE,
    lambda_ssim_error: float = C_AEE_LAMBDA_SSIM_ERROR,
    esmse_abs_scale: float = C_AEE_ESMSE_ABS_SCALE,
    epe_abs_scale: float = C_AEE_EPE_ABS_SCALE,
    ssim_error_abs_scale: float = C_AEE_SSIM_ERROR_ABS_SCALE,
) -> np.ndarray:
    """
    按用户定义计算一组样本的 C-AEE。

    输入：
        - esmse_values: 每个样本对应的超分辨率能量谱均方误差；
        - epe_values: 每个样本对应的 RAFT 平均端点误差；
        - ssim_values: 每个样本对应的 SR 图像 SSIM 相似性分数；
    输出：
        - 与输入同长度的一维数组；无效样本位置为 NaN。

    公式：
        C-AEE = λ1 * ESMSE_norm + λ2 * EPE_norm + λ3 * SSIM_error_norm

    当前归一化口径：
        ESMSE_norm      = ESMSE / esmse_abs_scale
        EPE_norm        = EPE / epe_abs_scale
        SSIM_error_norm = max(1 - SSIM, 0) / ssim_error_abs_scale

    这里已经不再使用当前样本集合的 min/max，因此不同实验只要 abs_scale 不变，
    C-AEE 就处在同一个绝对参考尺度上，越小越好。
    """
    esmse_arr = np.asarray(esmse_values, dtype=np.float32).reshape(-1)
    epe_arr = np.asarray(epe_values, dtype=np.float32).reshape(-1)
    if ssim_values is None:
        # 兼容旧外部脚本：新公式必须有 SSIM；若旧脚本没传，返回 NaN 而不是直接 TypeError。
        ssim_arr = np.full(esmse_arr.shape, np.nan, dtype=np.float32)
    else:
        ssim_arr = np.asarray(ssim_values, dtype=np.float32).reshape(-1)
    if esmse_arr.shape != epe_arr.shape or esmse_arr.shape != ssim_arr.shape:
        raise ValueError(
            f"ESMSE, EPE and SSIM values must have the same shape, got "
            f"{esmse_arr.shape}, {epe_arr.shape}, {ssim_arr.shape}"
        )

    c_aee = np.full(esmse_arr.shape, np.nan, dtype=np.float32)
    valid_mask = np.isfinite(esmse_arr) & np.isfinite(epe_arr) & np.isfinite(ssim_arr)
    if not np.any(valid_mask):
        return c_aee

    esmse_norm = absolute_error_normalize(esmse_arr[valid_mask], esmse_abs_scale)
    epe_norm = absolute_error_normalize(epe_arr[valid_mask], epe_abs_scale)
    ssim_error_norm = absolute_error_normalize(
        ssim_to_error(ssim_arr[valid_mask]),
        ssim_error_abs_scale,
    )
    c_aee[valid_mask] = (
        lambda_esmse * esmse_norm
        + lambda_epe * epe_norm
        + lambda_ssim_error * ssim_error_norm
    )
    return c_aee


def compute_c_aee_value(
    esmse_value,
    epe_value,
    ssim_value=None,
    *,
    ssim_error_value=None,
    lambda_esmse: float = C_AEE_LAMBDA_ESMSE,
    lambda_epe: float = C_AEE_LAMBDA_EPE,
    lambda_ssim_error: float = C_AEE_LAMBDA_SSIM_ERROR,
    esmse_abs_scale: float = C_AEE_ESMSE_ABS_SCALE,
    epe_abs_scale: float = C_AEE_EPE_ABS_SCALE,
    ssim_error_abs_scale: float = C_AEE_SSIM_ERROR_ABS_SCALE,
) -> float:
    """
    计算单个平均指标对应的 C-AEE。

    evaluate 的训练中验证只需要一个整体分数，因此应先求平均 ESMSE / 平均 EPE / 平均 SSIM，
    再调用这个函数得到最终 C-AEE，避免“先逐样本算 C-AEE 再平均”的口径混乱。

    参数说明：
        - ssim_value 是 SSIM 相似性，函数内部会转成 1-SSIM；
        - ssim_error_value 是已经算好的 1-SSIM，例如训练 validate_raft 中的 SSIMLoss。
          二者只需要传一个，优先使用 ssim_error_value。
    """
    if ssim_error_value is not None:
        ssim_value_for_formula = 1.0 - _safe_float(ssim_error_value)
    else:
        ssim_value_for_formula = ssim_value
    c_aee_arr = compute_c_aee_array(
        [esmse_value],
        [epe_value],
        [ssim_value_for_formula],
        lambda_esmse=lambda_esmse,
        lambda_epe=lambda_epe,
        lambda_ssim_error=lambda_ssim_error,
        esmse_abs_scale=esmse_abs_scale,
        epe_abs_scale=epe_abs_scale,
        ssim_error_abs_scale=ssim_error_abs_scale,
    )
    c_aee_value = float(c_aee_arr[0]) if c_aee_arr.size else float("nan")
    return c_aee_value if np.isfinite(c_aee_value) else float("nan")


def attach_c_aee_to_raft_rows(
    image_rows: list[dict],
    raft_rows: list[dict],
    sample_key_fields: tuple[str, ...],
    ese_key: str = "energy_spectrum_mse",
    aee_key: str = "VAL_AEE",
    ssim_key: str = "ssim",
    output_key: str = "VAL_C_AEE",
    lambda_esmse: float = C_AEE_LAMBDA_ESMSE,
    lambda_epe: float = C_AEE_LAMBDA_EPE,
    lambda_ssim_error: float = C_AEE_LAMBDA_SSIM_ERROR,
    esmse_abs_scale: float = C_AEE_ESMSE_ABS_SCALE,
    epe_abs_scale: float = C_AEE_EPE_ABS_SCALE,
    ssim_error_abs_scale: float = C_AEE_SSIM_ERROR_ABS_SCALE,
) -> None:
    """
    依据 sample key 把 image_pair 的 ESMSE/SSIM 与 RAFT 的 EPE 配对，并写回每条 RAFT 行。

    典型场景：
        - evaluate_all：同一个 sample 会先产生 previous / next 两条图像记录，
          再产生一条 RAFT 记录。这里先把 previous/next 的 ESMSE 和 SSIM 分别取平均，
          再和该 sample 的 EPE 配对计算 C-AEE；
        - test_all：同样是一条 sample 对应两条 image_pair 行和一条 RAFT 行。

    归一化口径：
        C-AEE 使用固定参考尺度的绝对归一化，不再使用当前 dataset 的 min/max：
        ESMSE / EPE / (1-SSIM) 先放缩到同一无量纲尺度，再按 λ1/λ2/λ3 加权。

    注意：
        image_rows 和 raft_rows 都是“原地修改”：
        - previous/next 图像行不写 C-AEE，避免一个样本被重复计两次；
        - 只有 RAFT 行会新增 output_key 字段。
    """
    if not raft_rows:
        return

    sample_to_esmse_values: dict[tuple, list[float]] = {}
    sample_to_ssim_values: dict[tuple, list[float]] = {}
    for row in image_rows:
        sample_key = tuple(row.get(field) for field in sample_key_fields)
        esmse_value = _safe_float(row.get(ese_key, float("nan")))
        ssim_value = _safe_float(row.get(ssim_key, float("nan")))
        if np.isfinite(esmse_value):
            sample_to_esmse_values.setdefault(sample_key, []).append(esmse_value)
        if np.isfinite(ssim_value):
            sample_to_ssim_values.setdefault(sample_key, []).append(ssim_value)

    esmse_values = []
    epe_values = []
    ssim_values = []
    for row in raft_rows:
        sample_key = tuple(row.get(field) for field in sample_key_fields)
        sample_esmse_list = sample_to_esmse_values.get(sample_key, [])
        sample_ssim_list = sample_to_ssim_values.get(sample_key, [])
        esmse_values.append(float(np.mean(sample_esmse_list)) if sample_esmse_list else float("nan"))
        ssim_values.append(float(np.mean(sample_ssim_list)) if sample_ssim_list else float("nan"))
        epe_values.append(_safe_float(row.get(aee_key, float("nan"))))

    c_aee_values = compute_c_aee_array(
        esmse_values,
        epe_values,
        ssim_values,
        lambda_esmse=lambda_esmse,
        lambda_epe=lambda_epe,
        lambda_ssim_error=lambda_ssim_error,
        esmse_abs_scale=esmse_abs_scale,
        epe_abs_scale=epe_abs_scale,
        ssim_error_abs_scale=ssim_error_abs_scale,
    )
    for row, c_aee_value in zip(raft_rows, c_aee_values):
        row[output_key] = float(c_aee_value) if np.isfinite(c_aee_value) else float("nan")
