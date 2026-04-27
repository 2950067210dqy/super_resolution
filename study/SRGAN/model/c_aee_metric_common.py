import numpy as np


# 用户指定的组合系数：C-AEE = λ * ESE_norm + (1 - λ) * AEE_norm。
# 这里固定放在公共模块，确保 evaluate / evaluate_all / test_all 三条路径完全同口径。
C_AEE_LAMBDA = 0.5


def _safe_float(value) -> float:
    """尽量把输入转成 float；失败时返回 NaN。"""
    try:
        return float(value)
    except Exception:
        return float("nan")


def min_max_normalize(values, eps: float = 1e-12) -> np.ndarray:
    """
    对一维数值序列做 min-max 归一化。

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


def compute_c_aee_array(
    ese_values,
    aee_values,
    lambda_value: float = C_AEE_LAMBDA,
) -> np.ndarray:
    """
    按用户定义计算一组样本的 C-AEE。

    输入：
        - ese_values: 每个样本对应的超分辨率能量谱均方误差；
        - aee_values: 每个样本对应的 RAFT 平均端点误差；
    输出：
        - 与输入同长度的一维数组；无效样本位置为 NaN。

    公式：
        C-AEE = λ * ESE_norm + (1 - λ) * AEE_norm
    """
    ese_arr = np.asarray(ese_values, dtype=np.float32).reshape(-1)
    aee_arr = np.asarray(aee_values, dtype=np.float32).reshape(-1)
    if ese_arr.shape != aee_arr.shape:
        raise ValueError(
            f"ESE values and AEE values must have the same shape, got "
            f"{ese_arr.shape} vs {aee_arr.shape}"
        )

    c_aee = np.full(ese_arr.shape, np.nan, dtype=np.float32)
    valid_mask = np.isfinite(ese_arr) & np.isfinite(aee_arr)
    if not np.any(valid_mask):
        return c_aee

    ese_norm = min_max_normalize(ese_arr[valid_mask])
    aee_norm = min_max_normalize(aee_arr[valid_mask])
    c_aee[valid_mask] = lambda_value * ese_norm + (1.0 - lambda_value) * aee_norm
    return c_aee


def attach_c_aee_to_raft_rows(
    image_rows: list[dict],
    raft_rows: list[dict],
    sample_key_fields: tuple[str, ...],
    ese_key: str = "energy_spectrum_mse",
    aee_key: str = "VAL_AEE",
    output_key: str = "VAL_C_AEE",
    lambda_value: float = C_AEE_LAMBDA,
) -> None:
    """
    依据 sample key 把 image_pair 的 ESE 与 RAFT 的 AEE 配对，并写回每条 RAFT 行。

    典型场景：
        - evaluate_all：同一个 sample 会先产生 previous / next 两条图像记录，
          再产生一条 RAFT 记录。这里先把 previous/next 的 ESE 取平均，
          再和该 sample 的 AEE 配对计算 C-AEE；
        - test_all：同样是一条 sample 对应两条 image_pair 行和一条 RAFT 行。

    注意：
        image_rows 和 raft_rows 都是“原地修改”：
        - previous/next 图像行不写 C-AEE，避免一个样本被重复计两次；
        - 只有 RAFT 行会新增 output_key 字段。
    """
    if not raft_rows:
        return

    sample_to_ese_values: dict[tuple, list[float]] = {}
    for row in image_rows:
        sample_key = tuple(row.get(field) for field in sample_key_fields)
        ese_value = _safe_float(row.get(ese_key, float("nan")))
        if np.isfinite(ese_value):
            sample_to_ese_values.setdefault(sample_key, []).append(ese_value)

    ese_values = []
    aee_values = []
    for row in raft_rows:
        sample_key = tuple(row.get(field) for field in sample_key_fields)
        sample_ese_list = sample_to_ese_values.get(sample_key, [])
        ese_values.append(float(np.mean(sample_ese_list)) if sample_ese_list else float("nan"))
        aee_values.append(_safe_float(row.get(aee_key, float("nan"))))

    c_aee_values = compute_c_aee_array(ese_values, aee_values, lambda_value=lambda_value)
    for row, c_aee_value in zip(raft_rows, c_aee_values):
        row[output_key] = float(c_aee_value) if np.isfinite(c_aee_value) else float("nan")
