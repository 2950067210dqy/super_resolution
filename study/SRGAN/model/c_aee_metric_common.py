import csv
from pathlib import Path

import numpy as np


# 用户指定的新 C-AEE 组合系数：
#   C-AEE = λ1 * ESMSE_norm + λ2 * EPE_norm + λ3 * SSIM_error_norm
# 其中：
#   ESMSE           = 超分辨率能量谱均方误差；
#   EPE             = RAFT 平均端点误差；
#   SSIM_error      = 1 - SSIM。
# 这些常量放在公共模块，确保 evaluate / evaluate_all / test_all 三条路径完全同口径。
# 当前比例按用户要求设置为 0.4 : 0.4 : 0.2：
#   - ESMSE_norm 占 40%，强调超分辨率频谱结构；
#   - EPE_norm 占 40%，强调 RAFT/PIV 位移端点误差；
#   - SSIM_error_norm 占 20%，保留图像结构相似性约束。
C_AEE_LAMBDA_ESMSE = 0.4
C_AEE_LAMBDA_EPE = 0.4
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


_MEAN_ROW_IDS = {"MEAN", "CLASS_MEAN", "ALL_MEAN"}


def _read_csv_rows(csv_path: str | Path) -> tuple[list[dict], list[str]]:
    """读取已有指标 CSV，并保留原始列顺序，便于重写时不破坏旧表头。"""
    path = Path(csv_path)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def _write_csv_rows(csv_path: str | Path, rows: list[dict], fieldnames: list[str]) -> None:
    """按原列顺序覆盖写回 CSV；新增的 C-AEE 列会追加到末尾。"""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _ensure_csv_field(fieldnames: list[str], rows: list[dict], field_name: str) -> list[str]:
    """
    确保 CSV 表头包含指定列。

    历史 CSV 可能还没有 VAL_C_AEE / C_AEE；重算模式需要能直接补列并覆盖回去。
    """
    merged = list(fieldnames)
    if field_name not in merged:
        merged.append(field_name)
    for row in rows:
        row.setdefault(field_name, "")
    return merged


def _is_mean_row(row: dict) -> bool:
    """识别 evaluate_all/test_all 写出的 MEAN / CLASS_MEAN 汇总行。"""
    for key in ("sample_id", "sample_index", "sid", "sample_key"):
        value = str(row.get(key, "")).strip().upper()
        if value in _MEAN_ROW_IDS:
            return True
    return False


def _finite_mean(values) -> float:
    """只对有限值求均值；没有有效值时返回 NaN。"""
    arr = np.asarray([_safe_float(value) for value in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size > 0 else float("nan")


def _guess_sample_key_fields(image_rows: list[dict], raft_rows: list[dict]) -> tuple[str, ...]:
    """
    自动判断 CSV 的样本键。

    evaluate_all 使用 class_name + sample_id；test_all 使用 dataset + sample_index。
    用自动判断可以让同一个重算函数同时覆盖 evaluate_all/test_all 的历史 CSV。
    """
    if not image_rows or not raft_rows:
        return ("class_name", "sample_id")
    image_keys = set(image_rows[0].keys())
    raft_keys = set(raft_rows[0].keys())
    for candidate in (("dataset", "sample_index"), ("class_name", "sample_id"), ("class_name", "sid")):
        if all(field in image_keys and field in raft_keys for field in candidate):
            return candidate
    return ("class_name", "sample_id")


def _guess_c_aee_column_names(raft_rows: list[dict], fieldnames: list[str]) -> tuple[str, str]:
    """
    自动判断 RAFT 误差列与 C-AEE 输出列。

    evaluate_all 的端点误差列叫 VAL_AEE，输出列叫 VAL_C_AEE；
    test_all 的端点误差列叫 epe，输出列叫 C_AEE。
    """
    available = set(fieldnames)
    if raft_rows:
        available.update(raft_rows[0].keys())
    if "epe" in available or "C_AEE" in available:
        return "epe", "C_AEE"
    return "VAL_AEE", "VAL_C_AEE"


def _refresh_c_aee_mean_rows(raft_rows: list[dict], output_key: str) -> None:
    """
    用逐 sample 的 C-AEE 平均值刷新 MEAN 行。

    attach_c_aee_to_raft_rows 会同时给 MEAN 行按“均值指标”算一次 C-AEE；
    但现有 metrics_raft.csv 的均值口径是“逐样本 C-AEE 再求均值”，这里保持同口径。
    """
    sample_values = [
        _safe_float(row.get(output_key, float("nan")))
        for row in raft_rows
        if not _is_mean_row(row)
    ]
    mean_value = _finite_mean(sample_values)
    if not np.isfinite(mean_value):
        return
    for row in raft_rows:
        if _is_mean_row(row):
            row[output_key] = mean_value


def recalculate_c_aee_csv_pair(
    image_csv_path: str | Path,
    raft_csv_path: str | Path,
    *,
    refresh_mean_rows: bool = True,
) -> dict:
    """
    读取一组 image_pair CSV + RAFT CSV，按当前公式重算 C-AEE 并覆盖 RAFT CSV。

    这个函数不重新跑模型、不读取图像，只依赖 CSV 中已经保存的：
        - image_pair: energy_spectrum_mse + ssim；
        - raft/flow: VAL_AEE 或 epe。
    因此适合在调整 C-AEE 权重后快速修正历史结果。
    """
    image_path = Path(image_csv_path)
    raft_path = Path(raft_csv_path)
    if not image_path.exists() or not raft_path.exists():
        return {
            "status": "missing",
            "image_csv": str(image_path),
            "raft_csv": str(raft_path),
            "updated_rows": 0,
        }

    image_rows, _ = _read_csv_rows(image_path)
    raft_rows, raft_fieldnames = _read_csv_rows(raft_path)
    if not image_rows or not raft_rows:
        return {
            "status": "empty",
            "image_csv": str(image_path),
            "raft_csv": str(raft_path),
            "updated_rows": 0,
        }

    aee_key, output_key = _guess_c_aee_column_names(raft_rows, raft_fieldnames)
    sample_key_fields = _guess_sample_key_fields(image_rows, raft_rows)
    raft_fieldnames = _ensure_csv_field(raft_fieldnames, raft_rows, output_key)

    before_values = [row.get(output_key, "") for row in raft_rows]
    attach_c_aee_to_raft_rows(
        image_rows=image_rows,
        raft_rows=raft_rows,
        sample_key_fields=sample_key_fields,
        ese_key="energy_spectrum_mse",
        aee_key=aee_key,
        ssim_key="ssim",
        output_key=output_key,
    )
    if refresh_mean_rows:
        # 普通 metrics_raft.csv 含逐 sample 行和 MEAN 行，MEAN 行应保持“逐样本 C-AEE 再求均值”的口径。
        _refresh_c_aee_mean_rows(raft_rows, output_key)
    _write_csv_rows(raft_path, raft_rows, raft_fieldnames)

    updated_rows = sum(
        str(before) != str(row.get(output_key, ""))
        for before, row in zip(before_values, raft_rows)
    )
    return {
        "status": "updated",
        "image_csv": str(image_path),
        "raft_csv": str(raft_path),
        "output_key": output_key,
        "updated_rows": updated_rows,
    }


def recalculate_c_aee_combined_csv(csv_path: str | Path) -> dict:
    """
    对 image_pair 和 RAFT 行同在一个文件里的 CSV 原地重算 C-AEE。

    evaluate_all 的 metrics_all.csv 通常是这种结构；如果文件里只有 RAFT 行，本函数会安全跳过。
    """
    path = Path(csv_path)
    if not path.exists():
        return {"status": "missing", "csv": str(path), "updated_rows": 0}

    rows, fieldnames = _read_csv_rows(path)
    if not rows:
        return {"status": "empty", "csv": str(path), "updated_rows": 0}

    pair_type_values = [str(row.get("pair_type", "")).strip().lower() for row in rows]
    image_rows = [
        row for row, pair_type in zip(rows, pair_type_values)
        if pair_type in {"previous", "next", "image_pair"}
    ]
    raft_rows = [
        row for row, pair_type in zip(rows, pair_type_values)
        if pair_type in {"raft", "flow"}
    ]
    if not image_rows or not raft_rows:
        return {"status": "skipped_no_pair_rows", "csv": str(path), "updated_rows": 0}

    aee_key, output_key = _guess_c_aee_column_names(raft_rows, fieldnames)
    sample_key_fields = _guess_sample_key_fields(image_rows, raft_rows)
    fieldnames = _ensure_csv_field(fieldnames, rows, output_key)

    before_values = [row.get(output_key, "") for row in raft_rows]
    attach_c_aee_to_raft_rows(
        image_rows=image_rows,
        raft_rows=raft_rows,
        sample_key_fields=sample_key_fields,
        ese_key="energy_spectrum_mse",
        aee_key=aee_key,
        ssim_key="ssim",
        output_key=output_key,
    )
    # combined metrics_all.csv 的 MEAN 行通常 pair_type=all，不属于 raft_rows；
    # 因此这里传入完整 rows，让总表里的 VAL_C_AEE 汇总行也同步刷新。
    _refresh_c_aee_mean_rows(rows, output_key)
    _write_csv_rows(path, rows, fieldnames)

    updated_rows = sum(
        str(before) != str(row.get(output_key, ""))
        for before, row in zip(before_values, raft_rows)
    )
    return {
        "status": "updated",
        "csv": str(path),
        "output_key": output_key,
        "updated_rows": updated_rows,
    }


def _refresh_test_all_summary_csv(output_root: Path) -> dict:
    """
    test_all 的 metrics_all.csv 只有各 dataset 的均值行。

    单独重算 dataset/metrics_raft.csv 后，同步把 summary 里的 mean_c_aee 更新成对应均值。
    """
    summary_path = output_root / "metrics_all.csv"
    if not summary_path.exists():
        return {"status": "missing", "csv": str(summary_path), "updated_rows": 0}
    rows, fieldnames = _read_csv_rows(summary_path)
    if "mean_c_aee" not in fieldnames:
        return {"status": "skipped_no_mean_c_aee", "csv": str(summary_path), "updated_rows": 0}

    updated_rows = 0
    for row in rows:
        dataset_name = str(row.get("dataset", "")).strip()
        if not dataset_name:
            continue
        raft_csv = output_root / dataset_name / "metrics_raft.csv"
        if not raft_csv.exists():
            continue
        raft_rows, _ = _read_csv_rows(raft_csv)
        mean_rows = [raft_row for raft_row in raft_rows if _is_mean_row(raft_row)]
        source_row = mean_rows[0] if mean_rows else None
        if source_row is None:
            continue
        new_value = source_row.get("C_AEE", source_row.get("VAL_C_AEE", ""))
        if str(row.get("mean_c_aee", "")) != str(new_value):
            row["mean_c_aee"] = new_value
            updated_rows += 1
    if updated_rows > 0:
        _write_csv_rows(summary_path, rows, fieldnames)
    return {"status": "updated" if updated_rows else "unchanged", "csv": str(summary_path), "updated_rows": updated_rows}


def recalculate_c_aee_for_metric_outputs(
    output_root: str | Path,
    metrics_csv_path: str | Path | None = None,
    logger=None,
) -> dict:
    """
    重算 evaluate_all/test_all 已有输出目录中的所有 C-AEE CSV。

    覆盖范围：
        - evaluate_all 根目录 metrics_all.csv；
        - evaluate_all 根目录 *_image_pair.csv + *_raft.csv；
        - evaluate_all/test_all 各类别或 dataset 子目录 metrics_image_pair.csv + metrics_raft.csv；
        - test_all 子目录 metrics.csv（它等同于 flow/RAFT 指标表）；
        - ALL_CLASS_IMAGE_PAIR.CSV + ALL_CLASS_flow.CSV；
        - test_all 的 metrics_all.csv 汇总 mean_c_aee。

    只改 CSV 中 C-AEE 相关列，不重新计算模型输出，也不保存任何图像/NPY。
    """
    root = Path(output_root)
    summary = {
        "output_root": str(root),
        "combined": [],
        "pairs": [],
        "all_class": [],
        "summary": [],
    }

    combined_paths: list[Path] = []
    if metrics_csv_path is not None:
        combined_paths.append(Path(metrics_csv_path))
    combined_paths.extend([root / "metrics.csv", root / "metrics_all.csv"])

    seen_combined: set[str] = set()
    for path in combined_paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen_combined:
            continue
        seen_combined.add(key)
        result = recalculate_c_aee_combined_csv(path)
        summary["combined"].append(result)
        if logger is not None:
            logger.info(f"[C-AEE recalc] combined csv: {result}")

    pairs: list[tuple[Path, Path]] = []

    def add_pair(image_path: Path, raft_path: Path) -> None:
        pairs.append((image_path, raft_path))

    if metrics_csv_path is not None:
        metric_path = Path(metrics_csv_path)
        add_pair(
            metric_path.with_name(f"{metric_path.stem}_image_pair{metric_path.suffix}"),
            metric_path.with_name(f"{metric_path.stem}_raft{metric_path.suffix}"),
        )

    add_pair(root / "metrics_image_pair.csv", root / "metrics_raft.csv")
    add_pair(root / "metrics_image_pair.csv", root / "metrics.csv")
    if root.exists():
        for image_path in root.rglob("metrics_image_pair.csv"):
            add_pair(image_path, image_path.with_name("metrics_raft.csv"))
            add_pair(image_path, image_path.with_name("metrics.csv"))

    seen_pairs: set[tuple[str, str]] = set()
    for image_path, raft_path in pairs:
        pair_key = (str(image_path.resolve()) if image_path.exists() else str(image_path),
                    str(raft_path.resolve()) if raft_path.exists() else str(raft_path))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        result = recalculate_c_aee_csv_pair(image_path, raft_path)
        summary["pairs"].append(result)
        if logger is not None:
            logger.info(f"[C-AEE recalc] split csv: {result}")

    # ALL_CLASS_flow.csv 只包含各类别的 CLASS_MEAN 行，不是普通逐 sample 表。
    # 因此它必须用 ALL_CLASS_IMAGE_PAIR.CSV 的类别级 ESMSE/SSIM 与自身类别级 AEE/EPE 配对，
    # 并且不要再用“非 MEAN 行均值”去覆盖这些 CLASS_MEAN 行。
    all_class_result = recalculate_c_aee_csv_pair(
        root / "ALL_CLASS_IMAGE_PAIR.CSV",
        root / "ALL_CLASS_flow.CSV",
        refresh_mean_rows=False,
    )
    summary["all_class"].append(all_class_result)
    if logger is not None:
        logger.info(f"[C-AEE recalc] ALL_CLASS_flow csv: {all_class_result}")

    test_summary = _refresh_test_all_summary_csv(root)
    summary["summary"].append(test_summary)
    if logger is not None:
        logger.info(f"[C-AEE recalc] summary csv: {test_summary}")
    return summary
