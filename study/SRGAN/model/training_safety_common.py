import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


class NonFiniteLossError(RuntimeError):
    """
    训练 batch 出现 NaN/Inf 损失时抛出的早停异常。

    这个异常只用于“当前 epoch 训练不可继续”的控制流：
    pipeline 捕获后会跳过本 epoch 的验证/保存，并加载上一个 epoch 已保存的模型进入
    evaluate_all / test_all，从而避免把已经污染的模型继续用于最终评估。
    """

    def __init__(self, report: dict[str, Any]):
        self.report = report
        invalid_losses = report.get("invalid_losses", {})
        message = (
            "[NonFiniteLoss] "
            f"epoch={report.get('epoch')} batch_index={report.get('batch_index')} "
            f"invalid_losses={invalid_losses}"
        )
        super().__init__(message)


def _json_safe(value: Any) -> Any:
    """
    把 tensor / numpy / Path 等对象转成 JSON 可保存的普通 Python 类型。

    早停报告需要尽量完整记录损失、权重和路径；这里集中做转换，避免每个 pipeline
    手写一套不完整的类型处理。
    """
    if torch.is_tensor(value):
        detached = value.detach().float().cpu()
        if detached.numel() == 1:
            return float(detached.item())
        finite = detached[torch.isfinite(detached)]
        return {
            "shape": list(detached.shape),
            "finite_count": int(finite.numel()),
            "nan_count": int(torch.isnan(detached).sum().item()),
            "inf_count": int(torch.isinf(detached).sum().item()),
            "mean": float(finite.mean().item()) if finite.numel() > 0 else float("nan"),
            "min": float(finite.min().item()) if finite.numel() > 0 else float("nan"),
            "max": float(finite.max().item()) if finite.numel() > 0 else float("nan"),
        }
    if isinstance(value, np.ndarray):
        return _json_safe(torch.from_numpy(value))
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value
    try:
        return float(value)
    except Exception:
        return str(value)


def _is_value_finite(value: Any) -> bool:
    """
    判断单个损失值是否有限。

    loss_dict 里既可能是 Python float，也可能是 tensor；这里统一判断，确保 NaN、+Inf、
    -Inf 都能被拦截。
    """
    if torch.is_tensor(value):
        return bool(torch.isfinite(value.detach()).all().item())
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def collect_loss_weight_snapshot(global_data) -> dict[str, Any]:
    """
    收集当前训练使用的损失权重和动态权重相关超参数。

    只挑选名字里包含 LAMBDA / WEIGHT / FAMO / WARMUP / WARMSTART 的大写变量，
    这样能覆盖对抗损失、flow warp 一致性、RAFT_EPE_WEIGHT、FAMO 开关等关键信息，
    又不会把所有路径和数据集配置都塞进早停报告。
    """
    snapshot: dict[str, Any] = {}
    esrgan_cfg = getattr(global_data, "esrgan", global_data)
    for name in dir(esrgan_cfg):
        if not name.isupper():
            continue
        if not any(token in name for token in ("LAMBDA", "WEIGHT", "FAMO", "WARMUP", "WARMSTART")):
            continue
        value = getattr(esrgan_cfg, name)
        if callable(value):
            continue
        snapshot[name] = _json_safe(value)
    return snapshot


def raise_if_nonfinite_losses(
    loss_dict: dict[str, Any],
    *,
    epoch: int,
    batch_index: int,
    class_name: str,
    data_type: str,
    scale: float,
    global_data,
) -> None:
    """
    检查当前 batch 的所有 loss 是否为有限值；发现 NaN/Inf 就触发早停异常。

    epoch 传入的是代码内部 0-based epoch，这里报告统一保存为 1-based，和日志/CSV
    中显示的 epoch 保持一致。
    """
    invalid_losses = {
        key: _json_safe(value)
        for key, value in loss_dict.items()
        if not _is_value_finite(value)
    }
    if not invalid_losses:
        return

    report = {
        "event": "non_finite_loss_early_stop",
        "epoch": int(epoch) + 1,
        "epoch_zero_based": int(epoch),
        "batch_index": int(batch_index),
        "class_name": class_name,
        "data_type": data_type,
        "scale": float(scale),
        "invalid_losses": invalid_losses,
        "all_losses": _json_safe(loss_dict),
        "loss_weights": collect_loss_weight_snapshot(global_data),
    }
    raise NonFiniteLossError(report)


def save_early_stop_report(
    report: dict[str, Any],
    output_dir: str | Path,
    *,
    logger=None,
) -> tuple[Path, Path]:
    """
    保存早停报告到 JSON 和 TXT。

    JSON 方便后续程序化读取；TXT 方便直接打开查看。文件名包含 epoch/batch，
    可以在同一实验目录中保留多次早停记录而不互相覆盖。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epoch = report.get("epoch", "unknown")
    batch_index = report.get("batch_index", "unknown")
    json_path = output_dir / f"early_stop_epoch_{epoch}_batch_{batch_index}.json"
    txt_path = output_dir / f"early_stop_epoch_{epoch}_batch_{batch_index}.txt"

    safe_report = _json_safe(report)
    json_text = json.dumps(safe_report, ensure_ascii=False, indent=2, sort_keys=True)
    json_path.write_text(json_text, encoding="utf-8")

    lines = [
        "Non-finite loss early stop report",
        f"epoch: {safe_report.get('epoch')}",
        f"batch_index: {safe_report.get('batch_index')}",
        f"class_name: {safe_report.get('class_name')}",
        f"data_type: {safe_report.get('data_type')}",
        f"scale: {safe_report.get('scale')}",
        "",
        "invalid_losses:",
    ]
    for key, value in safe_report.get("invalid_losses", {}).items():
        lines.append(f"  {key}: {value}")
    lines.extend(["", "all_losses:"])
    for key, value in safe_report.get("all_losses", {}).items():
        lines.append(f"  {key}: {value}")
    lines.extend(["", "loss_weights:"])
    for key, value in safe_report.get("loss_weights", {}).items():
        lines.append(f"  {key}: {value}")
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    if logger is not None:
        logger.error(f"[EarlyStop] non-finite loss report saved: {txt_path}")
        logger.error(f"[EarlyStop] non-finite loss json saved: {json_path}")
    return json_path, txt_path


def restore_model_from_checkpoint(model, checkpoint_path: str | Path, *, device, logger=None) -> bool:
    """
    从上一个 epoch 保存的模型权重恢复模型。

    返回 True 表示恢复成功；如果没有可用 checkpoint，返回 False，pipeline 仍会继续走
    evaluate_all/test_all，但日志和早停报告会清楚说明没有恢复来源。
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        if logger is not None:
            logger.error(f"[EarlyStop] cannot restore model, checkpoint not found: {checkpoint_path}")
        return False

    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as exc:
        # 加载失败时不让异常再次中断流程；早停报告里会记录 restored_last_checkpoint=False，
        # 后续 evaluate_all/test_all 仍按原流程执行，日志会明确提示需要人工检查 checkpoint。
        if logger is not None:
            logger.error(f"[EarlyStop] failed to restore model from {checkpoint_path}: {exc}")
        return False
    if logger is not None:
        logger.error(f"[EarlyStop] restored last saved model from: {checkpoint_path}")
    return True
