"""
直接评估 RAFT_CHECKPOINT/ckpt_256.tar 的入口脚本。

目标：
1. 复用 ESRuRAFT_PIV_Ground 中 TRAIN_MODE="hr_ground" 的前向逻辑：
   - previous / next 都直接使用真实 HR 图像；
   - 不经过 ESRGAN 或任何超分网络；
   - evaluate_all 仍然按 Ground 模型的统一输出格式保存 image_pair 和 flow 指标。
2. 只加载 RAFT checkpoint 并直接跑 evaluate_all，不训练、不初始化 wandb、不保存优化器。
"""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

import torch
from loguru import logger


# 允许用户直接运行：
#   python RAFT_CHECKPOINT/RAFT_MODEL_EVALUATE.py
# 此时 sys.path 默认只包含 RAFT_CHECKPOINT 目录，不一定能找到顶层 study 包。
# 这里从当前文件向上找到 super_reloution_project 根目录并加入 sys.path，
# 让下面的 study.SRGAN.* 绝对导入和其它模块保持一致。
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from study.SRGAN.data_load import filter_excluded_class_names, get_class_names, load_data
from study.SRGAN.model.ESRuRAFT_PIV_Ground.Module.PIV_ESRGAN_RAFT_Model import ESRuRAFT_PIV
from study.SRGAN.model.ESRuRAFT_PIV_Ground.evaluate import evaluate_all
from study.SRGAN.model.ESRuRAFT_PIV_Ground.global_class import global_data
from study.SRGAN.util.CSV_operator import CsvTable


# 默认 checkpoint 就放在当前 RAFT_CHECKPOINT 目录下。
# 这里等价于用户要求的 input_path_ckpt="./ckpt_256.tar"，
# 只是用绝对路径避免从不同工作目录启动脚本时找错文件。
INPUT_PATH_CKPT = CURRENT_FILE.with_name("ckpt_256.tar")

# checkpoint 权重/超参数报告默认保存到同目录 txt。
# 如果改了 INPUT_PATH_CKPT，实际保存路径会在 _load_raft_checkpoint 里按 checkpoint 名自动生成。
CHECKPOINT_INFO_TXT = CURRENT_FILE.with_name("ckpt_256_checkpoint_info.txt")

# 默认验证 validate_loader，和 ESRuRAFT_PIV_Ground pipeline 训练结束后的 evaluate_all 口径一致。
# 如果后续想评估 test_loader，只需要把这里改成 "test"。
EVALUATE_SPLIT = "validate"


def _resolve_device() -> torch.device:
    """
    返回评估设备。

    global_class 里默认是 cuda；如果当前机器没有 CUDA，则自动退回 CPU，
    避免在本地只想检查或小规模评估时直接因为设备不存在而失败。
    """
    configured_device = getattr(global_data.esrgan, "device", torch.device("cuda"))
    if configured_device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("[RAFT checkpoint evaluate] CUDA 不可用，自动切换到 CPU。")
        return torch.device("cpu")
    return configured_device


def _force_hr_ground_mode() -> None:
    """
    强制复用 ESRuRAFT_PIV_Ground 的 hr_ground 逻辑。

    evaluate_all 内部会调用 model.forward，而 Ground 模型的 forward 会根据
    global_data.esrgan.TRAIN_MODE 决定 RAFT 输入来源。这里必须在实例化模型前设置，
    因为 ESRuRAFT_PIV.__init__ 会读取并缓存 self.train_mode。
    """
    global_data.esrgan.TRAIN_MODE = "hr_ground"
    mode = global_data.esrgan.validate_train_mode()
    if mode != "hr_ground":
        raise ValueError(f"TRAIN_MODE 应为 hr_ground，当前为: {mode}")


def _format_checkpoint_value(value, depth: int = 0) -> str:
    """
    将 checkpoint 中非模型权重的信息格式化成短文本。

    checkpoint 里常见内容包括 epoch、lr、optimizer 参数、训练配置等。
    这里递归打印这些“超参数/元数据”，但对嵌套容器做深度和长度限制，
    避免一个很大的 optimizer state 或列表把日志刷到不可读。
    """
    if torch.is_tensor(value):
        shape = tuple(value.shape)
        if value.numel() == 1:
            return f"tensor(shape={shape}, dtype={value.dtype}, value={value.detach().cpu().item()!r})"
        return f"tensor(shape={shape}, dtype={value.dtype}, numel={value.numel()})"

    if isinstance(value, Mapping):
        keys = list(value.keys())
        if depth >= 2:
            return f"dict(keys={keys[:20]}, total_keys={len(keys)})"
        items = []
        for key in keys[:20]:
            items.append(f"{key!r}: {_format_checkpoint_value(value[key], depth + 1)}")
        suffix = ", ..." if len(keys) > 20 else ""
        return "{" + ", ".join(items) + suffix + "}"

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = list(value)
        if depth >= 2:
            return f"{type(value).__name__}(len={len(values)})"
        rendered = [_format_checkpoint_value(v, depth + 1) for v in values[:20]]
        suffix = ", ..." if len(values) > 20 else ""
        return f"{type(value).__name__}([" + ", ".join(rendered) + suffix + "])"

    return repr(value)


def _tensor_stats_text(tensor: torch.Tensor) -> str:
    """
    返回单个权重张量的基础统计量。

    不直接打印完整权重矩阵：RAFT 权重很多，完整数值会让日志巨大且难以定位问题。
    这里打印 shape/dtype/numel/均值/标准差/最小/最大，既能核对 checkpoint 内容，
    也能快速发现 NaN、Inf、全零权重等异常。
    """
    if tensor.numel() == 0:
        return "empty"

    with torch.no_grad():
        if tensor.is_complex():
            values = tensor.detach().abs().float().cpu()
            prefix = "abs_"
        else:
            values = tensor.detach().float().cpu()
            prefix = ""

        finite_mask = torch.isfinite(values)
        finite_count = int(finite_mask.sum().item())
        nan_count = int(torch.isnan(values).sum().item())
        inf_count = int(torch.isinf(values).sum().item())

        if finite_count == 0:
            return (
                f"{prefix}finite_count=0, nan_count={nan_count}, "
                f"inf_count={inf_count}"
            )

        finite_values = values[finite_mask]
        mean = float(finite_values.mean().item())
        std = float(finite_values.std(unbiased=False).item())
        min_value = float(finite_values.min().item())
        max_value = float(finite_values.max().item())
        return (
            f"{prefix}mean={mean:.8g}, {prefix}std={std:.8g}, "
            f"{prefix}min={min_value:.8g}, {prefix}max={max_value:.8g}, "
            f"finite_count={finite_count}, nan_count={nan_count}, inf_count={inf_count}"
        )


def _log_state_dict_details(state_dict: Mapping) -> None:
    """
    打印 model_state_dict 中所有权重/缓冲区的详细摘要。

    每个 key 都会输出一行，包含名称、shape、dtype、元素数量和基础统计量。
    额外汇总总张量数、总参数量、按 dtype/device 分布，方便确认 checkpoint 与模型规模。
    """
    total_tensors = 0
    total_numel = 0
    total_bytes = 0
    dtype_counts: dict[str, int] = {}
    device_counts: dict[str, int] = {}

    for value in state_dict.values():
        if not torch.is_tensor(value):
            continue
        total_tensors += 1
        total_numel += value.numel()
        total_bytes += value.numel() * value.element_size()
        dtype_counts[str(value.dtype)] = dtype_counts.get(str(value.dtype), 0) + 1
        device_counts[str(value.device)] = device_counts.get(str(value.device), 0) + 1

    logger.info(
        "[RAFT checkpoint evaluate] model_state_dict summary | "
        f"tensors={total_tensors}, numel={total_numel}, "
        f"size_mb={total_bytes / (1024 ** 2):.2f}, "
        f"dtypes={dtype_counts}, devices={device_counts}"
    )

    for name, value in state_dict.items():
        if torch.is_tensor(value):
            logger.info(
                "[RAFT checkpoint evaluate] weight | "
                f"name={name}, shape={tuple(value.shape)}, dtype={value.dtype}, "
                f"numel={value.numel()}, {_tensor_stats_text(value)}"
            )
        else:
            logger.info(
                "[RAFT checkpoint evaluate] state_dict non-tensor | "
                f"name={name}, value={_format_checkpoint_value(value)}"
            )


def _log_checkpoint_details(checkpoint: Mapping) -> None:
    """
    加载成功后打印 checkpoint 里的权重和相关超参数/元数据。

    输出分两部分：
    1. 顶层 key 以及除 model_state_dict 之外的配置、epoch、优化器等信息；
    2. model_state_dict 中每个权重张量的摘要。
    """
    keys = list(checkpoint.keys())
    logger.info(f"[RAFT checkpoint evaluate] checkpoint keys: {keys}")

    for key, value in checkpoint.items():
        if key == "model_state_dict":
            continue
        logger.info(
            "[RAFT checkpoint evaluate] checkpoint metadata | "
            f"{key}={_format_checkpoint_value(value)}"
        )

    state_dict = checkpoint.get("model_state_dict")
    if isinstance(state_dict, Mapping):
        _log_state_dict_details(state_dict)
    else:
        logger.warning(
            "[RAFT checkpoint evaluate] checkpoint['model_state_dict'] 不是 Mapping，"
            f"实际类型为 {type(state_dict)}，无法逐项打印权重。"
        )


def _load_raft_checkpoint(model: torch.nn.Module, input_path_ckpt: str | Path) -> None:
    """
    按用户指定方式加载裸 RAFT checkpoint。

    注意这里的 model 参数传入的是 ESRuRAFT_PIV 外壳里的 model.piv_RAFT，
    因此下面两行与要求一致：

        checkpoint = torch.load(input_path_ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])

    只额外加了 map_location，保证 checkpoint 可以被加载到当前评估设备。
    """
    input_path_ckpt = Path(input_path_ckpt)
    if not input_path_ckpt.exists():
        raise FileNotFoundError(f"找不到 RAFT checkpoint: {input_path_ckpt}")

    logger.info(f"[RAFT checkpoint evaluate] loading RAFT checkpoint: {input_path_ckpt}")
    checkpoint = torch.load(input_path_ckpt, map_location=_resolve_device())
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("[RAFT checkpoint evaluate] RAFT checkpoint loaded.")

    # 将“打印出来的 checkpoint 权重/超参数信息”同步保存成 txt。
    # 这里临时给 loguru 增加一个文件 sink，只包住 checkpoint 详情打印：
    # - 控制台仍然能看到同样内容；
    # - txt 中只包含 checkpoint 详情，不混入后续 evaluate_all 的长日志。
    checkpoint_info_path = input_path_ckpt.with_name(f"{input_path_ckpt.stem}_checkpoint_info.txt")
    if input_path_ckpt.resolve() == INPUT_PATH_CKPT.resolve():
        checkpoint_info_path = CHECKPOINT_INFO_TXT
    checkpoint_info_path.parent.mkdir(parents=True, exist_ok=True)
    sink_id = logger.add(
        str(checkpoint_info_path),
        level="INFO",
        format="{message}",
        mode="w",
        encoding="utf-8",
    )
    try:
        logger.info(f"[RAFT checkpoint evaluate] checkpoint info report: {checkpoint_info_path}")
        logger.info(f"[RAFT checkpoint evaluate] checkpoint source: {input_path_ckpt}")
        _log_checkpoint_details(checkpoint)
        logger.info(f"[RAFT checkpoint evaluate] checkpoint info txt saved: {checkpoint_info_path}")
    finally:
        logger.remove(sink_id)
    logger.info(f"[RAFT checkpoint evaluate] checkpoint info txt saved: {checkpoint_info_path}")


def _build_run_jobs() -> list[dict]:
    """
    复用 Ground pipeline 的类别评估方式。

    - mixed: 所有类别混合成一个 validate_loader，evaluate_all 内部再按真实 class_name 分桶输出；
    - all: 每个类别单独跑一遍；
    - single: 只跑 SINGLE_CLASS_NAME 指定的类别；
    - fixed: 使用 fixed train/validate list，其中 evaluate_all 默认跑固定验证列表。
    """
    mode = global_data.esrgan.validate_train_class_mode()
    available_class_names = get_class_names(global_data.esrgan.GR_DATA_ROOT_DIR)
    if mode in {"all", "mixed", "fixed"}:
        # 与 Ground pipeline 保持一致：评估入口也尊重 EXCLUDE_CLASS，避免 fixed/all/mixed 评估读入被排除类别。
        available_class_names = filter_excluded_class_names(
            available_class_names,
            global_data.esrgan.EXCLUDE_CLASS,
            context=f"RAFT_CHECKPOINT_EVALUATE:{mode}",
        )

    if mode == "fixed":
        # fixed 模式真实验证集由 FlowData_test.list 决定；先同步比例，便于日志/summary 保存真实划分。
        global_data.esrgan.update_fixed_split_rates()
        return [{"run_class_name": global_data.esrgan.FIXED_CLASS_TAG, "selected_classes": None}]

    if mode == "all":
        return [
            {"run_class_name": class_name, "selected_classes": [class_name]}
            for class_name in available_class_names
        ]

    if mode == "single":
        chosen = global_data.esrgan.SINGLE_CLASS_NAME
        if chosen is None:
            raise ValueError("TRAIN_CLASS_MODE='single' 时，请先设置 global_data.esrgan.SINGLE_CLASS_NAME。")
        if chosen not in available_class_names:
            raise ValueError(f"未知类别: {chosen}；可用类别: {available_class_names}")
        return [{"run_class_name": chosen, "selected_classes": [chosen]}]

    if mode == "mixed":
        return [{"run_class_name": global_data.esrgan.MIXED_CLASS_TAG, "selected_classes": None}]

    raise ValueError(f"TRAIN_CLASS_MODE 仅支持 {global_data.esrgan.TRAIN_CLASS_MODES}。")


def _pick_eval_loader(validate_loader, test_loader):
    """
    选择 evaluate_all 使用的数据划分。

    默认 validate，与原 Ground pipeline 训练结束后的全量验证保持一致。
    """
    split = str(EVALUATE_SPLIT).strip().lower()
    if split in {"validate", "val"}:
        return validate_loader
    if split == "test":
        return test_loader
    raise ValueError("EVALUATE_SPLIT 仅支持 'validate' 或 'test'。")


def _nested_get(mapping: Mapping | None, keys: Sequence[str]):
    """
    安全读取 batch 中的多层字段。

    DataLoader 返回的是嵌套 dict，例如 batch["image_pair"]["previous"]["lr_data"]。
    summary 只用于记录形状，不应该因为某个可选字段缺失影响 evaluate_all 主流程，
    所以这里使用安全读取，缺字段时返回 None。
    """
    current = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _shape_or_empty(value) -> tuple | str:
    """
    把 tensor / array 的 shape 转成可写入 CSV 的 tuple。

    CsvTable 最终会把 tuple 写成字符串；如果读取不到 shape，则返回空字符串，
    这样 summary CSV 的列结构仍然完整。
    """
    shape = getattr(value, "shape", None)
    return tuple(shape) if shape is not None else ""


def _pick_summary_shapes(data_loader) -> dict:
    """
    从验证 loader 取第一个 batch，只提取 summary 需要的形状信息。

    这个函数不会把 batch 送入模型，也不会改变 evaluate_all 的逻辑；
    DataLoader 每次 iter(data_loader) 都会创建新的迭代器，所以这里提前取一次
    首 batch 只用于记录 input_lr_shape / input_hr_shape / flow_shape。
    """
    try:
        sample_batch = next(iter(data_loader))
    except StopIteration:
        logger.warning("[RAFT checkpoint evaluate] eval_loader 为空，metrics summary 的输入形状将留空。")
        return {
            "batch_size": 0,
            "input_lr_shape": "",
            "input_hr_shape": "",
            "flow_shape": "",
        }

    lr_prev = _nested_get(sample_batch, ("image_pair", "previous", "lr_data"))
    hr_prev = _nested_get(sample_batch, ("image_pair", "previous", "gr_data"))
    flow_hr = _nested_get(sample_batch, ("flo", "gr_data"))

    # batch_size 优先从 LR 输入的第 0 维读取；如果 LR 缺失，再回退到配置值。
    # 这样在最后一个 batch 不满 BATCH_SIZE 时，CSV 记录的是实际参与评估的 batch 大小。
    batch_size = int(lr_prev.shape[0]) if torch.is_tensor(lr_prev) and lr_prev.ndim > 0 else global_data.esrgan.BATCH_SIZE
    return {
        "batch_size": batch_size,
        "input_lr_shape": _shape_or_empty(lr_prev),
        "input_hr_shape": _shape_or_empty(hr_prev),
        "flow_shape": _shape_or_empty(flow_hr),
    }


def _reset_cuda_peak_memory(device: torch.device) -> None:
    """
    在 evaluate_all 前重置 CUDA 峰值显存统计。

    这样 summary 里的 inference_gpu_memory_usage_gb 对应本次直接评估过程的峰值。
    如果当前使用 CPU，则这个函数什么也不做。
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)


def _cuda_peak_memory_gb(device: torch.device) -> float:
    """
    返回 evaluate_all 期间的 CUDA 峰值显存，单位 GB。

    CPU 评估没有 GPU 显存，统一记录为 0.0，便于后续 CSV 读取时保持数值类型。
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize(device)
    return float(torch.cuda.max_memory_allocated(device)) / (1024 ** 3)


def _build_run_metrics_summary(
    model: torch.nn.Module,
    device: torch.device,
    summary_shapes: dict,
    evaluate_all_seconds: float,
    evaluate_peak_memory_gb: float,
    evaluate_mean_row: Mapping | None,
) -> dict:
    """
    组装 _save_run_metrics_summary 需要的一行运行级指标。

    Ground pipeline 里的 metrics_summary 来自独立 profiling；这里是 checkpoint 直接验证脚本，
    为了不额外重复前向、不改变 evaluate_all 的验证逻辑，采用已经发生的 evaluate_all 运行结果：
    - inference_time_seconds 记录 evaluate_all 的端到端耗时；
    - inference_gpu_memory_usage_gb 记录 evaluate_all 期间的 CUDA 峰值显存；
    - FLOPs / training_step 相关列没有额外 profiling，写 NaN；
    - evaluate_mean_row 保留 evaluate_all 返回的均值指标，当前主要用于日志核对。
    """
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    metrics_summary = {
        "device": str(device),
        "batch_size": summary_shapes.get("batch_size", global_data.esrgan.BATCH_SIZE),
        "input_lr_shape": summary_shapes.get("input_lr_shape", ""),
        "input_hr_shape": summary_shapes.get("input_hr_shape", ""),
        "flow_shape": summary_shapes.get("flow_shape", ""),
        # 这个脚本只做直接评估，不进行训练，所以训练总时长固定为 0。
        "training_time_hours": 0.0,
        "inference_gpu_memory_usage_gb": evaluate_peak_memory_gb,
        # 不额外跑 profiler，避免为了 summary 再做一次耗时前向。
        "inference_flops_g": float("nan"),
        "inference_time_seconds": float(evaluate_all_seconds),
        "training_step_gpu_memory_usage_gb": float("nan"),
        "training_step_flops_g": float("nan"),
        "training_step_time_seconds": float("nan"),
        "trainable_params_m": float(trainable_params) / 1e6,
        "evaluate_mean_row": dict(evaluate_mean_row) if isinstance(evaluate_mean_row, Mapping) else {},
    }
    return metrics_summary


def _save_run_metrics_summary(class_name, data_type, scale, metrics_summary):
    """
    保存一次 checkpoint 直接评估的运行级 summary。

    这里保持 ESRuRAFT_PIV_Ground.pipeline._save_run_metrics_summary 的 CSV 列结构，
    文件仍写到 global_data.esrgan.OUT_PUT_DIR/metrics_summary.csv，方便和原训练流程输出合并查看。
    对于本脚本无法在不额外 profiling 的情况下得到的 FLOPs / training_step 指标，
    _build_run_metrics_summary 会显式写入 NaN，避免误把 evaluate_all 总耗时当作纯 forward FLOPs 统计。
    """
    summary_csv_path = f"{global_data.esrgan.OUT_PUT_DIR}/metrics_summary.csv"
    if global_data.esrgan.metricsSummaryCsvOperator is None:
        global_data.esrgan.metricsSummaryCsvOperator = CsvTable(
            file_path=summary_csv_path,
            columns=global_data.esrgan.METRICS_SUMMARY_COLUMNS,
        )
    else:
        global_data.esrgan.metricsSummaryCsvOperator.switch_file(summary_csv_path)

    row = {
        "run_name": global_data.esrgan.name,
        "description": global_data.esrgan.DESCRIPTION,
        "train_mode": global_data.esrgan.validate_train_mode(),
        "class_name": class_name,
        "data_type": data_type,
        "scale": int(scale * scale),
        "device": metrics_summary["device"],
        "batch_size": metrics_summary["batch_size"],
        "input_lr_shape": metrics_summary["input_lr_shape"],
        "input_hr_shape": metrics_summary["input_hr_shape"],
        "flow_shape": metrics_summary["flow_shape"],
        "training_time_hours": metrics_summary["training_time_hours"],
        "inference_gpu_memory_usage_gb": metrics_summary["inference_gpu_memory_usage_gb"],
        "inference_flops_g": metrics_summary["inference_flops_g"],
        "inference_time_seconds": metrics_summary["inference_time_seconds"],
        "training_step_gpu_memory_usage_gb": metrics_summary["training_step_gpu_memory_usage_gb"],
        "training_step_flops_g": metrics_summary["training_step_flops_g"],
        "training_step_time_seconds": metrics_summary["training_step_time_seconds"],
        "trainable_params_m": metrics_summary["trainable_params_m"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    global_data.esrgan.metricsSummaryCsvOperator.create(row)
    logger.info(f"[RAFT checkpoint evaluate] metrics summary CSV saved: {summary_csv_path}")

    # evaluate_all 的指标均值已经由 evaluate_all 自己写入 metrics_all_raft.csv；
    # 这里额外打一行日志，方便运行结束时快速看到本次 summary 对应的均值结果。
    evaluate_mean_row = metrics_summary.get("evaluate_mean_row", {})
    if evaluate_mean_row:
        logger.info(f"[RAFT checkpoint evaluate] evaluate_all mean metrics: {evaluate_mean_row}")


def main() -> None:
    """
    直接加载 RAFT checkpoint 并调用 evaluate_all。

    该函数不进入训练循环，也不创建优化器/调度器；所有样本输出、CSV、
    类别汇总和频谱/误差图仍由 ESRuRAFT_PIV_Ground.evaluate.evaluate_all 原样负责。
    """
    _force_hr_ground_mode()
    device = _resolve_device()
    global_data.esrgan.device = device

    data_type = "RAFT"
    run_jobs = _build_run_jobs()
    logger.info(
        f"[RAFT checkpoint evaluate] TRAIN_MODE={global_data.esrgan.TRAIN_MODE}, "
        f"split={EVALUATE_SPLIT}, device={device}, jobs={run_jobs}"
    )

    for job in run_jobs:
        class_name = job["run_class_name"]
        selected_classes = job["selected_classes"]

        for scale in global_data.esrgan.SCALES:
            logger.info(
                f"[RAFT checkpoint evaluate] loading data | class={class_name}, "
                f"selected_classes={selected_classes}, scale={scale}"
            )
            _, validate_loader, test_loader, _, _ = load_data(
                gr_data_root_dir=global_data.esrgan.GR_DATA_ROOT_DIR,
                lr_data_root_dir=f"{global_data.esrgan.LR_DATA_ROOT_DIR}/x{int(scale * scale)}/data",
                batch_size=global_data.esrgan.BATCH_SIZE,
                # 虽然这里只使用 validate/test loader，但这里仍沿用 Ground 配置，
                # 让 load_data 的整体构造逻辑和 ESRuRAFT_PIV_Ground pipeline 保持一致。
                # fixed 模式必须保留 FlowData_train.list 的顺序；虽然评估默认用验证集，
                # train loader 仍会被 load_data 构造，所以这里同样关闭 fixed 下的训练集 shuffle。
                shuffle=False if global_data.esrgan.validate_train_class_mode() == "fixed" else global_data.esrgan.SHUFFLE,
                target_size=global_data.esrgan.TARGET_SIZE,
                train_nums_rate=global_data.esrgan.Train_nums_rate,
                validate_nums_rate=global_data.esrgan.Validate_nums_rate,
                test_nums_rate=global_data.esrgan.Test_nums_rate,
                random_seed=global_data.esrgan.RANDOM_SEED,
                selected_classes=selected_classes,
                excluded_classes=global_data.esrgan.EXCLUDE_CLASS
                if global_data.esrgan.validate_train_class_mode() in {"all", "mixed", "fixed"}
                else None,
                class_sample_ratio=global_data.esrgan.CLASS_SAMPLE_RATIO,
                fixed_split=global_data.esrgan.validate_train_class_mode() == "fixed",
                fixed_train_list_path=global_data.esrgan.FIXED_TRAIN_LIST_PATH,
                fixed_validate_list_path=global_data.esrgan.FIXED_VALIDATE_LIST_PATH,
                return_test_loader=True,
            )
            eval_loader = _pick_eval_loader(validate_loader, test_loader)

            model = ESRuRAFT_PIV(
                inner_chanel=3,
                batch_size=global_data.esrgan.BATCH_SIZE,
            ).to(device)
            _load_raft_checkpoint(model.piv_RAFT, INPUT_PATH_CKPT)
            model.eval()
            summary_shapes = _pick_summary_shapes(eval_loader)

            output_root = (
                f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/"
                f"scale_{int(scale * scale)}/{global_data.esrgan.PREDICT_ALL_DIR}"
            )
            metrics_csv_path = (
                f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/"
                f"scale_{int(scale * scale)}/{global_data.esrgan.PREDICT_ALL_DIR}/"
                "metrics_all.csv"
            )

            logger.info(
                f"[RAFT checkpoint evaluate] evaluate_all start | class={class_name}, "
                f"scale={int(scale * scale)}, output={output_root}"
            )
            _reset_cuda_peak_memory(device)
            evaluate_start_time = time.perf_counter()
            evaluate_mean_row = evaluate_all(
                model=model,
                data_loader=eval_loader,
                class_name=class_name,
                data_type=data_type,
                SCALE=scale,
                output_root=output_root,
                metrics_csv_path=metrics_csv_path,
                stride=6,
            )
            evaluate_all_seconds = time.perf_counter() - evaluate_start_time
            evaluate_peak_memory_gb = _cuda_peak_memory_gb(device)
            run_metrics_summary = _build_run_metrics_summary(
                model=model,
                device=device,
                summary_shapes=summary_shapes,
                evaluate_all_seconds=evaluate_all_seconds,
                evaluate_peak_memory_gb=evaluate_peak_memory_gb,
                evaluate_mean_row=evaluate_mean_row,
            )
            _save_run_metrics_summary(
                class_name=class_name,
                data_type=data_type,
                scale=scale,
                metrics_summary=run_metrics_summary,
            )
            logger.info(
                f"[RAFT checkpoint evaluate] evaluate_all done | class={class_name}, "
                f"scale={int(scale * scale)}"
            )


if __name__ == "__main__":
    main()
