from loguru import logger

import os
import traceback
import time
from datetime import datetime

from pathlib import Path

import torch
import torch.nn as nn

from torchvision.models import vgg19
from tqdm import tqdm

import wandb
from study.SRGAN.compute_cost import (
    benchmark_latency_for_inputs,
    count_parameters,
    estimate_flops_for_inputs,
    measure_peak_memory_for_inputs,
)
from study.SRGAN.data_load import get_class_names, load_data, save_loaders_paths



from study.SRGAN.model.ESRuRAFT_PIV_Ground.Module.PIV_ESRGAN_RAFT_Model import ESRuRAFT_PIV
from study.SRGAN.model.ESRuRAFT_PIV_Ground.evaluate import evaluate, evaluate_all
from study.SRGAN.model.ESRuRAFT_PIV_Ground.global_class import global_data
from study.SRGAN.model.ESRuRAFT_PIV_Ground.train import esrgan_union_RAFT_train
from study.SRGAN.util.CSV_operator import CsvTable
from study.SRGAN.util.accumulator import Accumulator
from study.SRGAN.util.animator import Animator
try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


def _trainable_parameters_or_dummy(module: nn.Module, device: torch.device):
    """
    Ground/bicubic 模式下 Generator/Discriminator 可能是无参数的 nn.Identity。
    为了让 optimizer 构造、保存和恢复逻辑保持统一，这里在参数列表为空时返回一个
    不属于模型的 dummy 参数；它不会计入模型参数量，也不会改变训练语义。
    """
    params = [p for p in module.parameters() if p.requires_grad]
    if params:
        return params
    return [nn.Parameter(torch.zeros((), device=device), requires_grad=True)]


def _resolve_raft256_pretrain_path(path_text: str | Path) -> Path:
    """
    解析 ESRuRAFT_PIV_Ground 的 RAFT256 预训练 checkpoint 路径。

    global_class 里默认写成 "RAFT_CHECKPOINT/ckpt_256.tar"，表示相对 SRGAN 根目录。
    这里统一转为绝对路径，避免从 PyCharm、命令行、notebook 等不同 cwd 启动时找错文件。
    """
    path = Path(path_text)
    if path.is_absolute():
        return path
    srgan_root = Path(__file__).resolve().parents[2]
    return srgan_root / path


def _clone_optimizer_state_value(value):
    """
    深拷贝 optimizer/scheduler state 中的值。

    AdamW state 里既有 tensor(exp_avg/exp_avg_sq)，也有 step、lr、betas 等普通值；
    这里递归复制，避免直接引用 checkpoint 字典导致后续 optimizer 修改污染原始对象。
    """
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_optimizer_state_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_optimizer_state_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_optimizer_state_value(item) for item in value)
    return value


def _optimizer_state_is_compatible(state_entry: dict, target_param: torch.nn.Parameter) -> bool:
    """
    判断 checkpoint 中某个参数的 optimizer state 能否迁移到当前 RAFT128 参数。

    AdamW 的 exp_avg / exp_avg_sq 必须和目标参数 shape 完全一致；
    step 这种 0 维 tensor 或普通数值可以迁移。若不做这个检查，RAFT256 的
    update_block.mask.2 相关状态会在 optimizer.step() 时因为 576 vs 144 通道不匹配而报错。
    """
    for value in state_entry.values():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape != target_param.shape:
            return False
    return True


def _migrate_raft_optimizer_from_raft256_checkpoint(
    model: ESRuRAFT_PIV,
    optimizer: torch.optim.Optimizer | None,
    checkpoint: dict,
) -> None:
    """
    将 RAFT256 checkpoint 中可兼容的 AdamW 状态迁移到当前 RAFT128 optimizer。

    迁移策略：
    1. 用 checkpoint["model_state_dict"] 的 key 顺序和 optimizer param_groups 里的参数 id 建立源参数名映射；
    2. 用当前 model.piv_RAFT.named_parameters() 建立目标参数名映射；
    3. 只迁移“同名 + 权重 shape 相同 + optimizer state shape 相同”的参数状态；
    4. 不兼容参数保留当前 optimizer 的初始状态，后续训练会自然重新累计动量。
    """
    if optimizer is None or not bool(global_data.esrgan.RAFT128_INIT_FROM_RAFT256_OPTIMIZER):
        return

    source_state = checkpoint.get("model_state_dict")
    source_optimizer_state = checkpoint.get("optimizer_state_dict")
    if not isinstance(source_state, dict) or not isinstance(source_optimizer_state, dict):
        logger.warning("[ESRuRAFT_PIV_Ground] RAFT256 checkpoint 中没有可迁移的 optimizer_state_dict，跳过 optimizer 迁移。")
        return

    current_optimizer_state = optimizer.state_dict()
    source_param_ids = [
        param_id
        for group in source_optimizer_state.get("param_groups", [])
        for param_id in group.get("params", [])
    ]
    current_param_ids = [
        param_id
        for group in current_optimizer_state.get("param_groups", [])
        for param_id in group.get("params", [])
    ]
    target_named_params = list(model.piv_RAFT.named_parameters())
    target_name_to_id = {
        name: param_id
        for (name, _), param_id in zip(target_named_params, current_param_ids)
    }
    target_name_to_param = dict(target_named_params)

    # state_dict 基本按 named_parameters 的顺序保存。这里只保留当前 RAFT128 也存在的名字，
    # 即使 checkpoint 中未来出现 buffer 或额外字段，也不会参与 optimizer 状态迁移。
    source_param_names = [name for name in source_state.keys() if name in target_name_to_param]
    source_name_to_id = {
        name: param_id
        for name, param_id in zip(source_param_names, source_param_ids)
    }

    migrated_state = {}
    skipped = []
    for name, target_param in target_named_params:
        source_value = source_state.get(name)
        source_id = source_name_to_id.get(name)
        target_id = target_name_to_id.get(name)
        source_state_entry = source_optimizer_state.get("state", {}).get(source_id)
        if (
            source_id is not None
            and target_id is not None
            and torch.is_tensor(source_value)
            and source_value.shape == target_param.shape
            and isinstance(source_state_entry, dict)
            and _optimizer_state_is_compatible(source_state_entry, target_param)
        ):
            migrated_state[target_id] = _clone_optimizer_state_value(source_state_entry)
        else:
            source_shape = tuple(source_value.shape) if torch.is_tensor(source_value) else None
            skipped.append((name, source_shape, tuple(target_param.shape)))

    # param_groups 保留当前 optimizer 的目标参数 id；lr/betas/weight_decay 等超参数使用 checkpoint 中的值，
    # 这样“迁移 optimizer”时能延续 ckpt_256.tar 当时的学习率状态，同时不会引用旧模型参数 id。
    migrated_groups = []
    source_groups = source_optimizer_state.get("param_groups", [])
    for idx, current_group in enumerate(current_optimizer_state.get("param_groups", [])):
        migrated_group = dict(current_group)
        if idx < len(source_groups):
            for key, value in source_groups[idx].items():
                if key != "params":
                    migrated_group[key] = _clone_optimizer_state_value(value)
        migrated_group["params"] = list(current_group["params"])
        migrated_groups.append(migrated_group)

    optimizer.load_state_dict({
        "state": migrated_state,
        "param_groups": migrated_groups,
    })
    logger.info(
        "[ESRuRAFT_PIV_Ground] RAFT optimizer initialized from RAFT256 checkpoint | "
        f"migrated={len(migrated_state)}, skipped={len(skipped)}"
    )
    for name, source_shape, target_shape in skipped[:20]:
        logger.info(
            "[ESRuRAFT_PIV_Ground] skipped RAFT optimizer state | "
            f"name={name}, source_shape={source_shape}, target_shape={target_shape}"
        )


def _migrate_raft_scheduler_from_raft256_checkpoint(
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    checkpoint: dict,
) -> None:
    """
    将 RAFT256 checkpoint 中的 scheduler 状态迁移到当前 RAFT128 scheduler。

    ReduceLROnPlateau 的状态不包含参数形状，主要是 best、num_bad_epochs、last_epoch、_last_lr 等标量，
    因此可以直接 load_state_dict。optimizer 的参数状态仍由上面的安全迁移函数单独处理。
    """
    if scheduler is None or not bool(global_data.esrgan.RAFT128_INIT_FROM_RAFT256_SCHEDULER):
        return

    scheduler_state = checkpoint.get("scheduler_state_dict")
    if not isinstance(scheduler_state, dict):
        logger.warning("[ESRuRAFT_PIV_Ground] RAFT256 checkpoint 中没有 scheduler_state_dict，跳过 scheduler 迁移。")
        return
    scheduler.load_state_dict(scheduler_state)
    logger.info("[ESRuRAFT_PIV_Ground] RAFT scheduler initialized from RAFT256 checkpoint.")


def _maybe_migrate_raft_optimizer_scheduler_from_raft256(
    model: ESRuRAFT_PIV,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    optimizer_loaded_from_resume: bool,
    scheduler_loaded_from_resume: bool,
) -> None:
    """
    在 ESRuRAFT_PIV_Ground 中可选迁移 RAFT256 的 optimizer/scheduler 到 RAFT128。

    只在以下条件同时满足时执行：
    - RAFT_MODEL_TYPE="RAFT128"；
    - RAFT128_INIT_FROM_RAFT256=True；
    - 当前实验没有成功恢复自己的 RAFT optimizer/scheduler。

    这样既满足“用 ckpt_256 做 RAFT128 预训练初始化”的需求，又不会覆盖已有断点恢复。
    """
    if global_data.esrgan.validate_raft_model_type() != "raft128":
        return
    if not bool(global_data.esrgan.RAFT128_INIT_FROM_RAFT256):
        return

    checkpoint_path = _resolve_raft256_pretrain_path(global_data.esrgan.RAFT128_INIT_FROM_RAFT256_CKPT)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"RAFT256 预训练 checkpoint 不存在: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"RAFT256 checkpoint 格式不正确: {checkpoint_path}")

    if optimizer_loaded_from_resume:
        logger.info("[ESRuRAFT_PIV_Ground] 当前实验已恢复 RAFT optimizer，跳过 RAFT256 optimizer 迁移。")
    else:
        _migrate_raft_optimizer_from_raft256_checkpoint(model, optimizer, checkpoint)

    if scheduler_loaded_from_resume:
        logger.info("[ESRuRAFT_PIV_Ground] 当前实验已恢复 RAFT scheduler，跳过 RAFT256 scheduler 迁移。")
    else:
        _migrate_raft_scheduler_from_raft256_checkpoint(scheduler, checkpoint)


class ESRuRAFTPIVInferenceWrapper(nn.Module):
    """为联合模型 profiling 提供纯推理接口，避免把训练损失计算带进基准测试。"""

    def __init__(self, model: ESRuRAFT_PIV):
        super().__init__()
        self.model = model

    def forward(self, input_lr_prev, input_lr_next, input_gr_prev, input_gr_next, flowl0):
        # profiling 时只关心“模型推理本身”的开销，
        # 所以这里只走当前 TRAIN_MODE 对应的图像来源 + RAFT 推理，不计算图像/GAN 损失。
        if self.model.train_mode == "lr_ground":
            pred_prev = self.model._resize_image_to_target(input_lr_prev, input_gr_prev, mode="nearest")
            pred_next = self.model._resize_image_to_target(input_lr_next, input_gr_next, mode="nearest")
            # lr_ground 不经过超分网络，但 profiling 时也要和训练一致：
            # 先把 LR 用最近邻插值对齐到 HR/RAFT 尺寸，再送入 RAFT。
            raft_prev_source = pred_prev
            raft_next_source = pred_next
        elif self.model.train_mode == "bicubic":
            # bicubic 是传统插值超分 baseline，同样不经过 Generator。
            # 这里用双三次插值把 LR 放大到 HR/RAFT 尺寸，再把结果送入 RAFT；
            # profiling 口径必须和真实 train_step 保持一致。
            pred_prev = self.model._resize_image_to_target(input_lr_prev, input_gr_prev, mode="bicubic")
            pred_next = self.model._resize_image_to_target(input_lr_next, input_gr_next, mode="bicubic")
            raft_prev_source = pred_prev
            raft_next_source = pred_next
        elif self.model.train_mode == "hr_ground":
            pred_prev = input_gr_prev
            pred_next = input_gr_next
            raft_prev_source = input_gr_prev
            raft_next_source = input_gr_next
        else:
            pred_prev = self.model.piv_esrgan_generator(input_lr_prev)
            pred_next = self.model.piv_esrgan_generator(input_lr_next)
            raft_prev_source = pred_prev
            raft_next_source = pred_next

        flow_predictions, _ = self.model._compute_raft_branch(
            raft_prev_source=raft_prev_source,
            raft_next_source=raft_next_source,
            flowl0=flowl0,
        )
        final_flow_prediction = flow_predictions[-1] if isinstance(flow_predictions, (list, tuple)) else flow_predictions
        return pred_prev, pred_next, final_flow_prediction


def select_single_class(available_class_names, preset_name=None):
    """
    单类别训练时选择类别：
    - preset_name 非空且合法：直接用它
    - 否则：终端交互选择
    """
    if preset_name is not None:
        if preset_name not in available_class_names:
            logger.error(f"SINGLE_CLASS_NAME='{preset_name}' 不在可用类别中: {available_class_names}")
            raise ValueError(
                f"SINGLE_CLASS_NAME='{preset_name}' 不在可用类别中: {available_class_names}"
            )
        logger.error( f"SINGLE_CLASS_NAME='{preset_name}' 不在可用类别中: {available_class_names}")
        return preset_name

    logger.info("请选择单类别训练目标：")
    for idx, cname in enumerate(available_class_names):
        logger.info(f"  [{idx}] {cname}")

    while True:
        raw = input("输入类别序号: ").strip()
        if raw.isdigit():
            i = int(raw)
            if 0 <= i < len(available_class_names):
                return available_class_names[i]
        logger.warning("输入无效，请重新输入。")


def _extract_profile_inputs(batch, device):
    # 从 dataloader 的一个真实 batch 中抽出 profiling 所需输入，
    # 保证统计出来的耗时 / FLOPs / 显存更接近真实训练数据分布。
    lr_prev = batch["image_pair"]["previous"]["lr_data"].to(device, non_blocking=True)
    hr_prev = batch["image_pair"]["previous"]["gr_data"].to(device, non_blocking=True)
    lr_next = batch["image_pair"]["next"]["lr_data"].to(device, non_blocking=True)
    hr_next = batch["image_pair"]["next"]["gr_data"].to(device, non_blocking=True)
    flow_hr = batch["flo"]["gr_data"].to(device, non_blocking=True)
    flow_hr_uv = flow_hr[:, :2, :, :]
    return (lr_prev, hr_prev, lr_next, hr_next, flow_hr_uv)

def _profile_esru_raft_piv_model(model, sample_batch, device, warmup=5, iters=20):
    was_training = model.training
    lr_prev, hr_prev, lr_next, hr_next, flow_hr_uv = _extract_profile_inputs(sample_batch, device)
    # 包一层纯推理 wrapper，避免 forward 里的损失计算把 profiling 结果放大。
    inference_model = ESRuRAFTPIVInferenceWrapper(model).to(device, non_blocking=True)
    inference_model.eval()

    inputs = (lr_prev, lr_next, hr_prev, hr_next, flow_hr_uv)
    # 这里记录的是“可训练参数”，更符合实验表格里常见的模型规模口径。
    _, trainable_params = count_parameters(model)
    flops = estimate_flops_for_inputs(inference_model, inputs, device)
    latency_ms, _ = benchmark_latency_for_inputs(inference_model, inputs, device, warmup=warmup, iters=iters)
    peak_memory_mb = measure_peak_memory_for_inputs(inference_model, inputs, device)

    # profiling 结束后恢复模型原本的 train/eval 状态，避免影响后续流程。
    if was_training:
        model.train()
    else:
        model.eval()
    return {
        "device": str(device),
        "batch_size": int(lr_prev.shape[0]),
        "input_lr_shape": tuple(lr_prev.shape),
        "input_hr_shape": tuple(hr_prev.shape),
        "flow_shape": tuple(flow_hr_uv.shape),
        "gpu_memory_usage_gb": float(peak_memory_mb) / 1024.0,
        "flops_g": float(flops) / 1e9,
        "inference_time_seconds": float(latency_ms) / 1000.0,
        "trainable_params_m": float(trainable_params) / 1e6,
    }


def _save_run_metrics_summary(class_name, data_type, scale, metrics_summary):
    # 汇总指标单独落一份 CSV，和每个 epoch 的 loss CSV 分开，后续做论文表格更方便。
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
        "gpu_memory_usage_gb": metrics_summary["gpu_memory_usage_gb"],
        "flops_g": metrics_summary["flops_g"],
        "inference_time_seconds": metrics_summary["inference_time_seconds"],
        "trainable_params_m": metrics_summary["trainable_params_m"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    global_data.esrgan.metricsSummaryCsvOperator.create(row)
    logger.info(f"Metrics summary CSV saved to {summary_csv_path}")

def _pick_profile_batch(validate_loader, train_loader):
    try:
        # 优先用验证集首个 batch 统计推理指标，更贴近“推理”场景。
        return next(iter(validate_loader))
    except StopIteration:
        logger.warning("validate_loader 为空，profiling 将回退到 train_loader 的首个 batch。")
        return next(iter(train_loader))


def _model_parameters_are_finite(model, model_name: str) -> bool:
    for name, param in model.named_parameters():
        if param is not None and not torch.isfinite(param).all():
            logger.error(f"{model_name} parameter has NaN/Inf: {name}; skip saving model checkpoint.")
            return False
    return True


def main():
    logger.add(
        f"{global_data.esrgan.OUT_PUT_DIR}/running_log/running.log",
        rotation="100 MB",
        retention="30 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {process.name} | {thread.name} | {name}:{module}:{line} | {message}",
        enqueue=True,
        backtrace=True,
        diagnose=True,

    )
    # 记录整个 pipeline.main() 的开始时间。
    # 这个时间写入 global_data.esrgan.START_TIME，表示“本次主流程运行开始”，
    # 不等同于后面单次训练阶段的 training_start_time。
    start_time = time.time()
    global_data.esrgan.START_TIME = start_time

    # 保存超参数
    global_data.esrgan.save_hyper_parameters_txt(f"{global_data.esrgan.OUT_PUT_DIR}/hyper_parameters.txt")
    # 获取类别名
    available_class_names = get_class_names(global_data.esrgan.GR_DATA_ROOT_DIR)

    logger.info(f"一共{len(available_class_names)}个类别：{available_class_names}")

    # 训练模式: all | single | mixed
    mode = global_data.esrgan.TRAIN_CLASS_MODE.lower().strip()
    if mode not in {"all", "single", "mixed"}:
        logger.error(f'TRAIN_CLASS_MODE 仅支持 all/single/mixed，当前为: {global_data.esrgan.TRAIN_CLASS_MODE}')
        raise ValueError(f"TRAIN_CLASS_MODE 仅支持 all/single/mixed，当前为: {global_data.esrgan.TRAIN_CLASS_MODE}")

    run_jobs = []
    if mode == "all":
        # 每个类别读取数据并且训练验证和保存模型
        for class_name in available_class_names:
            run_jobs.append({"run_class_name": class_name, "selected_classes": [class_name]})
    elif mode == "single":
        chosen = select_single_class(available_class_names, global_data.esrgan.SINGLE_CLASS_NAME)
        # 每个类别读取数据并且训练验证和保存模型
        run_jobs.append({"run_class_name": chosen, "selected_classes": [chosen]})
    else:
        # 每个类别读取数据并且训练验证和保存模型
        run_jobs.append({"run_class_name": global_data.esrgan.MIXED_CLASS_TAG, "selected_classes": None})

    for job in run_jobs:
        class_name = job["run_class_name"]
        selected_classes = job["selected_classes"]

        # 几倍上采样倍率来训练
        for SCALE in global_data.esrgan.SCALES:
            # 获取数据 自动根据类别划分数据集并读取，每个类别都安装比例划分训练集和验证集
            # 根据类别和上采样读取数据
            train_loader, validate_loader, test_loader, class_names, samples = load_data(
                gr_data_root_dir=global_data.esrgan.GR_DATA_ROOT_DIR,
                lr_data_root_dir=f"{global_data.esrgan.LR_DATA_ROOT_DIR}/x{int(SCALE * SCALE)}/data",
                batch_size=global_data.esrgan.BATCH_SIZE,
                shuffle=global_data.esrgan.SHUFFLE,
                target_size=global_data.esrgan.TARGET_SIZE,
                train_nums_rate=global_data.esrgan.Train_nums_rate,
                validate_nums_rate=global_data.esrgan.Validate_nums_rate,
                test_nums_rate=global_data.esrgan.Test_nums_rate,
                random_seed=global_data.esrgan.RANDOM_SEED,
                selected_classes=selected_classes,
                class_sample_ratio=global_data.esrgan.CLASS_SAMPLE_RATIO,
                return_test_loader=True
            )
            # 每个类别的图像对和flo文件分别训练验证和保存模型 ！！！！！已经去除
            data_type = "RAFT"
            # Start a new wandb run to track this script.
            wandb.init(
                entity="2950067210-usst",
                project="esrgan",
                name=f"{global_data.esrgan.name}_{global_data.esrgan.DESCRIPTION}_{class_name}_{data_type}",
                config={
                    "createTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "IS_AUTO_DL": global_data.esrgan.IS_AUTO_DL,
                    "epochs": global_data.esrgan.EPOCH_NUMS,
                    "batch_size": global_data.esrgan.BATCH_SIZE,
                    "lr_G":global_data.esrgan. G_LR,
                    "lr_D": global_data.esrgan.D_LR,
                    "RANDOM_SEED":global_data.esrgan. RANDOM_SEED,
                    "SCALE": SCALE,
                    "SHUFFLE": global_data.esrgan.SHUFFLE,
                    "LAMBDA_ADVERSARIAL": global_data.esrgan.LAMBDA_ADVERSARIAL,
                    "LAMBDA_PIXEL_L1": global_data.esrgan.LAMBDA_PIXEL_L1,
                    "LAMBDA_PIXEL_MSE": global_data.esrgan.LAMBDA_PIXEL_MSE,
                    "SAVE_AS_GRAY": global_data.esrgan.SAVE_AS_GRAY,
                    "weight_decay": global_data.esrgan.weight_decay,
                    "g_optimizer_betas": global_data.esrgan.g_optimizer_betas,
                    "d_optimizer_betas": global_data.esrgan.d_optimizer_betas,
                    "Train_nums_rate": global_data.esrgan.Train_nums_rate,
                    "Validate_nums_rate": global_data.esrgan.Validate_nums_rate,
                    "Test_nums_rate": global_data.esrgan.Test_nums_rate,
                    "train_class_mode": mode,
                    "TRAIN_MODE": global_data.esrgan.validate_train_mode(),
                    "selected_classes": selected_classes if selected_classes is not None else "ALL_MIXED",
                },
            )

            # 创建文件夹
            Path(f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.LOSS_DIR}").mkdir(
                parents=True, exist_ok=True)
            Path(f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.TRAINING_DIR}").mkdir(
                parents=True, exist_ok=True)
            Path(f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.MODEL_DIR}").mkdir(
                parents=True, exist_ok=True)
            Path(f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.PREDICT_DIR}").mkdir(
                parents=True, exist_ok=True)
            Path(f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.PREDICT_ALL_DIR}").mkdir(
                parents=True, exist_ok=True)
            Path(f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.LOG_DIR}").mkdir(
                parents=True, exist_ok=True)

            #初始化训练日志
            logger.add(
                f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.LOG_DIR}/training.log",
                rotation="100 MB",
                retention="30 days",
                level="DEBUG",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {process.name} | {thread.name} | {name}:{module}:{line} | {message}",
                enqueue=True,
                backtrace=True,
                diagnose=True,

            )
            animator = Animator(xlabel='epoch', xlim=[1, global_data.esrgan.EPOCH_NUMS], ylim=[0, 0.5],
                                legend=global_data.esrgan.loss_label + global_data.esrgan.validate_label)

            ESRuRAFT_PIV_model = ESRuRAFT_PIV(inner_chanel=3,batch_size=global_data.esrgan.BATCH_SIZE).to(global_data.esrgan.device, non_blocking=True)
            if global_data.esrgan.csvOperator is None:
                global_data.esrgan.csvOperator = CsvTable(
                    file_path=f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.LOSS_DIR}/loss_{class_name} _{data_type}_scale_{int(SCALE * SCALE)}.csv",
                    columns=global_data.esrgan.CSV_COLUMNS)
            else:
                global_data.esrgan.csvOperator.switch_file(
                    file_path=f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.LOSS_DIR}/loss_{class_name} _{data_type}_scale_{int(SCALE * SCALE)}.csv")
            if global_data.esrgan.IS_LOAD_EXISTS_MODEL:
                ESRuRAFT_PIV_model_save_path = f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.MODEL_DIR}/ESRuRAFT_PIV_model_{global_data.esrgan.name}.pth"
                if os.path.exists(ESRuRAFT_PIV_model_save_path):
                    ESRuRAFT_PIV_model.load_state_dict(torch.load(ESRuRAFT_PIV_model_save_path, map_location=global_data.esrgan.device))
                    logger.info(f"Loaded pretrained model ESRuRAFT_PIV_model from {ESRuRAFT_PIV_model_save_path}")
                else:
                    logger.info("No pretrained model ESRuRAFT_PIV_model found. Starting training from scratch.")



            ESRuRAFT_PIV_model_g_optimizer = torch.optim.Adam(_trainable_parameters_or_dummy(ESRuRAFT_PIV_model.piv_esrgan_generator, global_data.esrgan.device), lr=global_data.esrgan.G_LR, betas=global_data.esrgan.g_optimizer_betas,
                                           weight_decay=global_data.esrgan.weight_decay)
            ESRuRAFT_PIV_model_d_optimizer = torch.optim.Adam(_trainable_parameters_or_dummy(ESRuRAFT_PIV_model.piv_esrgan_discriminator, global_data.esrgan.device), lr=global_data.esrgan.D_LR, betas=global_data.esrgan.d_optimizer_betas,
                                           weight_decay=global_data.esrgan.weight_decay)

            ESRuRAFT_PIV_model_RAFT_optimizeroptimizer = torch.optim.AdamW(ESRuRAFT_PIV_model.piv_RAFT.parameters(), lr=global_data.esrgan.RAFT_LR, betas=global_data.esrgan.RAFT_optimizer_betas,)
            # 这两个标记用来保护已有断点恢复：
            # 如果当前实验自己的 RAFT optimizer/scheduler 已经从 OUT_PUT_DIR 成功恢复，
            # 后面就不再用 RAFT256 checkpoint 覆盖它们。
            raft_optimizer_loaded_from_resume = False
            raft_scheduler_loaded_from_resume = False

            #是否读取之前存储的优化器
            if global_data.esrgan.IS_LOAD_EXISTS_MODEL:
                ESRuRAFT_PIV_g_optimizer_save_path = f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.MODEL_DIR}/ESRuRAFT_PIV_g_optimizer_{global_data.esrgan.name}.pth"
                if os.path.exists(ESRuRAFT_PIV_g_optimizer_save_path):
                    ESRuRAFT_PIV_model_g_optimizer.load_state_dict(torch.load(ESRuRAFT_PIV_g_optimizer_save_path, map_location=global_data.esrgan.device))
                    logger.info(f"Loaded pretrained optimizer ESRuRAFT_PIV_g_optimizer from {ESRuRAFT_PIV_g_optimizer_save_path}")
                else:
                    logger.info("No pretrained optimizer ESRuRAFT_PIV_g_optimizer found. Starting training from scratch.")

                ESRuRAFT_PIV_d_optimizer_save_path = f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.MODEL_DIR}/ESRuRAFT_PIV_d_optimizer_{global_data.esrgan.name}.pth"
                if os.path.exists(ESRuRAFT_PIV_g_optimizer_save_path):
                    ESRuRAFT_PIV_model_d_optimizer.load_state_dict(
                        torch.load(ESRuRAFT_PIV_d_optimizer_save_path, map_location=global_data.esrgan.device))
                    logger.info(
                        f"Loaded pretrained optimizer ESRuRAFT_PIV_d_optimizer from {ESRuRAFT_PIV_d_optimizer_save_path}")
                else:
                    logger.info(
                        "No pretrained optimizer ESRuRAFT_PIV_d_optimizer found. Starting training from scratch.")

                ESRuRAFT_PIV_RAFT_optimizer_save_path = f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.MODEL_DIR}/ESRuRAFT_PIV_RAFT_optimizer_{global_data.esrgan.name}.pth"
                if os.path.exists(ESRuRAFT_PIV_RAFT_optimizer_save_path):
                    ESRuRAFT_PIV_model_RAFT_optimizeroptimizer.load_state_dict(
                        torch.load(ESRuRAFT_PIV_RAFT_optimizer_save_path, map_location=global_data.esrgan.device))
                    raft_optimizer_loaded_from_resume = True
                    logger.info(
                        f"Loaded pretrained optimizer ESRuRAFT_PIV_RAFT_optimizer from {ESRuRAFT_PIV_RAFT_optimizer_save_path}")
                else:
                    logger.info(
                        "No pretrained optimizer ESRuRAFT_PIV_RAFT_optimizer found. Starting training from scratch.")

            #动态学习率 基于监控指标动态调整学习率的调度器
            g_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(ESRuRAFT_PIV_model_g_optimizer, 'min', factor=global_data.esrgan.G_LR_reduce_factor,
                                                                   patience=global_data.esrgan.G_LR_patience_level, min_lr=global_data.esrgan.G_LR_min)
            d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(ESRuRAFT_PIV_model_d_optimizer, 'min', factor=global_data.esrgan.D_LR_reduce_factor,
                                                                   patience=global_data.esrgan.D_LR_patience_level, min_lr=global_data.esrgan.D_LR_min)
            raft_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(ESRuRAFT_PIV_model_RAFT_optimizeroptimizer, 'min', factor=global_data.esrgan.RAFT_LR_reduce_factor,
                                                                   patience=global_data.esrgan.RAFT_LR_patience_level, min_lr=global_data.esrgan.RAFT_LR_min)
            # 是否读取之前存储的动态学习率器
            if global_data.esrgan.IS_LOAD_EXISTS_MODEL:
                ESRuRAFT_PIV_g_scheduler_save_path = f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.MODEL_DIR}/ESRuRAFT_PIV_g_scheduler_{global_data.esrgan.name}.pth"
                if os.path.exists(ESRuRAFT_PIV_g_scheduler_save_path):
                    g_scheduler.load_state_dict(torch.load(ESRuRAFT_PIV_g_scheduler_save_path, map_location=global_data.esrgan.device))
                    logger.info(f"Loaded pretrained optimizer ESRuRAFT_PIV_g_scheduler from {ESRuRAFT_PIV_g_scheduler_save_path}")
                else:
                    logger.info("No pretrained optimizer ESRuRAFT_PIV_g_scheduler found. Starting training from scratch.")

                ESRuRAFT_PIV_d_scheduler_save_path = f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.MODEL_DIR}/ESRuRAFT_PIV_d_scheduler_{global_data.esrgan.name}.pth"
                if os.path.exists(ESRuRAFT_PIV_d_scheduler_save_path):
                    d_scheduler.load_state_dict(
                        torch.load(ESRuRAFT_PIV_d_scheduler_save_path, map_location=global_data.esrgan.device))
                    logger.info(
                        f"Loaded pretrained optimizer ESRuRAFT_PIV_d_scheduler from {ESRuRAFT_PIV_d_scheduler_save_path}")
                else:
                    logger.info(
                        "No pretrained optimizer ESRuRAFT_PIV_d_scheduler found. Starting training from scratch.")

                ESRuRAFT_PIV_raft_scheduler_save_path = f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.MODEL_DIR}/ESRuRAFT_PIV_raft_scheduler_{global_data.esrgan.name}.pth"
                if os.path.exists(ESRuRAFT_PIV_raft_scheduler_save_path):
                    raft_scheduler.load_state_dict(
                        torch.load(ESRuRAFT_PIV_raft_scheduler_save_path, map_location=global_data.esrgan.device))
                    raft_scheduler_loaded_from_resume = True
                    logger.info(
                        f"Loaded pretrained optimizer ESRuRAFT_PIV_raft_scheduler from {ESRuRAFT_PIV_raft_scheduler_save_path}")
                else:
                    logger.info(
                        "No pretrained optimizer ESRuRAFT_PIV_raft_scheduler found. Starting training from scratch.")

            # ESRuRAFT_PIV_Ground 专用：当 RAFT_MODEL_TYPE="RAFT128" 且 RAFT128_INIT_FROM_RAFT256=True 时，
            # 除了模型权重，RAFT optimizer / scheduler 也可以从 ckpt_256.tar 做安全迁移。
            # 已经成功恢复当前实验断点时，上面的 loaded_from_resume 标记会阻止这里覆盖恢复结果。
            _maybe_migrate_raft_optimizer_scheduler_from_raft256(
                model=ESRuRAFT_PIV_model,
                optimizer=ESRuRAFT_PIV_model_RAFT_optimizeroptimizer,
                scheduler=raft_scheduler,
                optimizer_loaded_from_resume=raft_optimizer_loaded_from_resume,
                scheduler_loaded_from_resume=raft_scheduler_loaded_from_resume,
            )

            ESRuRAFT_PIV_model_scaler = GradScaler()


            # 这里单独记录“当前这次训练”的起始时间，
            # 不覆盖 global_data.esrgan.START_TIME。
            # global_data.esrgan.START_TIME 表示程序/流程整体开始运行的时间，
            # training_start_time 只用于统计本次训练耗时（小时）。
            training_start_time = time.time()
            # 轮数
            """
            训练 start
            """
            for epoch in range(global_data.esrgan.START_EPOCH-1,global_data.esrgan.EPOCH_NUMS):
                # 动态更新三类损失权重。
                # 这些权重在每个 epoch 开始时刷新一次，本 epoch 内所有 batch 保持一致：
                # 1. LAMBDA_ADVERSARIAL: 生成器对抗损失权重，前半程从 0.0005 增长到 0.02。
                # 2. LAMBDA_FLOW_WARP_CONSISTENCY: GT-flow warp 图像对一致性损失权重，前半程从 0.012 增长到 1.2。
                # 3. RAFT_EPE_WEIGHT: Generator 侧 RAFT EPE 反作用权重，后半程从 1 增长到 3。
                current_dynamic_weights = global_data.esrgan.update_dynamic_loss_weights(epoch)
                logger.info(
                    "[Train] Epoch {}/{}: current dynamic loss weights | "
                    "LAMBDA_ADVERSARIAL={:.6f}, "
                    "LAMBDA_FLOW_WARP_CONSISTENCY={:.6f}, "
                    "RAFT_EPE_WEIGHT={:.6f}".format(
                        epoch + 1,
                        global_data.esrgan.EPOCH_NUMS,
                        current_dynamic_weights["lambda_adversarial"],
                        current_dynamic_weights["lambda_flow_warp_consistency"],
                        current_dynamic_weights["raft_epe_weight"],
                    )
                )
                ESRuRAFT_PIV_model.train()  # 确保在训练模式
                metric = Accumulator(len(global_data.esrgan.loss_label))
                train_progress_bar = tqdm(train_loader,
                                          desc=f"Epoch [{epoch + 1}/{global_data.esrgan.EPOCH_NUMS}] {class_name} {data_type} scale_{int(SCALE * SCALE)} Training",
                                          unit="batch", dynamic_ncols=True,
                                        ascii=True,
                                        leave=True,

                                          )

                for i, batch in enumerate(train_progress_bar):
                    """RAFT 联合训练"""
                    esrgan_union_RAFT_train(
                        epoch=epoch,
                        batch=batch, i=i,
                        g_optimizer=ESRuRAFT_PIV_model_g_optimizer,
                        d_optimizer=ESRuRAFT_PIV_model_d_optimizer,
                        RAFT_optimizer = ESRuRAFT_PIV_model_RAFT_optimizeroptimizer,
                        scaler = ESRuRAFT_PIV_model_scaler,

                        model = ESRuRAFT_PIV_model,
                        train_progress_bar=train_progress_bar,
                        metric=metric, data_type=data_type, device=global_data.esrgan.device, class_name=class_name,
                        SCALE=SCALE

                    )
                # 每轮结束后评价一次 验证集只取一轮batch
                avg_val_mse_loss,avg_val_ssim_loss,avg_psnr,avg_val_energy_spectrum_mse,avg_val_aee,avg_val_norm_aee_per100=evaluate(epoch=epoch, class_name=class_name, data_type=data_type, device=global_data.esrgan.device,
                         model = ESRuRAFT_PIV_model, animator=animator, validate_loader=validate_loader,
                         loss_label=global_data.esrgan.loss_label,validate_label=global_data.esrgan. validate_label, SCALE=SCALE,
                         csvOperator=global_data.esrgan.csvOperator,metric=metric,train_loader_lens=len(train_loader))

                #动态学习率step
                # g_scheduler.step(avg_val_energy_spectrum_mse)
                raft_scheduler.step(avg_val_aee)
                #保存动态学习率器
                ESRuRAFT_PIV_g_scheduler_save_path = f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.MODEL_DIR}/ESRuRAFT_PIV_g_scheduler_{global_data.esrgan.name}.pth"
                ESRuRAFT_PIV_d_scheduler_save_path = f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.MODEL_DIR}/ESRuRAFT_PIV_d_scheduler_{global_data.esrgan.name}.pth"
                ESRuRAFT_PIV_raft_scheduler_save_path = f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.MODEL_DIR}/ESRuRAFT_PIV_raft_scheduler_{global_data.esrgan.name}.pth"
                torch.save(g_scheduler.state_dict(), ESRuRAFT_PIV_g_scheduler_save_path)
                logger.info(
                    f"{class_name} {data_type} |g_scheduler saved: v -> {ESRuRAFT_PIV_g_scheduler_save_path}")
                torch.save(d_scheduler.state_dict(), ESRuRAFT_PIV_d_scheduler_save_path)
                logger.info(
                    f"{class_name} {data_type} |d_scheduler saved: v -> {ESRuRAFT_PIV_d_scheduler_save_path}")
                torch.save(raft_scheduler.state_dict(),
                           ESRuRAFT_PIV_raft_scheduler_save_path)
                logger.info(
                    f"{class_name} {data_type} |RAFT_scheduler saved: v -> {ESRuRAFT_PIV_raft_scheduler_save_path}")
                # 保存优化器
                ESRuRAFT_PIV_g_optimizer_save_path = f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.MODEL_DIR}/ESRuRAFT_PIV_g_optimizer_{global_data.esrgan.name}.pth"
                ESRuRAFT_PIV_d_optimizer_save_path = f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.MODEL_DIR}/ESRuRAFT_PIV_d_optimizer_{global_data.esrgan.name}.pth"
                ESRuRAFT_PIV_RAFT_optimizer_save_path = f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.MODEL_DIR}/ESRuRAFT_PIV_RAFT_optimizer_{global_data.esrgan.name}.pth"
                torch.save(ESRuRAFT_PIV_model_g_optimizer.state_dict(), ESRuRAFT_PIV_g_optimizer_save_path)
                logger.info(
                    f"{class_name} {data_type} |g_optimizer saved: v -> {ESRuRAFT_PIV_g_optimizer_save_path}")
                torch.save(ESRuRAFT_PIV_model_d_optimizer.state_dict(), ESRuRAFT_PIV_d_optimizer_save_path)
                logger.info(
                    f"{class_name} {data_type} |d_optimizer saved: v -> {ESRuRAFT_PIV_d_optimizer_save_path}")
                torch.save(ESRuRAFT_PIV_model_RAFT_optimizeroptimizer.state_dict(), ESRuRAFT_PIV_RAFT_optimizer_save_path)
                logger.info(
                    f"{class_name} {data_type} |RAFT_optimizer saved: v -> {ESRuRAFT_PIV_RAFT_optimizer_save_path}")
                # 保存模型
                model_save_path = f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.MODEL_DIR}/ESRuRAFT_PIV_model_{global_data.esrgan.name}.pth"
                if _model_parameters_are_finite(ESRuRAFT_PIV_model, "ESRuRAFT_PIV_model"):
                    torch.save(ESRuRAFT_PIV_model.state_dict(), model_save_path)
                    logger.info(
                        f"{class_name} {data_type} |Models saved: v -> {model_save_path}")

            training_end_time = time.time()
            # 训练总时长统一换算成小时，直接对应你要记录的“训练时间（小时）”。
            training_time_hours = (training_end_time - training_start_time) / 3600.0
            sample_batch = _pick_profile_batch(validate_loader, train_loader)
            # 训练结束后再做一次统一 profiling，得到显存 / FLOPs / 推理时间 / 参数量。
            metrics_summary = _profile_esru_raft_piv_model(
                model=ESRuRAFT_PIV_model,
                sample_batch=sample_batch,
                device=global_data.esrgan.device,
            )
            metrics_summary["training_time_hours"] = training_time_hours
            # 把这次实验的整体指标写入实验级汇总 CSV。
            _save_run_metrics_summary(
                class_name=class_name,
                data_type=data_type,
                scale=SCALE,
                metrics_summary=metrics_summary,
            )

            wandb.finish()
            """
            训练 end
            """

            """
            验证集全部验证一遍 start
            """
            evaluate_all(
                model = ESRuRAFT_PIV_model,
                data_loader=validate_loader,
                class_name=class_name,
                data_type=data_type,
                SCALE=SCALE,
                output_root=f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.PREDICT_ALL_DIR}",
                metrics_csv_path=f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.PREDICT_ALL_DIR}/metrics_all.csv",
                stride=6,
            )
            """
            验证集全部验证一遍 end
            """
            """
            保存 训练集 验证集 测试集的引用地址json合集 方便查看用了哪些数据 而且也可以重新load 
            """
            save_loaders_paths(f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/datas_splits.json", train_loader=train_loader,validate_loader=validate_loader, test_loader=test_loader)


    global_data.esrgan.END_TIME = time.time()
    logger.info(f"一共运行：{global_data.esrgan.END_TIME - global_data.esrgan.START_TIME}秒")
    #如果是autodl 运行完就直接关机

if __name__ =="__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"{e}\n{traceback.format_exc()}")
    finally:
        if global_data.esrgan.IS_AUTO_DL:
            os.system("/usr/bin/shutdown")
