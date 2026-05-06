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
from study.SRGAN.data_load import filter_excluded_class_names, get_class_names, load_data, save_loaders_paths



from study.SRGAN.model.ESRuRAFT_PIV.Module.PIV_ESRGAN_RAFT_Model import ESRuRAFT_PIV
from study.SRGAN.model.ESRuRAFT_PIV.evaluate import evaluate, evaluate_all
from study.SRGAN.model.ESRuRAFT_PIV.test import test_all
from study.SRGAN.util.famo_weight_logger import save_famo_weight_snapshot
from study.SRGAN.model.ESRuRAFT_PIV.global_class import global_data
from study.SRGAN.model.ESRuRAFT_PIV.train import esrgan_union_RAFT_train
from study.SRGAN.util.CSV_operator import CsvTable
from study.SRGAN.util.accumulator import Accumulator
from study.SRGAN.util.animator import Animator
from study.SRGAN.util.famo_weight_logger import append_famo_weight_row, save_famo_weight_plot
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


def _get_class_names_compat(gr_data_root_dir: str, dataset_name: str):
    """兼容新旧 `get_class_names()` 签名。"""
    try:
        return get_class_names(gr_data_root_dir, dataset_name=dataset_name)
    except TypeError as exc:
        if "dataset_name" not in str(exc):
            raise
        if str(dataset_name).strip().lower() == "class_2":
            logger.warning(
                "Imported get_class_names() does not support dataset_name keyword yet; "
                "fallback to class_2 pseudo class name. Please make sure data_load.py is also updated."
            )
            return [getattr(global_data.esrgan, "CLASS2_PSEUDO_CLASS_NAME", "problem_class2_raft_piv")]
        return get_class_names(gr_data_root_dir)


def _load_data_compat(**kwargs):
    """兼容新旧 `load_data()` 签名。"""
    try:
        return load_data(**kwargs)
    except TypeError as exc:
        message = str(exc)
        compatibility_keys = (
            "dataset_name",
            "scale_factor",
            "class2_train_tfrecord",
            "class2_train_tfrecord_idx",
            "class2_validate_tfrecord",
            "class2_validate_tfrecord_idx",
        )
        if not any(key in message for key in compatibility_keys):
            raise
        dataset_name = str(kwargs.get("dataset_name", "class_1")).strip().lower()
        if dataset_name == "class_2":
            raise RuntimeError(
                "当前导入的 data_load.load_data 仍是旧签名，缺少 class_2 TFRecord 支持。"
                "请同步更新 study/SRGAN/data_load.py 后再运行。"
            ) from exc
        fallback_kwargs = dict(kwargs)
        for key in compatibility_keys:
            fallback_kwargs.pop(key, None)
        logger.warning(
            "Imported load_data() does not support the new dataset_name/class_2 kwargs yet; "
            "fallback to legacy class_1 call signature."
        )
        return load_data(**fallback_kwargs)


def _format_dynamic_weight_for_log(value):
    """
    将动态损失权重格式化为日志字符串。

    FAMO 开启时，global_class.update_dynamic_loss_weights(...) 会返回 managed_by_famo；
    这个字符串不能用 {:.6f} 格式化，所以这里统一做兼容。
    """
    if isinstance(value, str):
        return value
    return f"{float(value):.6f}"


class ESRuRAFTPIVInferenceWrapper(nn.Module):
    """为联合模型 profiling 提供纯 forward 推理接口，只统计模型输出，不包含 loss/backward/优化器。"""

    def __init__(self, model: ESRuRAFT_PIV):
        super().__init__()
        self.model = model

    def forward(self, input_lr_prev, input_lr_next, flowl0):
        # Inference Time 的标准口径只包含 forward。
        # 这里走 Generator + RAFT 输出路径，不计算 SR loss、RAFT loss、判别器 loss，也不做 backward。
        if hasattr(self.model.piv_esrgan_generator, "forward_pair"):
            pred_prev, pred_next = self.model.piv_esrgan_generator.forward_pair(input_lr_prev, input_lr_next)
        else:
            pred_prev = self.model.piv_esrgan_generator(input_lr_prev)
            pred_next = self.model.piv_esrgan_generator(input_lr_next)

        raft_prev = self.model._to_raft_frame(pred_prev)
        raft_next = self.model._to_raft_frame(pred_next)
        raft_input = torch.cat([raft_prev, raft_next], dim=1)
        raft_flow_gt = self.model._to_raft_flow_gt(flowl0)
        flow_predictions, _ = self.model.piv_RAFT(raft_input, raft_flow_gt)
        final_flow_prediction = flow_predictions[-1] if isinstance(flow_predictions, (list, tuple)) else flow_predictions
        return pred_prev, pred_next, final_flow_prediction


class ESRuRAFTPIVTrainingStepWrapper(nn.Module):
    """为联合模型 profiling 提供完整 train_step 接口，把训练损失与反向传播都纳入统计。"""

    def __init__(self, model: ESRuRAFT_PIV):
        super().__init__()
        self.model = model
        # profiling 只想测一次真实 train_step 的计算成本，不希望改变训练好的模型参数。
        # 因此这里创建 lr=0 的临时优化器：backward / optimizer.step 的流程照常执行，
        # 但参数值不会因为 profiling 发生更新。
        self.g_optimizer = torch.optim.Adam(
            self.model.piv_esrgan_generator.parameters(),
            lr=0.0,
            betas=global_data.esrgan.g_optimizer_betas,
        )
        self.d_optimizer = torch.optim.Adam(
            self.model.piv_esrgan_discriminator.parameters(),
            lr=0.0,
            betas=global_data.esrgan.d_optimizer_betas,
        )
        self.raft_optimizer = torch.optim.AdamW(
            self.model.piv_RAFT.parameters(),
            lr=0.0,
            betas=global_data.esrgan.RAFT_optimizer_betas,
        )

    def forward(self, input_lr_prev, input_lr_next, input_gr_prev, input_gr_next, flowl0):
        # 这里故意调用完整 train_step，而不是纯 forward。
        # 这样显存 / FLOPs / 时间会包含：
        # 1. Generator 前向与 SR 损失
        # 2. RAFT 前向与光流损失
        # 3. Generator / RAFT / Discriminator 的 backward
        # 4. 三个 optimizer.step 的训练流程开销
        pred_prev, pred_next, final_flow_prediction, loss_dict = self.model.train_step(
            input_lr_prev=input_lr_prev,
            input_lr_next=input_lr_next,
            input_gr_prev=input_gr_prev,
            input_gr_next=input_gr_next,
            flowl0=flowl0,
            generator_optimizer=self.g_optimizer,
            raft_optimizer=self.raft_optimizer,
            d_optimizer=self.d_optimizer,
            scaler=None,
            # 训练结束后的 profiling 按“完整训练阶段”统计，因此启用对抗分支。
            is_adversarial=True,
        )
        return pred_prev, pred_next, final_flow_prediction, loss_dict["g_loss"], loss_dict["raft_loss"]


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


def _snapshot_model_buffers(model):
    # BatchNorm 等层的 running_mean / running_var 属于 buffer，不是参数。
    # profiling 时 model.train() 会更新这些 buffer；这里先克隆一份，结束后恢复，
    # 避免一次基准测试污染已经训练好的模型状态。
    return {name: buffer.detach().clone() for name, buffer in model.named_buffers()}


def _restore_model_buffers(model, buffer_snapshot):
    # 将 profiling 前保存的 buffer 原样拷回去。
    # 使用 copy_ 可以保持原 buffer 对象不变，只恢复其中的数值。
    for name, buffer in model.named_buffers():
        if name in buffer_snapshot:
            buffer.copy_(buffer_snapshot[name])


def _clear_model_grads(model):
    # train_step 会产生梯度；profiling 结束后清空，避免后续保存或评估看到残留 grad。
    for param in model.parameters():
        param.grad = None


def _estimate_training_step_flops(profile_model, inputs, device):
    # torch.profiler 的 FLOPs 统计需要真正执行一次训练步骤。
    # 这里不能使用 torch.no_grad()，因为 train_step 内部包含 backward。
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_flops=True,
    ) as prof:
        _ = profile_model(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
    total_flops = 0
    for event in prof.key_averages():
        event_flops = getattr(event, "flops", 0)
        if event_flops is not None:
            total_flops += event_flops
    return total_flops


def _benchmark_training_step_latency(profile_model, inputs, device, warmup, iters):
    # 训练步骤计时需要包含 forward、loss、backward、optimizer.step。
    # 每轮调用都会走完整 train_step，因此不能包 no_grad。
    batch_size = inputs[0].shape[0] if inputs and hasattr(inputs[0], "shape") else 1
    for _ in range(warmup):
        _ = profile_model(*inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        _ = profile_model(*inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    latency_ms = elapsed * 1000.0 / iters
    throughput = iters * batch_size / elapsed
    return latency_ms, throughput


def _measure_training_step_peak_memory(profile_model, inputs, device):
    # 峰值显存统计同样执行完整 train_step，包含训练损失、反向传播和优化器步骤。
    if device.type != "cuda":
        return 0.0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    _ = profile_model(*inputs)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)

def _profile_esru_raft_piv_model(model, sample_batch, device, warmup=5, iters=20):
    was_training = model.training
    lr_prev, hr_prev, lr_next, hr_next, flow_hr_uv = _extract_profile_inputs(sample_batch, device)
    # 第一套口径：纯 forward 推理指标。
    # 这套结果对应论文表格里常见的 Inference Time / Inference FLOPs。
    inference_model = ESRuRAFTPIVInferenceWrapper(model).to(device, non_blocking=True)
    inference_model.eval()
    inference_inputs = (lr_prev, lr_next, flow_hr_uv)

    # 第二套口径：完整训练 step 指标。
    # 这套结果包含 forward、loss、backward、optimizer.step，更接近单个 batch 的训练成本。
    training_step_model = ESRuRAFTPIVTrainingStepWrapper(model).to(device, non_blocking=True)
    training_step_model.train()
    training_step_inputs = (lr_prev, lr_next, hr_prev, hr_next, flow_hr_uv)

    # 这里记录的是“可训练参数”，更符合实验表格里常见的模型规模口径。
    _, trainable_params = count_parameters(model)
    buffer_snapshot = _snapshot_model_buffers(model)
    try:
        inference_flops = estimate_flops_for_inputs(inference_model, inference_inputs, device)
        inference_latency_ms, _ = benchmark_latency_for_inputs(
            inference_model,
            inference_inputs,
            device,
            warmup=warmup,
            iters=iters,
        )
        inference_peak_memory_mb = measure_peak_memory_for_inputs(inference_model, inference_inputs, device)

        training_step_flops = _estimate_training_step_flops(training_step_model, training_step_inputs, device)
        training_step_latency_ms, _ = _benchmark_training_step_latency(
            training_step_model,
            training_step_inputs,
            device,
            warmup=warmup,
            iters=iters,
        )
        training_step_peak_memory_mb = _measure_training_step_peak_memory(training_step_model, training_step_inputs, device)
    finally:
        # 无论 profiling 是否中途失败，都尽量恢复模型状态，避免影响后续保存 / evaluate_all。
        _restore_model_buffers(model, buffer_snapshot)
        _clear_model_grads(model)

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
        "inference_gpu_memory_usage_gb": float(inference_peak_memory_mb) / 1024.0,
        "inference_flops_g": float(inference_flops) / 1e9,
        "inference_time_seconds": float(inference_latency_ms) / 1000.0,
        "training_step_gpu_memory_usage_gb": float(training_step_peak_memory_mb) / 1024.0,
        "training_step_flops_g": float(training_step_flops) / 1e9,
        "training_step_time_seconds": float(training_step_latency_ms) / 1000.0,
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
    dataset_name = global_data.esrgan.validate_dataset_name()
    # 获取类别名；class_1 扫目录，class_2 返回一个伪类别名，表示“整套 TFRecord 就是一个训练任务”。
    available_class_names = _get_class_names_compat(global_data.esrgan.GR_DATA_ROOT_DIR, dataset_name=dataset_name)

    logger.info(f"一共{len(available_class_names)}个类别：{available_class_names}")

    # 训练模式: all | single | mixed | fixed
    # 统一通过 global_class 的校验入口读取，避免新增 fixed 后各 pipeline 的字符串判断不一致。
    mode = global_data.esrgan.validate_train_class_mode()
    if dataset_name == "class_2":
        # class_2 的 train/validate 已经由两个 TFRecord 固定定义；这里把比例超参数改写成真实样本占比。
        global_data.esrgan.update_dataset_split_rates()
        global_data.esrgan.save_hyper_parameters_txt(f"{global_data.esrgan.OUT_PUT_DIR}/hyper_parameters.txt")
    elif mode == "fixed":
        # fixed 模式下，真实划分由 FlowData_train/test.list 决定。
        # 这里先按 list 行数同步比例超参数，只用于日志、wandb 和 hyper_parameters.txt 展示。
        global_data.esrgan.update_dataset_split_rates()
        # hyper_parameters.txt 在上面已经按默认比例写过一次；fixed 需要覆盖为 list 行数反推后的真实比例。
        global_data.esrgan.save_hyper_parameters_txt(f"{global_data.esrgan.OUT_PUT_DIR}/hyper_parameters.txt")
    if dataset_name == "class_1" and mode in {"all", "mixed", "fixed"}:
        # EXCLUDE_CLASS 只对 all/mixed/fixed 生效：这里先过滤任务列表，
        # 后面 load_data 也会收到同一份排除配置，确保真正的数据收集阶段不读取这些类别。
        available_class_names = filter_excluded_class_names(
            available_class_names,
            global_data.esrgan.EXCLUDE_CLASS,
            context=f"{global_data.esrgan.name}:{mode}",
        )
        logger.info(f"排除类别后剩余{len(available_class_names)}个类别：{available_class_names}")

    run_jobs = []
    if dataset_name == "class_2":
        logger.info("DATA_SET=class_2：跳过按类别拆任务，直接使用 TFRecord train/validate 作为单个训练任务。")
        run_jobs.append({"run_class_name": available_class_names[0], "selected_classes": None})
    elif mode == "all":
        # 每个类别读取数据并且训练验证和保存模型
        for class_name in available_class_names:
            run_jobs.append({"run_class_name": class_name, "selected_classes": [class_name]})
    elif mode == "single":
        chosen = select_single_class(available_class_names, global_data.esrgan.SINGLE_CLASS_NAME)
        # 每个类别读取数据并且训练验证和保存模型
        run_jobs.append({"run_class_name": chosen, "selected_classes": [chosen]})
    elif mode == "mixed":
        # 每个类别读取数据并且训练验证和保存模型
        run_jobs.append({"run_class_name": global_data.esrgan.MIXED_CLASS_TAG, "selected_classes": None})
    else:
        # fixed 只训练一个任务：先收集所有类别，再由 data_load 按两个 list 文件固定筛选 train/validate。
        run_jobs.append({"run_class_name": global_data.esrgan.FIXED_CLASS_TAG, "selected_classes": None})

    for job in run_jobs:
        class_name = job["run_class_name"]
        selected_classes = job["selected_classes"]

        # 几倍上采样倍率来训练
        for SCALE in global_data.esrgan.SCALES:
            # 获取数据 自动根据类别划分数据集并读取，每个类别都安装比例划分训练集和验证集
            # 根据类别和上采样读取数据
            train_loader, validate_loader, test_loader, class_names, samples = _load_data_compat(
                gr_data_root_dir=global_data.esrgan.GR_DATA_ROOT_DIR,
                lr_data_root_dir=f"{global_data.esrgan.LR_DATA_ROOT_DIR}/x{int(SCALE * SCALE)}/data",
                batch_size=global_data.esrgan.BATCH_SIZE,
                # fixed 模式必须保留 FlowData_train.list 的顺序，因此无论 SHUFFLE 如何配置都传 False。
                shuffle=False if (dataset_name == "class_1" and mode == "fixed") else global_data.esrgan.SHUFFLE,
                target_size=global_data.esrgan.TARGET_SIZE,
                train_nums_rate=global_data.esrgan.Train_nums_rate,
                validate_nums_rate=global_data.esrgan.Validate_nums_rate,
                test_nums_rate=global_data.esrgan.Test_nums_rate,
                random_seed=global_data.esrgan.RANDOM_SEED,
                selected_classes=selected_classes,
                excluded_classes=global_data.esrgan.EXCLUDE_CLASS if (dataset_name == "class_1" and mode in {"all", "mixed", "fixed"}) else None,
                class_sample_ratio=global_data.esrgan.CLASS_SAMPLE_RATIO,
                fixed_split=dataset_name == "class_1" and mode == "fixed",
                fixed_train_list_path=global_data.esrgan.FIXED_TRAIN_LIST_PATH,
                fixed_validate_list_path=global_data.esrgan.FIXED_VALIDATE_LIST_PATH,
                dataset_name=dataset_name,
                scale_factor=int(SCALE * SCALE),
                class2_train_tfrecord=global_data.esrgan.CLASS2_TRAIN_TFRECORD,
                class2_train_tfrecord_idx=global_data.esrgan.CLASS2_TRAIN_TFRECORD_IDX,
                class2_validate_tfrecord=global_data.esrgan.CLASS2_VALIDATE_TFRECORD,
                class2_validate_tfrecord_idx=global_data.esrgan.CLASS2_VALIDATE_TFRECORD_IDX,
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
                    "train_mode": mode,
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



            ESRuRAFT_PIV_model_g_optimizer = torch.optim.Adam(ESRuRAFT_PIV_model.piv_esrgan_generator.parameters(), lr=global_data.esrgan.G_LR, betas=global_data.esrgan.g_optimizer_betas,
                                           weight_decay=global_data.esrgan.weight_decay)
            ESRuRAFT_PIV_model_d_optimizer = torch.optim.Adam(ESRuRAFT_PIV_model.piv_esrgan_discriminator.parameters(), lr=global_data.esrgan.D_LR, betas=global_data.esrgan.d_optimizer_betas,
                                           weight_decay=global_data.esrgan.weight_decay)

            ESRuRAFT_PIV_model_RAFT_optimizeroptimizer = torch.optim.AdamW(ESRuRAFT_PIV_model.piv_RAFT.parameters(), lr=global_data.esrgan.RAFT_LR, betas=global_data.esrgan.RAFT_optimizer_betas,)

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
                    logger.info(
                        f"Loaded pretrained optimizer ESRuRAFT_PIV_raft_scheduler from {ESRuRAFT_PIV_raft_scheduler_save_path}")
                else:
                    logger.info(
                        "No pretrained optimizer ESRuRAFT_PIV_raft_scheduler found. Starting training from scratch.")

            ESRuRAFT_PIV_model_scaler = GradScaler()


            # 这里单独记录“当前这次训练”的起始时间，
            # 不覆盖 global_data.esrgan.START_TIME。
            # global_data.esrgan.START_TIME 表示程序/流程整体开始运行的时间，
            # training_start_time 只用于统计本次训练耗时（小时）。
            training_start_time = time.time()
            # FAMO 权重独立保存路径。
            # 不放入原 loss CSV，避免破坏 global_class.loss_label 和 metric.add(...) 的固定顺序。
            scale_output_dir = Path(
                f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}"
            )
            famo_weight_csv_path = scale_output_dir / global_data.esrgan.LOSS_DIR.strip("/") / "famo_weights.csv"
            famo_weight_plot_path = scale_output_dir / global_data.esrgan.LOSS_DIR.strip("/") / "famo_weights.png"
            # 轮数
            """
            训练 start
            """
            training_enabled = bool(getattr(global_data.esrgan, "IS_training", True))
            if not training_enabled:
                logger.info("IS_training=False，跳过训练循环，直接进入 evaluate_all/test_all。")
            # 保持原训练循环主体不变：开关关闭时只把 epoch_range 置空，后续验证/测试流程仍继续执行。
            epoch_range = (
                range(global_data.esrgan.START_EPOCH - 1, global_data.esrgan.EPOCH_NUMS)
                if training_enabled
                else range(0)
            )
            for epoch in epoch_range:
                # 动态更新三类损失权重。
                # 这些权重在每个 epoch 开始时刷新一次，本 epoch 内所有 batch 保持一致：
                # 1. LAMBDA_ADVERSARIAL: 生成器对抗损失权重，前半程从 0.0005 增长到 0.02。
                # 2. LAMBDA_FLOW_WARP_CONSISTENCY: GT-flow warp 图像对一致性损失权重，前半程从 0.012 增长到 1.2。
                # 3. RAFT_EPE_WEIGHT: Generator 侧 RAFT EPE 反作用权重，后半程从 1 增长到 3。
                current_dynamic_weights = global_data.esrgan.update_dynamic_loss_weights(epoch)
                logger.info(
                    "[Train] Epoch {}/{}: current dynamic loss weights | "
                    "LAMBDA_ADVERSARIAL={}, "
                    "LAMBDA_FLOW_WARP_CONSISTENCY={}, "
                    "RAFT_EPE_WEIGHT={}".format(
                        epoch + 1,
                        global_data.esrgan.EPOCH_NUMS,
                        _format_dynamic_weight_for_log(current_dynamic_weights["lambda_adversarial"]),
                        _format_dynamic_weight_for_log(current_dynamic_weights["lambda_flow_warp_consistency"]),
                        _format_dynamic_weight_for_log(current_dynamic_weights["raft_epe_weight"]),
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
                    loss_dict = esrgan_union_RAFT_train(
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
                    # 每个 batch 记录一次 FAMO 权重。
                    # FAMO 的权重是按训练 step 更新的，逐 batch 保存才能看到真实变化曲线。
                    append_famo_weight_row(
                        famo_weight_csv_path,
                        epoch=epoch,
                        batch_index=i,
                        global_step=epoch * len(train_loader) + i,
                        loss_dict=loss_dict,
                    )
                # 每轮结束后评价一次 验证集只取一轮batch
                avg_val_mse_loss,avg_val_ssim_loss,avg_psnr,avg_val_energy_spectrum_mse,avg_val_aee,avg_val_norm_aee_per100=evaluate(epoch=epoch, class_name=class_name, data_type=data_type, device=global_data.esrgan.device,
                         model = ESRuRAFT_PIV_model, animator=animator, validate_loader=validate_loader,
                         loss_label=global_data.esrgan.loss_label,validate_label=global_data.esrgan. validate_label, SCALE=SCALE,
                         csvOperator=global_data.esrgan.csvOperator,metric=metric,train_loader_lens=len(train_loader))
                # 每个 epoch 结束后刷新一次折线图。
                # 这样不会在 batch 内频繁画图拖慢训练，同时训练中断时也能保留已经完成 epoch 的图。
                save_famo_weight_plot(
                    famo_weight_csv_path,
                    famo_weight_plot_path,
                    title=f"ESRuRAFT_PIV FAMO Weights | {class_name} {data_type} scale_{int(SCALE * SCALE)}",
                )

                # FAMO 权重快照只在 global_data.esrgan.USE_FAMO=True 且模型持有 generator_famo 时写入。
                # 关闭 FAMO 时该函数会直接返回，因此不会改变普通手动权重训练的输出行为。
                save_famo_weight_snapshot(
                    model=ESRuRAFT_PIV_model,
                    epoch=epoch + 1,
                    output_dir=f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.LOSS_DIR}",
                    file_prefix=f"famo_weights_ESRuRAFT_PIV_{global_data.esrgan.name}",
                    logger=logger,
                )

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
            if global_data.esrgan.IS_VALIDATE_ALL:
                # 完整验证可能比较耗时，使用 IS_VALIDATE_ALL 做总开关；默认 True，保持原有流程。
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
            else:
                logger.info(
                    "跳过 evaluate_all："
                    f" IS_VALIDATE_ALL={global_data.esrgan.IS_VALIDATE_ALL}"
                )
            """
            验证集全部验证一遍 end
            """
            """
            RAFT256-PIV 风格 TFRecord 全数据集测试 start
            """
            if global_data.esrgan.IS_TEST:
                # test_all 使用单 GPU 顺序测试 TEST_DATASETS；默认 IS_TEST=False，不改变原训练流程。
                test_all(
                    model=ESRuRAFT_PIV_model,
                    class_name=class_name,
                    data_type=data_type,
                    SCALE=SCALE,
                    device=global_data.esrgan.device,
                )
            else:
                logger.info("IS_TEST=False，跳过 test_all TFRecord 全数据集测试。")
            """
            RAFT256-PIV 风格 TFRecord 全数据集测试 end
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












