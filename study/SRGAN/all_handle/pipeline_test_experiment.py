from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image, ImageDraw, ImageFont


# 允许用户在 SRGAN 根目录直接执行：
#     python all_handle/pipeline_test_experiment.py
# 现有 pipeline_test.py 多数依赖 study.SRGAN 绝对导入；这里主动把项目根目录
# D:\WorkSpace\super_resolution_project 加入 sys.path，避免工作目录变化导致导入失败。
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from study.SRGAN.data_downscal import read_flo
from study.SRGAN.model.c_aee_metric_common import attach_c_aee_to_raft_rows
from study.SRGAN.model.pipeline_test_common import (
    _extract_model_state_dict,
    _load_state_dict_checked,
    _normalize_state_dict_keys_for_model,
)
from study.SRGAN.model.tfrecord_test_common import (
    _as_numpy_batch,
    _compute_flow_error_maps_np,
    _compute_flow_metric_row,
    _compute_image_metric_row,
    _energy_spectrum_curves,
    _energy_spectrum_mse_from_curves,
    _flow_uv_to_uvw_np,
    _adapt_image_channels_for_model,
    _collapse_image_to_single_channel_for_test,
    _omega_star_from_uv_np,
    _fold_weighted_patches,
    _infer_model_image_channels,
    _last_flow_prediction,
    _pad_full_frame_for_sliding,
    _plot_regular,
    _save_mean_spectrum,
    _save_flow_visual_artifacts,
    _save_image_outputs,
    _save_sample_plots,
    _sliding_full_coverage_size,
    _window_2d,
    _write_csv,
    _write_rows_with_mean,
)


# =========================
# 固定实验输入与输出配置
# =========================
# DATA_ROOT_DIR = Path(
#     r"D:\BaiduSyncdisk\AYanJiuSheng\data\train_datas\root\autodl-tmp\train_datas"
# )
DATA_ROOT_DIR = Path(
    r"/study_datas/train_datas/root/autodl-tmp/train_datas/"
)
# 用户指定的六组实验图片已经由 experiment_handle/pre_handle.py 预处理完成；
# 本脚本只读取这些结果，不重新减背景，也不重新生成 OpenPIV flo。
# EXPERIMENT_HANDLE_DIR = Path(r"D:\BaiduSyncdisk\AYanJiuSheng\data\train_datas\experiment_handle")
EXPERIMENT_HANDLE_DIR = Path(r"/study_datas/train_datas/experiment_handle")

# 用户指定的各对比实验根目录。目录结构为：
#   train_datas/{experiment_dir}/{class_1|class_2}/{mixed_all_classes|problem_class2_raft_piv}/RAFT/scale_x/train_model
# TRAIN_DATAS_ROOT = Path(r"D:\BaiduSyncdisk\AYanJiuSheng\data\train_datas\root\autodl-tmp\train_datas")
TRAIN_DATAS_ROOT = Path(r"/study_datas/train_datas/root/autodl-tmp/train_datas")

# all_handle 合并图也放在同一批训练结果根目录下，避免写到代码仓库或 Linux 默认路径。
MERGED_OUTPUT_DIR = Path(r"/study_datas/train_all_datas/") / "experiment_all_handle"
PROGRESS_STATUS_PATH = TRAIN_DATAS_ROOT / "experiment_test_progress.json"

# 和用户给定语义保持一致：img1 是 previous，img2 是 next。
EXPERIMENT_SAMPLES = (
    ("start_1", "exp_0042"),
    ("start_2", "exp_0043"),
    ("peak_1", "exp_0152"),
    ("peak_2", "exp_0153"),
    ("end_1", "exp_0802"),
    ("end_2", "exp_0801"),
)

# ESRuRAFT_PIV_Ground 里的这两个实验不是神经网络 checkpoint baseline：
# 它们只复用 Ground 分支的输入整理、bicubic 上采样和传统 PIV/光流估计代码。
# 因此对应 train_model 目录没有 *_model_*.pth 是正常情况，发现任务时不能把它们跳过。
TRADITIONAL_GROUND_TRAIN_MODES = {"bicubic_widim", "bicubic_hs"}
TRADITIONAL_CHECKPOINT_LABEL = "traditional_no_checkpoint"

# 实验图片没有对应 HR 原图；它们本身就是要直接送入模型的低分辨率输入。
# 因此这里的 64 是 LR 坐标里的窗口大小：先把实验 previous/next 切成多个 64x64 LR patch，
# 每个 patch 直接作为 input_lr_prev/input_lr_next，模型输出 SR patch 后再按倍率 fold 回完整 SR 图。
EXPERIMENT_MODEL_INPUT_SIZE = 64
# 不做重叠融合：模型吃 64x64 LR patch，相邻 patch 也按 64 像素步长铺块。
EXPERIMENT_MODEL_PATCH_SHIFT = 64
# 背景 mask 阈值作用在插值后的 HR 参考图上。低于该值的区域认为是黑背景，
# SR 输出拼接后会被压回 0，避免模型在黑区“造”白点。
EXPERIMENT_BACKGROUND_MASK_THRESHOLD = 10.0 / 256.0
# mask 在 HR 尺寸上做少量膨胀，保留颗粒边缘和晕影，避免把真实弱颗粒直接切掉。
EXPERIMENT_BACKGROUND_MASK_DILATE = 3

# _save_sample_plots 会用 matplotlib 绘制 SR 图、flow 和误差图。
# 实验图作为 LR 输入后，输出尺寸会变成原图的 scale_factor 倍；如果一次把六组样本
# 全部交给绘图函数，matplotlib 可能在同一阶段累计很多大图对象，导致 Linux 直接 Killed。
# 因此保留原始尺寸和输出结构，但按 sample_0000...sample_0005 逐个保存并及时释放内存。
EXPERIMENT_SAVE_HEAVY_SAMPLE_PLOTS = True
# 实验图没有真实 HR，颗粒误差是 SR - 插值 HR，数值通常很小；单独缩小实验误差图色条，
# 避免沿用普通测试的 [-2, 2] 导致误差图几乎全白。
EXPERIMENT_PARTICLE_ERROR_COLORBAR_LIMIT = 0.15
# 局部放大区域，使用 HR/SR 坐标的相对位置：(x_ratio, y_ratio, width_ratio, height_ratio)。
# 这些区域覆盖喷流主体、右侧稀疏颗粒和下方颗粒边缘，便于直观看 LR 与 SR 的差异。
EXPERIMENT_PARTICLE_ZOOM_REGIONS = (
    (0.18, 0.38, 0.08, 0.10),
    (0.42, 0.45, 0.08, 0.10),
    (0.72, 0.42, 0.08, 0.10),
)

# 遇到 {"detail":"Bad Request"} 时按用户要求不中断，先重试。
# 这里仍设置上限，避免外部服务长期异常时脚本无限占用 GPU；每次重试都会写日志。
BAD_REQUEST_MAX_RETRIES = 5
BAD_REQUEST_RETRY_SLEEP_SECONDS = 2.0


@dataclass(frozen=True)
class ExperimentSample:
    """一组实验时刻样本，包含 previous/next 两张图和对应 OpenPIV flo 光流。"""

    stage: str
    group_name: str
    previous_path: Path
    next_path: Path
    flow_path: Path


@dataclass(frozen=True)
class ModelJob:
    """一个待测试的 checkpoint 任务，由 train_model 目录反推出输出位置和模型分支。"""

    experiment_dir_name: str
    class_name: str
    run_class_name: str
    data_type: str
    scale_dir_name: str
    scale: float
    train_model_dir: Path
    checkpoint_path: Path | None
    output_root: Path
    branch_name: str
    requires_checkpoint: bool = True


@dataclass(frozen=True)
class BranchRuntime:
    """封装不同实验分支的 global_data 和模型构造函数。"""

    global_data: object
    model_factory: Callable[[float], torch.nn.Module]


def _is_bad_request_exception(exc: Exception) -> bool:
    """识别用户特别点名的 {"detail":"Bad Request"} 异常文本。"""

    text = f"{exc!r}\n{exc}"
    return "Bad Request" in text or '"detail"' in text and "Bad Request" in text


def _retry_bad_request(fn: Callable, context: str):
    """
    对容易触发外部服务/后端临时 Bad Request 的步骤做局部重试。

    其它异常仍立即抛出，避免真正的模型结构、文件缺失、显存等错误被重试掩盖。
    """

    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            if not _is_bad_request_exception(exc):
                raise
            attempt += 1
            logger.warning(
                "[experiment_test] {} 遇到 Bad Request，第 {}/{} 次重试。",
                context,
                attempt,
                BAD_REQUEST_MAX_RETRIES,
            )
            if attempt >= BAD_REQUEST_MAX_RETRIES:
                raise
            time.sleep(BAD_REQUEST_RETRY_SLEEP_SECONDS)


def _progress_log_interval(total: int, max_updates: int = 10) -> int:
    """按总量估算日志间隔，让长任务有进度提示，又避免每个 patch 都刷屏。"""

    return max(1, int(math.ceil(max(1, total) / max(1, max_updates))))


def _format_duration(seconds: float) -> str:
    """把秒数格式化成 h/m/s，方便日志里阅读运行时间和剩余时间。"""

    seconds = max(0.0, float(seconds))
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _estimate_eta(elapsed_seconds: float, completed: int, total: int) -> str:
    """根据当前进度估算剩余时间；进度还太少时返回 unknown。"""

    completed = int(completed)
    total = int(total)
    if completed <= 0 or total <= 0 or completed > total:
        return "unknown"
    remaining = max(0, total - completed)
    seconds_per_item = float(elapsed_seconds) / completed
    return _format_duration(seconds_per_item * remaining)


def _write_progress_status(**status) -> None:
    """
    立即写入当前运行状态，避免进程被系统 Killed 时完全没有 CSV/日志线索。

    这个文件只保存最近一次状态，路径固定在 train_datas 根目录：
    experiment_test_progress.json。
    """

    payload = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **status,
    }
    try:
        PROGRESS_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROGRESS_STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("[experiment_test] failed to write progress status: {}", exc)


def _read_gray_image_unit(path: Path) -> np.ndarray:
    """
    读取预处理后的 bmp，并按 test_all 的 TFRecord 口径缩放到浮点图像。

    run_test_all 中 target 会除以 256；这里保持一致，而不是除以 255，
    让实验三阶段样本与参考 test_all 的输入数值范围完全对齐。
    """

    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    return arr / 256.0


def _load_experiment_samples(input_dir: Path = EXPERIMENT_HANDLE_DIR) -> list[ExperimentSample]:
    """读取用户指定的 start/peak/end 各两组 previous-next-flow 实验样本。"""

    logger.info("[experiment_test] loading experiment samples from {}", input_dir)
    samples: list[ExperimentSample] = []
    for stage, group_name in EXPERIMENT_SAMPLES:
        sample = ExperimentSample(
            stage=stage,
            group_name=group_name,
            previous_path=input_dir / f"{group_name}_img1.bmp",
            next_path=input_dir / f"{group_name}_img2.bmp",
            flow_path=input_dir / f"{group_name}_flow.flo",
        )
        missing = [
            str(path)
            for path in (sample.previous_path, sample.next_path, sample.flow_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(f"[experiment_test] 缺少实验样本文件：{missing}")
        samples.append(sample)
        logger.info(
            "[experiment_test] sample ready: stage={} group={} prev={} next={} flow={}",
            stage,
            group_name,
            sample.previous_path.name,
            sample.next_path.name,
            sample.flow_path.name,
        )
    logger.info("[experiment_test] loaded {} experiment sample groups", len(samples))
    return samples


def _samples_to_tensors(samples: list[ExperimentSample], device: torch.device):
    """
    将多组实验样本整理成 test_all 兼容的张量：
        images: [B, 2, H, W]，第 0 通道 previous，第 1 通道 next；
        flows:  [B, 2, H, W]，第 0 通道 u，第 1 通道 v。
    """

    image_items = []
    flow_items = []
    for sample in samples:
        previous = _read_gray_image_unit(sample.previous_path)
        next_image = _read_gray_image_unit(sample.next_path)
        flow = read_flo(sample.flow_path).astype(np.float32, copy=False)
        if flow.ndim != 3 or flow.shape[2] != 2:
            raise ValueError(f"[experiment_test] flo 必须是 HxWx2：{sample.flow_path}, shape={flow.shape}")
        if previous.shape != next_image.shape or previous.shape != flow.shape[:2]:
            raise ValueError(
                "[experiment_test] 图像与 flo 尺寸不一致："
                f"{sample.group_name}: prev={previous.shape}, next={next_image.shape}, flow={flow.shape}"
            )
        image_items.append(np.stack([previous, next_image], axis=0))
        flow_items.append(np.moveaxis(flow, -1, 0))
        logger.info(
            "[experiment_test] tensor sample: {} image={} flow={}",
            sample.group_name,
            previous.shape,
            flow.shape,
        )

    images = torch.from_numpy(np.stack(image_items).astype(np.float32)).to(device=device)
    flows = torch.from_numpy(np.stack(flow_items).astype(np.float32)).to(device=device)
    logger.info(
        "[experiment_test] moved samples to {}: images={} flows={}",
        device,
        tuple(images.shape),
        tuple(flows.shape),
    )
    return images, flows


def _parse_train_model_dir(train_model_dir: Path) -> dict:
    """
    从 train_model 目录反推实验、class、任务名、data_type 和 scale。

    期望路径：
      {root}/{experiment}/{class}/{run_class}/{data_type}/{scale_dir}/train_model
    """

    rel = train_model_dir.relative_to(TRAIN_DATAS_ROOT)
    parts = rel.parts
    if len(parts) < 6:
        raise ValueError(f"[experiment_test] train_model 路径层级不足：{train_model_dir}")
    experiment_dir_name, class_name, run_class_name, data_type, scale_dir_name = parts[:5]
    if parts[-1] != "train_model":
        raise ValueError(f"[experiment_test] 不是 train_model 目录：{train_model_dir}")
    if not scale_dir_name.startswith("scale_"):
        raise ValueError(f"[experiment_test] 无法识别倍率目录：{train_model_dir}")
    # scale_4 表示 SCALE=2；scale_8 表示 SCALE=sqrt(8)，和训练 pipeline 的 scale_{SCALE*SCALE} 一致。
    # 如果平方根正好是整数，必须保留成 int；部分模型构造里会把 sr_scale 传给
    # range/PixelShuffle 等只接受整数的逻辑，传 2.0 会触发 "'float' object cannot be interpreted as an integer"。
    scale_square = float(scale_dir_name.split("_", 1)[1])
    scale_root = math.sqrt(scale_square)
    scale_value = int(scale_root) if float(scale_root).is_integer() else scale_root
    return {
        "experiment_dir_name": experiment_dir_name,
        "class_name": class_name,
        "run_class_name": run_class_name,
        "data_type": data_type,
        "scale_dir_name": scale_dir_name,
        "scale": scale_value,
    }


def _checkpoint_for_train_model_dir(train_model_dir: Path) -> Path | None:
    """选择 train_model 中最新的 *_model_*.pth；传统 baseline 允许返回 None。"""

    matches = sorted(
        train_model_dir.glob("*_model_*.pth"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _ground_train_mode_for_experiment(experiment_dir_name: str) -> str | None:
    """
    从 ESRuRAFT_PIV_Groundv_xxx 目录名中取出 TRAIN_MODE。

    传统方法目录也是这个命名体系，例如 ESRuRAFT_PIV_Groundv_bicubic_hs；
    后面会把取出的 bicubic_hs 写回 global_data.esrgan.TRAIN_MODE，让模型 forward
    自动走 Ground 分支里已经实现好的传统 HS/WIDIM 代码。
    """

    prefix = "ESRuRAFT_PIV_Groundv_"
    if not experiment_dir_name.startswith(prefix):
        return None
    return experiment_dir_name.replace(prefix, "", 1)


def _is_traditional_ground_experiment(experiment_dir_name: str) -> bool:
    """判断当前实验目录是否属于无 checkpoint 的 Ground 传统 PIV baseline。"""

    return _ground_train_mode_for_experiment(experiment_dir_name) in TRADITIONAL_GROUND_TRAIN_MODES


def _checkpoint_label(job: ModelJob) -> str:
    """统一把 checkpoint 路径写成可读文本；传统 baseline 用固定标签避免误解为漏文件。"""

    return str(job.checkpoint_path) if job.checkpoint_path is not None else TRADITIONAL_CHECKPOINT_LABEL


def _branch_name_for_experiment(experiment_dir_name: str) -> str:
    """根据实验目录名识别应该使用哪个模型分支构造 checkpoint 对应网络。"""

    if experiment_dir_name.startswith("ESRuRAFT_PIV_Groundv_"):
        return "ESRuRAFT_PIV_Ground"
    if experiment_dir_name.startswith("PIV_A_Esrgan"):
        return "PIV_A_Esrgan"
    raise ValueError(f"[experiment_test] 暂不支持的实验目录：{experiment_dir_name}")


def _discover_model_jobs(root: Path = TRAIN_DATAS_ROOT) -> tuple[list[ModelJob], list[dict]]:
    """
    扫描所有 train_model 目录，生成需要测试的模型任务。

    一般训练实验必须有 checkpoint；bicubic_hs/bicubic_widim 是传统 baseline，
    没有 checkpoint 也要作为任务执行，并在 _load_model_for_job 中跳过权重加载。
    """

    jobs: list[ModelJob] = []
    skipped: list[dict] = []
    for train_model_dir in sorted(root.rglob("train_model")):
        meta = _parse_train_model_dir(train_model_dir)
        checkpoint_path = _checkpoint_for_train_model_dir(train_model_dir)
        branch_name = _branch_name_for_experiment(meta["experiment_dir_name"])
        is_traditional_baseline = _is_traditional_ground_experiment(meta["experiment_dir_name"])
        if checkpoint_path is None and not is_traditional_baseline:
            skipped.append({**meta, "train_model_dir": str(train_model_dir), "reason": "no *_model_*.pth"})
            continue
        output_root = train_model_dir.parent / "experiment"
        jobs.append(
            ModelJob(
                experiment_dir_name=meta["experiment_dir_name"],
                class_name=meta["class_name"],
                run_class_name=meta["run_class_name"],
                data_type=meta["data_type"],
                scale_dir_name=meta["scale_dir_name"],
                scale=meta["scale"],
                train_model_dir=train_model_dir,
                checkpoint_path=checkpoint_path,
                output_root=output_root,
                branch_name=branch_name,
                requires_checkpoint=not is_traditional_baseline,
            )
        )
    return jobs, skipped


def _runtime_for_job(job: ModelJob) -> BranchRuntime:
    """
    创建当前 job 的模型构造环境。

    注意：模型模块内部会读取各自分支的 global_data，因此必须在实例化模型前把 name、
    OUT_PUT_DIR、DATA_SET、SCALES、TRAIN_MODE 等关键字段同步为 checkpoint 所在目录的值。
    """

    if job.branch_name == "ESRuRAFT_PIV_Ground":
        from study.SRGAN.model.ESRuRAFT_PIV_Ground.Module.PIV_ESRGAN_RAFT_Model import ESRuRAFT_PIV
        from study.SRGAN.model.ESRuRAFT_PIV_Ground.global_class import global_data

        train_mode = _ground_train_mode_for_experiment(job.experiment_dir_name)
        if train_mode is None:
            raise ValueError(f"[experiment_test] 无法从 Ground 实验目录解析 TRAIN_MODE：{job.experiment_dir_name}")
        global_data.esrgan.TRAIN_MODE = train_mode
        global_data.esrgan.name = job.experiment_dir_name
        global_data.esrgan.DATA_SET = job.class_name
        global_data.esrgan.OUT_PUT_DIR = str(TRAIN_DATAS_ROOT / job.experiment_dir_name / job.class_name)
        global_data.esrgan.SCALES = [job.scale]

        def factory(scale: float):
            return ESRuRAFT_PIV(
                inner_chanel=3,
                batch_size=global_data.esrgan.BATCH_SIZE,
                sr_scale=scale,
            )

        return BranchRuntime(global_data=global_data, model_factory=factory)

    if job.branch_name == "PIV_A_Esrgan":
        from study.SRGAN.model.PIV_A_Esrgan.Module.PIV_ESRGAN_RAFT_Model import ESRuRAFT_PIV
        from study.SRGAN.model.PIV_A_Esrgan.global_class import global_data

        global_data.esrgan.name = job.experiment_dir_name
        global_data.esrgan.DATA_SET = job.class_name
        global_data.esrgan.OUT_PUT_DIR = str(TRAIN_DATAS_ROOT / job.experiment_dir_name / job.class_name)
        global_data.esrgan.SCALES = [job.scale]
        global_data.esrgan.USE_RAFT = True

        def factory(_scale: float):
            return ESRuRAFT_PIV(
                inner_chanel=3,
                batch_size=global_data.esrgan.BATCH_SIZE,
                scale=job.scale,
            )

        return BranchRuntime(global_data=global_data, model_factory=factory)

    raise ValueError(f"[experiment_test] 暂不支持的模型分支：{job.branch_name}")


def _test_args_from_global(global_data) -> dict:
    """
    复用 run_test_all 的可视化/滑窗参数名。

    这样实验三阶段输出的 sample 图、flow 图、色条和 test_all 保持同一套参数。
    """

    cfg = global_data.esrgan
    return {
        "offset": int(getattr(cfg, "TEST_OFFSET", 256)),
        "shift": int(getattr(cfg, "TEST_SHIFT", 64)),
        "split_size": int(getattr(cfg, "TEST_SPLIT_SIZE", 1)),
        "amp": bool(getattr(cfg, "TEST_AMP", False)),
        "plot_results": bool(getattr(cfg, "TEST_PLOT_RESULTS", True)),
        "displacement_cmap": getattr(cfg, "TEST_DISPLACEMENT_CMAP", "viridis"),
        "regular_flow_cmap": getattr(cfg, "TEST_REGULAR_FLOW_CMAP", "jet"),
        "method_label": getattr(cfg, "name", "Current method"),
        "save_npy": bool(getattr(cfg, "IS_SAVE_NPY", False)),
        "global_data": global_data,
        "particle_error_colorbar_limit": getattr(cfg, "PARTICLE_ERROR_COLORBAR_LIMIT", 1.0),
        "flow_error_colorbar_limit": getattr(cfg, "FLOW_ERROR_COLORBAR_LIMIT", 0.5),
        "vorticity_quiver_stride": getattr(cfg, "VORTICITY_QUIVER_STRIDE", None),
        "tbl_profile_column_ratios": getattr(cfg, "TBL_PROFILE_COLUMN_RATIOS", (0.15, 0.265, 0.83)),
        "tbl_profile_region_names": getattr(cfg, "TBL_PROFILE_REGION_NAMES", ("Laminar", "Transition", "Turbulent")),
        "tbl_profile_y_limit": getattr(cfg, "TBL_PROFILE_Y_LIMIT", 200),
        "tbl_profile_sample_crop_width": getattr(cfg, "TBL_PROFILE_SAMPLE_CROP_WIDTH", 256),
    }


def _scale_factor_for_job(job: ModelJob) -> int:
    """
    把目录里的 SCALE 还原成 data_downscal.py 使用的下采样倍率。

    训练目录命名是 scale_{SCALE*SCALE}：例如 scale_4 表示 SCALE=2，
    真实 LR/HR 尺寸倍率是 4。这里用 round 而不是直接 int，避免 sqrt(8)
    这类浮点数出现 7.999999 被截断成 7。
    """

    return max(1, int(round(float(job.scale) * float(job.scale))))


def _apply_experiment_patch_geometry(test_args: dict, job: ModelJob) -> dict:
    """
    为实验图片强制设置 64x64 LR 输入 patch，并沿用 TWCF 的 pad/unfold/fold 机制。

    实验图没有 HR 原图，offset/shift 必须是 LR 坐标，不能再乘下采样倍率。
    边缘不足 64 的区域由公共 padding 逻辑补齐；模型预测出的 SR patch 会在
    _predict_experiment_lr_full_frame_with_folding 中按 HR 坐标 fold，再裁回 LR 原图
    放大后的完整 SR 尺寸。
    """

    patched_args = dict(test_args)
    patched_args["offset"] = int(EXPERIMENT_MODEL_INPUT_SIZE)
    patched_args["shift"] = int(EXPERIMENT_MODEL_PATCH_SHIFT)
    patched_args["particle_error_colorbar_limit"] = float(EXPERIMENT_PARTICLE_ERROR_COLORBAR_LIMIT)
    # 实验数据没有真实 HR，comparison.png 里的 HR 是 LR 插值得到的参考图。
    # 因此颗粒误差面板只展示 SR-HR 误差本身，不再叠加 ESMSE 文本，避免把伪参考指标误读为真实精度。
    patched_args["particle_error_show_metric"] = False
    # 只有 scale_8 的实验大图会在 image_outputs 的全幅 FFT/统计/matplotlib 绘图阶段触发 OOM；
    # 其他倍率仍保留完整 test_all 图像输出，避免为了 scale_8 的内存问题改变正常倍率的产物结构。
    patched_args["experiment_low_memory_image_outputs"] = str(job.scale_dir_name) == "scale_8"
    patched_args["experiment_low_memory_particle_zoom"] = str(job.scale_dir_name) == "scale_8"
    return patched_args


def _upsample_lr_image_for_reference(image_lr: torch.Tensor, factor: int) -> torch.Tensor:
    """
    将实验 LR 图像放大到模型输出尺寸，作为没有 HR 原图时的图像参考。

    这个参考只用于保持 metrics_image_pair.csv、sample 图和能谱图的 test_all 输出结构；
    它不是实验真实 HR ground truth。
    """

    if factor <= 1:
        return image_lr.detach().clone()
    h, w = image_lr.shape[-2:]
    return F.interpolate(image_lr, size=(h * factor, w * factor), mode="bicubic", align_corners=False)


def _upsample_lr_flow_for_reference(flow_lr: torch.Tensor, factor: int) -> torch.Tensor:
    """
    将低分辨率 OpenPIV flo 参考流放大到 SR 输出尺寸。

    flow 的 u/v 是“像素位移”。当空间坐标从 LR 放大到 SR 时，同一个物理位移对应的
    像素数也要乘以 factor；否则会把 SR 坐标下的预测流和 LR 坐标下的参考流直接相减。
    """

    if factor <= 1:
        return flow_lr.detach().clone()
    h, w = flow_lr.shape[-2:]
    flow_hr = F.interpolate(flow_lr, size=(h * factor, w * factor), mode="bilinear", align_corners=True)
    return flow_hr * float(factor)


def _background_mask_from_hr_reference(hr_reference: torch.Tensor) -> torch.Tensor:
    """
    根据插值后的 HR 参考颗粒图生成 HR/SR 坐标背景 mask。

    用户指定实验没有真实 HR，因此 prev_hr/next_hr 是 LR 实验原图插值得到的参考图。
    这里就在这张 HR 参考图上区分前景/背景：HR 参考图为背景的位置，最终 SR 输出也强制为黑。
    这样 mask 的空间结构和 SR 输出同尺寸，不再使用 LR mask nearest 放大造成块状边缘。
    """

    mask_hr = (hr_reference > float(EXPERIMENT_BACKGROUND_MASK_THRESHOLD)).to(dtype=torch.float32)
    dilate = int(EXPERIMENT_BACKGROUND_MASK_DILATE)
    if dilate > 1:
        if dilate % 2 == 0:
            dilate += 1
        mask_hr = F.max_pool2d(mask_hr, kernel_size=dilate, stride=1, padding=dilate // 2)
    return mask_hr.detach().cpu()


def _predict_experiment_lr_patch(
    model: torch.nn.Module,
    images_lr: torch.Tensor,
    flows_lr: torch.Tensor,
    factor: int,
    device: torch.device,
):
    """
    对一批 64x64 实验 LR patch 做模型推理。

    与 tfrecord_test_common._predict_patch 的关键区别：
    - 那个函数假设输入 images 是 HR patch，并会先下采样得到 LR；
    - 这里用户明确说明实验图就是 LR，因此 previous/next 直接作为 input_lr_*。

    模型 forward 仍需要 input_gr_* 和 flowl0 来计算内部损失/日志。由于实验没有 HR 原图，
    这里用 bicubic 放大的 LR 图像作为伪 HR 参考，用插值并缩放位移后的 OpenPIV flo 作为
    SR 坐标参考流；这些参考不会改变模型输入，只是让现有 forward 和 test_all 输出结构可复用。
    """

    prev_lr = images_lr[:, 0:1, :, :]
    next_lr = images_lr[:, 1:2, :, :]
    prev_hr_ref = _upsample_lr_image_for_reference(prev_lr, factor)
    next_hr_ref = _upsample_lr_image_for_reference(next_lr, factor)
    flow_hr_ref = _upsample_lr_flow_for_reference(flows_lr, factor)

    expected_channels = _infer_model_image_channels(model)
    prev_lr_for_model = _adapt_image_channels_for_model(prev_lr, expected_channels)
    next_lr_for_model = _adapt_image_channels_for_model(next_lr, expected_channels)
    prev_hr_for_model = _adapt_image_channels_for_model(prev_hr_ref, expected_channels)
    next_hr_for_model = _adapt_image_channels_for_model(next_hr_ref, expected_channels)

    pred_prev, pred_next, flow_predictions, _ = model(
        input_lr_prev=prev_lr_for_model,
        input_lr_next=next_lr_for_model,
        input_gr_prev=prev_hr_for_model,
        input_gr_next=next_hr_for_model,
        flowl0=flow_hr_ref,
        flow_init=None,
        is_adversarial=False,
    )
    return {
        "flow": _last_flow_prediction(flow_predictions),
        "flow_reference": flow_hr_ref,
        "prev_lr": prev_lr,
        "next_lr": next_lr,
        "prev_hr": prev_hr_ref,
        "next_hr": next_hr_ref,
        "pred_prev": _collapse_image_to_single_channel_for_test(pred_prev),
        "pred_next": _collapse_image_to_single_channel_for_test(pred_next),
    }


def _predict_experiment_lr_full_frame_with_folding(
    model: torch.nn.Module,
    images_lr: torch.Tensor,
    flows_lr: torch.Tensor,
    factor: int,
    device: torch.device,
    test_args: dict,
    progress_context: dict | None = None,
) -> dict:
    """
    将任意尺寸实验 LR 图切成 64x64 patch，预测后合并回完整 SR 尺寸。

    padding/unfold/fold 的骨架沿用 TWCF 大图逻辑，但坐标系改成 LR：
    1. LR 实验图 pad 到 64 的整数覆盖；
    2. unfold 得到多个 64x64 LR patch，直接送入模型；
    3. 每个 patch 输出 factor 倍大小的 SR 图和 SR 坐标 flow；
    4. 按 HR/SR 坐标 fold 回去，并裁到原始 LR 图放大后的尺寸。
    """

    start_time = time.perf_counter()
    offset_lr = int(test_args["offset"])
    shift_lr = int(test_args["shift"])
    split_size = int(test_args["split_size"])

    B, C, original_h_lr, original_w_lr = images_lr.size()
    padded_h_lr = _sliding_full_coverage_size(original_h_lr, offset_lr, shift_lr)
    padded_w_lr = _sliding_full_coverage_size(original_w_lr, offset_lr, shift_lr)
    images_lr_original = images_lr

    images_lr = _pad_full_frame_for_sliding(images_lr, padded_h_lr, padded_w_lr)
    flows_lr = _pad_full_frame_for_sliding(flows_lr, padded_h_lr, padded_w_lr)
    _, _, H_lr, W_lr = images_lr.size()
    num_y = (H_lr - offset_lr) // shift_lr + 1
    num_x = (W_lr - offset_lr) // shift_lr + 1
    total_patches = int(B * num_y * num_x)
    logger.info(
        "[experiment_test] LR folding setup: samples={} original_lr={}x{} padded_lr={}x{} grid={}x{} patches={} split_size={}",
        B,
        original_h_lr,
        original_w_lr,
        H_lr,
        W_lr,
        num_y,
        num_x,
        total_patches,
        split_size,
    )
    _write_progress_status(
        stage="patch_setup",
        **(progress_context or {}),
        original_lr_shape=[int(original_h_lr), int(original_w_lr)],
        padded_lr_shape=[int(H_lr), int(W_lr)],
        patch_grid=[int(num_y), int(num_x)],
        total_patches=total_patches,
        split_size=split_size,
    )

    image_patches = images_lr.unfold(3, offset_lr, shift_lr).unfold(2, offset_lr, shift_lr).permute(0, 2, 3, 1, 5, 4)
    image_patches = image_patches.reshape((-1, C, offset_lr, offset_lr))
    flow_patches = flows_lr.unfold(3, offset_lr, shift_lr).unfold(2, offset_lr, shift_lr).permute(0, 2, 3, 1, 5, 4)
    flow_patches = flow_patches.reshape((-1, 2, offset_lr, offset_lr))

    offset_hr = offset_lr * factor
    original_h_hr = original_h_lr * factor
    original_w_hr = original_w_lr * factor
    # 不再把所有 patch 输出留在 GPU list 中最后一次性 cat/fold。
    # 这里恢复为原始 64x64 无重叠铺块：每个 patch 推理完成后直接写回最终大图对应位置。
    predicted_flow_cpu = torch.empty((B, 2, original_h_hr, original_w_hr), dtype=torch.float32, device="cpu")
    flow_reference_cpu = torch.empty((B, 2, original_h_hr, original_w_hr), dtype=torch.float32, device="cpu")
    pred_prev_cpu = torch.empty((B, 1, original_h_hr, original_w_hr), dtype=torch.float32, device="cpu")
    pred_next_cpu = torch.empty((B, 1, original_h_hr, original_w_hr), dtype=torch.float32, device="cpu")

    image_splits = torch.split(image_patches, split_size, dim=0)
    flow_splits = torch.split(flow_patches, split_size, dim=0)
    total_batches = len(image_splits)
    log_every = _progress_log_interval(total_batches)
    patches_per_sample = num_y * num_x
    for batch_idx, (image_patch, flow_patch) in enumerate(zip(image_splits, flow_splits), start=1):
        if batch_idx == 1 or batch_idx == total_batches or batch_idx % log_every == 0:
            elapsed = time.perf_counter() - start_time
            patches_done = min(batch_idx * split_size, total_patches)
            logger.info(
                "[experiment_test] patch inference progress: batch {}/{} patches_done={}/{} current_batch={} elapsed={} eta={}",
                batch_idx,
                total_batches,
                patches_done,
                total_patches,
                image_patch.shape[0],
                _format_duration(elapsed),
                _estimate_eta(elapsed, batch_idx, total_batches),
            )
            _write_progress_status(
                stage="patch_inference",
                **(progress_context or {}),
                batch_index=batch_idx,
                total_batches=total_batches,
                patches_done=patches_done,
                total_patches=total_patches,
                elapsed=_format_duration(elapsed),
                eta=_estimate_eta(elapsed, batch_idx, total_batches),
            )
        patch_result = _predict_experiment_lr_patch(model, image_patch, flow_patch, factor, device)
        batch_start_flat = (batch_idx - 1) * split_size
        for local_idx in range(int(image_patch.shape[0])):
            flat_idx = batch_start_flat + local_idx
            sample_idx = flat_idx // patches_per_sample
            sample_patch_idx = flat_idx % patches_per_sample
            patch_y = sample_patch_idx // num_x
            patch_x = sample_patch_idx % num_x
            y0 = int(patch_y * offset_hr)
            x0 = int(patch_x * offset_hr)
            y1 = min(y0 + int(patch_result["flow"].shape[-2]), original_h_hr)
            x1 = min(x0 + int(patch_result["flow"].shape[-1]), original_w_hr)
            if y0 >= original_h_hr or x0 >= original_w_hr:
                continue
            crop_h = y1 - y0
            crop_w = x1 - x0
            predicted_flow_cpu[sample_idx, :, y0:y1, x0:x1] = (
                patch_result["flow"][local_idx, :, :crop_h, :crop_w].detach().cpu()
            )
            flow_reference_cpu[sample_idx, :, y0:y1, x0:x1] = (
                patch_result["flow_reference"][local_idx, :, :crop_h, :crop_w].detach().cpu()
            )
            pred_prev_cpu[sample_idx, :, y0:y1, x0:x1] = (
                patch_result["pred_prev"][local_idx, :, :crop_h, :crop_w].detach().cpu()
            )
            pred_next_cpu[sample_idx, :, y0:y1, x0:x1] = (
                patch_result["pred_next"][local_idx, :, :crop_h, :crop_w].detach().cpu()
            )
        del patch_result
    prev_hr_reference_cpu = _upsample_lr_image_for_reference(images_lr_original[:, 0:1, :, :], factor).detach().cpu()
    next_hr_reference_cpu = _upsample_lr_image_for_reference(images_lr_original[:, 1:2, :, :], factor).detach().cpu()
    prev_mask_hr = _background_mask_from_hr_reference(prev_hr_reference_cpu)
    next_mask_hr = _background_mask_from_hr_reference(next_hr_reference_cpu)
    flow_foreground_mask_hr = torch.clamp(prev_mask_hr + next_mask_hr, max=1.0)
    pred_prev_cpu = pred_prev_cpu * prev_mask_hr
    pred_next_cpu = pred_next_cpu * next_mask_hr
    # flow 是模型 forward 中已经基于未 mask 的 SR 图算出的；为了让最终实验输出和
    # “HR 背景区域全黑”的颗粒图一致，背景区域的预测/参考 flow 都压成 0。
    # 使用 prev/next 的并集作为前景，避免运动后某一帧刚好变暗的真实颗粒被误删。
    predicted_flow_cpu = predicted_flow_cpu * flow_foreground_mask_hr
    flow_reference_cpu = flow_reference_cpu * flow_foreground_mask_hr
    logger.info(
        "[experiment_test] 64x64 patch stitching finished: runtime={} sr_size={}x{} flow_shape={} mask_threshold={:.6f}",
        _format_duration(time.perf_counter() - start_time),
        original_h_hr,
        original_w_hr,
        tuple(predicted_flow_cpu.shape),
        EXPERIMENT_BACKGROUND_MASK_THRESHOLD,
    )

    return {
        "flow": predicted_flow_cpu,
        "flow_reference": flow_reference_cpu,
        "prev_lr": images_lr_original[:, 0:1, :, :],
        "next_lr": images_lr_original[:, 1:2, :, :],
        "prev_hr": prev_hr_reference_cpu,
        "next_hr": next_hr_reference_cpu,
        "pred_prev": pred_prev_cpu,
        "pred_next": pred_next_cpu,
    }


def _load_model_for_job(job: ModelJob, device: torch.device):
    """
    构造模型并按任务类型加载权重。

    神经网络实验继续严格加载 checkpoint，避免结构错配被静默吞掉；
    bicubic_hs/bicubic_widim 属于 Ground 分支里的传统无参数 baseline，只需要模型外壳
    提供 forward 入口和统一输出格式，所以这里显式跳过 torch.load。
    """

    runtime = _runtime_for_job(job)
    model = runtime.model_factory(job.scale).to(device, non_blocking=(device.type == "cuda"))
    if not job.requires_checkpoint:
        logger.info(
            "[experiment_test] skip checkpoint load for traditional baseline: {} ({})",
            job.experiment_dir_name,
            _checkpoint_label(job),
        )
        model.eval()
        return model, runtime.global_data
    if job.checkpoint_path is None:
        raise FileNotFoundError(f"[experiment_test] 缺少 checkpoint：{job.train_model_dir}")
    checkpoint_obj = torch.load(job.checkpoint_path, map_location=device)
    state_dict = _extract_model_state_dict(checkpoint_obj)
    state_dict = _normalize_state_dict_keys_for_model(state_dict, model)
    _load_state_dict_checked(model, state_dict, job.checkpoint_path)
    model.eval()
    return model, runtime.global_data


def _annotate_row(row: dict, sample: ExperimentSample) -> dict:
    """给 test_all 同款指标行补充实验三阶段来源，便于回查原始图片。"""

    row = dict(row)
    row.update(
        {
            "time_stage": sample.stage,
            "group_name": sample.group_name,
            "previous_image": sample.previous_path.name,
            "next_image": sample.next_path.name,
            "flow_file": sample.flow_path.name,
        }
    )
    return row


def _save_experiment_metadata(dataset_dir: Path, samples: list[ExperimentSample], job: ModelJob) -> None:
    """
    写入样本索引到真实 exp_XXXX 名称的映射。

    all_handle 依赖 sample_0000 这类统一目录名做横向合并；metadata 负责保留
    sample_0000 = start_1/exp_0042 这种实验语义。
    """

    rows = []
    for idx, sample in enumerate(samples):
        rows.append(
            {
                "sample_index": idx,
                "sample_dir": f"sample_{idx:04d}",
                "time_stage": sample.stage,
                "group_name": sample.group_name,
                "previous_image": sample.previous_path.name,
                "next_image": sample.next_path.name,
                "flow_file": sample.flow_path.name,
            }
        )
    _write_csv(dataset_dir / "experiment_samples.csv", rows)
    metadata = {
        "experiment_dir_name": job.experiment_dir_name,
        "class_name": job.class_name,
        "run_class_name": job.run_class_name,
        "data_type": job.data_type,
        "scale_dir_name": job.scale_dir_name,
        "checkpoint_path": _checkpoint_label(job),
        "requires_checkpoint": job.requires_checkpoint,
        "samples": rows,
    }
    (dataset_dir / "experiment_samples.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _slice_prediction_payload_for_sample(prediction: dict, sample_index: int) -> dict:
    """
    为单个 sample plot 截取一份 batch=1 的 image_payload。

    _save_sample_plots 的内部仍按 test_all 的 batch 结构读取 pred_prev/pred_next/prev_hr 等字段；
    这里不改变任何图像尺寸和图像结构，只把 batch 维度从 3 个实验时刻缩成当前 1 个时刻，
    让 matplotlib 每次只处理一组大图，降低峰值内存。
    """

    sliced = {}
    for key, value in prediction.items():
        if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] > sample_index:
            sliced[key] = value[sample_index:sample_index + 1]
        elif isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] > sample_index:
            sliced[key] = value[sample_index:sample_index + 1]
        else:
            sliced[key] = value
    return sliced


def _finite_percentile_limits(*arrays: np.ndarray, fallback: float = 1.0) -> tuple[float, float]:
    """为单通道位移图计算稳定色标范围，避免少数极端值把整张图压暗。"""

    values = []
    for arr in arrays:
        flat = np.asarray(arr, dtype=np.float32).reshape(-1)
        flat = flat[np.isfinite(flat)]
        if flat.size:
            values.append(flat)
    if not values:
        return -float(fallback), float(fallback)
    merged = np.concatenate(values)
    vmin = float(np.percentile(merged, 1))
    vmax = float(np.percentile(merged, 99))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1.0e-6:
        center = float(np.mean(merged)) if merged.size else 0.0
        return center - float(fallback), center + float(fallback)
    return vmin, vmax


def _array_to_rgb_panel(arr: np.ndarray, vmin: float, vmax: float, cmap_name: str) -> Image.Image:
    """
    把二维数组转成 RGB 面板图。

    为了避免 regular overview 被 matplotlib 大画布杀进程，这里用轻量 numpy+PIL 做颜色映射。
    支持常用的 jet 位移图和 bwr 误差图；其它 cmap 名称回退到灰度。
    """

    field = np.asarray(arr, dtype=np.float32)
    normalized = (field - float(vmin)) / max(float(vmax) - float(vmin), 1.0e-6)
    normalized = np.clip(np.nan_to_num(normalized, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)

    if str(cmap_name).lower() == "jet":
        r = np.clip(1.5 - np.abs(4.0 * normalized - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * normalized - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * normalized - 1.0), 0.0, 1.0)
    elif str(cmap_name).lower() == "bwr":
        # 0=blue, 0.5=white, 1=red，适合 Pred-GT 误差。
        r = np.where(normalized < 0.5, 2.0 * normalized, 1.0)
        g = np.where(normalized < 0.5, 2.0 * normalized, 2.0 * (1.0 - normalized))
        b = np.where(normalized < 0.5, 1.0, 2.0 * (1.0 - normalized))
    else:
        r = g = b = normalized

    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB")


def _compose_three_panel_image(
    out_path: Path,
    pred: np.ndarray,
    reference: np.ndarray,
    error: np.ndarray,
    component_name: str,
    displacement_cmap: str,
    error_limit: float,
) -> None:
    """
    保存单个分量的轻量总览图：Pred / Reference / Error。

    这是原 regular overview 的低内存拆分版：
    - U 分量保存一张图；
    - V 分量保存一张图；
    - 不缩放、不裁剪、不改变原始场尺寸，只是不再把 U/V 同时塞进一个 matplotlib figure。
    """

    disp_vmin, disp_vmax = _finite_percentile_limits(pred, reference, fallback=1.0)
    err_limit = float(error_limit) if np.isfinite(error_limit) and float(error_limit) > 0 else 0.5
    panels = [
        (f"Pred {component_name}", _array_to_rgb_panel(pred, disp_vmin, disp_vmax, displacement_cmap)),
        (f"Reference {component_name}", _array_to_rgb_panel(reference, disp_vmin, disp_vmax, displacement_cmap)),
        (f"Error {component_name}", _array_to_rgb_panel(error, -err_limit, err_limit, "bwr")),
    ]

    panel_w, panel_h = panels[0][1].size
    title_h = 34
    gap = 8
    canvas = Image.new("RGB", (panel_w * 3 + gap * 2, panel_h + title_h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for idx, (title, panel) in enumerate(panels):
        x0 = idx * (panel_w + gap)
        draw.text((x0 + 8, 10), title, fill=(0, 0, 0), font=font)
        canvas.paste(panel, (x0, title_h))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _save_split_regular_overview(
    dataset_dir: Path,
    dataset_name: str,
    sample_index: int,
    predicted_sample_np: np.ndarray,
    flow_reference_sample_np: np.ndarray,
    regular_flow_cmap: str,
    flow_error_colorbar_limit: float,
) -> None:
    """把 regular overview 拆成 U/误差 和 V/误差 两张图，降低保存阶段内存峰值。"""

    u_pred = predicted_sample_np[0]
    v_pred = predicted_sample_np[1]
    u_ref = flow_reference_sample_np[0]
    v_ref = flow_reference_sample_np[1]
    _compose_three_panel_image(
        dataset_dir / f"{dataset_name}_sample_{sample_index:04d}_u.png",
        u_pred,
        u_ref,
        u_pred - u_ref,
        "U",
        regular_flow_cmap,
        flow_error_colorbar_limit,
    )
    gc.collect()
    _compose_three_panel_image(
        dataset_dir / f"{dataset_name}_sample_{sample_index:04d}_v.png",
        v_pred,
        v_ref,
        v_pred - v_ref,
        "V",
        regular_flow_cmap,
        flow_error_colorbar_limit,
    )


def _tensor_or_array_to_2d_unit(value) -> np.ndarray | None:
    """把 image_payload 中的 batch=1 图像转成 0-1 的二维 numpy。"""

    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        return None
    return np.clip(arr.astype(np.float32, copy=False), 0.0, 1.0)


def _region_to_bounds(region: tuple[float, float, float, float], height: int, width: int) -> tuple[int, int, int, int]:
    """把相对区域转换成 y0/y1/x0/x1，并裁到图像范围内。"""

    x_ratio, y_ratio, w_ratio, h_ratio = region
    crop_w = max(8, int(round(width * float(w_ratio))))
    crop_h = max(8, int(round(height * float(h_ratio))))
    cx = int(round(width * float(x_ratio)))
    cy = int(round(height * float(y_ratio)))
    x0 = max(0, min(width - crop_w, cx - crop_w // 2))
    y0 = max(0, min(height - crop_h, cy - crop_h // 2))
    return y0, y0 + crop_h, x0, x0 + crop_w


def _scale_bounds(bounds: tuple[int, int, int, int], source_hw: tuple[int, int], target_hw: tuple[int, int]) -> tuple[int, int, int, int]:
    """把 HR/SR 坐标框缩放到 LR 坐标，用于裁 LR 局部图。"""

    y0, y1, x0, x1 = bounds
    src_h, src_w = source_hw
    dst_h, dst_w = target_hw
    return (
        max(0, min(dst_h - 1, int(round(y0 * dst_h / src_h)))),
        max(1, min(dst_h, int(round(y1 * dst_h / src_h)))),
        max(0, min(dst_w - 1, int(round(x0 * dst_w / src_w)))),
        max(1, min(dst_w, int(round(x1 * dst_w / src_w)))),
    )


def _draw_rectangles_on_gray(image: np.ndarray, regions: list[tuple[int, int, int, int]]) -> np.ndarray:
    """在灰度整图上画红框，返回 RGB 数组。"""

    gray = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    for y0, y1, x0, x1 in regions:
        rgb[y0:y0 + 3, x0:x1, :] = (255, 0, 0)
        rgb[max(y1 - 3, y0):y1, x0:x1, :] = (255, 0, 0)
        rgb[y0:y1, x0:x0 + 3, :] = (255, 0, 0)
        rgb[y0:y1, max(x1 - 3, x0):x1, :] = (255, 0, 0)
    return rgb


def _save_single_experiment_particle_zoom(
    sample_image_dir: Path,
    time_name: str,
    lr_img: np.ndarray,
    hr_img: np.ndarray,
    sr_img: np.ndarray,
    low_memory: bool = False,
) -> None:
    """
    保存单模型实验颗粒局部放大对比图。

    顶部是带红框的整张 HR/SR 参考，下面每行只放 HR/SR 两个局部放大块。
    这里不再放 LR 和 SR-HR 误差：实验 LR 与 HR/SR 尺寸不一致，误差又基于插值伪 HR，
    局部图的目标是直接看超分颗粒细节，所以保留最直观的 HR/SR 对照。
    """

    if hr_img is None or sr_img is None:
        return

    h, w = hr_img.shape
    regions = [_region_to_bounds(region, h, w) for region in EXPERIMENT_PARTICLE_ZOOM_REGIONS]
    if not low_memory:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        n_regions = len(regions)
        fig = plt.figure(figsize=(11, 3.0 + 3.1 * n_regions), dpi=150, facecolor="w")
        gs = fig.add_gridspec(n_regions + 1, 2, height_ratios=[1.0] + [1.25] * n_regions, hspace=0.16, wspace=0.04)

        for col, (title, arr) in enumerate((("HR with regions", hr_img), ("SR with regions", sr_img))):
            ax = fig.add_subplot(gs[0, col])
            ax.imshow(arr, cmap="gray", vmin=0.0, vmax=1.0)
            for idx, (y0, y1, x0, x1) in enumerate(regions, start=1):
                ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="red", linewidth=1.2))
                ax.text(x0, y0, str(idx), color="white", fontsize=8, bbox={"facecolor": "red", "edgecolor": "none", "pad": 1})
            ax.set_title(title)
            ax.axis("off")

        for row, bounds in enumerate(regions, start=1):
            y0, y1, x0, x1 = bounds
            panels = (
                (f"Region {row} HR", hr_img[y0:y1, x0:x1], "gray", 0.0, 1.0),
                (f"Region {row} SR", sr_img[y0:y1, x0:x1], "gray", 0.0, 1.0),
            )
            for col, (title, arr, cmap, vmin, vmax) in enumerate(panels):
                ax = fig.add_subplot(gs[row, col])
                ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title(title)
                ax.axis("off")

        sample_image_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(sample_image_dir / f"{time_name}_particle_zoom_comparison.png", bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
        return

    sample_image_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image, ImageDraw

    def to_gray_image(arr: np.ndarray) -> Image.Image:
        """用 PIL 直接构造灰度图，避免 matplotlib 为超大实验图生成 RGBA 画布。"""

        gray = np.clip(np.squeeze(arr).astype(np.float32, copy=False), 0.0, 1.0)
        return Image.fromarray((gray * 255.0).astype(np.uint8), mode="L")

    def add_label(image: Image.Image, text: str) -> Image.Image:
        """给拼图小块加灰底文字，保持和原有可视化的标签风格接近。"""

        image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
        bbox = draw.textbbox((0, 0), text)
        draw.rectangle((0, 0, bbox[2] + 8, bbox[3] + 6), fill=(220, 220, 220))
        draw.text((4, 3), text, fill=(0, 0, 0))
        return image

    def overview_with_regions(arr: np.ndarray, title: str, max_width: int = 720) -> Image.Image:
        """先用 numpy 步进抽样生成小预览，再画红框，避免整张 scale_8 大图进入 PIL resize。"""

        source = np.squeeze(arr)
        step = max(1, int(math.ceil(source.shape[1] / max_width)))
        preview = to_gray_image(source[::step, ::step]).convert("RGB")
        scale_x = preview.width / max(1, source.shape[1])
        scale_y = preview.height / max(1, source.shape[0])
        draw = ImageDraw.Draw(preview)
        for idx, (y0, y1, x0, x1) in enumerate(regions, start=1):
            rect = (
                int(round(x0 * scale_x)),
                int(round(y0 * scale_y)),
                int(round(x1 * scale_x)),
                int(round(y1 * scale_y)),
            )
            draw.rectangle(rect, outline=(255, 0, 0), width=2)
            draw.rectangle((rect[0], rect[1], rect[0] + 16, rect[1] + 14), fill=(255, 0, 0))
            draw.text((rect[0] + 3, rect[1] + 1), str(idx), fill=(255, 255, 255))
        return add_label(preview, title)

    def crop_region(arr: np.ndarray, bounds: tuple[int, int, int, int], title: str, max_width: int = 720) -> Image.Image:
        """裁出局部颗粒区域并放大到固定预览宽度，便于直接比较 HR/SR 颗粒形态。"""

        y0, y1, x0, x1 = bounds
        crop = to_gray_image(arr[y0:y1, x0:x1]).convert("RGB")
        scale = max_width / max(1, crop.width)
        crop = crop.resize(
            (max_width, max(1, int(round(crop.height * scale)))),
            Image.Resampling.NEAREST,
        )
        return add_label(crop, title)

    panels: list[tuple[Image.Image, Image.Image]] = [
        (overview_with_regions(hr_img, "HR with regions"), overview_with_regions(sr_img, "SR with regions"))
    ]
    for idx, bounds in enumerate(regions, start=1):
        panels.append(
            (
                crop_region(hr_img, bounds, f"Region {idx} HR"),
                crop_region(sr_img, bounds, f"Region {idx} SR"),
            )
        )

    margin, gap_x, gap_y = 20, 14, 12
    col_w = max(max(left.width, right.width) for left, right in panels)
    row_heights = [max(left.height, right.height) for left, right in panels]
    canvas = Image.new(
        "RGB",
        (margin * 2 + col_w * 2 + gap_x, margin * 2 + sum(row_heights) + gap_y * (len(panels) - 1)),
        "white",
    )
    y = margin
    for row_h, (left, right) in zip(row_heights, panels):
        canvas.paste(left, (margin, y))
        canvas.paste(right, (margin + col_w + gap_x, y))
        y += row_h + gap_y
    canvas.save(sample_image_dir / f"{time_name}_particle_zoom_comparison.png")


def _save_experiment_particle_zoom_comparisons(
    dataset_dir: Path,
    sample_index: int,
    image_payload_sample: dict,
    low_memory: bool = False,
) -> None:
    """为 previous/next 分别保存局部放大颗粒对比图。"""

    sample_image_dir = dataset_dir / "images" / f"sample_{sample_index:04d}"
    specs = (
        ("prev", "prev_lr", "prev_hr", "pred_prev"),
        ("next", "next_lr", "next_hr", "pred_next"),
    )
    for time_name, lr_key, hr_key, sr_key in specs:
        _save_single_experiment_particle_zoom(
            sample_image_dir,
            time_name,
            _tensor_or_array_to_2d_unit(image_payload_sample.get(lr_key)),
            _tensor_or_array_to_2d_unit(image_payload_sample.get(hr_key)),
            _tensor_or_array_to_2d_unit(image_payload_sample.get(sr_key)),
            low_memory=low_memory,
        )


def _save_experiment_sample_plot_artifacts_one_by_one(
    dataset_name: str,
    dataset_dir: Path,
    sample_index: int,
    predicted_sample_np: np.ndarray,
    flow_reference_sample_np: np.ndarray,
    image_payload_sample: dict,
    test_args: dict,
    job: ModelJob,
    sample: ExperimentSample,
) -> None:
    """
    按 artifact 粒度保存单个实验样本的图，避免 _save_sample_plots 的大包装调用不透明。

    输出路径和文件结构仍沿用 test_all：
    - image/sample_xxxx/... 由 _save_image_outputs 负责；
    - flow/sample_xxxx/... 由 _save_flow_visual_artifacts 负责；
    - experiment_sample_xxxx.png 由 _plot_regular 负责。

    这样不改变图片结构，只把“一个 sample 内的一堆图”拆成一步一步保存，并在每步后释放内存。
    如果再次卡住，日志会精确停在 image_outputs / flow_artifacts / regular_overview 哪一步。
    """

    plot_start = time.perf_counter()
    regular_flow_cmap = str(test_args.get("regular_flow_cmap", "jet"))
    flow_error_colorbar_limit = float(test_args.get("flow_error_colorbar_limit", 0.5))
    save_npy = bool(test_args.get("save_npy", False))

    logger.info("[experiment_test] sample plot step: image outputs -> sample_{:04d}", sample_index)
    _write_progress_status(
        stage="saving_sample_plot_image_outputs",
        experiment_dir_name=job.experiment_dir_name,
        class_name=job.class_name,
        run_class_name=job.run_class_name,
        scale_dir_name=job.scale_dir_name,
        sample_index=sample_index,
        sample_stage=sample.stage,
        group_name=sample.group_name,
        elapsed=_format_duration(time.perf_counter() - plot_start),
    )
    _save_image_outputs(dataset_name, dataset_dir, image_payload_sample, sample_index, plot_args=test_args)
    logger.info("[experiment_test] sample plot step: particle zoom comparisons -> sample_{:04d}", sample_index)
    _save_experiment_particle_zoom_comparisons(
        dataset_dir,
        sample_index,
        image_payload_sample,
        low_memory=bool(test_args.get("experiment_low_memory_particle_zoom", False)),
    )
    gc.collect()

    logger.info("[experiment_test] sample plot step: flow artifacts -> sample_{:04d}", sample_index)
    _write_progress_status(
        stage="saving_sample_plot_flow_artifacts",
        experiment_dir_name=job.experiment_dir_name,
        class_name=job.class_name,
        run_class_name=job.run_class_name,
        scale_dir_name=job.scale_dir_name,
        sample_index=sample_index,
        sample_stage=sample.stage,
        group_name=sample.group_name,
        elapsed=_format_duration(time.perf_counter() - plot_start),
    )
    _save_flow_visual_artifacts(
        dataset_dir / "flow" / f"sample_{sample_index:04d}",
        predicted_sample_np,
        flow_reference_sample_np,
        cmap_name=regular_flow_cmap,
        quiver_stride=test_args.get("vorticity_quiver_stride", None),
        save_npy=save_npy,
        dataset_name=dataset_name,
        mask_2d=None,
        tbl_y_limit=None,
        flow_error_colorbar_limit=flow_error_colorbar_limit,
        global_data=test_args.get("global_data", None),
    )
    gc.collect()

    logger.info("[experiment_test] sample plot step: split regular overview U/V -> sample_{:04d}", sample_index)
    _write_progress_status(
        stage="saving_sample_plot_split_regular_overview",
        experiment_dir_name=job.experiment_dir_name,
        class_name=job.class_name,
        run_class_name=job.run_class_name,
        scale_dir_name=job.scale_dir_name,
        sample_index=sample_index,
        sample_stage=sample.stage,
        group_name=sample.group_name,
        elapsed=_format_duration(time.perf_counter() - plot_start),
    )
    _save_split_regular_overview(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        sample_index=sample_index,
        predicted_sample_np=predicted_sample_np,
        flow_reference_sample_np=flow_reference_sample_np,
        regular_flow_cmap=regular_flow_cmap,
        flow_error_colorbar_limit=flow_error_colorbar_limit,
    )
    gc.collect()
    logger.info(
        "[experiment_test] sample plot finished: sample_{:04d} runtime={}",
        sample_index,
        _format_duration(time.perf_counter() - plot_start),
    )


def _run_single_job(job: ModelJob, samples: list[ExperimentSample], device: torch.device) -> dict:
    """
    对一个 checkpoint 跑 start/peak/end 各两组实验样本，并保存 test_all 同款产物。

    输出目录固定为：
      {experiment}/{class}/{run_class}/{data_type}/{scale}/experiment/experiment/
    第一层 experiment 是与 test_all 同层级的 split 名；第二层 experiment 是 dataset/category 名，
    这与 run_test_all 的 test_all/{dataset_name}/... 结构保持一致。
    """

    logger.info(
        "[experiment_test] start job: exp={} class={} run={} scale={} checkpoint={}",
        job.experiment_dir_name,
        job.class_name,
        job.run_class_name,
        job.scale_dir_name,
        _checkpoint_label(job),
    )
    job_start_time = time.perf_counter()
    logger.info("[experiment_test] loading runtime/model for {}", job.experiment_dir_name)
    model, branch_global_data = _load_model_for_job(job, device)
    logger.info("[experiment_test] model ready for {}", job.experiment_dir_name)
    scale_factor = _scale_factor_for_job(job)
    test_args = _apply_experiment_patch_geometry(_test_args_from_global(branch_global_data), job)
    logger.info(
        "[experiment_test] experiment LR patch geometry: input_patch={}x{} SR_patch={}x{} lr_shift={} scale_factor={}",
        EXPERIMENT_MODEL_INPUT_SIZE,
        EXPERIMENT_MODEL_INPUT_SIZE,
        EXPERIMENT_MODEL_INPUT_SIZE * scale_factor,
        EXPERIMENT_MODEL_INPUT_SIZE * scale_factor,
        test_args["shift"],
        scale_factor,
    )
    dataset_name = "experiment"
    dataset_dir = job.output_root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    logger.info("[experiment_test] output directory ready: {}", dataset_dir)

    images, flows = _samples_to_tensors(samples, device)
    logger.info("[experiment_test] start LR patch inference for {} sample groups", len(samples))
    with torch.no_grad():
        prediction = _retry_bad_request(
            lambda: _predict_experiment_lr_full_frame_with_folding(
                model,
                images,
                flows,
                scale_factor,
                device,
                test_args,
                progress_context={
                    "experiment_dir_name": job.experiment_dir_name,
                    "class_name": job.class_name,
                    "run_class_name": job.run_class_name,
                    "scale_dir_name": job.scale_dir_name,
                },
            ),
            context=f"{job.experiment_dir_name}/{job.class_name}/{job.scale_dir_name}",
        )
    logger.info("[experiment_test] inference finished, moving tensors to numpy for metrics")

    predicted_flows = prediction["flow"]
    predicted_np = predicted_flows.detach().cpu().numpy().astype(np.float32, copy=False)
    flow_np = prediction["flow_reference"].detach().cpu().numpy().astype(np.float32, copy=False)
    pred_prev_np = _as_numpy_batch(prediction["pred_prev"].clamp(0, 1))
    pred_next_np = _as_numpy_batch(prediction["pred_next"].clamp(0, 1))
    prev_hr_np = _as_numpy_batch(prediction["prev_hr"].clamp(0, 1))
    next_hr_np = _as_numpy_batch(prediction["next_hr"].clamp(0, 1))

    image_rows = []
    raft_rows = []
    image_pred_curves = []
    image_gt_curves = []
    flow_pred_curves = []
    flow_gt_curves = []
    epe_array = np.full((len(samples),), np.nan, dtype=np.float32)
    norm_aee_per100_array = np.full((len(samples),), np.nan, dtype=np.float32)
    results = np.zeros((len(samples), 4, predicted_np.shape[-2], predicted_np.shape[-1]), dtype=np.float32)
    results[:, 0:2] = predicted_np[:, 0:2]
    results[:, 2:4] = flow_np[:, 0:2]

    for local_idx, sample in enumerate(samples):
        logger.info(
            "[experiment_test] computing metrics {}/{}: {} ({})",
            local_idx + 1,
            len(samples),
            sample.stage,
            sample.group_name,
        )
        prev_pred_curve, prev_gt_curve = _energy_spectrum_curves(pred_prev_np[local_idx], prev_hr_np[local_idx])
        next_pred_curve, next_gt_curve = _energy_spectrum_curves(pred_next_np[local_idx], next_hr_np[local_idx])
        image_pred_curves.extend([prev_pred_curve, next_pred_curve])
        image_gt_curves.extend([prev_gt_curve, next_gt_curve])

        image_rows.append(
            _annotate_row(
                _compute_image_metric_row(
                    dataset_name,
                    local_idx,
                    "previous",
                    pred_prev_np[local_idx],
                    prev_hr_np[local_idx],
                ),
                sample,
            )
        )
        image_rows.append(
            _annotate_row(
                _compute_image_metric_row(
                    dataset_name,
                    local_idx,
                    "next",
                    pred_next_np[local_idx],
                    next_hr_np[local_idx],
                ),
                sample,
            )
        )

        flow_row = _compute_flow_metric_row(dataset_name, local_idx, predicted_np[local_idx], flow_np[local_idx])
        flow_pred_uvw = _flow_uv_to_uvw_np(predicted_np[local_idx])
        flow_gt_uvw = _flow_uv_to_uvw_np(flow_np[local_idx])
        flow_pred_curve, flow_gt_curve = _energy_spectrum_curves(flow_pred_uvw, flow_gt_uvw)
        flow_row["energy_spectrum_mse"] = _energy_spectrum_mse_from_curves(flow_pred_curve, flow_gt_curve)
        flow_row = _annotate_row(flow_row, sample)
        raft_rows.append(flow_row)
        flow_pred_curves.append(flow_pred_curve)
        flow_gt_curves.append(flow_gt_curve)

        epe_array[local_idx] = float(flow_row["epe"])
        norm_aee_per100_array[local_idx] = float(flow_row["NORM_AEE_PER100PIXEL"])

        # 预先计算误差图，既用于确认维度，也让保存出的 flow/sample_xxxx 目录和 test_all 语义一致。
        # 具体 PNG/NPY 写盘仍交给 _save_sample_plots 里的 _save_flow_visual_artifacts。
        _compute_flow_error_maps_np(predicted_np[local_idx], flow_np[local_idx])
        _omega_star_from_uv_np(predicted_np[local_idx, 0], predicted_np[local_idx, 1])

    attach_c_aee_to_raft_rows(
        image_rows,
        raft_rows,
        # c_aee_metric_common 通过 sample key 把两条 image_pair 行
        # (previous/next) 与同一个样本的一条 RAFT 行配对；这里的 dataset 固定为
        # experiment，sample_index 对应 EXPERIMENT_SAMPLES 中固定的六组实验样本顺序。
        ("dataset", "sample_index"),
        ese_key="energy_spectrum_mse",
        ssim_key="ssim",
        aee_key="epe",
        output_key="C_AEE",
    )

    image_mean_row = _write_rows_with_mean(
        dataset_dir / "metrics_image_pair.csv",
        image_rows,
        {"dataset": dataset_name, "sample_index": "MEAN", "pair_type": "IMAGE_PAIR"},
        ["mse", "psnr", "energy_spectrum_mse", "r2", "ssim", "tke_acc", "nrmse"],
        global_data=branch_global_data,
    )
    raft_mean_row = _write_rows_with_mean(
        dataset_dir / "metrics_raft.csv",
        raft_rows,
        {"dataset": dataset_name, "sample_index": "MEAN", "pair_type": "RAFT"},
        ["epe", "NORM_AEE_PER100PIXEL", "energy_spectrum_mse", "C_AEE"],
        global_data=branch_global_data,
    )
    _write_csv(dataset_dir / "metrics.csv", raft_rows + ([raft_mean_row] if raft_mean_row else []))

    summary_rows = [
        {
            "dataset": dataset_name,
            "class_name": job.class_name,
            "run_class_name": job.run_class_name,
            "data_type": job.data_type,
            "scale": job.scale,
            "num_samples": len(samples),
            "mean_epe": float(raft_mean_row["epe"]) if raft_mean_row else float("nan"),
            "mean_norm_aee_per100pixel": (
                float(raft_mean_row["NORM_AEE_PER100PIXEL"]) if raft_mean_row else float("nan")
            ),
            "mean_c_aee": float(raft_mean_row["C_AEE"]) if raft_mean_row else float("nan"),
            "image_mse_mean": float(image_mean_row["mse"]) if image_mean_row else float("nan"),
            "image_psnr_mean": float(image_mean_row["psnr"]) if image_mean_row else float("nan"),
            "image_energy_spectrum_mse_mean": (
                float(image_mean_row["energy_spectrum_mse"]) if image_mean_row else float("nan")
            ),
            "flow_energy_spectrum_mse_mean": (
                float(raft_mean_row["energy_spectrum_mse"]) if raft_mean_row else float("nan")
            ),
            "image_r2_mean": float(image_mean_row["r2"]) if image_mean_row else float("nan"),
            "image_ssim_mean": float(image_mean_row["ssim"]) if image_mean_row else float("nan"),
            "image_tke_acc_mean": float(image_mean_row["tke_acc"]) if image_mean_row else float("nan"),
            "image_nrmse_mean": float(image_mean_row["nrmse"]) if image_mean_row else float("nan"),
            "metrics_image_pair_csv": str(dataset_dir / "metrics_image_pair.csv"),
            "metrics_raft_csv": str(dataset_dir / "metrics_raft.csv"),
        }
    ]
    _write_csv(job.output_root / "metrics_all.csv", summary_rows)
    _write_csv(job.output_root / "ALL_CLASS_IMAGE_PAIR.CSV", [image_mean_row] if image_mean_row else [])
    _write_csv(job.output_root / "ALL_CLASS_flow.CSV", [raft_mean_row] if raft_mean_row else [])
    logger.info("[experiment_test] metric CSV files written under {}", job.output_root)

    np.save(dataset_dir / "results.npy", results)
    np.save(dataset_dir / "epe_array.npy", epe_array)
    np.save(dataset_dir / "norm_aee_per100_array.npy", norm_aee_per100_array)
    _save_mean_spectrum(
        image_pred_curves,
        image_gt_curves,
        dataset_dir,
        title="experiment Image Pair Mean Energy Spectrum",
        file_prefix="image_pair",
        save_npy=True,
        global_data=branch_global_data,
        also_save_legacy_names=True,
    )
    _save_mean_spectrum(
        flow_pred_curves,
        flow_gt_curves,
        dataset_dir,
        title="experiment Flow Mean Energy Spectrum",
        file_prefix="flow",
        save_npy=True,
        global_data=branch_global_data,
        also_save_legacy_names=False,
    )

    if bool(test_args["plot_results"]) and EXPERIMENT_SAVE_HEAVY_SAMPLE_PLOTS:
        logger.info("[experiment_test] saving sample plots one by one for {}", dataset_dir)
        for plot_idx, sample in enumerate(samples):
            logger.info(
                "[experiment_test] saving sample plot {}/{}: {} ({})",
                plot_idx + 1,
                len(samples),
                sample.stage,
                sample.group_name,
            )
            _write_progress_status(
                stage="saving_sample_plot",
                experiment_dir_name=job.experiment_dir_name,
                class_name=job.class_name,
                run_class_name=job.run_class_name,
                scale_dir_name=job.scale_dir_name,
                output_dir=str(dataset_dir),
                sample_index=plot_idx,
                sample_stage=sample.stage,
                group_name=sample.group_name,
            )
            _save_experiment_sample_plot_artifacts_one_by_one(
                dataset_name=dataset_name,
                dataset_dir=dataset_dir,
                sample_index=plot_idx,
                predicted_sample_np=predicted_np[plot_idx],
                flow_reference_sample_np=flow_np[plot_idx],
                image_payload_sample=_slice_prediction_payload_for_sample(prediction, plot_idx),
                test_args=test_args,
                job=job,
                sample=sample,
            )
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    elif bool(test_args["plot_results"]):
        logger.warning(
            "[experiment_test] skip heavy sample plots to avoid OS Killed: {}. "
            "Set EXPERIMENT_SAVE_HEAVY_SAMPLE_PLOTS=True only for small/manual runs.",
            dataset_dir,
        )
    _save_experiment_metadata(dataset_dir, samples, job)
    logger.info(
        "[experiment_test] finished job: runtime={} -> {}",
        _format_duration(time.perf_counter() - job_start_time),
        dataset_dir,
    )
    return summary_rows[0]


def _merge_with_all_handle() -> None:
    """
    用 all_handle/pipeline.py 完全相同的核心类合并 experiment 结果。

    这里不重新实现任何拼图逻辑，只临时覆盖 DATA_ROOT_DIR/SPLIT_NAMES/OUTPUT_ROOT_DIR，
    然后调用 AllHandlePipeline.run_all()，因此合并行为与 all_handle/pipeline.py 保持一致。
    """

    from study.SRGAN.all_handle.global_class import global_data as all_handle_global_data
    from study.SRGAN.all_handle.pipeline_core import AllHandlePipeline

    merge_start_time = time.perf_counter()
    logger.info(
        "[experiment_test] start all_handle merge: data_root={} output={}",
        TRAIN_DATAS_ROOT,
        MERGED_OUTPUT_DIR,
    )
    cfg = all_handle_global_data.all_handle
    original_values = {
        "DATA_ROOT_DIR": cfg.DATA_ROOT_DIR,
        "SPLIT_NAMES": cfg.SPLIT_NAMES,
        "OUTPUT_ROOT_DIR": cfg.OUTPUT_ROOT_DIR,
        "CATEGORY_FILTER": getattr(cfg, "CATEGORY_FILTER", None),
        "OUTPUT_STAGE_FILTER": getattr(cfg, "OUTPUT_STAGE_FILTER", None),
        "PARTICLE_ERROR_COLORBAR_LIMIT": getattr(cfg, "PARTICLE_ERROR_COLORBAR_LIMIT", "auto"),
    }
    try:
        cfg.DATA_ROOT_DIR = TRAIN_DATAS_ROOT
        cfg.SPLIT_NAMES = ("experiment",)
        cfg.OUTPUT_ROOT_DIR = MERGED_OUTPUT_DIR
        cfg.CATEGORY_FILTER = ("experiment",)
        # all_handle/global_class.py 当前可能为了调试被设置成只跑少数阶段，
        # 例如 ("tbl_profile_overlay", "particle_stats_metrics", "flow_u_epe_hist_overlay")。
        # experiment 合并必须和 all_handle/pipeline.py 的完整输出逻辑一致，因此这里临时
        # 打开全部阶段：01_energy_spectrum、02_error_maps、03_error_histograms、
        # 04_composite_panels，以及 run_all 末尾固定写出的 05_metric_tables。
        cfg.OUTPUT_STAGE_FILTER = "all"
        # 实验图的颗粒误差是 SR - 插值 HR，数值很小；单独缩小 experiment 合并图色条，
        # 避免 auto 被少数极端像素撑大后整行误差图发白。
        cfg.PARTICLE_ERROR_COLORBAR_LIMIT = (
            -float(EXPERIMENT_PARTICLE_ERROR_COLORBAR_LIMIT),
            float(EXPERIMENT_PARTICLE_ERROR_COLORBAR_LIMIT),
        )
        pipeline = AllHandlePipeline(cfg, enable_plotting=True)
        pipeline.run_all()
        logger.info(
            "[experiment_test] all_handle merge finished: runtime={}",
            _format_duration(time.perf_counter() - merge_start_time),
        )
    finally:
        for key, value in original_values.items():
            setattr(cfg, key, value)


def _write_skipped_jobs(path: Path, skipped: list[dict]) -> None:
    """记录没有 checkpoint 或不支持分支的目录，避免静默漏掉实验。"""

    if not skipped:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in skipped for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(skipped)
    logger.warning("[experiment_test] skipped {} jobs; details written to {}", len(skipped), path)


def run_experiment_tests(skip_merge: bool = False, only_merge: bool = False) -> list[dict]:
    """主流程：跑三阶段实验测试，并按需调用 all_handle 合并。"""

    total_start_time = time.perf_counter()
    logger.info("[experiment_test] pipeline started: skip_merge={} only_merge={}", skip_merge, only_merge)
    if only_merge:
        _merge_with_all_handle()
        return []

    samples = _load_experiment_samples()
    jobs, skipped = _discover_model_jobs()
    _write_skipped_jobs(TRAIN_DATAS_ROOT / "experiment_skipped_jobs.csv", skipped)
    if not jobs:
        raise FileNotFoundError(f"[experiment_test] 没有在 {TRAIN_DATAS_ROOT} 下找到可测试 checkpoint。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "[experiment_test] discovered jobs: runnable={} skipped={} device={}",
        len(jobs),
        len(skipped),
        device,
    )
    summaries = []
    errors = []
    for job_idx, job in enumerate(jobs, start=1):
        elapsed_total = time.perf_counter() - total_start_time
        logger.info(
            "[experiment_test] job progress {}/{}: {} / {} / {} / {} elapsed={} eta={}",
            job_idx,
            len(jobs),
            job.experiment_dir_name,
            job.class_name,
            job.run_class_name,
            job.scale_dir_name,
            _format_duration(elapsed_total),
            _estimate_eta(elapsed_total, job_idx - 1, len(jobs)) if job_idx > 1 else "unknown",
        )
        _write_progress_status(
            stage="job_start",
            job_index=job_idx,
            total_jobs=len(jobs),
            experiment_dir_name=job.experiment_dir_name,
            class_name=job.class_name,
            run_class_name=job.run_class_name,
            scale_dir_name=job.scale_dir_name,
            checkpoint_path=_checkpoint_label(job),
            elapsed=_format_duration(elapsed_total),
        )
        try:
            summaries.append(_run_single_job(job, samples, device))
            logger.info("[experiment_test] job succeeded {}/{}", job_idx, len(jobs))
            _write_progress_status(
                stage="job_succeeded",
                job_index=job_idx,
                total_jobs=len(jobs),
                experiment_dir_name=job.experiment_dir_name,
                class_name=job.class_name,
                run_class_name=job.run_class_name,
                scale_dir_name=job.scale_dir_name,
                success_count=len(summaries),
                error_count=len(errors),
            )
        except Exception as exc:
            logger.error("[experiment_test] job failed: {}\n{}", job, traceback.format_exc())
            errors.append(
                {
                    "experiment_dir_name": job.experiment_dir_name,
                    "class_name": job.class_name,
                    "run_class_name": job.run_class_name,
                    "scale_dir_name": job.scale_dir_name,
                    "checkpoint_path": _checkpoint_label(job),
                    "requires_checkpoint": job.requires_checkpoint,
                    "error": repr(exc),
                }
            )
            logger.warning("[experiment_test] job failed {}/{}; continuing with next job", job_idx, len(jobs))
            _write_progress_status(
                stage="job_failed",
                job_index=job_idx,
                total_jobs=len(jobs),
                experiment_dir_name=job.experiment_dir_name,
                class_name=job.class_name,
                run_class_name=job.run_class_name,
                scale_dir_name=job.scale_dir_name,
                error=repr(exc),
                success_count=len(summaries),
                error_count=len(errors),
            )
        finally:
            # 每个 job 后立刻落盘，避免系统级 Killed 时没有 summary/error CSV。
            _write_csv(TRAIN_DATAS_ROOT / "experiment_test_summary.csv", summaries)
            _write_csv(TRAIN_DATAS_ROOT / "experiment_test_errors.csv", errors)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    _write_csv(TRAIN_DATAS_ROOT / "experiment_test_summary.csv", summaries)
    _write_csv(TRAIN_DATAS_ROOT / "experiment_test_errors.csv", errors)
    logger.info(
        "[experiment_test] inference stage finished: success={} errors={} summary={} errors_csv={}",
        len(summaries),
        len(errors),
        TRAIN_DATAS_ROOT / "experiment_test_summary.csv",
        TRAIN_DATAS_ROOT / "experiment_test_errors.csv",
    )
    if not skip_merge:
        _merge_with_all_handle()
    logger.info(
        "[experiment_test] pipeline finished: total_runtime={}",
        _format_duration(time.perf_counter() - total_start_time),
    )
    return summaries


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run trained comparison models on the three experiment_handle particle-spray samples."
    )
    parser.add_argument("--skip-merge", action="store_true", help="Only run experiment tests; do not call all_handle merge.")
    parser.add_argument("--only-merge", action="store_true", help="Skip model inference and only merge existing experiment outputs.")
    return parser


def main() -> list[dict]:
    args = build_arg_parser().parse_args()
    return run_experiment_tests(skip_merge=args.skip_merge, only_merge=args.only_merge)


if __name__ == "__main__":
    main()
