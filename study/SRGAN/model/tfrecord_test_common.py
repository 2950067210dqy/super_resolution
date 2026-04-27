import csv
import math
import time
from pathlib import Path
from subprocess import call

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

from study.SRGAN.model.c_aee_metric_common import attach_c_aee_to_raft_rows
from study.SRGAN.data_downscal import DOWNSAMPLE_METHOD, INTERPOLATION_MODE, downsample_tif

try:
    autocast = torch.cuda.amp.autocast
except Exception:
    # 兼容旧版 PyTorch：如果没有 AMP，就保留 with autocast(enabled=False) 的调用形式。
    class autocast:
        def __init__(self, enabled=False):
            self.enabled = enabled

        def __enter__(self):
            return None

        def __exit__(self, *args):
            return False


_CACHED_2D_WINDOWS = {}


def _match_common_channels(pred_chw, gt_chw):
    """
    对齐预测/真值的公共通道数，避免不同分支的 SR 输出通道数不一致时直接报 shape 错误。

    说明：
        test_all 的 HR 输入大多是单通道颗粒图，但个别生成器可能输出多通道结果。
        evaluate_all 的指标本质上都只依赖“可对齐的公共通道”，因此这里统一截到两者共同拥有的最小通道数。
    """
    pred = np.asarray(pred_chw, dtype=np.float32)
    gt = np.asarray(gt_chw, dtype=np.float32)
    if pred.ndim != 3 or gt.ndim != 3:
        raise ValueError(f"Expected CHW arrays, got pred={pred.shape}, gt={gt.shape}")
    cnum = min(int(pred.shape[0]), int(gt.shape[0]))
    if cnum <= 0:
        raise ValueError(f"No common channels for metrics: pred={pred.shape}, gt={gt.shape}")
    return pred[:cnum], gt[:cnum]


def _mse(pred_chw, gt_chw):
    """计算均方误差 MSE。"""
    pred, gt = _match_common_channels(pred_chw, gt_chw)
    return float(np.mean((pred - gt) ** 2))


def _psnr_from_mse(mse):
    """由 MSE 计算 PSNR，假设图像范围已经在 [0,1]。"""
    return float("inf") if mse == 0 else 20.0 * math.log10(1.0 / math.sqrt(mse))


def _r2_score(pred_chw, gt_chw, eps=1e-12):
    """计算决定系数 R^2。"""
    pred, gt = _match_common_channels(pred_chw, gt_chw)
    y_true = gt.reshape(-1)
    y_pred = pred.reshape(-1)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / (ss_tot + eps)


def _nrmse(pred_chw, gt_chw, eps=1e-12):
    """按真值范围归一化的 RMSE。"""
    pred, gt = _match_common_channels(pred_chw, gt_chw)
    rmse = math.sqrt(float(np.mean((pred - gt) ** 2)))
    den = float(np.max(gt) - np.min(gt))
    return rmse / (den + eps)


def _ssim_score(pred_chw, gt_chw):
    """
    计算 SSIM，按公共通道逐通道求值后取平均。

    优先使用 skimage；如果环境里没有 skimage，则回退到一个简化版 SSIM 公式，
    保证 test_all 在最小依赖环境里也能把指标跑完。
    """
    pred, gt = _match_common_channels(pred_chw, gt_chw)
    try:
        from skimage.metrics import structural_similarity as sk_ssim

        vals = []
        for c in range(pred.shape[0]):
            p = pred[c]
            g = gt[c]
            dr = float(np.max(g) - np.min(g))
            dr = dr if dr > 1e-12 else 1.0
            vals.append(float(sk_ssim(g, p, data_range=dr)))
        return float(np.mean(vals))
    except Exception:
        vals = []
        C1, C2 = 0.01**2, 0.03**2
        for c in range(pred.shape[0]):
            x = pred[c]
            y = gt[c]
            mx, my = float(np.mean(x)), float(np.mean(y))
            vx, vy = float(np.var(x)), float(np.var(y))
            cov = float(np.mean((x - mx) * (y - my)))
            num = (2 * mx * my + C1) * (2 * cov + C2)
            den = (mx * mx + my * my + C1) * (vx + vy + C2)
            vals.append(num / den if den != 0 else 0.0)
        return float(np.mean(vals))


def _tke_reconstruction_accuracy(pred_chw, gt_chw, eps=1e-12):
    """
    计算 TKE 重建精度。

    对单通道 SR 图像，这个指标没有物理意义，因此会返回 nan；
    这与 evaluate_all 的行为保持一致。
    """
    pred, gt = _match_common_channels(pred_chw, gt_chw)
    if pred.shape[0] < 2 or gt.shape[0] < 2:
        return float("nan")
    up, vp = pred[0], pred[1]
    ug, vg = gt[0], gt[1]
    up_p = up - np.mean(up)
    vp_p = vp - np.mean(vp)
    ug_p = ug - np.mean(ug)
    vg_p = vg - np.mean(vg)
    tke_p = 0.5 * float(np.mean(up_p ** 2 + vp_p ** 2))
    tke_g = 0.5 * float(np.mean(ug_p ** 2 + vg_p ** 2))
    return 1.0 - abs(tke_p - tke_g) / (abs(tke_g) + eps)


def _radial_spectrum(ch2d):
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


def _energy_spectrum_curves(pred_chw, gt_chw):
    """计算公共通道上的平均径向能量谱曲线。"""
    pred, gt = _match_common_channels(pred_chw, gt_chw)
    pred_specs = []
    gt_specs = []
    min_len = None
    for c in range(pred.shape[0]):
        sp = _radial_spectrum(pred[c])
        sg = _radial_spectrum(gt[c])
        n = min(len(sp), len(sg))
        min_len = n if min_len is None else min(min_len, n)
        pred_specs.append(sp[:n])
        gt_specs.append(sg[:n])
    pred_curve = np.mean(np.stack([x[:min_len] for x in pred_specs], axis=0), axis=0)
    gt_curve = np.mean(np.stack([x[:min_len] for x in gt_specs], axis=0), axis=0)
    return pred_curve, gt_curve


def _energy_spectrum_mse_from_curves(pred_curve, gt_curve):
    """对已算好的谱曲线计算 log1p 频谱差 MSE。"""
    return float(np.mean((np.log1p(pred_curve) - np.log1p(gt_curve)) ** 2))


def _compute_aee_from_chw(pred_chw, gt_chw):
    """
    计算单样本 CHW 流场的 AEE。

    这里和 evaluate_all 保持同口径：AEE 等价于像素级 EPE 的平均值。
    """
    pred, gt = _match_common_channels(pred_chw, gt_chw)
    if pred.shape[0] < 2 or gt.shape[0] < 2:
        return float("nan")
    du = pred[0] - gt[0]
    dv = pred[1] - gt[1]
    epe = np.sqrt(du * du + dv * dv)
    return float(np.mean(epe))


def _mean_sum_per_100_pixels(values_1d, group_size=100):
    """
    将一维误差序列按每 100 个像素分组求和，再对所有满 100 像素分组和取平均。

    这就是 evaluate_all 里 NORM_AEE_PER100PIXEL 的统计口径。
    最后一组不足 100 个像素时直接丢弃，不参与统计。
    """
    values = np.asarray(values_1d, dtype=np.float32).reshape(-1)
    if values.size < group_size:
        return float("nan")
    usable_count = (values.size // group_size) * group_size
    if usable_count <= 0:
        return float("nan")
    values = values[:usable_count].reshape(-1, group_size)
    group_sums = np.sum(values, axis=1, dtype=np.float32)
    return float(np.mean(group_sums, dtype=np.float32)) if group_sums.size > 0 else float("nan")


def _compute_norm_aee_per100_from_chw(pred_chw, gt_chw):
    """
    用单样本 CHW 流场计算“每 100 个像素 EPE 累加值的平均”。
    """
    pred, gt = _match_common_channels(pred_chw, gt_chw)
    if pred.shape[0] < 2 or gt.shape[0] < 2:
        return float("nan")
    du = pred[0] - gt[0]
    dv = pred[1] - gt[1]
    epe = np.sqrt(du * du + dv * dv)
    return _mean_sum_per_100_pixels(epe, group_size=100)


def _compute_image_metric_row(dataset_name, sample_index, pair_type, pred_chw, gt_chw):
    """
    计算单个 previous/next SR 图像对的指标行。

    字段对齐 evaluate_all：mse / psnr / energy_spectrum_mse / r2 / ssim / tke_acc / nrmse。
    """
    pred_curve, gt_curve = _energy_spectrum_curves(pred_chw, gt_chw)
    mse = _mse(pred_chw, gt_chw)
    return {
        "dataset": dataset_name,
        "sample_index": sample_index,
        "pair_type": pair_type,
        "mse": mse,
        "psnr": _psnr_from_mse(mse),
        "energy_spectrum_mse": _energy_spectrum_mse_from_curves(pred_curve, gt_curve),
        "r2": _r2_score(pred_chw, gt_chw),
        "ssim": _ssim_score(pred_chw, gt_chw),
        "tke_acc": _tke_reconstruction_accuracy(pred_chw, gt_chw),
        "nrmse": _nrmse(pred_chw, gt_chw),
    }


def _compute_flow_metric_row(dataset_name, sample_index, pred_chw, gt_chw):
    """
    计算单个样本的流场指标行。

    按用户当前需求，test_all 的 RAFT 指标包含：
        1. EPE
        2. NORM_AEE_PER100PIXEL
        3. C_AEE

    其中 C_AEE 依赖整个 dataset 的 min-max 归一化，所以这里先放占位值，
    等整套 dataset 全部跑完后再统一回填真实结果。
    """
    return {
        "dataset": dataset_name,
        "sample_index": sample_index,
        "pair_type": "RAFT",
        "epe": _compute_aee_from_chw(pred_chw, gt_chw),
        "NORM_AEE_PER100PIXEL": _compute_norm_aee_per100_from_chw(pred_chw, gt_chw),
        # C-AEE 依赖“整套测试样本内”的 min-max 归一化，
        # 因此这里先占位为 NaN，等 dataset 全部跑完后再统一回填。
        "C_AEE": float("nan"),
    }


def _build_mean_row(rows, fixed_fields, metric_keys):
    """根据给定指标列生成一行均值记录。"""
    mean_row = dict(fixed_fields)
    for key in metric_keys:
        vals = []
        for row in rows:
            value = row.get(key, float("nan"))
            try:
                value = float(value)
            except Exception:
                value = float("nan")
            if np.isfinite(value):
                vals.append(value)
        mean_row[key] = float(np.mean(vals)) if vals else float("nan")
    return mean_row


def _write_rows_with_mean(path, rows, fixed_fields, metric_keys):
    """
    写入明细行 + 均值行。

    返回值：
        mean_row: 方便调用方继续把 dataset 级均值拼到 root metrics_all.csv。
    """
    if not rows:
        return None
    mean_row = _build_mean_row(rows, fixed_fields, metric_keys)
    _write_csv(path, rows + [mean_row])
    return mean_row


def _find_first_conv2d(module):
    """
    在模块里递归找到第一层 Conv2d。

    test_all 需要知道“这个分支的 SR 生成器训练时期待几通道图像”。
    对当前四个分支来说，最可靠的信息就是生成器第一层卷积的 in_channels。
    """
    if module is None:
        return None
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return child
    return None


def _infer_model_image_channels(model):
    """
    推断当前模型在 SR 分支期望的图像通道数。

    背景：
        - TFRecord 测试集里的 target 存的是 2 通道：[prev_gray, next_gray]；
        - 但这几个分支训练时的 Generator 常常按 inner_chanel=3 初始化，
          即 previous/next 每帧都会被当成 3 通道图像来处理；
        - 如果 test_all 直接喂 1 通道 patch，就会在第一层卷积处报
          “expected input to have 3 channels, but got 1 channels”。

    因此这里统一从 Generator 第一层卷积读取 in_channels，作为测试阶段的适配目标。
    如果模型上没有 Generator，则回退到 1，保持最保守行为。
    """
    generator = getattr(model, "piv_esrgan_generator", None)
    first_conv = _find_first_conv2d(generator)
    if first_conv is None:
        return 1
    return int(first_conv.in_channels)


def _adapt_image_channels_for_model(image_bchw, expected_channels):
    """
    把测试图像适配到模型训练时的通道口径。

    当前 test_all 的原始图片是单通道颗粒图 [B, 1, H, W]。如果模型训练时使用
    inner_chanel=3，那么这里会把单通道复制成 3 通道，再送入 Generator / VGG / GAN
    分支，保证前向逻辑和训练时一致。

    说明：
        - 1 -> 3：直接 repeat，最符合“灰度图复制到 RGB 三通道”的训练习惯；
        - N -> 1：对通道取均值，保持灰度意义；
        - 其他情况：尽量按 repeat + 截断适配，避免因为测试集通道数和实验口径不同而崩溃。
    """
    if image_bchw.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape={tuple(image_bchw.shape)}")

    current_channels = int(image_bchw.shape[1])
    expected_channels = int(expected_channels)
    if current_channels == expected_channels:
        return image_bchw
    if expected_channels <= 0:
        raise ValueError(f"expected_channels must be positive, got {expected_channels}")

    if current_channels == 1 and expected_channels > 1:
        return image_bchw.repeat(1, expected_channels, 1, 1)
    if expected_channels == 1 and current_channels > 1:
        return image_bchw.mean(dim=1, keepdim=True)
    if expected_channels < current_channels:
        return image_bchw[:, :expected_channels, :, :]

    repeat_times = int(math.ceil(expected_channels / current_channels))
    return image_bchw.repeat(1, repeat_times, 1, 1)[:, :expected_channels, :, :]


def _collapse_image_to_single_channel_for_test(image_bchw):
    """
    把模型输出图像压回 test_all 统一使用的单通道口径。

    test_all 的保存图、图像指标、对比图都围绕“单通道 prev / next 颗粒图”设计。
    因此如果某个分支的 Generator 输出为 3 通道，这里统一做通道均值，得到单通道图：
        1. 与 TFRecord 的原始 ground truth 口径一致；
        2. 不改模型内部 RAFT / loss 的真实计算逻辑；
        3. 避免后续可视化代码只取第 0 通道时混入通道偏差。
    """
    if image_bchw.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape={tuple(image_bchw.shape)}")
    if int(image_bchw.shape[1]) == 1:
        return image_bchw
    return image_bchw.mean(dim=1, keepdim=True)


def _load_dali_modules():
    """
    延迟导入 DALI。

    pipeline 导入 test_all 时不一定马上运行 TFRecord 测试；把 DALI 放到运行阶段导入，
    可以避免没有安装 DALI 的普通训练/验证环境在 import 阶段就失败。
    """
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.tfrecord as tfrec
    from nvidia.dali.plugin.pytorch import DALIGenericIterator

    return Pipeline, ops, tfrec, DALIGenericIterator


def _triang(window_size):
    """生成与 scipy.signal.triang 等价的三角窗，SciPy 新旧版本路径不同，因此这里做兼容。"""
    try:
        from scipy.signal import triang

        return triang(window_size)
    except Exception:
        try:
            from scipy.signal.windows import triang

            return triang(window_size)
        except Exception:
            return np.bartlett(window_size)


def _spline_window(window_size, power=2):
    """RAFT256-PIV 测试脚本同款 squared spline 窗，用于重叠 patch 加权融合。"""
    intersection = int(window_size / 4)
    wind_outer = (abs(2 * _triang(window_size)) ** power) / 2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2 * (_triang(window_size) - 1)) ** power) / 2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind


def _window_2d(window_size, power=2):
    """缓存 2D spline window，避免每个 batch 重复计算。"""
    key = f"{window_size}_{power}"
    if key not in _CACHED_2D_WINDOWS:
        wind = _spline_window(window_size, power)
        wind = np.expand_dims(np.expand_dims(wind, -1), -1)
        _CACHED_2D_WINDOWS[key] = wind * wind.transpose(1, 0, 2)
    return _CACHED_2D_WINDOWS[key]


def _build_tfrecord_pipeline_class():
    """
    构造和 RAFT256-PIV_test.py 保持一致的 DALI TFRecord 读取 pipeline。

    TFRecord 中：
    - target: [2, H, W]，第 0 通道是 prev，第 1 通道是 next
    - flow:   [2, H, W]，光流真值
    - label:  保留读取但 test_all 不参与计算，保持原脚本数据结构不变
    """
    Pipeline, ops, tfrec, _ = _load_dali_modules()

    class TFRecordPipeline(Pipeline):
        def __init__(
            self,
            batch_size,
            num_threads,
            device_id,
            tfrecord,
            tfrecord_idx,
            image_shape,
            label_shape,
            is_shuffle=False,
        ):
            super().__init__(
                batch_size,
                num_threads,
                device_id,
                exec_pipelined=False,
                exec_async=False,
            )
            self.input = ops.TFRecordReader(
                path=tfrecord,
                index_path=tfrecord_idx,
                random_shuffle=is_shuffle,
                pad_last_batch=True,
                shard_id=0,
                num_shards=1,
                features={
                    "target": tfrec.FixedLenFeature([], tfrec.string, ""),
                    "label": tfrec.FixedLenFeature([], tfrec.string, ""),
                    "flow": tfrec.FixedLenFeature([], tfrec.string, ""),
                },
            )
            self.decode = ops.PythonFunction(function=self.extract_view, num_outputs=1)
            self.reshape_image = ops.Reshape(shape=image_shape)
            self.reshape_label = ops.Reshape(shape=label_shape)

        def extract_view(self, data):
            return data.view("<f4")

        def define_graph(self):
            inputs = self.input(name="Reader")
            images = self.reshape_image(self.decode(inputs["target"]))
            labels = self.reshape_label(self.decode(inputs["label"]))
            flows = self.reshape_image(self.decode(inputs["flow"]))
            return images, labels, flows

    return TFRecordPipeline


def _as_cuda_device(device):
    """把 global_data 中的 device 配置统一转换成 torch.device 与 DALI 需要的 int gpu id。"""
    if device is None:
        device = torch.device("cuda", 0)
    elif isinstance(device, int):
        device = torch.device("cuda", device)
    elif isinstance(device, str):
        device = torch.device(device)

    if device.type != "cuda":
        raise RuntimeError("test_all 按单 GPU CUDA 流程运行，请将 global_data.esrgan.device 设置为 cuda。")

    if device.index is None:
        device_id = torch.cuda.current_device()
        device = torch.device("cuda", device_id)
    else:
        device_id = device.index

    torch.cuda.set_device(device_id)
    return device, device_id


def _resolve_path(raw_path, global_data):
    """
    解析全局变量中的 TFRecord / idx / TWCF 辅助文件路径。

    优先使用用户配置的原路径；如果是相对路径，再依次尝试当前工作目录、SRGAN 根目录、
    study 根目录和 AUTODL_DATA_PATH。这样既兼容 RAFT256-PIV_test.py 的 ../data 写法，
    也方便在不同启动目录下运行 pipeline。
    """
    path = Path(raw_path)
    if path.is_absolute():
        return path

    this_file = Path(__file__).resolve()
    candidates = [
        Path.cwd() / path,
        this_file.parents[1] / path,  # .../study/SRGAN
        this_file.parents[2] / path,  # .../study
    ]
    autodl_data_path = getattr(global_data.esrgan, "AUTODL_DATA_PATH", None)
    if autodl_data_path:
        candidates.append(Path(autodl_data_path) / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _ensure_tfrecord_idx(tfrecord_path, idx_path, global_data):
    """idx 不存在时复用 RAFT256-PIV_test.py 的 tfrecord2idx 生成方式。"""
    if idx_path.exists():
        return

    script = getattr(global_data.esrgan, "TEST_TFRECORD2IDX_SCRIPT", "tfrecord2idx")
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"[test_all] idx 不存在，开始生成：{idx_path}")
    call([script, str(tfrecord_path), str(idx_path)])


def _build_dali_iterator(dataset_cfg, global_data, device_id):
    """根据单个 dataset 的全局配置创建 DALI iterator。"""
    _, _, _, DALIGenericIterator = _load_dali_modules()
    TFRecordPipeline = _build_tfrecord_pipeline_class()

    tfrecord_path = _resolve_path(dataset_cfg["test_tfrecord"], global_data)
    idx_path = _resolve_path(dataset_cfg["test_tfrecord_idx"], global_data)
    _ensure_tfrecord_idx(tfrecord_path, idx_path, global_data)

    batch_size = int(getattr(global_data.esrgan, "TEST_BATCH_SIZE", 1))
    num_threads = int(getattr(global_data.esrgan, "TEST_NUM_THREADS", 8))
    image_shape = [2, int(dataset_cfg["image_height"]), int(dataset_cfg["image_width"])]
    label_shape = list(dataset_cfg.get("label_shape", [12]))

    pipe = TFRecordPipeline(
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        tfrecord=str(tfrecord_path),
        tfrecord_idx=str(idx_path),
        is_shuffle=False,
        image_shape=image_shape,
        label_shape=label_shape,
    )
    pipe.build()

    size = int(pipe.epoch_size("Reader"))
    iterator = DALIGenericIterator(
        pipe,
        ["target", "label", "flow"],
        size=size,
        last_batch_padded=True,
        fill_last_batch=False,
        auto_reset=True,
    )
    return iterator, size


def _scale_factor_from_scale(scale):
    """训练目录一直使用 scale_{int(SCALE * SCALE)}，测试下采样也保持同一倍率定义。"""
    return max(1, int(scale * scale))


def _make_lr_from_hr_tensor(hr_tensor, factor, device):
    """
    用 data_downscal.py 的 downsample_tif 生成 LR。

    输入是已经归一化后的 HR tensor，形状 [B, 1, H, W]；为了严格复用项目下采样逻辑，
    每个 sample 每个通道转 numpy 后调用 downsample_tif，再转回 CUDA tensor。
    """
    if factor <= 1:
        return hr_tensor.detach().clone()

    lr_images = []
    for sample in hr_tensor.detach().cpu().numpy():
        # sample 形状是 [1, H, W]，downsample_tif 接收单张 2D 图，因此取第 0 通道。
        lr_hw = downsample_tif(
            sample[0],
            factor=factor,
            expand_to_original=False,
            method=DOWNSAMPLE_METHOD,
            interpolation_mode=INTERPOLATION_MODE,
        )
        lr_images.append(lr_hw[None, ...])

    lr = np.stack(lr_images, axis=0).astype(np.float32, copy=False)
    return torch.from_numpy(lr).to(device=device, dtype=hr_tensor.dtype)


def _last_flow_prediction(flow_predictions):
    """兼容 list/tuple/tensor 三种返回形式，统一取最终迭代的 flow。"""
    if isinstance(flow_predictions, (list, tuple)):
        return flow_predictions[-1]
    return flow_predictions


def _predict_patch(model, images_hr, flows_hr, factor, device, flow_init=None):
    """
    对一个 256x256 patch 或普通 256x256 样本做联合模型推理。

    target 的第 0 通道是 prev，第 1 通道是 next；这里先拆 HR，再用 data_downscal.py
    生成 prev_lr/next_lr，最后仍通过 PIV_ESRGAN_RAFT_Model.forward 得到 SR+RAFT 预测。

    额外兼容：
        当前四个分支训练时很多 Generator 是按 inner_chanel=3 初始化的，但 test_all 的
        TFRecord target 是灰度双帧 [prev_gray, next_gray]。因此这里会：
        1. 保留原始单通道 prev/next，供 test_all 的图片保存和指标统计使用；
        2. 自动把 prev/next 的 LR/HR 复制到模型期望的通道数后再送入 forward；
        3. 模型生成出的多通道 SR 图再压回单通道，统一交给 test_all 做后处理。
    """
    # TFRecord 原始双帧是单通道灰度图；这两份 tensor 会原样保留给保存图片/算指标。
    prev_hr = images_hr[:, 0:1, :, :]
    next_hr = images_hr[:, 1:2, :, :]
    prev_lr = _make_lr_from_hr_tensor(prev_hr, factor, device)
    next_lr = _make_lr_from_hr_tensor(next_hr, factor, device)

    # 生成器/VGG/GAN 分支前向必须吃到与训练时一致的通道数，否则会在第一层卷积报错。
    expected_channels = _infer_model_image_channels(model)
    prev_hr_for_model = _adapt_image_channels_for_model(prev_hr, expected_channels)
    next_hr_for_model = _adapt_image_channels_for_model(next_hr, expected_channels)
    prev_lr_for_model = _adapt_image_channels_for_model(prev_lr, expected_channels)
    next_lr_for_model = _adapt_image_channels_for_model(next_lr, expected_channels)

    pred_prev, pred_next, flow_predictions, _ = model(
        input_lr_prev=prev_lr_for_model,
        input_lr_next=next_lr_for_model,
        input_gr_prev=prev_hr_for_model,
        input_gr_next=next_hr_for_model,
        flowl0=flows_hr,
        flow_init=flow_init,
        is_adversarial=False,
    )
    # test_all 的图像输出、对比图和 SR 图像指标都以单通道颗粒图为基准；
    # 这里把多通道 Generator 输出压回单通道，避免后续保存/评估继续带着 3 通道口径。
    pred_prev = _collapse_image_to_single_channel_for_test(pred_prev)
    pred_next = _collapse_image_to_single_channel_for_test(pred_next)
    return {
        "flow": _last_flow_prediction(flow_predictions),  # 最终 RAFT flow 预测
        "prev_lr": prev_lr,  # 用 data_downscal.py 从 HR prev 生成的 LR prev
        "next_lr": next_lr,  # 用 data_downscal.py 从 HR next 生成的 LR next
        "prev_hr": prev_hr,  # TFRecord target 第 0 通道，对应原始 HR prev
        "next_hr": next_hr,  # TFRecord target 第 1 通道，对应原始 HR next
        "pred_prev": pred_prev,  # 联合模型输出的 SR/generated prev
        "pred_next": pred_next,  # 联合模型输出的 SR/generated next
    }


def _fold_weighted_patches(patch_tensor, B, C, H, W, num_y, num_x, offset, shift, window):
    """
    将 patch 预测按 RAFT256-PIV_test.py 的 spline window 加权 fold 回原图尺寸。

    flow 使用 C=2，SR 图像使用 C=1；抽成公共函数后，tbl/twcf 的 flow 和生成图
    都能使用同一套重叠区域融合逻辑，避免图片输出和 flow 输出的拼接方式不一致。
    """
    weighted_patches = patch_tensor * window
    weighted_patches = weighted_patches.reshape((B, num_y, num_x, C, offset, offset)).permute(0, 3, 1, 2, 4, 5)
    weighted_patches = weighted_patches.contiguous().view(B, C, -1, offset * offset)
    weighted_patches = weighted_patches.permute(0, 1, 3, 2)
    weighted_patches = weighted_patches.contiguous().view(B, C * offset * offset, -1)
    folded = F.fold(weighted_patches, output_size=(H, W), kernel_size=offset, stride=shift)

    mask_source = torch.ones((B, C, H, W), device=patch_tensor.device, dtype=patch_tensor.dtype)
    mask_patches = mask_source.unfold(3, offset, shift).unfold(2, offset, shift)
    mask_patches = mask_patches.contiguous().view(B, C, -1, offset, offset)
    mask_patches = mask_patches * window
    mask_patches = mask_patches.view(B, C, -1, offset * offset)
    mask_patches = mask_patches.permute(0, 1, 3, 2)
    mask_patches = mask_patches.contiguous().view(B, C * offset * offset, -1)
    folding_mask = F.fold(mask_patches, output_size=(H, W), kernel_size=offset, stride=shift)

    return folded / folding_mask


def _predict_full_frame_with_folding(model, images, flows, factor, device, test_args):
    """
    复刻 RAFT256-PIV_test.py 对 tbl/twcf 的滑窗推理。

    不同点：这里不是直接把 HR 图送 RAFT，而是每个 HR patch 先按当前工程的
    data_downscal.py 逻辑生成 LR patch，再走联合 ESRGAN+RAFT 模型。
    """
    offset = int(test_args["offset"])
    shift = int(test_args["shift"])
    split_size = int(test_args["split_size"])

    B, C, H, W = images.size()
    num_y = int(H / shift - (offset / shift - 1))
    num_x = int(W / shift - (offset / shift - 1))

    predicted_flows = torch.zeros_like(flows, device=device)

    patches = images.unfold(3, offset, shift).unfold(2, offset, shift).permute(0, 2, 3, 1, 5, 4)
    patches = patches.reshape((-1, 2, offset, offset))
    flow_patches = flows.unfold(3, offset, shift).unfold(2, offset, shift).permute(0, 2, 3, 1, 5, 4)
    flow_patches = flow_patches.reshape((-1, 2, offset, offset))

    predicted_flow_patches = predicted_flows.unfold(3, offset, shift).unfold(2, offset, shift).permute(0, 2, 3, 1, 5, 4)
    predicted_flow_patches = predicted_flow_patches.reshape((-1, 2, offset, offset))

    patch_flow_outputs = []
    patch_pred_prev_outputs = []
    patch_pred_next_outputs = []
    split_patches = torch.split(patches, split_size, dim=0)
    split_flows = torch.split(flow_patches, split_size, dim=0)
    split_flow_init = torch.split(predicted_flow_patches, split_size, dim=0)
    for patch, flow_patch, flow_init_patch in zip(split_patches, split_flows, split_flow_init):
        patch_result = _predict_patch(model, patch, flow_patch, factor, device, flow_init=flow_init_patch)
        patch_flow_outputs.append(patch_result["flow"])
        patch_pred_prev_outputs.append(patch_result["pred_prev"])
        patch_pred_next_outputs.append(patch_result["pred_next"])

    window = torch.from_numpy(np.squeeze(_window_2d(window_size=offset, power=2))).to(device=device, dtype=images.dtype)
    predicted_flows = _fold_weighted_patches(
        torch.cat(patch_flow_outputs, dim=0), B, C, H, W, num_y, num_x, offset, shift, window
    )
    pred_prev = _fold_weighted_patches(
        torch.cat(patch_pred_prev_outputs, dim=0), B, 1, H, W, num_y, num_x, offset, shift, window
    )
    pred_next = _fold_weighted_patches(
        torch.cat(patch_pred_next_outputs, dim=0), B, 1, H, W, num_y, num_x, offset, shift, window
    )

    return {
        "flow": predicted_flows,
        # tbl/twcf 的 LR 图保存整张 full-frame 下采样结果；SR/generated 图则来自 patch SR 融合。
        "prev_lr": _make_lr_from_hr_tensor(images[:, 0:1, :, :], factor, device),
        "next_lr": _make_lr_from_hr_tensor(images[:, 1:2, :, :], factor, device),
        "prev_hr": images[:, 0:1, :, :],
        "next_hr": images[:, 1:2, :, :],
        "pred_prev": pred_prev,
        "pred_next": pred_next,
    }


def _mask_field_for_plot(field_2d, mask_2d=None):
    """
    按 mask 把无效区域转成 masked array，绘图时显示为空白而不是被最低值颜色染满。
    """
    field = np.asarray(field_2d, dtype=np.float32)
    if mask_2d is None:
        return field
    mask = np.asarray(mask_2d)
    if mask.shape != field.shape:
        return field
    return np.ma.masked_where(mask <= 0, field)


def _plot_field_with_colorbar(ax, field_2d, title, cmap_name, vmin, vmax, label):
    """给单个子图统一绘制位移场和色条。"""
    im = ax.imshow(field_2d, origin="lower", cmap=cmap_name, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=14)
    ax.axis("off")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.set_ylabel(label, fontsize=12)
    return im


def _plot_twcf(out_path, u_pred, v_pred, piv_results, mask_twcf, sample_index, cmap_name="viridis"):
    """
    保存 TWCF 的 PascalPIV 对比图。

    修改点：
        旧版色条使用 Greys，视觉上是黑白图；这里改成与用户给定示例一致的 viridis 风格色条，
        让位移高低在紫-蓝-绿-黄之间连续过渡，更适合看边界层速度梯度。
    """
    ref_index = min(int(sample_index), int(piv_results.shape[0]) - 1)
    u_pascal = np.asarray(piv_results[ref_index, 0, :, :], dtype=np.float32)
    v_pascal = np.asarray(piv_results[ref_index, 1, :, :], dtype=np.float32)
    mask_2d = np.asarray(mask_twcf, dtype=np.float32) if mask_twcf is not None else None

    fig, axes = plt.subplots(2, 2, figsize=(24, 16), dpi=120, facecolor="w", edgecolor="k")
    _plot_field_with_colorbar(
        axes[0, 0], _mask_field_for_plot(u_pascal, mask_2d), "PascalPIV U",
        cmap_name, vmin=-2, vmax=12, label="displacement [px]"
    )
    _plot_field_with_colorbar(
        axes[0, 1], _mask_field_for_plot(v_pascal, mask_2d), "PascalPIV V",
        cmap_name, vmin=-1, vmax=1, label="displacement [px]"
    )
    _plot_field_with_colorbar(
        axes[1, 0], _mask_field_for_plot(u_pred, mask_2d), "Current U",
        cmap_name, vmin=-2, vmax=12, label="displacement [px]"
    )
    _plot_field_with_colorbar(
        axes[1, 1], _mask_field_for_plot(v_pred, mask_2d), "Current V",
        cmap_name, vmin=-1, vmax=1, label="displacement [px]"
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_tbl(out_path, u_pred, v_pred, u_gt, v_gt, cmap_name="viridis"):
    """
    保存 TBL 全图测试的预测/真值对比图。

    这里同样把原来的黑白色条改成 viridis，便于直接和论文式位移场热图保持一致。
    """
    fig, axes = plt.subplots(2, 2, figsize=(24, 16), dpi=120, facecolor="w", edgecolor="k")
    _plot_field_with_colorbar(
        axes[0, 0], np.squeeze(u_pred), "Pred U",
        cmap_name, vmin=2, vmax=8, label="displacement [px]"
    )
    _plot_field_with_colorbar(
        axes[0, 1], np.squeeze(v_pred), "Pred V",
        cmap_name, vmin=-0.5, vmax=0.5, label="displacement [px]"
    )
    _plot_field_with_colorbar(
        axes[1, 0], np.squeeze(u_gt), "GT U",
        cmap_name, vmin=2, vmax=8, label="displacement [px]"
    )
    _plot_field_with_colorbar(
        axes[1, 1], np.squeeze(v_gt), "GT V",
        cmap_name, vmin=-0.5, vmax=0.5, label="displacement [px]"
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _resolve_twcf_profile_columns(width, column_ratios):
    """把 TWCF 论文风格图的三个剖面位置从比例转换成像素列坐标。"""
    cols = []
    for ratio in column_ratios:
        ratio = float(ratio)
        col = int(round((width - 1) * ratio))
        cols.append(int(np.clip(col, 0, width - 1)))
    return cols


def _save_twcf_profile_artifacts(
    dataset_dir,
    sample_index,
    u_pred,
    u_gt,
    mask_twcf,
    method_label,
    cmap_name="viridis",
    column_ratios=(0.15, 0.24, 0.83),
    region_names=("Laminar", "Transition", "Turbulent"),
):
    """
    保存 TWCF 的论文风格剖面对比图，并把关键数据落成 .npy 便于和其他方法后处理比较。

    图像结构：
        1. 上半部分：GT 水平位移场 + 三条红色虚线采样位置；
        2. 下半部分：三个位置的 y-方向位移剖面，只画 GT 与当前方法。
    """
    analysis_dir = dataset_dir / "twcf_profile_analysis" / f"sample_{sample_index:04d}"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    u_pred = np.asarray(u_pred, dtype=np.float32)
    u_gt = np.asarray(u_gt, dtype=np.float32)
    mask_2d = None if mask_twcf is None else np.asarray(mask_twcf, dtype=np.float32)
    columns = _resolve_twcf_profile_columns(u_gt.shape[1], column_ratios)
    y_positions = np.arange(u_gt.shape[0], dtype=np.float32)

    profile_gt = np.full((len(columns), u_gt.shape[0]), np.nan, dtype=np.float32)
    profile_pred = np.full((len(columns), u_gt.shape[0]), np.nan, dtype=np.float32)
    for idx, col in enumerate(columns):
        if mask_2d is not None and mask_2d.shape == u_gt.shape:
            valid = mask_2d[:, col] > 0
        else:
            valid = np.ones((u_gt.shape[0],), dtype=bool)
        profile_gt[idx, valid] = u_gt[valid, col]
        profile_pred[idx, valid] = u_pred[valid, col]

    # 保存原始场和已经抽好的剖面，后续其他方法只要在同样列位置上取 profile 就能直接对比。
    np.save(analysis_dir / "u_gt.npy", u_gt.astype(np.float32))
    np.save(analysis_dir / "u_pred.npy", u_pred.astype(np.float32))
    if mask_2d is not None:
        np.save(analysis_dir / "mask.npy", mask_2d.astype(np.float32))
    np.save(analysis_dir / "profile_columns.npy", np.asarray(columns, dtype=np.int32))
    np.save(analysis_dir / "profile_y_positions.npy", y_positions)
    np.save(analysis_dir / "profile_gt.npy", profile_gt)
    np.save(analysis_dir / "profile_pred.npy", profile_pred)

    masked_gt = _mask_field_for_plot(u_gt, mask_2d)
    if np.ma.isMaskedArray(masked_gt):
        valid_values = masked_gt.compressed()
    else:
        valid_values = np.asarray(masked_gt).reshape(-1)
        valid_values = valid_values[np.isfinite(valid_values)]
    if valid_values.size > 0:
        vmin = float(np.nanpercentile(valid_values, 1))
        vmax = float(np.nanpercentile(valid_values, 99))
    else:
        vmin, vmax = float(np.min(u_gt)), float(np.max(u_gt))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0.0, 1.0

    fig = plt.figure(figsize=(14, 14), dpi=160, facecolor="w", edgecolor="k")
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.8], hspace=0.35, wspace=0.25)

    ax_top = fig.add_subplot(grid[0, :])
    im = ax_top.imshow(masked_gt, origin="lower", cmap=cmap_name, vmin=vmin, vmax=vmax, aspect="auto")
    ax_top.set_title("Ground truth of horizontal direction", fontsize=18)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    for idx, col in enumerate(columns):
        ax_top.axvline(col, color="red", linestyle="--", linewidth=2.0)
        label = region_names[idx] if idx < len(region_names) else f"Region {idx + 1}"
        ax_top.text(
            col,
            u_gt.shape[0] - 8,
            label,
            color="red",
            fontsize=16,
            ha="center",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.5},
        )
    cbar = fig.colorbar(im, ax=ax_top, orientation="horizontal", fraction=0.08, pad=0.18)
    cbar.set_label("Displacement[px]", fontsize=15)

    x_values = np.concatenate(
        [
            profile_gt[np.isfinite(profile_gt)],
            profile_pred[np.isfinite(profile_pred)],
        ],
        axis=0,
    ) if np.isfinite(profile_gt).any() or np.isfinite(profile_pred).any() else np.asarray([0.0, 1.0], dtype=np.float32)
    x_min = float(np.nanmin(x_values))
    x_max = float(np.nanmax(x_values))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
        x_min, x_max = 0.0, 1.0
    x_pad = 0.05 * max(x_max - x_min, 1.0)

    for idx, col in enumerate(columns):
        ax = fig.add_subplot(grid[1, idx])
        valid_gt = np.isfinite(profile_gt[idx])
        valid_pred = np.isfinite(profile_pred[idx])
        if np.any(valid_gt):
            ax.plot(
                profile_gt[idx][valid_gt],
                y_positions[valid_gt],
                color="black",
                linewidth=2.0,
                linestyle=(0, (6, 3)),
                label="GT",
            )
        if np.any(valid_pred):
            ax.plot(
                profile_pred[idx][valid_pred],
                y_positions[valid_pred],
                color="red",
                linewidth=2.2,
                linestyle="-",
                label=method_label,
            )
        title = region_names[idx] if idx < len(region_names) else f"Region {idx + 1}"
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.set_xlabel("Displacement[px]", fontsize=12)
        if idx == 0:
            ax.set_ylabel("y-position[px]", fontsize=12)
            ax.legend(loc="upper left", fontsize=11, frameon=True)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(0, u_gt.shape[0] - 1)
        ax.grid(alpha=0.15)

    fig.savefig(analysis_dir / "twcf_profile_compare.png", bbox_inches="tight")
    plt.close(fig)


def _plot_regular(out_path, u_pred, v_pred, u_gt, v_gt):
    """保存 256x256 数据集的 u/v 预测、真值和误差图。"""
    min_val_u, max_val_u = -4, 4
    min_val_v, max_val_v = -4, 4

    plt.figure(num=None, figsize=(24, 16), dpi=120, facecolor="w", edgecolor="k")
    plt.subplot(3, 2, 1)
    plt.pcolor(u_pred, cmap="Greys", vmin=min_val_u, vmax=max_val_u)
    plt.axis("off")
    plt.colorbar().ax.set_ylabel("displacement [px]", fontsize=14)
    plt.subplot(3, 2, 3)
    plt.pcolor(u_gt, cmap="Greys", vmin=min_val_u, vmax=max_val_u)
    plt.axis("off")
    plt.colorbar().ax.set_ylabel("displacement [px]", fontsize=14)
    plt.subplot(3, 2, 5)
    plt.pcolor(u_pred - u_gt, cmap="bwr", vmin=-0.25, vmax=0.25)
    plt.axis("off")
    plt.colorbar().ax.set_ylabel("abs. error [px]", fontsize=14)
    plt.subplot(3, 2, 2)
    plt.pcolor(v_pred, cmap="Greys", vmin=min_val_v, vmax=max_val_v)
    plt.axis("off")
    plt.colorbar().ax.set_ylabel("displacement [px]", fontsize=14)
    plt.subplot(3, 2, 4)
    plt.pcolor(v_gt, cmap="Greys", vmin=min_val_v, vmax=max_val_v)
    plt.axis("off")
    plt.colorbar().ax.set_ylabel("displacement [px]", fontsize=14)
    plt.subplot(3, 2, 6)
    plt.pcolor(v_pred - v_gt, cmap="bwr", vmin=-0.25, vmax=0.25)
    plt.axis("off")
    plt.colorbar().ax.set_ylabel("abs. error [px]", fontsize=14)
    plt.savefig(out_path)
    plt.close()


def _as_numpy_batch(tensor):
    """把 [B, C, H, W] tensor 转成 float32 numpy，供图片保存函数复用。"""
    return tensor.detach().cpu().numpy().astype(np.float32, copy=False)


def _clip_image_for_display(arr):
    """图片保存前做有限值清理和 [0, 1] 裁剪，避免异常值把整张灰度图拉黑/拉白。"""
    arr = np.squeeze(arr).astype(np.float32, copy=False)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(arr, 0.0, 1.0)


def _resize_image_for_display(arr, out_hw):
    """
    将 LR 图临时插值到 HR 尺寸，仅用于合并对比图的视觉对齐。

    单独保存的 *_lr.png 仍然保留真实低分辨率尺寸；这里放大只是为了让 LR/HR/SR
    能在同一张 comparison 图里并排比较。
    """
    arr = np.squeeze(arr).astype(np.float32, copy=False)
    if tuple(arr.shape[-2:]) == tuple(out_hw):
        return arr
    tensor = torch.from_numpy(arr)[None, None, :, :].float()
    resized = F.interpolate(tensor, size=out_hw, mode="bicubic", align_corners=False)
    return resized.squeeze(0).squeeze(0).numpy()


def _save_gray_image(path, arr):
    """保存单通道灰度图，路径父目录自动创建。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(str(path), _clip_image_for_display(arr), cmap="gray", vmin=0.0, vmax=1.0)


def _plot_image_comparison(out_path, sample_images):
    """
    保存 prev/next 的 LR、原图 HR、生成 SR 合并对比图。

    画布为 2 行 3 列：第一行 previous，第二行 next；列依次为 LR、HR、Generated。
    """
    frames = [
        ("prev", sample_images["prev_lr"], sample_images["prev_hr"], sample_images["pred_prev"]),
        ("next", sample_images["next_lr"], sample_images["next_hr"], sample_images["pred_next"]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=140, facecolor="w")
    for row_idx, (frame_name, lr_img, hr_img, sr_img) in enumerate(frames):
        hr_hw = np.squeeze(hr_img).shape[-2:]
        lr_for_compare = _resize_image_for_display(lr_img, hr_hw)
        panels = [
            ("LR", lr_for_compare),
            ("Original HR", hr_img),
            ("Generated SR", sr_img),
        ]
        for col_idx, (title, arr) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.imshow(_clip_image_for_display(arr), cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_title(f"{frame_name} {title}", fontsize=10)
            ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_image_outputs(dataset_dir, image_payload, start_index):
    """
    保存每个 sample 的 LR、原始 HR、生成 SR 图，以及一张合并对比图。

    目录结构：
    dataset_dir/images/sample_0000/
        prev_lr.png / prev_hr.png / prev_sr.png
        next_lr.png / next_hr.png / next_sr.png
        comparison.png
    """
    payload_np = {
        "prev_lr": _as_numpy_batch(image_payload["prev_lr"]),
        "next_lr": _as_numpy_batch(image_payload["next_lr"]),
        "prev_hr": _as_numpy_batch(image_payload["prev_hr"]),
        "next_hr": _as_numpy_batch(image_payload["next_hr"]),
        "pred_prev": _as_numpy_batch(image_payload["pred_prev"]),
        "pred_next": _as_numpy_batch(image_payload["pred_next"]),
    }

    batch_size = payload_np["prev_hr"].shape[0]
    for local_idx in range(batch_size):
        sample_index = start_index + local_idx
        sample_dir = dataset_dir / "images" / f"sample_{sample_index:04d}"
        sample_images = {
            "prev_lr": payload_np["prev_lr"][local_idx, 0],
            "next_lr": payload_np["next_lr"][local_idx, 0],
            "prev_hr": payload_np["prev_hr"][local_idx, 0],
            "next_hr": payload_np["next_hr"][local_idx, 0],
            "pred_prev": payload_np["pred_prev"][local_idx, 0],
            "pred_next": payload_np["pred_next"][local_idx, 0],
        }

        _save_gray_image(sample_dir / "prev_lr.png", sample_images["prev_lr"])
        _save_gray_image(sample_dir / "prev_hr.png", sample_images["prev_hr"])
        _save_gray_image(sample_dir / "prev_sr.png", sample_images["pred_prev"])
        _save_gray_image(sample_dir / "next_lr.png", sample_images["next_lr"])
        _save_gray_image(sample_dir / "next_hr.png", sample_images["next_hr"])
        _save_gray_image(sample_dir / "next_sr.png", sample_images["pred_next"])
        _plot_image_comparison(sample_dir / "comparison.png", sample_images)


def _save_sample_plots(
    dataset_name,
    dataset_dir,
    predicted_np,
    flow_np,
    start_index,
    twcf_payload,
    image_payload=None,
    plot_args=None,
):
    """
    按 dataset 类别分文件夹保存每个 sample 的 flow 可视化图，并可选保存 LR/HR/SR 图片。

    plot_args 用于把 test_all 的一些可视化超参数往下传，例如：
        - displacement_cmap
        - method_label
        - twcf_profile_column_ratios
        - twcf_profile_region_names
    """
    plot_args = plot_args or {}
    displacement_cmap = str(plot_args.get("displacement_cmap", "viridis"))
    method_label = str(plot_args.get("method_label", "Current method"))
    twcf_profile_column_ratios = tuple(plot_args.get("twcf_profile_column_ratios", (0.15, 0.24, 0.83)))
    twcf_profile_region_names = tuple(plot_args.get("twcf_profile_region_names", ("Laminar", "Transition", "Turbulent")))
    if image_payload is not None:
        _save_image_outputs(dataset_dir, image_payload, start_index)

    for local_idx in range(predicted_np.shape[0]):
        sample_index = start_index + local_idx
        out_path = dataset_dir / f"{dataset_name}_sample_{sample_index:04d}.png"
        u_pred = predicted_np[local_idx, 0, :, :]
        v_pred = predicted_np[local_idx, 1, :, :]
        u_gt = flow_np[local_idx, 0, :, :]
        v_gt = flow_np[local_idx, 1, :, :]

        if dataset_name == "twcf":
            _plot_twcf(
                out_path,
                u_pred,
                v_pred,
                twcf_payload["piv_results"],
                twcf_payload["mask"],
                sample_index,
                cmap_name=displacement_cmap,
            )
            # 额外生成用户要求的论文风格 TWCF 剖面对比图，并把关键场/剖面存成 .npy。
            _save_twcf_profile_artifacts(
                dataset_dir=dataset_dir,
                sample_index=sample_index,
                u_pred=u_pred,
                u_gt=u_gt,
                mask_twcf=twcf_payload["mask"],
                method_label=method_label,
                cmap_name=displacement_cmap,
                column_ratios=twcf_profile_column_ratios,
                region_names=twcf_profile_region_names,
            )
        elif dataset_name == "tbl":
            _plot_tbl(out_path, u_pred, v_pred, u_gt, v_gt, cmap_name=displacement_cmap)
        else:
            _plot_regular(out_path, u_pred, v_pred, u_gt, v_gt)


def _write_csv(path, rows):
    """写入 dataset 级或 test_all 汇总指标。"""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_test_all(model, global_data, class_name, data_type, SCALE, device=None):
    """
    四个分支共享的 test_all 实现。

    行为：
    - 单 GPU，不启用 DDP/多进程；
    - 顺序测试 TEST_DATASETS 中定义的全部 dataset；
    - target 两通道按 prev/next 拆分，并用 data_downscal.py 生成 prev_lr/next_lr；
    - 输出目录为 OUT_PUT_DIR/class_name/data_type/scale_x/TEST_DIR/dataset_name。
    """
    if not getattr(global_data.esrgan, "IS_TEST", False):
        logger.info("[test_all] IS_TEST=False，跳过额外 TFRecord 全数据集测试。")
        return []

    if model is None:
        logger.warning("[test_all] model=None，跳过测试。")
        return []

    if hasattr(model, "module"):
        model = model.module

    device, device_id = _as_cuda_device(device or getattr(global_data.esrgan, "device", None))
    test_datasets = getattr(global_data.esrgan, "TEST_DATASETS", {})
    if not test_datasets:
        logger.warning("[test_all] TEST_DATASETS 为空，跳过测试。")
        return []

    if not getattr(global_data.esrgan, "is_TEST_CLASS3", False):
        # class3 对应 RAFT256-PIV_test.py 中的 tbl/twcf 大图数据集。
        # 默认跳过它们，避免 test_all 一打开就触发长时间全图滑窗测试。
        test_datasets = {
            name: cfg
            for name, cfg in test_datasets.items()
            if name not in {"tbl", "twcf"}
        }

    factor = _scale_factor_from_scale(SCALE)
    test_base_dir = Path(
        f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.TEST_DIR}"
    )
    test_base_dir.mkdir(parents=True, exist_ok=True)

    test_args = {
        "split_size": getattr(global_data.esrgan, "TEST_SPLIT_SIZE", 1),
        "offset": getattr(global_data.esrgan, "TEST_OFFSET", 256),
        "shift": getattr(global_data.esrgan, "TEST_SHIFT", 64),
        "amp": getattr(global_data.esrgan, "TEST_AMP", False),
        "plot_results": getattr(global_data.esrgan, "TEST_PLOT_RESULTS", True),
        # viridis 与用户提供的彩色位移色条最接近；这里用 getattr 预留后续配置入口，
        # 这样不改 global_class 也能工作，未来如果用户想换别的 cmap，可以直接在全局变量补同名字段。
        "displacement_cmap": getattr(global_data.esrgan, "TEST_DISPLACEMENT_CMAP", "viridis"),
        "method_label": getattr(global_data.esrgan, "name", "Current method"),
        "twcf_profile_column_ratios": getattr(global_data.esrgan, "TWCF_PROFILE_COLUMN_RATIOS", (0.15, 0.24, 0.83)),
        "twcf_profile_region_names": getattr(
            global_data.esrgan,
            "TWCF_PROFILE_REGION_NAMES",
            ("Laminar", "Transition", "Turbulent"),
        ),
    }

    twcf_payload = None
    if "twcf" in test_datasets:
        piv_path = _resolve_path(getattr(global_data.esrgan, "PIV_RESULTS_TWCF_PATH"), global_data)
        mask_path = _resolve_path(getattr(global_data.esrgan, "MASK_TWCF_PATH"), global_data)
        twcf_payload = {
            "piv_results": np.load(piv_path),
            "mask": np.load(mask_path),
        }

    was_training = model.training
    model.eval()
    summary_rows = []

    try:
        with torch.no_grad():
            for dataset_name, dataset_cfg in test_datasets.items():
                dataset_dir = test_base_dir / dataset_name
                dataset_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"[test_all] 开始测试 dataset={dataset_name}，输出目录：{dataset_dir}")

                test_iterator, dataset_size = _build_dali_iterator(dataset_cfg, global_data, device_id)
                loader_len = int(math.ceil(dataset_size / int(getattr(global_data.esrgan, "TEST_BATCH_SIZE", 1))))
                progress = tqdm(enumerate(test_iterator), total=loader_len, leave=False)

                height = int(dataset_cfg["image_height"])
                width = int(dataset_cfg["image_width"])
                results = np.empty((dataset_size, 4, height, width), dtype=np.float32)
                epe_array = np.empty((dataset_size,), dtype=np.float32)
                norm_aee_per100_array = np.empty((dataset_size,), dtype=np.float32)
                dataset_image_rows = []
                dataset_raft_rows = []

                total_samples = 0
                sum_epe = 0.0
                start_time = time.time()

                for i_batch, sample_batched in progress:
                    t0 = time.time()
                    local_dict = sample_batched[0]

                    # 与 RAFT256-PIV_test.py 保持一致：target / 256，且 target[0]=prev、target[1]=next。
                    images = local_dict["target"].type(torch.FloatTensor).cuda(device_id) / 256
                    flows = local_dict["flow"].type(torch.FloatTensor).cuda(device_id)

                    batch_size = int(images.size(0))
                    sample_start = i_batch * batch_size
                    valid_size = min(batch_size, dataset_size - sample_start)
                    if valid_size <= 0:
                        continue

                    images = images[:valid_size]
                    flows = flows[:valid_size]

                    with autocast(enabled=bool(test_args["amp"])):
                        if dataset_name in {"tbl", "twcf"}:
                            prediction = _predict_full_frame_with_folding(
                                model, images, flows, factor, device, test_args
                            )
                        else:
                            prediction = _predict_patch(model, images, flows, factor, device)
                        predicted_flows = prediction["flow"]

                    epe_per_sample = torch.sum((predicted_flows - flows) ** 2, dim=1).sqrt().flatten(1).mean(dim=1)
                    batch_epe = float(epe_per_sample.mean().item())
                    total_samples += valid_size
                    sum_epe += batch_epe * valid_size

                    predicted_np = predicted_flows.detach().cpu().numpy().astype(np.float32, copy=False)
                    flow_np = flows.detach().cpu().numpy().astype(np.float32, copy=False)
                    # 图像指标和 evaluate_all 保持一致：统一按 [0,1] 范围统计。
                    pred_prev_np = _as_numpy_batch(prediction["pred_prev"].clamp(0, 1))
                    pred_next_np = _as_numpy_batch(prediction["pred_next"].clamp(0, 1))
                    prev_hr_np = _as_numpy_batch(prediction["prev_hr"].clamp(0, 1))
                    next_hr_np = _as_numpy_batch(prediction["next_hr"].clamp(0, 1))
                    sample_end = sample_start + valid_size
                    results[sample_start:sample_end, 0, :, :] = predicted_np[:, 0, :, :]
                    results[sample_start:sample_end, 1, :, :] = predicted_np[:, 1, :, :]
                    results[sample_start:sample_end, 2, :, :] = flow_np[:, 0, :, :]
                    results[sample_start:sample_end, 3, :, :] = flow_np[:, 1, :, :]
                    epe_array[sample_start:sample_end] = epe_per_sample.detach().cpu().numpy().astype(np.float32)
                    # 逐样本补充 test_all 的 SR 图像指标和流场额外指标。
                    for local_idx in range(valid_size):
                        sample_index = sample_start + local_idx
                        dataset_image_rows.append(
                            _compute_image_metric_row(
                                dataset_name,
                                sample_index,
                                "previous",
                                pred_prev_np[local_idx],
                                prev_hr_np[local_idx],
                            )
                        )
                        dataset_image_rows.append(
                            _compute_image_metric_row(
                                dataset_name,
                                sample_index,
                                "next",
                                pred_next_np[local_idx],
                                next_hr_np[local_idx],
                            )
                        )
                        flow_row = _compute_flow_metric_row(
                            dataset_name,
                            sample_index,
                            predicted_np[local_idx],
                            flow_np[local_idx],
                        )
                        dataset_raft_rows.append(flow_row)
                        norm_aee_per100_array[sample_index] = float(flow_row["NORM_AEE_PER100PIXEL"])

                    if bool(test_args["plot_results"]):
                        _save_sample_plots(
                            dataset_name,
                            dataset_dir,
                            predicted_np,
                            flow_np,
                            sample_start,
                            twcf_payload if dataset_name == "twcf" else None,
                            image_payload=prediction,
                            plot_args=test_args,
                        )

                    logger.info(
                        f"[test_all] dataset={dataset_name}, batch={i_batch}, "
                        f"samples={total_samples}/{dataset_size}, mean_epe={sum_epe / total_samples:.6f}, "
                        f"time={time.time() - t0:.2f}s"
                    )

                mean_epe = float(np.mean(epe_array[:total_samples])) if total_samples else float("nan")
                elapsed = time.time() - start_time
                results_path = dataset_dir / "results.npy"
                epe_path = dataset_dir / "epe_array.npy"
                norm_aee_per100_path = dataset_dir / "norm_aee_per100_array.npy"
                np.save(results_path, results)
                np.save(epe_path, epe_array)
                np.save(norm_aee_per100_path, norm_aee_per100_array)

                # C-AEE 需要把 same sample 的 previous/next 图像 ESE 与该 sample 的 RAFT AEE 配对，
                # 并在当前 dataset 内做 min-max 归一化后再组合，因此必须放到整个 dataset 收集完成后统一回填。
                attach_c_aee_to_raft_rows(
                    image_rows=dataset_image_rows,
                    raft_rows=dataset_raft_rows,
                    sample_key_fields=("dataset", "sample_index"),
                    ese_key="energy_spectrum_mse",
                    aee_key="epe",
                    output_key="C_AEE",
                )

                image_metric_keys = ["mse", "psnr", "energy_spectrum_mse", "r2", "ssim", "tke_acc", "nrmse"]
                raft_metric_keys = ["epe", "NORM_AEE_PER100PIXEL", "C_AEE"]
                image_mean_row = _write_rows_with_mean(
                    dataset_dir / "metrics_image_pair.csv",
                    dataset_image_rows,
                    fixed_fields={
                        "dataset": dataset_name,
                        "sample_index": "MEAN",
                        "pair_type": "all",
                    },
                    metric_keys=image_metric_keys,
                )
                raft_mean_row = _write_rows_with_mean(
                    dataset_dir / "metrics_raft.csv",
                    dataset_raft_rows,
                    fixed_fields={
                        "dataset": dataset_name,
                        "sample_index": "MEAN",
                        "pair_type": "RAFT",
                    },
                    metric_keys=raft_metric_keys,
                )
                # 兼容原有只看 metrics.csv 的脚本：继续保留这个名字，并让它等同于 flow/RAFT 指标表。
                if dataset_raft_rows:
                    _write_rows_with_mean(
                        dataset_dir / "metrics.csv",
                        dataset_raft_rows,
                        fixed_fields={
                            "dataset": dataset_name,
                            "sample_index": "MEAN",
                            "pair_type": "RAFT",
                        },
                        metric_keys=raft_metric_keys,
                    )

                summary_rows.append(
                    {
                        "dataset": dataset_name,
                        "samples": total_samples,
                        "mean_epe": mean_epe,
                        "mean_norm_aee_per100pixel": (
                            float(raft_mean_row["NORM_AEE_PER100PIXEL"]) if raft_mean_row is not None else float("nan")
                        ),
                        "mean_c_aee": float(raft_mean_row["C_AEE"]) if raft_mean_row is not None else float("nan"),
                        "image_mse_mean": float(image_mean_row["mse"]) if image_mean_row is not None else float("nan"),
                        "image_psnr_mean": float(image_mean_row["psnr"]) if image_mean_row is not None else float("nan"),
                        "image_energy_spectrum_mse_mean": (
                            float(image_mean_row["energy_spectrum_mse"]) if image_mean_row is not None else float("nan")
                        ),
                        "image_r2_mean": float(image_mean_row["r2"]) if image_mean_row is not None else float("nan"),
                        "image_ssim_mean": float(image_mean_row["ssim"]) if image_mean_row is not None else float("nan"),
                        "image_tke_acc_mean": float(image_mean_row["tke_acc"]) if image_mean_row is not None else float("nan"),
                        "image_nrmse_mean": float(image_mean_row["nrmse"]) if image_mean_row is not None else float("nan"),
                        "elapsed_seconds": elapsed,
                        "results_npy": str(results_path),
                        "epe_npy": str(epe_path),
                        "norm_aee_per100_npy": str(norm_aee_per100_path),
                        "metrics_image_pair_csv": str(dataset_dir / "metrics_image_pair.csv"),
                        "metrics_raft_csv": str(dataset_dir / "metrics_raft.csv"),
                    }
                )
                logger.info(
                    f"[test_all] 完成 dataset={dataset_name}, samples={total_samples}, "
                    f"mean_epe={mean_epe:.6f}, "
                    f"mean_norm_aee_per100={summary_rows[-1]['mean_norm_aee_per100pixel']:.6f}, "
                    f"mean_c_aee={summary_rows[-1]['mean_c_aee']:.6f}, "
                    f"elapsed={elapsed:.2f}s"
                )

        _write_csv(test_base_dir / "metrics_all.csv", summary_rows)
        return summary_rows
    finally:
        if was_training:
            model.train()
