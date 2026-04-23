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
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

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
    """
    prev_hr = images_hr[:, 0:1, :, :]
    next_hr = images_hr[:, 1:2, :, :]
    prev_lr = _make_lr_from_hr_tensor(prev_hr, factor, device)
    next_lr = _make_lr_from_hr_tensor(next_hr, factor, device)

    pred_prev, pred_next, flow_predictions, _ = model(
        input_lr_prev=prev_lr,
        input_lr_next=next_lr,
        input_gr_prev=prev_hr,
        input_gr_next=next_hr,
        flowl0=flows_hr,
        flow_init=flow_init,
        is_adversarial=False,
    )
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


def _plot_twcf(out_path, u_pred, v_pred, piv_results, mask_twcf, sample_index):
    """保存 TWCF 的 PascalPIV 对比图，布局沿用 RAFT256-PIV_test.py。"""
    u_pascal = piv_results[sample_index, 0, :, :]
    v_pascal = piv_results[sample_index, 1, :, :]

    plt.figure(num=None, figsize=(24, 16), dpi=120, facecolor="w", edgecolor="k")
    plt.subplot(2, 2, 1)
    plt.pcolor(np.squeeze(u_pascal), cmap="Greys", vmin=-2, vmax=12)
    plt.axis("off")
    plt.colorbar().ax.set_ylabel("displacement [px]", fontsize=14)
    plt.subplot(2, 2, 2)
    plt.pcolor(np.squeeze(v_pascal), cmap="Greys", vmin=-1, vmax=1)
    plt.axis("off")
    plt.colorbar().ax.set_ylabel("displacement [px]", fontsize=14)
    plt.subplot(2, 2, 3)
    plt.pcolor(u_pred * mask_twcf, cmap="Greys", vmin=-2, vmax=12)
    plt.axis("off")
    plt.colorbar().ax.set_ylabel("displacement [px]", fontsize=14)
    plt.subplot(2, 2, 4)
    plt.pcolor(v_pred * mask_twcf, cmap="Greys", vmin=-1, vmax=1)
    plt.axis("off")
    plt.colorbar().ax.set_ylabel("displacement [px]", fontsize=14)
    plt.savefig(out_path)
    plt.close()


def _plot_tbl(out_path, u_pred, v_pred, u_gt, v_gt):
    """保存 TBL 全图测试的预测/真值对比图。"""
    plt.figure(num=None, figsize=(24, 16), dpi=120, facecolor="w", edgecolor="k")
    plt.subplot(2, 2, 1)
    plt.pcolor(np.squeeze(u_pred), cmap="Greys", vmin=2, vmax=8)
    plt.axis("off")
    plt.colorbar().ax.set_ylabel("displacement [px]", fontsize=14)
    plt.subplot(2, 2, 2)
    plt.pcolor(np.squeeze(v_pred), cmap="Greys", vmin=-0.5, vmax=0.5)
    plt.axis("off")
    plt.colorbar().ax.set_ylabel("displacement [px]", fontsize=14)
    plt.subplot(2, 2, 3)
    plt.pcolor(u_gt, cmap="Greys", vmin=2, vmax=8)
    plt.axis("off")
    plt.colorbar().ax.set_ylabel("displacement [px]", fontsize=14)
    plt.subplot(2, 2, 4)
    plt.pcolor(v_gt, cmap="Greys", vmin=-0.5, vmax=0.5)
    plt.axis("off")
    plt.colorbar().ax.set_ylabel("displacement [px]", fontsize=14)
    plt.savefig(out_path)
    plt.close()


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


def _save_sample_plots(dataset_name, dataset_dir, predicted_np, flow_np, start_index, twcf_payload, image_payload=None):
    """按 dataset 类别分文件夹保存每个 sample 的 flow 可视化图，并可选保存 LR/HR/SR 图片。"""
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
            _plot_twcf(out_path, u_pred, v_pred, twcf_payload["piv_results"], twcf_payload["mask"], sample_index)
        elif dataset_name == "tbl":
            _plot_tbl(out_path, u_pred, v_pred, u_gt, v_gt)
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
                    sample_end = sample_start + valid_size
                    results[sample_start:sample_end, 0, :, :] = predicted_np[:, 0, :, :]
                    results[sample_start:sample_end, 1, :, :] = predicted_np[:, 1, :, :]
                    results[sample_start:sample_end, 2, :, :] = flow_np[:, 0, :, :]
                    results[sample_start:sample_end, 3, :, :] = flow_np[:, 1, :, :]
                    epe_array[sample_start:sample_end] = epe_per_sample.detach().cpu().numpy().astype(np.float32)

                    if bool(test_args["plot_results"]):
                        _save_sample_plots(
                            dataset_name,
                            dataset_dir,
                            predicted_np,
                            flow_np,
                            sample_start,
                            twcf_payload if dataset_name == "twcf" else None,
                            image_payload=prediction,
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
                np.save(results_path, results)
                np.save(epe_path, epe_array)

                dataset_rows = [
                    {
                        "dataset": dataset_name,
                        "sample_index": i,
                        "epe": float(epe_array[i]),
                    }
                    for i in range(total_samples)
                ]
                _write_csv(dataset_dir / "metrics.csv", dataset_rows)

                summary_rows.append(
                    {
                        "dataset": dataset_name,
                        "samples": total_samples,
                        "mean_epe": mean_epe,
                        "elapsed_seconds": elapsed,
                        "results_npy": str(results_path),
                        "epe_npy": str(epe_path),
                    }
                )
                logger.info(
                    f"[test_all] 完成 dataset={dataset_name}, samples={total_samples}, "
                    f"mean_epe={mean_epe:.6f}, elapsed={elapsed:.2f}s"
                )

        _write_csv(test_base_dir / "metrics_all.csv", summary_rows)
        return summary_rows
    finally:
        if was_training:
            model.train()
