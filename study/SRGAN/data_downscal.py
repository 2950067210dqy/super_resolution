
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量对 .tif/.tiff 图像和 .flo 光流文件生成 LR 数据，并额外保存“对比产物”：
1) 若干 flo 原文件与其对应下采样 flo 的：
   - 可视化图像（HR/LR/配对图）
   - 完整矩阵信息文件（U/V 全矩阵）
2) 若干图像（tif/tiff）原图与下采样图的配对图片

输出规则：
- 不使用命令行参数，直接改脚本内配置；
- 输出目录为输入目录父目录下的 <输入目录名>_lr/；
- 不同倍率输出到 x4/、x8/ 等子目录；
- 输出文件名为 <原文件名>_lr.<ext>；
- 若 EXPAND_TO_ORIGINAL=True，则先下采样，再用最近邻 repeat 扩回原始 HxW。
"""

from __future__ import annotations
from loguru import logger

import struct
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover
    logger.error('需要 Pillow 才能读写 tif 文件：pip install pillow')
    raise RuntimeError("需要 Pillow 才能读写 tif 文件：pip install pillow") from exc


# =========================
# 配置（请在这里改）
# =========================
# 运行环境是否是autoDL
IS_AUTO_DL = True
INPUT_DIRS = [
    rf"D:\BaiduSyncdisk\AYanJiuSheng\data\sr_dataset\class_1",
    rf"D:\BaiduSyncdisk\AYanJiuSheng\data\sr_dataset\class_2",
] if not IS_AUTO_DL else [
    rf"/root/autodl-tmp/study_datas/sr_dataset/class_1",
    rf"/root/autodl-tmp/study_datas/sr_dataset/class_2",
]

#下采样倍数
FACTORS = [4, 8, 16]

# "maxpool" 或 "interpolate"
DOWNSAMPLE_METHOD = "interpolate"

# 仅在 DOWNSAMPLE_METHOD == "interpolate" 时生效："bilinear" 或 "bicubic"
INTERPOLATION_MODE = "bicubic"

# True: 先下采样再扩回原图尺寸；False: 保持下采样尺寸输出
EXPAND_TO_ORIGINAL = False

FLO_MAGIC = 202021.25

# ---------- 对比产物配置 ----------
SAVE_COMPARE_ARTIFACTS = True

# 保存几个 flo 源文件的对比产物（每个文件会按所有 factor 保存）
FLO_COMPARE_SAMPLES = 3

# 保存几个 tif/tiff 源文件的配对图（每个文件会按所有 factor 保存）
IMAGE_COMPARE_SAMPLES = 3

# 对比产物目录：<输入目录名>_lr/compare_artifacts/x{factor}/...
COMPARE_DIR_NAME = "compare_artifacts"


def crop_to_multiple_hw(arr: np.ndarray, factor: int) -> np.ndarray:
    """将数组最后两维(H,W)裁剪为factor的整数倍，避免下采样维度不整除。"""
    h, w = arr.shape[-2], arr.shape[-1]
    hh = (h // factor) * factor
    ww = (w // factor) * factor
    if hh == 0 or ww == 0:
        logger.error(f'factor={factor} 对形状 (...,{h},{w}) 太大，裁剪后会变成 0')
        raise ValueError(f"factor={factor} 对形状 (...,{h},{w}) 太大，裁剪后会变成 0")
    return arr[..., :hh, :ww]


def max_pool_last2(arr: np.ndarray, factor: int) -> np.ndarray:
    """对最后两维做factor×factor最大池化下采样。"""
    h, w = arr.shape[-2], arr.shape[-1]
    if h % factor != 0 or w % factor != 0:
        logger.error(f'形状 (...,{h},{w}) 不能被 factor={factor} 整除')
        raise ValueError(f"形状 (...,{h},{w}) 不能被 factor={factor} 整除")
    reshaped = arr.reshape(arr.shape[:-2] + (h // factor, factor, w // factor, factor))
    return reshaped.max(axis=(-1, -3))


def expand_back_with_repeat(arr: np.ndarray, factor: int, out_h: int, out_w: int) -> np.ndarray:
    """将下采样结果用最近邻repeat放大回指定尺寸(out_h,out_w)。"""
    expanded = np.repeat(np.repeat(arr, factor, axis=-2), factor, axis=-1)
    hh, ww = expanded.shape[-2], expanded.shape[-1]

    if hh < out_h:
        pad_h = out_h - hh
        expanded = np.concatenate(
            [expanded, np.repeat(expanded[..., -1:, :], pad_h, axis=-2)],
            axis=-2,
        )
    if ww < out_w:
        pad_w = out_w - ww
        expanded = np.concatenate(
            [expanded, np.repeat(expanded[..., :, -1:], pad_w, axis=-1)],
            axis=-1,
        )
    return expanded[..., :out_h, :out_w]


def _pil_resample(interpolation_mode: str) -> int:
    """把字符串插值模式映射为Pillow的重采样常量。"""
    interpolation_mode = interpolation_mode.lower()
    if interpolation_mode == "bilinear":
        return Image.Resampling.BILINEAR
    if interpolation_mode == "bicubic":
        return Image.Resampling.BICUBIC
    logger.error(f'不支持的 INTERPOLATION_MODE={interpolation_mode}，仅支持 bilinear / bicubic')
    raise ValueError(f"不支持的 INTERPOLATION_MODE={interpolation_mode}，仅支持 bilinear / bicubic")


def resize_2d_with_pillow(arr2d: np.ndarray, out_hw: tuple[int, int], interpolation_mode: str) -> np.ndarray:
    """使用Pillow对单通道2D数组按指定插值方式缩放到目标尺寸。"""
    out_h, out_w = out_hw
    if out_h <= 0 or out_w <= 0:
        logger.error(f'非法输出尺寸：{out_hw}')
        raise ValueError(f"非法输出尺寸：{out_hw}")

    src_dtype = arr2d.dtype
    resample = _pil_resample(interpolation_mode)

    if np.issubdtype(src_dtype, np.integer):
        pil_img = Image.fromarray(arr2d)
        resized = np.array(pil_img.resize((out_w, out_h), resample=resample))
        return resized.astype(src_dtype, copy=False)

    work = arr2d.astype(np.float32, copy=False)
    pil_img = Image.fromarray(work, mode="F")
    resized = np.array(pil_img.resize((out_w, out_h), resample=resample), dtype=np.float32)
    return resized.astype(src_dtype, copy=False)


def interpolate_last2(arr: np.ndarray, factor: int, interpolation_mode: str) -> np.ndarray:
    """对数组最后两维按factor插值下采样，支持多通道/多批次展平处理。"""

    h, w = arr.shape[-2], arr.shape[-1]
    out_h = h // factor
    out_w = w // factor
    if out_h == 0 or out_w == 0:
        logger.error(f'factor={factor} 对形状 (...,{h},{w}) 太大，下采样后会变成 0')
        raise ValueError(f"factor={factor} 对形状 (...,{h},{w}) 太大，下采样后会变成 0")

    flat = arr.reshape((-1, h, w))
    resized = [resize_2d_with_pillow(ch, (out_h, out_w), interpolation_mode) for ch in flat]
    return np.stack(resized, axis=0).reshape(arr.shape[:-2] + (out_h, out_w))


def downsample_hw(arr_hw_or_chw: np.ndarray, factor: int, expand_to_original: bool, method: str, interpolation_mode: str) -> np.ndarray:
    """按method(maxpool/interpolate)对最后两维下采样，可选再放大回原尺寸。"""
    h, w = arr_hw_or_chw.shape[-2], arr_hw_or_chw.shape[-1]
    method = method.lower()

    if method == "maxpool":
        ready = crop_to_multiple_hw(arr_hw_or_chw, factor)
        lr = max_pool_last2(ready, factor)
    elif method == "interpolate":
        lr = interpolate_last2(arr_hw_or_chw, factor, interpolation_mode)
    else:
        logger.error(f'不支持的 DOWNSAMPLE_METHOD={method}，仅支持 maxpool / interpolate')
        raise ValueError(f"不支持的 DOWNSAMPLE_METHOD={method}，仅支持 maxpool / interpolate")

    if not expand_to_original:
        return lr
    return expand_back_with_repeat(lr, factor, h, w)


def read_tif(path: Path) -> np.ndarray:
    """读取.tif/.tiff图像为numpy数组。"""
    with Image.open(path) as img:
        return np.array(img)


def write_tif(path: Path, arr: np.ndarray) -> None:
    """将numpy数组写入tif文件，自动创建父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def read_flo(path: Path) -> np.ndarray:
    """读取Middlebury .flo文件，返回HxWx2(float32)光流数组。"""
    with path.open("rb") as f:
        magic = struct.unpack("f", f.read(4))[0]
        if abs(magic - FLO_MAGIC) > 1e-4:
            logger.error(f'{path} 的 .flo magic 不正确：{magic}')
            raise ValueError(f"{path} 的 .flo magic 不正确：{magic}")
        w = struct.unpack("i", f.read(4))[0]
        h = struct.unpack("i", f.read(4))[0]
        data = np.fromfile(f, np.float32, count=2 * w * h)

    if data.size != 2 * w * h:
        logger.error(f'{path} 的 .flo 数据长度异常')
        raise ValueError(f"{path} 的 .flo 数据长度异常")
    return data.reshape(h, w, 2)


def write_flo(path: Path, flow: np.ndarray) -> None:
    """将HxWx2光流数组按.flo格式写盘。"""
    if flow.ndim != 3 or flow.shape[2] != 2:
        logger.error('flow 必须是 HxWx2')
        raise ValueError("flow 必须是 HxWx2")
    h, w, _ = flow.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("f", FLO_MAGIC))
        f.write(struct.pack("i", w))
        f.write(struct.pack("i", h))
        flow.astype(np.float32).tofile(f)


def downsample_tif(arr: np.ndarray, factor: int, expand_to_original: bool, method: str, interpolation_mode: str) -> np.ndarray:
    """对2D或3D图像执行下采样并保持原dtype。"""
    if arr.ndim == 2:
        lr = downsample_hw(arr, factor, expand_to_original, method, interpolation_mode)
        return lr.astype(arr.dtype, copy=False)

    if arr.ndim == 3:
        chw = np.moveaxis(arr, -1, 0)
        lr_chw = downsample_hw(chw, factor, expand_to_original, method, interpolation_mode)
        lr = np.moveaxis(lr_chw, 0, -1)
        return lr.astype(arr.dtype, copy=False)

    logger.error(f'不支持的 tif shape：{arr.shape}')
    raise ValueError(f"不支持的 tif shape：{arr.shape}")


def downsample_flo(flow: np.ndarray, factor: int, expand_to_original: bool, method: str, interpolation_mode: str) -> np.ndarray:
    """对HxWx2光流执行下采样并返回float32。"""
    chw = np.moveaxis(flow, -1, 0)
    lr_chw = downsample_hw(chw, factor, expand_to_original, method, interpolation_mode)
    lr = np.moveaxis(lr_chw, 0, -1)
    return lr.astype(np.float32, copy=False)


def iter_target_files(input_dirs: Iterable[Path]) -> Iterable[Path]:
    """递归遍历输入目录，产出.tif/.tiff/.flo文件路径。"""
    exts = {".tif", ".tiff", ".flo"}
    for root in input_dirs:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                yield p


def output_path_for(src: Path, input_root: Path, factor: int) -> Path:
    """生成下采样结果输出路径：<root>_lr/x{factor}/.../<stem>_lr<suffix>。"""
    output_root = input_root.parent / f"{input_root.name}_lr"
    rel_parent = src.parent.relative_to(input_root)
    target_dir = output_root / f"x{factor}" / rel_parent
    name = f"{src.stem}_lr{src.suffix.lower()}"
    return target_dir / name


def compare_dir_for(src: Path, input_root: Path, factor: int) -> Path:
    """生成对比产物目录路径：.../compare_artifacts/x{factor}/...。"""
    output_root = input_root.parent / f"{input_root.name}_lr"
    rel_parent = src.parent.relative_to(input_root)
    safe_suffix = src.suffix.lower().replace(".", "")
    return output_root / COMPARE_DIR_NAME / f"x{factor}" / rel_parent / f"{src.stem}_{safe_suffix}"


def resize_nearest_image(arr: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """用最近邻插值缩放2D/3D图像到目标尺寸。"""
    out_h, out_w = out_hw
    if arr.ndim == 2:
        pil = Image.fromarray(arr)
        return np.array(pil.resize((out_w, out_h), resample=Image.Resampling.NEAREST))
    if arr.ndim == 3:
        pil = Image.fromarray(arr)
        return np.array(pil.resize((out_w, out_h), resample=Image.Resampling.NEAREST))
    logger.error(f'不支持 resize 的图像 shape: {arr.shape}')
    raise ValueError(f"不支持 resize 的图像 shape: {arr.shape}")


def to_uint8_preview(arr: np.ndarray) -> np.ndarray:
    """
    将 2D/3D 图像转成 HxWx3 uint8 预览图
    """
    if arr.ndim == 2:
        x = arr.astype(np.float32, copy=False)
        mn, mx = float(np.min(x)), float(np.max(x))
        if mx > mn:
            x = (x - mn) / (mx - mn)
        else:
            x = np.zeros_like(x, dtype=np.float32)
        g = (x * 255.0).clip(0, 255).astype(np.uint8)
        return np.stack([g, g, g], axis=-1)

    if arr.ndim == 3:
        if arr.shape[2] == 1:
            return np.repeat(to_uint8_preview(arr[..., 0])[..., :1], 3, axis=2)
        if arr.shape[2] >= 3:
            x = arr[..., :3].astype(np.float32, copy=False)
            mn, mx = float(np.min(x)), float(np.max(x))
            if mx > mn:
                x = (x - mn) / (mx - mn)
            else:
                x = np.zeros_like(x, dtype=np.float32)
            return (x * 255.0).clip(0, 255).astype(np.uint8)

    logger.error(f'不支持预览转换的图像 shape: {arr.shape}')
    raise ValueError(f"不支持预览转换的图像 shape: {arr.shape}")


def hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    将HSV(0~1)转换为RGB(0~1)。
    h,s,v in [0,1], return rgb in [0,1], shape HxWx3
    """
    h = np.mod(h, 1.0)
    i = np.floor(h * 6.0).astype(np.int32)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    i_mod = i % 6
    r = np.empty_like(v)
    g = np.empty_like(v)
    b = np.empty_like(v)

    m = i_mod == 0
    r[m], g[m], b[m] = v[m], t[m], p[m]
    m = i_mod == 1
    r[m], g[m], b[m] = q[m], v[m], p[m]
    m = i_mod == 2
    r[m], g[m], b[m] = p[m], v[m], t[m]
    m = i_mod == 3
    r[m], g[m], b[m] = p[m], q[m], v[m]
    m = i_mod == 4
    r[m], g[m], b[m] = t[m], p[m], v[m]
    m = i_mod == 5
    r[m], g[m], b[m] = v[m], p[m], q[m]

    return np.stack([r, g, b], axis=-1)


def flow_to_color(flow: np.ndarray, mag_clip_percentile: float = 99.0) -> np.ndarray:
    """
    将HxWx2光流转换为HSV色轮可视化RGB图(uint8)。
    将 HxWx2 光流可视化为 RGB(uint8)
    """
    if flow.ndim != 3 or flow.shape[2] != 2:
        logger.error(f'flow 必须是 HxWx2，实际: {flow.shape}')
        raise ValueError(f"flow 必须是 HxWx2，实际: {flow.shape}")

    u = flow[..., 0].astype(np.float32, copy=False)
    v = flow[..., 1].astype(np.float32, copy=False)
    mag = np.sqrt(u * u + v * v)
    ang = np.arctan2(v, u)  # [-pi, pi]

    h = (ang + np.pi) / (2.0 * np.pi)
    s = np.ones_like(h, dtype=np.float32)

    clip_val = float(np.percentile(mag, mag_clip_percentile))
    clip_val = max(clip_val, 1e-6)
    val = np.clip(mag / clip_val, 0.0, 1.0)

    rgb = hsv_to_rgb(h, s, val)
    return (rgb * 255.0).clip(0, 255).astype(np.uint8)


def save_pair_png(left_rgb: np.ndarray, right_rgb: np.ndarray, out_path: Path) -> None:
    """左右拼接两张RGB图并保存为PNG。"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if left_rgb.shape[0] != right_rgb.shape[0]:
        right_rgb = resize_nearest_image(right_rgb, (left_rgb.shape[0], right_rgb.shape[1]))
    pair = np.concatenate([left_rgb, right_rgb], axis=1)
    Image.fromarray(pair).save(out_path)


def save_flo_matrix_txt(path: Path, flow: np.ndarray, title: str) -> None:
    """将光流U/V通道完整矩阵以文本形式保存，便于核对数值。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write(f"shape={flow.shape}, dtype={flow.dtype}\n\n")
        with np.printoptions(threshold=np.inf, linewidth=240, precision=7, suppress=False):
            f.write("U channel (flow[..., 0]):\n")
            f.write(np.array2string(flow[..., 0], separator=", "))
            f.write("\n\nV channel (flow[..., 1]):\n")
            f.write(np.array2string(flow[..., 1], separator=", "))
            f.write("\n")


def save_flo_artifacts(src: Path, in_root: Path, factor: int, hr_flow: np.ndarray, lr_flow: np.ndarray) -> None:
    """
    保存flo对比产物：HR/LR可视化、配对图、矩阵txt与npy及说明文件。
    保存 flo 对比产物：
    - HR/LR 可视化图
    - HR vs LR(上采样到HR尺寸) 配对图
    - HR/LR 完整矩阵 txt + npy
    """
    cdir = compare_dir_for(src, in_root, factor)
    cdir.mkdir(parents=True, exist_ok=True)

    hr_vis = flow_to_color(hr_flow)
    lr_vis = flow_to_color(lr_flow)

    h, w = hr_flow.shape[:2]
    lr_vis_up = resize_nearest_image(lr_vis, (h, w))

    Image.fromarray(hr_vis).save(cdir / "flo_hr_vis.png")
    Image.fromarray(lr_vis).save(cdir / "flo_lr_vis.png")
    save_pair_png(hr_vis, lr_vis_up, cdir / "flo_pair_hr_vs_lr_up.png")

    save_flo_matrix_txt(cdir / "flo_hr_matrix.txt", hr_flow, title="HR flow full matrix")
    save_flo_matrix_txt(cdir / "flo_lr_matrix.txt", lr_flow, title="LR flow full matrix")

    np.save(cdir / "flo_hr_matrix.npy", hr_flow.astype(np.float32, copy=False))
    np.save(cdir / "flo_lr_matrix.npy", lr_flow.astype(np.float32, copy=False))

    with (cdir / "readme.txt").open("w", encoding="utf-8") as f:
        f.write(f"source: {src}\n")
        f.write(f"factor: x{factor}\n")
        f.write("files:\n")
        f.write("- flo_hr_vis.png\n")
        f.write("- flo_lr_vis.png\n")
        f.write("- flo_pair_hr_vs_lr_up.png\n")
        f.write("- flo_hr_matrix.txt / flo_lr_matrix.txt (完整矩阵)\n")
        f.write("- flo_hr_matrix.npy / flo_lr_matrix.npy\n")


def save_image_pair_artifacts(src: Path, in_root: Path, factor: int, hr_img: np.ndarray, lr_img: np.ndarray) -> None:
    """
    保存图像对比产物：HR预览、LR预览、以及HR vs 上采样LR配对图
    保存图像对配对图（HR vs LR上采样到HR尺寸）
    """
    cdir = compare_dir_for(src, in_root, factor)
    cdir.mkdir(parents=True, exist_ok=True)

    hr_prev = to_uint8_preview(hr_img)
    lr_prev = to_uint8_preview(lr_img)
    lr_prev_up = resize_nearest_image(lr_prev, (hr_prev.shape[0], hr_prev.shape[1]))

    Image.fromarray(hr_prev).save(cdir / "img_hr_preview.png")
    Image.fromarray(lr_prev).save(cdir / "img_lr_preview.png")
    save_pair_png(hr_prev, lr_prev_up, cdir / "img_pair_hr_vs_lr_up.png")


def process_one_file(
    src: Path,
    in_root: Path,
    factors: list[int],
    save_flo_compare: bool,
    save_img_compare: bool,
) -> None:
    """按文件类型(tif/tiff/flo)执行多倍率下采样与可选对比产物保存。"""
    ext = src.suffix.lower()

    if ext in {".tif", ".tiff"}:
        arr = read_tif(src)
        for factor in factors:
            out = output_path_for(src, in_root, factor)
            lr = downsample_tif(arr, factor, EXPAND_TO_ORIGINAL, DOWNSAMPLE_METHOD, INTERPOLATION_MODE)
            write_tif(out, lr)
            logger.info(f"[OK] {src} -> {out} (x{factor}, method={DOWNSAMPLE_METHOD}, {arr.shape} -> {lr.shape})")

            if SAVE_COMPARE_ARTIFACTS and save_img_compare:
                save_image_pair_artifacts(src, in_root, factor, arr, lr)
        return

    if ext == ".flo":
        flow = read_flo(src)
        for factor in factors:
            out = output_path_for(src, in_root, factor)
            lr = downsample_flo(flow, factor, EXPAND_TO_ORIGINAL, DOWNSAMPLE_METHOD, INTERPOLATION_MODE)
            write_flo(out, lr)
            logger.info(f"[OK] {src} -> {out} (x{factor}, method={DOWNSAMPLE_METHOD}, {flow.shape} -> {lr.shape})")

            if SAVE_COMPARE_ARTIFACTS and save_flo_compare:
                save_flo_artifacts(src, in_root, factor, flow, lr)
        return

    logger.warning(f"[SKIP] 不支持的文件类型：{src}")


def _validate_config() -> tuple[list[Path], list[int]]:
    """校验配置有效性并返回规范化后的输入目录列表与倍率列表。"""
    if not INPUT_DIRS:
        logger.error('请在脚本里设置 INPUT_DIRS（至少一个文件夹路径）')
        raise ValueError("请在脚本里设置 INPUT_DIRS（至少一个文件夹路径）")

    input_dirs = [Path(p).expanduser().resolve() for p in INPUT_DIRS]
    for d in input_dirs:
        if not d.exists() or not d.is_dir():
            logger.error(f'输入目录不存在或不是目录：{d}')
            raise ValueError(f"输入目录不存在或不是目录：{d}")

    factors = [int(x) for x in FACTORS]
    if not factors or any(x < 1 for x in factors):
        logger.error(f'FACTORS 配置不合法：{FACTORS}')
        raise ValueError(f"FACTORS 配置不合法：{FACTORS}")

    method = DOWNSAMPLE_METHOD.lower()
    if method not in {"maxpool", "interpolate"}:
        logger.error(f'DOWNSAMPLE_METHOD 配置不合法：{DOWNSAMPLE_METHOD}，仅支持 maxpool / interpolate')
        raise ValueError(f"DOWNSAMPLE_METHOD 配置不合法：{DOWNSAMPLE_METHOD}，仅支持 maxpool / interpolate")

    if method == "interpolate" and INTERPOLATION_MODE.lower() not in {"bilinear", "bicubic"}:
        logger.error('INTERPOLATION_MODE 配置不合法，仅支持 bilinear / bicubic')
        raise ValueError("INTERPOLATION_MODE 配置不合法，仅支持 bilinear / bicubic")

    if FLO_COMPARE_SAMPLES < 0 or IMAGE_COMPARE_SAMPLES < 0:
        logger.error('FLO_COMPARE_SAMPLES / IMAGE_COMPARE_SAMPLES 不能小于 0')
        raise ValueError("FLO_COMPARE_SAMPLES / IMAGE_COMPARE_SAMPLES 不能小于 0")

    return input_dirs, factors


def main() -> None:
    """主流程：遍历输入目录、处理文件、统计并打印汇总结果。"""
    logger.add(
        f"./data_downscal/log/running.log",
        rotation="100 MB",
        retention="30 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {process.name} | {thread.name} | {name}:{module}:{line} | {message}",
        enqueue=True,
        backtrace=True,
        diagnose=True,

    )
    input_dirs, factors = _validate_config()

    total = 0
    flo_compare_count = 0
    img_compare_count = 0

    for in_root in input_dirs:
        for src in iter_target_files([in_root]):
            ext = src.suffix.lower()

            save_flo_compare = False
            save_img_compare = False

            if ext == ".flo" and flo_compare_count < FLO_COMPARE_SAMPLES:
                save_flo_compare = True
                flo_compare_count += 1

            if ext in {".tif", ".tiff"} and img_compare_count < IMAGE_COMPARE_SAMPLES:
                save_img_compare = True
                img_compare_count += 1

            process_one_file(
                src=src,
                in_root=in_root,
                factors=factors,
                save_flo_compare=save_flo_compare,
                save_img_compare=save_img_compare,
            )
            total += 1

    logger.info(f"[DONE] 处理完成，源文件数量：{total}")
    logger.info(f"[DONE] 下采样方式：{DOWNSAMPLE_METHOD}")
    if DOWNSAMPLE_METHOD.lower() == "interpolate":
        logger.info(f"[DONE] 插值模式：{INTERPOLATION_MODE}")

    logger.info(f"[DONE] 保存 flo 对比产物的源文件数：{flo_compare_count} (上限={FLO_COMPARE_SAMPLES})")
    logger.info(f"[DONE] 保存图像对配对图的源文件数：{img_compare_count} (上限={IMAGE_COMPARE_SAMPLES})")

    for in_root in input_dirs:
        logger.info(f"[DONE] 输出根目录：{in_root.parent / (in_root.name + '_lr')}")
        logger.info(f"[DONE] 对比产物目录：{in_root.parent / (in_root.name + '_lr') / COMPARE_DIR_NAME}")


if __name__ == "__main__":
    main()
