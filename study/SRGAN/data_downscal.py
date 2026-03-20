#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量对 .tif/.tiff 图像和 .flo 光流文件生成 LR 数据。

支持两种下采样方式：
1) 最大池化：`DOWNSAMPLE_METHOD = "maxpool"`
2) 插值下采样：`DOWNSAMPLE_METHOD = "interpolate"`
   - 插值模式可选：`INTERPOLATION_MODE = "bilinear"` 或 `"bicubic"`

输出规则：
1) 不使用命令行参数，直接改脚本内配置；
2) 输出目录为输入目录父目录下的 `<输入目录名>_lr/`；
3) 不同倍率输出到 `x4/`、`x8/` 等子目录；
4) 输出文件名为 `<原文件名>_lr.<ext>`；
5) 若 `EXPAND_TO_ORIGINAL=True`，则先下采样，再用最近邻 repeat 扩回原始 HxW；
   对不能整除 factor 的边缘，复制最后一行/列补齐到原始尺寸。
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover
    raise RuntimeError("需要 Pillow 才能读写 tif 文件：pip install pillow") from exc


# =========================
# 配置（请在这里改）
# =========================

INPUT_DIRS = [
    r"D:\BaiduSyncdisk\AYanJiuSheng\data\sr_dataset\class_1",
    r"D:\BaiduSyncdisk\AYanJiuSheng\data\sr_dataset\class_2",
]

FACTORS = [4, 8, 16]

# "maxpool" 或 "interpolate"
DOWNSAMPLE_METHOD = "interpolate"

# 仅在 DOWNSAMPLE_METHOD == "interpolate" 时生效：
# "bilinear" 或 "bicubic"
INTERPOLATION_MODE = "bicubic"

EXPAND_TO_ORIGINAL = False

FLO_MAGIC = 202021.25


def crop_to_multiple_hw(arr: np.ndarray, factor: int) -> np.ndarray:
    h, w = arr.shape[-2], arr.shape[-1]
    hh = (h // factor) * factor
    ww = (w // factor) * factor
    if hh == 0 or ww == 0:
        raise ValueError(f"factor={factor} 对形状 (...,{h},{w}) 太大，裁剪后会变成 0")
    return arr[..., :hh, :ww]


def max_pool_last2(arr: np.ndarray, factor: int) -> np.ndarray:
    h, w = arr.shape[-2], arr.shape[-1]
    if h % factor != 0 or w % factor != 0:
        raise ValueError(f"形状 (...,{h},{w}) 不能被 factor={factor} 整除")
    reshaped = arr.reshape(arr.shape[:-2] + (h // factor, factor, w // factor, factor))
    return reshaped.max(axis=(-1, -3))


def expand_back_with_repeat(arr: np.ndarray, factor: int, out_h: int, out_w: int) -> np.ndarray:
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
    interpolation_mode = interpolation_mode.lower()
    if interpolation_mode == "bilinear":
        return Image.Resampling.BILINEAR
    if interpolation_mode == "bicubic":
        return Image.Resampling.BICUBIC
    raise ValueError(
        f"不支持的 INTERPOLATION_MODE={interpolation_mode}，仅支持 bilinear / bicubic"
    )


def resize_2d_with_pillow(arr2d: np.ndarray, out_hw: tuple[int, int], interpolation_mode: str) -> np.ndarray:
    out_h, out_w = out_hw
    if out_h <= 0 or out_w <= 0:
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
    h, w = arr.shape[-2], arr.shape[-1]
    out_h = h // factor
    out_w = w // factor
    if out_h == 0 or out_w == 0:
        raise ValueError(f"factor={factor} 对形状 (...,{h},{w}) 太大，下采样后会变成 0")

    flat = arr.reshape((-1, h, w))
    resized = [
        resize_2d_with_pillow(channel_2d, (out_h, out_w), interpolation_mode)
        for channel_2d in flat
    ]
    return np.stack(resized, axis=0).reshape(arr.shape[:-2] + (out_h, out_w))


def downsample_hw(
    arr_hw_or_chw: np.ndarray,
    factor: int,
    expand_to_original: bool,
    method: str,
    interpolation_mode: str,
) -> np.ndarray:
    h, w = arr_hw_or_chw.shape[-2], arr_hw_or_chw.shape[-1]
    method = method.lower()

    if method == "maxpool":
        ready = crop_to_multiple_hw(arr_hw_or_chw, factor)
        lr = max_pool_last2(ready, factor)
    elif method == "interpolate":
        lr = interpolate_last2(arr_hw_or_chw, factor, interpolation_mode)
    else:
        raise ValueError(f"不支持的 DOWNSAMPLE_METHOD={method}，仅支持 maxpool / interpolate")

    if not expand_to_original:
        return lr

    return expand_back_with_repeat(lr, factor, h, w)


def read_tif(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img)


def write_tif(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def read_flo(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        magic = struct.unpack("f", f.read(4))[0]
        if abs(magic - FLO_MAGIC) > 1e-4:
            raise ValueError(f"{path} 的 .flo magic 不正确：{magic}")
        w = struct.unpack("i", f.read(4))[0]
        h = struct.unpack("i", f.read(4))[0]
        data = np.fromfile(f, np.float32, count=2 * w * h)
    if data.size != 2 * w * h:
        raise ValueError(f"{path} 的 .flo 数据长度异常")
    return data.reshape(h, w, 2)


def write_flo(path: Path, flow: np.ndarray) -> None:
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError("flow 必须是 HxWx2")
    h, w, _ = flow.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("f", FLO_MAGIC))
        f.write(struct.pack("i", w))
        f.write(struct.pack("i", h))
        flow.astype(np.float32).tofile(f)


def downsample_tif(
    arr: np.ndarray,
    factor: int,
    expand_to_original: bool,
    method: str,
    interpolation_mode: str,
) -> np.ndarray:
    if arr.ndim == 2:
        lr = downsample_hw(arr, factor, expand_to_original, method, interpolation_mode)
        return lr.astype(arr.dtype, copy=False)

    if arr.ndim == 3:
        chw = np.moveaxis(arr, -1, 0)
        lr_chw = downsample_hw(chw, factor, expand_to_original, method, interpolation_mode)
        lr = np.moveaxis(lr_chw, 0, -1)
        return lr.astype(arr.dtype, copy=False)

    raise ValueError(f"不支持的 tif shape：{arr.shape}")


def downsample_flo(
    flow: np.ndarray,
    factor: int,
    expand_to_original: bool,
    method: str,
    interpolation_mode: str,
) -> np.ndarray:
    chw = np.moveaxis(flow, -1, 0)
    lr_chw = downsample_hw(chw, factor, expand_to_original, method, interpolation_mode)
    lr = np.moveaxis(lr_chw, 0, -1)
    return lr.astype(np.float32, copy=False)


def iter_target_files(input_dirs: Iterable[Path]) -> Iterable[Path]:
    exts = {".tif", ".tiff", ".flo"}
    for root in input_dirs:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                yield p


def output_path_for(src: Path, input_root: Path, factor: int) -> Path:
    output_root = input_root.parent / f"{input_root.name}_lr"
    rel_parent = src.parent.relative_to(input_root)
    target_dir = output_root / f"x{factor}" / rel_parent
    name = f"{src.stem}_lr{src.suffix.lower()}"
    return target_dir / name


def process_one_file(src: Path, in_root: Path, factors: list[int]) -> None:
    ext = src.suffix.lower()
    if ext in {".tif", ".tiff"}:
        arr = read_tif(src)
        for factor in factors:
            out = output_path_for(src, in_root, factor)
            lr = downsample_tif(
                arr,
                factor,
                EXPAND_TO_ORIGINAL,
                DOWNSAMPLE_METHOD,
                INTERPOLATION_MODE,
            )
            write_tif(out, lr)
            print(
                f"[OK] {src} -> {out} "
                f"(x{factor}, method={DOWNSAMPLE_METHOD}, {arr.shape} -> {lr.shape})"
            )
        return

    if ext == ".flo":
        flow = read_flo(src)
        for factor in factors:
            out = output_path_for(src, in_root, factor)
            lr = downsample_flo(
                flow,
                factor,
                EXPAND_TO_ORIGINAL,
                DOWNSAMPLE_METHOD,
                INTERPOLATION_MODE,
            )
            write_flo(out, lr)
            print(
                f"[OK] {src} -> {out} "
                f"(x{factor}, method={DOWNSAMPLE_METHOD}, {flow.shape} -> {lr.shape})"
            )
        return

    print(f"[SKIP] 不支持的文件类型：{src}")


def _validate_config() -> tuple[list[Path], list[int]]:
    if not INPUT_DIRS:
        raise ValueError("请在脚本里设置 INPUT_DIRS（至少一个文件夹路径）")

    input_dirs = [Path(p).expanduser().resolve() for p in INPUT_DIRS]
    for d in input_dirs:
        if not d.exists() or not d.is_dir():
            raise ValueError(f"输入目录不存在或不是目录：{d}")

    factors = [int(x) for x in FACTORS]
    if not factors or any(x < 1 for x in factors):
        raise ValueError(f"FACTORS 配置不合法：{FACTORS}")

    method = DOWNSAMPLE_METHOD.lower()
    if method not in {"maxpool", "interpolate"}:
        raise ValueError(
            f"DOWNSAMPLE_METHOD 配置不合法：{DOWNSAMPLE_METHOD}，仅支持 maxpool / interpolate"
        )

    if method == "interpolate" and INTERPOLATION_MODE.lower() not in {"bilinear", "bicubic"}:
        raise ValueError(
            "INTERPOLATION_MODE 配置不合法，仅支持 bilinear / bicubic"
        )

    return input_dirs, factors


def main() -> None:
    input_dirs, factors = _validate_config()

    total = 0
    for in_root in input_dirs:
        for src in iter_target_files([in_root]):
            process_one_file(src, in_root, factors)
            total += 1

    print(f"[DONE] 处理完成，源文件数量：{total}")
    print(f"[DONE] 下采样方式：{DOWNSAMPLE_METHOD}")
    if DOWNSAMPLE_METHOD.lower() == "interpolate":
        print(f"[DONE] 插值模式：{INTERPOLATION_MODE}")
    for in_root in input_dirs:
        print(f"[DONE] 输出根目录：{in_root.parent / (in_root.name + '_lr')}")


if __name__ == "__main__":
    main()
