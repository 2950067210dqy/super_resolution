#!/usr/bin/env python3  # 使用系统环境中的 python3 来运行脚本
# -*- coding: utf-8 -*-  # 声明源码编码为 UTF-8（便于中文注释）
"""
批量对 .tif 图像和 .flo 光流文件做“最大池化（max-pooling）下采样”，并把结果再扩充回原始分辨率。  # 脚本说明（中文）

核心需求（按你的描述实现）：  # 功能概述
1) 不使用命令行参数（argparse），而是用脚本内变量赋值配置；  # 配置方式
2) 输出目录在输入目录的“父目录”下新建一个同名 + `_lr` 的文件夹（同级目录）；  # 输出位置规则
   例如：输入根目录为 `...\cliny\`，输出根目录为 `...\cliny_lr\`；  # 示例
3) 同一份输入可生成多个倍率（例如 x4、x8），为避免同名冲突，按倍率分到子目录 `x4/`、`x8/`；  # 倍率分目录
4) 输出文件名为 `<原文件名>_lr.<ext>`；  # 文件命名规则
5) 生成的 LR 结果会“扩充回原始 HxW shape”，确保输出 shape 与输入一致：  # shape 一致规则
   - 先 max-pool 下采样；  # 第一步
   - 再用最近邻 repeat 扩回；  # 第二步
   - 若原始尺寸不能整除 factor，则对边缘做复制填充补齐到原始尺寸。  # 边缘补齐策略
"""  # 文档字符串结束

from __future__ import annotations  # 允许在类型注解中使用更现代的写法（例如前向引用）

import struct  # 用于按二进制结构读写 .flo 头部
from pathlib import Path  # 用于跨平台路径处理
from typing import Iterable  # 用于类型注解：可迭代对象

import numpy as np  # 数值计算（池化、repeat、拼接等）

try:  # 尝试导入 Pillow，用于读写 tif/tiff
    from PIL import Image  # Pillow 图像读写
except Exception as exc:  # pragma: no cover  # 如果 Pillow 不存在或导入失败
    raise RuntimeError("需要 Pillow 才能读写 tif 文件：pip install pillow") from exc  # 抛出更友好的错误信息


# =========================  # 分隔线：配置区
# 配置（请在这里改）  # 配置说明
# =========================  # 分隔线：配置区

INPUT_DIRS = [  # 输入根目录列表（会递归遍历其中的 tif/tiff/flo 文件）
    r"D:\BaiduSyncdisk\AYanJiuSheng\data\sr_dataset\class_1",
    r"D:\BaiduSyncdisk\AYanJiuSheng\data\sr_dataset\class_2",
]  # 输入根目录列表结束

FACTORS = [4, 8,16]  # 下采样倍率列表，例如 [4, 8] 表示生成 x4 与 x8 两套结果

EXPAND_TO_ORIGINAL = False  # 是否把下采样结果“扩充回原始 HxW”（True：输出与输入同 shape；False：输出为真正的低分辨率）


FLO_MAGIC = 202021.25  # .flo 文件格式的 magic number（Middlebury optical flow）


def crop_to_multiple_hw(arr: np.ndarray, factor: int) -> np.ndarray:  # 把数组在 H/W 方向裁剪到 factor 的整数倍
    h, w = arr.shape[-2], arr.shape[-1]  # 取最后两维作为高度 H 和宽度 W（兼容 CHW 或 HW）
    hh = (h // factor) * factor  # 可整除 factor 的最大高度
    ww = (w // factor) * factor  # 可整除 factor 的最大宽度
    if hh == 0 or ww == 0:  # 如果裁剪后变成 0，说明 factor 太大
        raise ValueError(f"factor={factor} 对形状 (...,{h},{w}) 太大，裁剪后会变成 0")  # 报错提示
    return arr[..., :hh, :ww]  # 在最后两维裁剪（不改变通道等前置维度）


def max_pool_last2(arr: np.ndarray, factor: int) -> np.ndarray:  # 对数组最后两维做 factor x factor 的最大池化
    h, w = arr.shape[-2], arr.shape[-1]  # 取最后两维 H/W
    if h % factor != 0 or w % factor != 0:  # 最大池化要求能整除
        raise ValueError(f"形状 (...,{h},{w}) 不能被 factor={factor} 整除")  # 报错提示
    reshaped = arr.reshape(arr.shape[:-2] + (h // factor, factor, w // factor, factor))  # 变形为分块视图
    return reshaped.max(axis=(-1, -3))  # 在每个 block 内取最大值（对 factor 维度做 max）


def downsample_hw(arr_hw_or_chw: np.ndarray, factor: int, expand_to_original: bool) -> np.ndarray:  # 下采样（可选：再扩回原始 H/W）
    h, w = arr_hw_or_chw.shape[-2], arr_hw_or_chw.shape[-1]  # 记录原始 H/W，供可选扩回
    ready = crop_to_multiple_hw(arr_hw_or_chw, factor)  # 为池化裁剪到可整除的大小
    pooled = max_pool_last2(ready, factor)  # 做最大池化下采样，得到更小的 H'/W'

    if not expand_to_original:  # 如果不扩回原始分辨率
        return pooled  # 直接返回真正的低分辨率结果（H' = floor(H/f)*, W' = floor(W/f)*）

    expanded = np.repeat(np.repeat(pooled, factor, axis=-2), factor, axis=-1)  # 最近邻 repeat 扩回到裁剪后的大小
    hh, ww = expanded.shape[-2], expanded.shape[-1]  # 扩回后的 H/W（此时可能小于原始 h/w）

    if hh < h:  # 如果高度不足（因为裁剪导致）
        pad_h = h - hh  # 需要补齐的高度行数
        expanded = np.concatenate(  # 在高度方向拼接补齐
            [expanded, np.repeat(expanded[..., -1:, :], pad_h, axis=-2)],  # 复制最后一行 pad_h 次
            axis=-2,  # 在高度维度拼接
        )  # 高度补齐结束
    if ww < w:  # 如果宽度不足（因为裁剪导致）
        pad_w = w - ww  # 需要补齐的宽度列数
        expanded = np.concatenate(  # 在宽度方向拼接补齐
            [expanded, np.repeat(expanded[..., :, -1:], pad_w, axis=-1)],  # 复制最后一列 pad_w 次
            axis=-1,  # 在宽度维度拼接
        )  # 宽度补齐结束

    return expanded  # 返回扩回到原始 H/W 的结果（shape 与输入一致）


def read_tif(path: Path) -> np.ndarray:  # 读取 tif/tiff 文件为 numpy 数组
    with Image.open(path) as img:  # 用 Pillow 打开图像
        return np.array(img)  # 转成 numpy 数组（可能是 HxW 或 HxWxC）


def write_tif(path: Path, arr: np.ndarray) -> None:  # 写入 tif/tiff 文件
    path.parent.mkdir(parents=True, exist_ok=True)  # 自动创建输出目录
    Image.fromarray(arr).save(path)  # 把 numpy 数组写回图像文件


def read_flo(path: Path) -> np.ndarray:  # 读取 .flo 光流文件（返回 HxWx2 的 float32）
    with path.open("rb") as f:  # 以二进制方式打开文件
        magic = struct.unpack("f", f.read(4))[0]  # 读取 4 字节 float 的 magic
        if abs(magic - FLO_MAGIC) > 1e-4:  # 检查 magic 是否匹配
            raise ValueError(f"{path} 的 .flo magic 不正确：{magic}")  # magic 不对则报错
        w = struct.unpack("i", f.read(4))[0]  # 读取宽度（int32）
        h = struct.unpack("i", f.read(4))[0]  # 读取高度（int32）
        data = np.fromfile(f, np.float32, count=2 * w * h)  # 读取剩余数据（每像素 2 个 float）
    if data.size != 2 * w * h:  # 校验数据长度是否正确
        raise ValueError(f"{path} 的 .flo 数据长度异常")  # 数据长度不对则报错
    return data.reshape(h, w, 2)  # reshape 成 HxWx2（u,v）


def write_flo(path: Path, flow: np.ndarray) -> None:  # 写入 .flo 光流文件
    if flow.ndim != 3 or flow.shape[2] != 2:  # 校验输入是否为 HxWx2
        raise ValueError("flow 必须是 HxWx2")  # 不符合则报错
    h, w, _ = flow.shape  # 取 H/W
    path.parent.mkdir(parents=True, exist_ok=True)  # 自动创建输出目录
    with path.open("wb") as f:  # 以二进制方式写入
        f.write(struct.pack("f", FLO_MAGIC))  # 写 magic（float32）
        f.write(struct.pack("i", w))  # 写宽度（int32）
        f.write(struct.pack("i", h))  # 写高度（int32）
        flow.astype(np.float32).tofile(f)  # 写入光流数据（float32，按行展开）


def downsample_tif(arr: np.ndarray, factor: int, expand_to_original: bool) -> np.ndarray:  # 对 tif 数组下采样（可选：扩回原 shape）
    if arr.ndim == 2:  # 情况 1：灰度图 HxW
        pooled = downsample_hw(arr, factor, expand_to_original)  # 对 H/W 做池化（可选：扩回）
        return pooled.astype(arr.dtype, copy=False)  # 保持原 dtype（比如 uint8）

    if arr.ndim == 3:  # 情况 2：彩色/多通道 HxWxC
        chw = np.moveaxis(arr, -1, 0)  # 变成 CxHxW，便于对最后两维做池化
        pooled_chw = downsample_hw(chw, factor, expand_to_original)  # 对每个通道独立池化（可选：扩回）
        pooled = np.moveaxis(pooled_chw, 0, -1)  # 再变回 HxWxC
        return pooled.astype(arr.dtype, copy=False)  # 保持原 dtype

    raise ValueError(f"不支持的 tif shape：{arr.shape}")  # 其它维度暂不支持


def downsample_flo(flow: np.ndarray, factor: int, expand_to_original: bool) -> np.ndarray:  # 对 flo 光流下采样（可选：扩回原 shape）
    chw = np.moveaxis(flow, -1, 0)  # HxWx2 -> 2xHxW
    pooled_chw = downsample_hw(chw, factor, expand_to_original)  # 对两个分量分别池化（可选：扩回）
    pooled = np.moveaxis(pooled_chw, 0, -1)  # 2xHxW -> HxWx2
    return pooled.astype(np.float32, copy=False)  # flo 固定输出 float32


def iter_target_files(input_dirs: Iterable[Path]) -> Iterable[Path]:  # 递归遍历输入目录，产出 tif/tiff/flo 文件路径
    exts = {".tif", ".tiff", ".flo"}  # 支持的扩展名集合
    for root in input_dirs:  # 遍历每个输入根目录
        for p in root.rglob("*"):  # 递归遍历所有子路径
            if p.is_file() and p.suffix.lower() in exts:  # 只要文件且扩展名匹配
                yield p  # 产出该文件路径


def output_path_for(src: Path, input_root: Path, factor: int) -> Path:  # 计算输出文件路径（放到 <input_root>_lr 同级目录）
    output_root = input_root.parent / f"{input_root.name}_lr"  # 输出根目录：输入根目录的父目录下的同名 _lr 文件夹
    rel_parent = src.parent.relative_to(input_root)  # 计算 src 的相对父路径（保持目录结构）
    target_dir = output_root / f"x{factor}" / rel_parent  # 目标目录：按倍率分到 x{factor}/ 子目录
    name = f"{src.stem}_lr{src.suffix.lower()}"  # 输出文件名：原名 + _lr + 原扩展名（小写）
    return target_dir / name  # 返回完整输出路径


def process_one_file(src: Path, in_root: Path, factors: list[int]) -> None:  # 处理单个文件（生成多个倍率输出）
    ext = src.suffix.lower()  # 取扩展名（小写）
    if ext in {".tif", ".tiff"}:  # 如果是 tif/tiff
        arr = read_tif(src)  # 读取图像
        for f in factors:  # 遍历每个倍率
            out = output_path_for(src, in_root, f)  # 计算输出路径
            lr = downsample_tif(arr, f, EXPAND_TO_ORIGINAL)  # 下采样（由超参数决定是否扩回原 shape）
            write_tif(out, lr)  # 写入输出
            print(f"[OK] {src} -> {out} (x{f}, {arr.shape} -> {lr.shape})")  # 打印处理信息
        return  # tif 处理结束

    if ext == ".flo":  # 如果是 flo
        flow = read_flo(src)  # 读取光流
        for f in factors:  # 遍历每个倍率
            out = output_path_for(src, in_root, f)  # 计算输出路径
            lr = downsample_flo(flow, f, EXPAND_TO_ORIGINAL)  # 下采样（由超参数决定是否扩回原 shape）
            write_flo(out, lr)  # 写入输出
            print(f"[OK] {src} -> {out} (x{f}, {flow.shape} -> {lr.shape})")  # 打印处理信息
        return  # flo 处理结束

    print(f"[SKIP] 不支持的文件类型：{src}")  # 其它扩展名跳过（理论上不会进来）


def _validate_config() -> tuple[list[Path], list[int]]:  # 校验配置并返回解析后的输入目录与倍率
    if not INPUT_DIRS:  # 如果没有配置输入目录
        raise ValueError("请在脚本里设置 INPUT_DIRS（至少一个文件夹路径）")  # 提示用户修改配置

    input_dirs = [Path(p).expanduser().resolve() for p in INPUT_DIRS]  # 把输入目录字符串转成绝对 Path
    for d in input_dirs:  # 校验每个输入目录
        if not d.exists() or not d.is_dir():  # 不存在或不是目录
            raise ValueError(f"输入目录不存在或不是目录：{d}")  # 抛错

    factors = [int(x) for x in FACTORS]  # 把倍率转为 int
    if not factors or any(x < 1 for x in factors):  # 倍率列表不能为空，且都要 >=1
        raise ValueError(f"FACTORS 配置不合法：{FACTORS}")  # 抛错

    return input_dirs, factors  # 返回校验通过的配置


def main() -> None:  # 主函数入口
    input_dirs, factors = _validate_config()  # 读取并校验配置

    total = 0  # 计数：处理的源文件数量
    for in_root in input_dirs:  # 遍历每个输入根目录
        for src in iter_target_files([in_root]):  # 遍历该目录下所有目标文件
            process_one_file(src, in_root, factors)  # 处理该文件（生成多个倍率输出）
            total += 1  # 源文件计数 +1

    print(f"[DONE] 处理完成，源文件数量：{total}")  # 打印总处理文件数
    for in_root in input_dirs:  # 遍历输入根目录
        print(f"[DONE] 输出根目录：{in_root.parent / (in_root.name + '_lr')}")  # 打印每个输入目录对应的输出目录


if __name__ == "__main__":  # Python 脚本入口判断（直接运行时为 True）
    main()  # 执行主函数
