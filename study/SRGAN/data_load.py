
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from loguru import logger

import hashlib
import json
import pickle
import re
from functools import lru_cache
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


"""
训练时 batch 的读取方式：

for batch in train_dataloader:
    lr_img = batch["image_pair"]["lr_data"]
    gr_img = batch["image_pair"]["gr_data"]

    lr_flo = batch["flo"]["lr_data"]
    gr_flo = batch["flo"]["gr_data"]
"""


# ==============================
# 基础配置
# ==============================

# 真实数据根路径
GR_DATA_ROOT_DIR = r"/study_datas/sr_dataset/class_1/data"
# 低分辨率数据根路径
LR_DATA_ROOT_DIR = rf"/study_datas/sr_dataset/class_1_lr/x{4}/data"

# Middlebury `.flo` 文件的 magic number。
FLO_MAGIC = 202021.25

# 认为是图像的扩展名集合。
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
# 从文件名提取样本主键的正则规则
PAIR_KEY_PATTERNS = (
    re.compile(r"^(.*?)(?:[_-]?ti[_-]?\d+)$"),
    re.compile(r"^(.*?)(?:[_-]?t[_-]?i[_-]?\d+)$"),
    re.compile(r"^(.*?)(?:[_-]?(?:img|image|frame)[_-]?\d+)$"),
    re.compile(r"^(.*?)(?:[_-]?\d+)$"),
)
# 统一样本键时剥离后缀的规则
LR_SUFFIX_PATTERN = re.compile(r"([_-]lr)+$")
FLOW_LR_SUFFIX_PATTERN = re.compile(r"([_-]flow)+([_-]lr)+$")
FLOW_SUFFIX_PATTERN = re.compile(r"([_-]flow)+$")


def resolve_lr_data_root_dir(lr_data_root_dir: str, lr_data_variant: str = "default") -> str:
    """
    根据传入参数选择 LR 数据根目录。

    参数:
        lr_data_root_dir: str
            当前训练代码传入的 LR 根目录，通常是原始 `_lr` 路径。
        lr_data_variant: str
            LR 数据变体标识。
            - `default` / `original` / `lr`: 使用原始 `_lr` 数据。
            - `particle` / `lr_particle`: 使用颗粒退化生成的 `_lr_particle` 数据。

    返回:
        str
            最终实际要使用的 LR 数据根目录。

    说明:
        这个函数只负责做路径字符串映射，不检查目录是否存在；
        目录合法性检查由后续 `ensure_valid_root_dir` 负责。
    """
    variant = (lr_data_variant or "default").lower().strip()
    if variant in {"default", "original", "lr"}:
        return lr_data_root_dir.replace("_lr_particle", "_lr")
    if variant in {"particle", "lr_particle"}:
        if "_lr_particle" in lr_data_root_dir:
            return lr_data_root_dir
        if "_lr" in lr_data_root_dir:
            return lr_data_root_dir.replace("_lr", "_lr_particle", 1)
        return f"{lr_data_root_dir}_particle"
    logger.error(f"Unsupported lr_data_variant: {lr_data_variant}")
    raise ValueError(f"Unsupported lr_data_variant: {lr_data_variant}")


# ==============================
# 通用工具
# ==============================

def progress_iter(iterable, desc: str, total: int | None = None):
    """
    给迭代器包一层进度显示。

    - 如果环境中安装了 `tqdm`，显示进度条
    - 否则退化成普通迭代，同时打印提示信息
    """
    if tqdm is not None:
        return tqdm(iterable, desc=desc, total=total, leave=False)

    logger.info(f"[Progress] {desc}")
    return iterable


def log_info(message: str, verbose: bool) -> None:
    """
    按需打印日志，避免大数据集下逐文件输出拖慢速度。减少非必要 I/O。
    """
    if verbose:
        logger.info(message)


def _normalize_image_array(image: np.ndarray, image_path: str) -> np.ndarray:
    """
    把读取出的图像统一整理成 H x W x 3 的 numpy 数组。处理灰度/通道前置/带 alpha 等情况。
    """
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    elif image.ndim == 3 and image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4):
        image = np.moveaxis(image, 0, -1)

    if image.ndim != 3:
        logger.error(f'Unsupported image shape: {image.shape}, file: {image_path}')
        raise ValueError(f"Unsupported image shape: {image.shape}, file: {image_path}")

    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] >= 4:
        image = image[..., :3]

    return np.ascontiguousarray(image)


@lru_cache(maxsize=512)
def _read_rgb_image_cached(path_str: str) -> np.ndarray:
    """带 LRU 缓存的 RGB 图像读取。"""
    return _normalize_image_array(tifffile.imread(path_str), path_str)


@lru_cache(maxsize=512)
def _read_flo_cached(path_str: str) -> np.ndarray:
    """带 LRU 缓存的 .flo 读取，并补第三通道 magnitude。"""
    path = Path(path_str)
    header = np.fromfile(path, dtype=np.float32, count=1)
    if header.size != 1:
        logger.error(f'Incomplete .flo file header: {path}')
        raise ValueError(f"Incomplete .flo file header: {path}")

    magic = float(header[0])
    if abs(magic - FLO_MAGIC) > 1e-4:
        logger.error(f'Invalid .flo magic number: {path}')
        raise ValueError(f"Invalid .flo magic number: {path}")

    dims = np.fromfile(path, dtype=np.int32, count=2, offset=4)
    if dims.size != 2:
        logger.error(f'Failed to read width/height from .flo file: {path}')
        raise ValueError(f"Failed to read width/height from .flo file: {path}")

    width, height = int(dims[0]), int(dims[1])
    flow_uv = np.fromfile(path, dtype=np.float32, offset=12)
    expected_size = 2 * width * height
    if flow_uv.size != expected_size:
        logger.error(f'Incomplete .flo data: {path}, expected {expected_size}, got {flow_uv.size}')
        raise ValueError(
            f"Incomplete .flo data: {path}, expected {expected_size}, got {flow_uv.size}"
        )

    flow_uv = flow_uv.reshape(height, width, 2)
    flow_magnitude = np.linalg.norm(flow_uv, axis=2, keepdims=True)
    return np.concatenate([flow_uv, flow_magnitude], axis=2)


def build_metadata_cache_path(
    cache_dir: Path,
    gr_root: Path,
    lr_root: Path,
    class_names: list[str],
) -> Path:
    """
    为当前数据配置生成稳定的元数据缓存文件路径。根据根目录、类别及目录时间戳生成稳定元数据缓存路径。
    """
    fingerprint_parts = [str(gr_root), str(lr_root)]
    for class_name in class_names:
        gr_class_dir = gr_root / class_name / class_name
        lr_class_dir = lr_root / class_name / class_name
        gr_mtime = gr_class_dir.stat().st_mtime_ns if gr_class_dir.exists() else 0
        lr_mtime = lr_class_dir.stat().st_mtime_ns if lr_class_dir.exists() else 0
        fingerprint_parts.append(f"{class_name}:{gr_mtime}:{lr_mtime}")

    digest = hashlib.md5("|".join(fingerprint_parts).encode("utf-8")).hexdigest()
    return cache_dir / f"sr_samples_{digest}.pkl"


def build_tensor_cache_root(
    cache_dir: Path,
    gr_root: Path,
    lr_root: Path,
    class_names: list[str],
    target_size: tuple[int, int] | None,
) -> Path:
    """
    为当前数据配置生成张量缓存目录。根据数据配置生成样本张量缓存根目录。
    """
    fingerprint = hashlib.md5(
        "|".join(
            [
                str(gr_root),
                str(lr_root),
                ",".join(class_names),
                str(target_size),
            ]
        ).encode("utf-8")
    ).hexdigest()
    return cache_dir / f"tensor_cache_{fingerprint}"


def build_sample_tensor_cache_path(
    cache_root: Path,
    sample: dict,
) -> Path:
    """
    为单个样本生成稳定的张量缓存文件路径。
    """
    path_parts = [sample["class_name"], sample["sample_key"]]
    for path in sample["image_pair"]["gr_paths"]:
        path_parts.append(f"gri:{path.stat().st_mtime_ns}:{path}")
    for path in sample["image_pair"]["lr_paths"]:
        path_parts.append(f"lri:{path.stat().st_mtime_ns}:{path}")
    for path in sample["flo"]["gr_paths"]:
        path_parts.append(f"grf:{path.stat().st_mtime_ns}:{path}")
    for path in sample["flo"]["lr_paths"]:
        path_parts.append(f"lrf:{path.stat().st_mtime_ns}:{path}")

    digest = hashlib.md5("|".join(path_parts).encode("utf-8")).hexdigest()
    return cache_root / sample["class_name"] / f"{digest}.pt"


def dump_samples_cache(cache_path: Path, samples: list[dict]) -> None:
    """将样本元数据序列化保存到 pkl。"""
    serializable_samples = []
    for sample in samples:
        serializable_samples.append(
            {
                "class_name": sample["class_name"],
                "sample_key": sample["sample_key"],
                "image_pair": {
                    "gr_paths": [str(path) for path in sample["image_pair"]["gr_paths"]],
                    "lr_paths": [str(path) for path in sample["image_pair"]["lr_paths"]],
                },
                "flo": {
                    "gr_paths": [str(path) for path in sample["flo"]["gr_paths"]],
                    "lr_paths": [str(path) for path in sample["flo"]["lr_paths"]],
                },
            }
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as file:
        pickle.dump(serializable_samples, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_samples_cache(cache_path: Path) -> list[dict]:
    """从 pkl 反序列化样本元数据并恢复 Path 对象。"""
    with cache_path.open("rb") as file:
        cached_samples = pickle.load(file)

    samples: list[dict] = []
    for sample in cached_samples:
        samples.append(
            {
                "class_name": sample["class_name"],
                "sample_key": sample["sample_key"],
                "image_pair": {
                    "gr_paths": [Path(path) for path in sample["image_pair"]["gr_paths"]],
                    "lr_paths": [Path(path) for path in sample["image_pair"]["lr_paths"]],
                },
                "flo": {
                    "gr_paths": [Path(path) for path in sample["flo"]["gr_paths"]],
                    "lr_paths": [Path(path) for path in sample["flo"]["lr_paths"]],
                },
            }
        )
    return samples


def ensure_valid_root_dir(root_dir: str, root_name: str) -> Path:
    """
    校验根目录是否合法，并统一转换成绝对路径。
    """
    if not root_dir:
        logger.error(f'Please pass {root_name}.')
        raise ValueError(f"Please pass {root_name}.")

    root_path = Path(root_dir).expanduser().resolve()
    if not root_path.exists() or not root_path.is_dir():
        logger.error(f'Invalid {root_name}: {root_path}')
        raise ValueError(f"Invalid {root_name}: {root_path}")

    return root_path


# ==============================
# flo 读取与预处理
# ==============================

def read_flo(path: Path) -> np.ndarray:
    """
    读取 `.flo` 光流文件。

    原始 `.flo` 通常只有 2 个通道：
    - u: 水平方向分量
    - v: 垂直方向分量

    这里额外补 1 个物理量通道：
    - magnitude = sqrt(u^2 + v^2)

    最终返回形状为 `H x W x 3` 的数组。
    """
    return _read_flo_cached(str(path))


class FloToTensor:
    """
    把 `H x W x 3` 的光流数组转换为 `3 x H x W` 张量。
    """

    def __call__(self, flow: np.ndarray) -> torch.Tensor:
        if flow.ndim != 3 or flow.shape[-1] != 3:
            logger.error(f'Unexpected flow shape: {flow.shape}')
            raise ValueError(f"Unexpected flow shape: {flow.shape}")

        flow = np.transpose(flow.astype(np.float32, copy=False), (2, 0, 1))
        return torch.from_numpy(flow)


class FlowResize:
    """
    使用双线性插值把 flo 数据 resize 到目标尺寸。
    """

    def __init__(self, target_size: tuple[int, int]) -> None:
        self.target_size = target_size

    def __call__(self, flow: np.ndarray) -> np.ndarray:
        flow_tensor = torch.from_numpy(np.transpose(flow.astype(np.float32, copy=False), (2, 0, 1)))
        flow_tensor = torch.nn.functional.interpolate(
            flow_tensor.unsqueeze(0),
            size=self.target_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return np.transpose(flow_tensor.numpy(), (1, 2, 0))


# ==============================
# 类别发现与筛选
# ==============================

def discover_class_names(data_root: Path) -> list[str]:
    """
    从单个根目录中发现类别名。

    目录结构要求是：
    root/
        class_name/
            class_name/
                data files...
    """
    class_names: list[str] = []
    logger.info(f"[Scan] Discovering class folders under: {data_root}")

    for candidate in progress_iter(sorted(data_root.iterdir()), desc="Discover classes"):
        if candidate.is_dir() and (candidate / candidate.name).is_dir():
            class_names.append(candidate.name)
            logger.info(f"[Scan] Found class: {candidate.name}")

    return class_names


def get_class_names(gr_data_root_dir: str) -> list[str]:
    """
    对外公开的类别名获取函数（已弃用 lr_data_root_dir）。

    当前行为：
    - 仅扫描 GR 根目录
    - 返回 GR 拥有的类别名（按字典序）
    """
    gr_root = ensure_valid_root_dir(gr_data_root_dir, "gr_data_root_dir")



    class_names = discover_class_names(gr_root)

    if not class_names:
        logger.error(f'No class folders found under GR root: {gr_root}')
        raise ValueError(f"No class folders found under GR root: {gr_root}")

    return sorted(class_names)


def normalize_selected_classes(
    available_class_names: list[str],
    selected_classes: str | list[str] | tuple[str, ...] | None,
) -> list[str]:
    """
    标准化外部传入的类别筛选参数。标准化并校验 selected_classes 参数。

    支持：
    - None
    - 单个类名字符串
    - 类名列表 / 元组
    """
    if selected_classes is None:
        return available_class_names

    if isinstance(selected_classes, str):
        normalized = [selected_classes]
    else:
        normalized = list(selected_classes)

    if not normalized:
        logger.error('selected_classes cannot be empty.')
        raise ValueError("selected_classes cannot be empty.")

    invalid_names = [name for name in normalized if name not in available_class_names]
    if invalid_names:
        logger.error(f'Unknown class names: {invalid_names}. Available classes: {available_class_names}')
        raise ValueError(
            f"Unknown class names: {invalid_names}. Available classes: {available_class_names}"
        )

    return normalized


# ==============================
# 样本键标准化
# ==============================

def infer_image_pair_key(file_stem: str) -> str:
    """
    从图像文件名里提取图像对的“样本主键”。

    例如：
    - `sample_ti0`, `sample_ti1` -> `sample`
    - `sample-ti-0`, `sample-ti-1` -> `sample`
    """
    stem = file_stem.lower()
    stem = LR_SUFFIX_PATTERN.sub("", stem)
    for pattern in PAIR_KEY_PATTERNS:
        match = pattern.match(stem)
        if match and match.group(1):
            return match.group(1)
    return stem


def normalize_pair_key(sample_key: str) -> str:
    """
    统一配对时使用的样本键。统一样本键格式并去除 lr/flow 等后缀。

    主要处理几类常见尾缀：
    - `_lr`
    - `_flow`
    - `_flow_lr`
    """
    normalized_key = sample_key.lower()
    normalized_key = FLOW_LR_SUFFIX_PATTERN.sub("", normalized_key)
    normalized_key = LR_SUFFIX_PATTERN.sub("", normalized_key)
    normalized_key = FLOW_SUFFIX_PATTERN.sub("", normalized_key)
    return normalized_key


# ==============================
# 单根目录样本收集
# ==============================

def collect_root_class_samples(
    class_name: str,
    class_dir: Path,
    domain_name: str,
    verbose: bool = False,
) -> list[dict]:
    """
    在单个类别目录收集完整样本（2张图+1个flo）。
    """
    image_groups: dict[str, list[Path]] = defaultdict(list)
    flo_map: dict[str, Path] = {}

    if not class_dir.exists() or not class_dir.is_dir():
        logger.error(f'{domain_name} class directory does not exist: {class_dir}')
        raise ValueError(f"{domain_name} class directory does not exist: {class_dir}")

    class_files = sorted(class_dir.iterdir())
    logger.info(f"[Scan] Reading {domain_name} class '{class_name}' from: {class_dir}")
    logger.info(f"[Scan] Found {len(class_files)} entries in {domain_name} class '{class_name}'")

    for file_path in progress_iter(class_files, desc=f"Scan {domain_name}:{class_name}", total=len(class_files)):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()

        if suffix == ".flo":
            sample_key = normalize_pair_key(file_path.stem)
            log_info(
                f"[Collect] {domain_name}:{class_name} flo -> {file_path.name}, key={sample_key}",
                verbose,
            )
            flo_map[sample_key] = file_path
            continue

        if suffix in IMAGE_EXTENSIONS:
            sample_key = normalize_pair_key(infer_image_pair_key(file_path.stem))
            image_groups[sample_key].append(file_path)
            log_info(
                f"[Collect] {domain_name}:{class_name} image -> {file_path.name}, key={sample_key}",
                verbose,
            )

    samples: list[dict] = []
    all_keys = sorted(set(image_groups.keys()) | set(flo_map.keys()))
    incomplete_keys: list[tuple[str, str]] = []

    for sample_key in all_keys:
        image_paths = sorted(image_groups.get(sample_key, []))
        flo_path = flo_map.get(sample_key)

        if len(image_paths) != 2:
            incomplete_keys.append((sample_key, f"image_count={len(image_paths)}"))
            continue

        if flo_path is None:
            incomplete_keys.append((sample_key, "missing_flo"))
            continue

        log_info(
            f"[Pack] {domain_name}:{class_name} sample_key={sample_key} "
            f"-> images={[path.name for path in image_paths]}, flo={flo_path.name}",
            verbose,
        )
        samples.append(
            {
                "sample_key": sample_key,
                "image_paths": image_paths,
                "flo_path": flo_path,
            }
        )

    if incomplete_keys:
        logger.info(f"[Debug] Incomplete {domain_name} samples in class '{class_name}':")
        for sample_key, reason in incomplete_keys[:50]:
            logger.info(f"  key={sample_key}, reason={reason}")
        if len(incomplete_keys) > 50:
            logger.info(f"[Debug] ... and {len(incomplete_keys) - 50} more incomplete keys")

    logger.info(f"[Done] {domain_name} class '{class_name}' collected {len(samples)} complete samples")
    return samples


# ==============================
# GR/LR 配对与模态对齐
# ==============================

def pair_sr_class_samples(
    class_name: str,
    gr_class_dir: Path,
    lr_class_dir: Path,
    verbose: bool = False,
) -> list[dict]:
    """按 sample_key 对齐 GR 与 LR 样本，形成多模态配对样本。"""
    gr_samples = collect_root_class_samples(class_name, gr_class_dir, "GR", verbose=verbose)
    lr_samples = collect_root_class_samples(class_name, lr_class_dir, "LR", verbose=verbose)

    gr_map = {sample["sample_key"]: sample for sample in gr_samples}
    lr_map = {sample["sample_key"]: sample for sample in lr_samples}

    common_keys = sorted(set(gr_map.keys()) & set(lr_map.keys()))
    gr_only_keys = sorted(set(gr_map.keys()) - set(lr_map.keys()))
    lr_only_keys = sorted(set(lr_map.keys()) - set(gr_map.keys()))

    if gr_only_keys:
        logger.warning(f"[Warn] GR-only sample keys in class '{class_name}': {gr_only_keys[:20]}")
        if len(gr_only_keys) > 20:
            logger.warning(f"[Warn] ... and {len(gr_only_keys) - 20} more")
    if lr_only_keys:
        logger.warning(f"[Warn] LR-only sample keys in class '{class_name}': {lr_only_keys[:20]}")
        if len(lr_only_keys) > 20:
            logger.warning(f"[Warn] ... and {len(lr_only_keys) - 20} more")

    if not common_keys:
        logger.error(f"No paired GR/LR samples found for class '{class_name}'")
        raise ValueError(f"No paired GR/LR samples found for class '{class_name}'")

    paired_samples: list[dict] = []
    for sample_key in common_keys:
        paired_samples.append(
            {
                "class_name": class_name,
                "sample_key": sample_key,
                "image_pair": {
                    "gr_paths": gr_map[sample_key]["image_paths"],
                    "lr_paths": lr_map[sample_key]["image_paths"],
                },
                "flo": {
                    "gr_paths": [gr_map[sample_key]["flo_path"]],
                    "lr_paths": [lr_map[sample_key]["flo_path"]],
                },
            }
        )

    logger.info(f"[Done] Class '{class_name}' paired {len(paired_samples)} aligned SR samples")
    return paired_samples


# ==============================
# 按类别划分 train / val / test
# ==============================

def split_samples_by_class_three_way(
    samples: list[dict],
    class_names: list[str],
    train_nums_rate: float,
    validate_nums_rate: float,
    test_nums_rate: float,
    random_seed: int,
) -> tuple[list[dict], list[dict], list[dict], dict[str, dict[str, int]]]:
    """
    按类别分别随机划分 train/val/test，避免类分布偏移。
    """
    train_samples: list[dict] = []
    validate_samples: list[dict] = []
    test_samples: list[dict] = []
    split_summary: dict[str, dict[str, int]] = {}

    for class_offset, class_name in enumerate(class_names):
        class_samples = [sample for sample in samples if sample["class_name"] == class_name]
        class_total = len(class_samples)
        if class_total == 0:
            continue

        class_generator = torch.Generator().manual_seed(random_seed + class_offset)
        shuffled_indices = torch.randperm(class_total, generator=class_generator).tolist()
        shuffled_samples = [class_samples[index] for index in shuffled_indices]

        class_train_size = int(class_total * train_nums_rate)
        class_val_size = int(class_total * validate_nums_rate)
        class_test_size = class_total - class_train_size - class_val_size

        # 兜底：当样本够多时，比例>0 的集合尽量不为空
        if class_total >= 3:
            if train_nums_rate > 0 and class_train_size == 0:
                class_train_size = 1
            if validate_nums_rate > 0 and class_val_size == 0:
                class_val_size = 1
            class_test_size = class_total - class_train_size - class_val_size
            if test_nums_rate > 0 and class_test_size <= 0:
                class_test_size = 1
                if class_val_size > 1:
                    class_val_size -= 1
                else:
                    class_train_size = max(1, class_train_size - 1)

        train_samples.extend(shuffled_samples[:class_train_size])
        validate_samples.extend(shuffled_samples[class_train_size: class_train_size + class_val_size])
        test_samples.extend(shuffled_samples[class_train_size + class_val_size:])

        split_summary[class_name] = {
            "total": class_total,
            "train": class_train_size,
            "val": class_val_size,
            "test": class_test_size,
        }

    return train_samples, validate_samples, test_samples, split_summary


def split_samples_global(
    samples: list[dict],
    train_nums_rate: float,
    validate_nums_rate: float,
    test_nums_rate: float,
    random_seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    全局随机三划分：先打乱所有类别样本，再按 train/val/test 切分。
    """
    total = len(samples)
    if total == 0:
        return [], [], []

    g = torch.Generator().manual_seed(random_seed)
    indices = torch.randperm(total, generator=g).tolist()
    shuffled = [samples[i] for i in indices]

    n_train = int(total * train_nums_rate)
    n_val = int(total * validate_nums_rate)
    n_test = total - n_train - n_val

    if total >= 3:
        if train_nums_rate > 0 and n_train == 0:
            n_train = 1
        if validate_nums_rate > 0 and n_val == 0:
            n_val = 1
        n_test = total - n_train - n_val
        if test_nums_rate > 0 and n_test <= 0:
            n_test = 1
            if n_val > 1:
                n_val -= 1
            else:
                n_train = max(1, n_train - 1)

    train_samples = shuffled[:n_train]
    validate_samples = shuffled[n_train:n_train + n_val]
    test_samples = shuffled[n_train + n_val:]

    return train_samples, validate_samples, test_samples


def _bucket_class_name(
    sample_class_name: str | None,
    known_class_names: list[str] | tuple[str, ...] | set[str],
    other_name: str = "other",
) -> str:
    # 当 mixed 模式下 validate/test 中出现“未知类别”时，不报错，统一放到 other 桶里。
    # 这样评估阶段仍然能完整跑完，并且你能在输出目录里看到这些异常样本。
    if sample_class_name in known_class_names:
        return str(sample_class_name)
    return other_name


def group_samples_by_class(
    samples: list[dict],
    known_class_names: list[str],
    other_name: str = "other",
) -> dict[str, list[dict]]:
    # 这里不直接创建多个 DataLoader，而是先做“样本级分桶”。
    # 原因是训练/验证主流程仍然沿用单个 loader，改动更小、更稳定。
    grouped: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        bucket = _bucket_class_name(sample.get("class_name"), known_class_names, other_name=other_name)
        grouped[bucket].append(sample)

    ordered_grouped: dict[str, list[dict]] = {}
    for class_name in known_class_names:
        if grouped.get(class_name):
            ordered_grouped[class_name] = grouped[class_name]
    if grouped.get(other_name):
        ordered_grouped[other_name] = grouped[other_name]
    return ordered_grouped


def attach_grouped_class_metadata(
    dataset: Dataset,
    known_class_names: list[str],
    other_name: str = "other",
) -> None:
    # 动态给 dataset 挂元数据，而不是改 Dataset 构造参数。
    # 这样能兼容现有 DataLoader / collate_fn / pipeline，不需要级联改很多地方。
    grouped_samples = group_samples_by_class(dataset.samples, known_class_names, other_name=other_name)
    dataset.grouped_samples_by_class = grouped_samples
    dataset.grouped_class_names = list(grouped_samples.keys())
    dataset.known_class_names = list(known_class_names)
    dataset.other_class_name = other_name


# ==============================
# Dataset / collate
# ==============================

class SRPairedDataset(Dataset):
    """
    返回样本级对齐的 image_pair + flo 多模态数据集。
    每个样本包含两种模态：
    - image_pair (previous / next)
    - flo

    并且二者在样本级别严格一一对应。
    """

    def __init__(
        self,
        samples: list[dict],
        class_to_idx: dict[str, int],
        image_transform: transforms.Compose,
        flow_transform: transforms.Compose,
        verbose: bool = False,
        cache_file_reads: bool = True,
        cache_transformed_samples: bool = False,
        tensor_cache_root: Path | None = None,
    ) -> None:
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.image_transform = image_transform
        self.flow_transform = flow_transform
        self.verbose = verbose
        self.cache_file_reads = cache_file_reads
        self.cache_transformed_samples = cache_transformed_samples
        self.tensor_cache_root = tensor_cache_root
        self._sample_cache: dict[int, dict] = {}
        self._sample_metadata = [
            {
                "label": self.class_to_idx[sample["class_name"]],
                "class_name": sample["class_name"],
                "sample_key": sample["sample_key"],
                "image_pair_lr_paths": " | ".join(str(path) for path in sample["image_pair"]["lr_paths"]),
                "image_pair_gr_paths": " | ".join(str(path) for path in sample["image_pair"]["gr_paths"]),
                "flo_lr_paths": " | ".join(str(path) for path in sample["flo"]["lr_paths"]),
                "flo_gr_paths": " | ".join(str(path) for path in sample["flo"]["gr_paths"]),
            }
            for sample in samples
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image_pair(self, paths: list[Path]) -> dict:
        """
        读取两张 RGB 图像，拆成 previous / next 两部分。

        每张图是 `3 x H x W`：
        - previous: 第一张
        - next: 第二张
        """
        log_info(f"[Read] Loading image pair: {paths[0].name}, {paths[1].name}", self.verbose)
        image_tensors = []
        for image_path in paths:
            if self.cache_file_reads:
                image = _read_rgb_image_cached(str(image_path))
            else:
                image = _normalize_image_array(tifffile.imread(str(image_path)), str(image_path))
            image_tensors.append(self.image_transform(image))

        if len(image_tensors) != 2:
            logger.error(f'Expected exactly 2 images, got {len(image_tensors)}')
            raise ValueError(f"Expected exactly 2 images, got {len(image_tensors)}")

        return {
            "previous": image_tensors[0],
            "next": image_tensors[1],
        }

    def _split_old_6ch_image_pair(self, tensor_or_dict) -> dict:
        """
        兼容旧缓存：
        - 旧格式可能是 6xHxW tensor（两张图通道拼接）
        - 也可能已是 {"previous": ..., "next": ...}
        """
        if isinstance(tensor_or_dict, dict):
            if "previous" in tensor_or_dict and "next" in tensor_or_dict:
                return {
                    "previous": tensor_or_dict["previous"],
                    "next": tensor_or_dict["next"],
                }
            logger.error('Unsupported cached image pair dict format.')
            raise ValueError("Unsupported cached image pair dict format.")

        if not isinstance(tensor_or_dict, torch.Tensor):
            logger.error(f'Unsupported cached image pair type: {type(tensor_or_dict)}')
            raise ValueError(f"Unsupported cached image pair type: {type(tensor_or_dict)}")

        if tensor_or_dict.ndim != 3 or tensor_or_dict.shape[0] != 6:
            logger.error(f'Expected cached 6xHxW tensor, got {tuple(tensor_or_dict.shape)}')
            raise ValueError(f"Expected cached 6xHxW tensor, got {tuple(tensor_or_dict.shape)}")

        return {
            "previous": tensor_or_dict[:3, ...],
            "next": tensor_or_dict[3:, ...],
        }

    def _normalize_cached_image_pair(self, cached_image_pair: dict) -> dict:
        """
        统一把缓存里的 image_pair 转成新结构：
        {
          "previous": {"lr_data":..., "gr_data":...},
          "next": {"lr_data":..., "gr_data":...},
        }
        """
        # 新格式
        if (
            "previous" in cached_image_pair
            and "next" in cached_image_pair
            and isinstance(cached_image_pair["previous"], dict)
            and isinstance(cached_image_pair["next"], dict)
            and "lr_data" in cached_image_pair["previous"]
            and "gr_data" in cached_image_pair["previous"]
            and "lr_data" in cached_image_pair["next"]
            and "gr_data" in cached_image_pair["next"]
        ):
            return cached_image_pair

        # 旧格式: {"lr_data": <6ch or dict>, "gr_data": <6ch or dict>}
        if "lr_data" in cached_image_pair and "gr_data" in cached_image_pair:
            lr_pair = self._split_old_6ch_image_pair(cached_image_pair["lr_data"])
            gr_pair = self._split_old_6ch_image_pair(cached_image_pair["gr_data"])
            return {
                "previous": {
                    "lr_data": lr_pair["previous"],
                    "gr_data": gr_pair["previous"],
                },
                "next": {
                    "lr_data": lr_pair["next"],
                    "gr_data": gr_pair["next"],
                },
            }

        logger.error('Unsupported cached image_pair format.')
        raise ValueError("Unsupported cached image_pair format.")

    def _clone_image_pair(self, image_pair: dict) -> dict:
        """深拷贝 image_pair 张量，避免原地修改污染缓存。"""
        return {
            "previous": {
                "lr_data": image_pair["previous"]["lr_data"].clone(),
                "gr_data": image_pair["previous"]["gr_data"].clone(),
            },
            "next": {
                "lr_data": image_pair["next"]["lr_data"].clone(),
                "gr_data": image_pair["next"]["gr_data"].clone(),
            },
        }

    def __getitem__(self, index: int) -> dict:
        """获取单样本：优先缓存，缺失时读取并变换后返回。"""
        if self.cache_transformed_samples and index in self._sample_cache:
            cached = self._sample_cache[index]
            return {
                "image_pair": self._clone_image_pair(cached["image_pair"]),
                "flo": {
                    "lr_data": cached["flo"]["lr_data"].clone(),
                    "gr_data": cached["flo"]["gr_data"].clone(),
                },
                "label": cached["label"],
                "class_name": cached["class_name"],
                "sample_key": cached["sample_key"],
                "image_pair_lr_paths": cached["image_pair_lr_paths"],
                "image_pair_gr_paths": cached["image_pair_gr_paths"],
                "flo_lr_paths": cached["flo_lr_paths"],
                "flo_gr_paths": cached["flo_gr_paths"],
            }

        sample = self.samples[index]
        metadata = self._sample_metadata[index]
        tensor_cache_path = None

        if self.tensor_cache_root is not None:
            tensor_cache_path = build_sample_tensor_cache_path(self.tensor_cache_root, sample)
            if tensor_cache_path.exists():
                cached = torch.load(tensor_cache_path, map_location="cpu")
                normalized_image_pair = self._normalize_cached_image_pair(cached["image_pair"])

                item = {
                    "image_pair": normalized_image_pair,
                    "flo": {
                        "lr_data": cached["flo"]["lr_data"],
                        "gr_data": cached["flo"]["gr_data"],
                    },
                    "label": metadata["label"],
                    "class_name": metadata["class_name"],
                    "sample_key": metadata["sample_key"],
                    "image_pair_lr_paths": metadata["image_pair_lr_paths"],
                    "image_pair_gr_paths": metadata["image_pair_gr_paths"],
                    "flo_lr_paths": metadata["flo_lr_paths"],
                    "flo_gr_paths": metadata["flo_gr_paths"],
                }

                if self.cache_transformed_samples:
                    self._sample_cache[index] = {
                        "image_pair": self._clone_image_pair(item["image_pair"]),
                        "flo": {
                            "lr_data": item["flo"]["lr_data"].clone(),
                            "gr_data": item["flo"]["gr_data"].clone(),
                        },
                        "label": item["label"],
                        "class_name": item["class_name"],
                        "sample_key": item["sample_key"],
                        "image_pair_lr_paths": item["image_pair_lr_paths"],
                        "image_pair_gr_paths": item["image_pair_gr_paths"],
                        "flo_lr_paths": item["flo_lr_paths"],
                        "flo_gr_paths": item["flo_gr_paths"],
                    }
                return item

        lr_image_pair = self._load_image_pair(sample["image_pair"]["lr_paths"])
        gr_image_pair = self._load_image_pair(sample["image_pair"]["gr_paths"])

        item = {
            "image_pair": {
                "previous": {
                    "lr_data": lr_image_pair["previous"],
                    "gr_data": gr_image_pair["previous"],
                },
                "next": {
                    "lr_data": lr_image_pair["next"],
                    "gr_data": gr_image_pair["next"],
                },
            },
            "flo": {
                "lr_data": self.flow_transform(read_flo(sample["flo"]["lr_paths"][0])),
                "gr_data": self.flow_transform(read_flo(sample["flo"]["gr_paths"][0])),
            },
            "label": metadata["label"],
            "class_name": metadata["class_name"],
            "sample_key": metadata["sample_key"],
            "image_pair_lr_paths": metadata["image_pair_lr_paths"],
            "image_pair_gr_paths": metadata["image_pair_gr_paths"],
            "flo_lr_paths": metadata["flo_lr_paths"],
            "flo_gr_paths": metadata["flo_gr_paths"],
        }

        if tensor_cache_path is not None and not tensor_cache_path.exists():
            tensor_cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "image_pair": item["image_pair"],
                    "flo": {
                        "lr_data": item["flo"]["lr_data"],
                        "gr_data": item["flo"]["gr_data"],
                    },
                },
                tensor_cache_path,
            )

        if self.cache_transformed_samples:
            self._sample_cache[index] = {
                "image_pair": self._clone_image_pair(item["image_pair"]),
                "flo": {
                    "lr_data": item["flo"]["lr_data"].clone(),
                    "gr_data": item["flo"]["gr_data"].clone(),
                },
                "label": item["label"],
                "class_name": item["class_name"],
                "sample_key": item["sample_key"],
                "image_pair_lr_paths": item["image_pair_lr_paths"],
                "image_pair_gr_paths": item["image_pair_gr_paths"],
                "flo_lr_paths": item["flo_lr_paths"],
                "flo_gr_paths": item["flo_gr_paths"],
            }

        return item


def build_aligned_batch(samples: list[dict]) -> dict | None:
    """
    把样本列表堆叠为对齐 batch（image_pair/flo/标签/路径）。
    把一组已经一一对应的多模态样本整理成 batch。
    """
    if not samples:
        return None

    return {
        "image_pair": {
            "previous": {
                "lr_data": torch.stack(
                    [sample["image_pair"]["previous"]["lr_data"] for sample in samples], dim=0
                ),
                "gr_data": torch.stack(
                    [sample["image_pair"]["previous"]["gr_data"] for sample in samples], dim=0
                ),
            },
            "next": {
                "lr_data": torch.stack(
                    [sample["image_pair"]["next"]["lr_data"] for sample in samples], dim=0
                ),
                "gr_data": torch.stack(
                    [sample["image_pair"]["next"]["gr_data"] for sample in samples], dim=0
                ),
            },
        },
        "flo": {
            "lr_data": torch.stack([sample["flo"]["lr_data"] for sample in samples], dim=0),
            "gr_data": torch.stack([sample["flo"]["gr_data"] for sample in samples], dim=0),
        },
        "label": torch.tensor([sample["label"] for sample in samples], dtype=torch.long),
        "class_name": [sample["class_name"] for sample in samples],
        "sample_key": [sample["sample_key"] for sample in samples],
        "image_pair_lr_paths": [sample["image_pair_lr_paths"] for sample in samples],
        "image_pair_gr_paths": [sample["image_pair_gr_paths"] for sample in samples],
        "flo_lr_paths": [sample["flo_lr_paths"] for sample in samples],
        "flo_gr_paths": [sample["flo_gr_paths"] for sample in samples],
    }


def sr_paired_collate_fn(batch: list[dict]) -> dict | None:
    """
    DataLoader 的 collate_fn，返回对齐后的多模态 batch。
    collate 后返回一个对齐 batch：

    batch["image_pair"]["previous"]["lr_data"]
    batch["image_pair"]["previous"]["gr_data"]
    batch["image_pair"]["next"]["lr_data"]
    batch["image_pair"]["next"]["gr_data"]

    batch["flo"]["lr_data"]
    batch["flo"]["gr_data"]

    对于同一个 batch 下的第 i 个样本，
    image_pair 和 flo 一定对应同一个 sample_key。
    """
    return build_aligned_batch(batch)


# ==============================
# 调试工具
# ==============================

def print_batch_debug_info(batch: dict | None, batch_name: str = "batch") -> None:
    """
    打印一个 batch 的完整结构，方便调试。
    打印 batch 结构、形状、路径和 sample_key 对齐信息。
    """
    logger.info("=" * 80)
    logger.info(f"[Debug] {batch_name} structure")
    if batch is None:
        logger.info("[Debug] batch is None")
        logger.info("=" * 80)
        return

    logger.info(f"[Debug] top-level keys: {list(batch.keys())}")

    image_pair = batch.get("image_pair")
    flo = batch.get("flo")

    if image_pair is not None:
        logger.info("[Debug] image_pair sub-batch:")
        logger.info(f"  previous lr_data shape: {tuple(image_pair['previous']['lr_data'].shape)}")
        logger.info(f"  previous gr_data shape: {tuple(image_pair['previous']['gr_data'].shape)}")
        logger.info(f"  next lr_data shape: {tuple(image_pair['next']['lr_data'].shape)}")
        logger.info(f"  next gr_data shape: {tuple(image_pair['next']['gr_data'].shape)}")
    else:
        logger.info("[Debug] image_pair sub-batch: None")

    if flo is not None:
        logger.info("[Debug] flo sub-batch:")
        logger.info(f"  lr_data shape: {tuple(flo['lr_data'].shape)}")
        logger.info(f"  gr_data shape: {tuple(flo['gr_data'].shape)}")
    else:
        logger.info("[Debug] flo sub-batch: None")

    sample_keys = batch.get("sample_key", [])
    class_names = batch.get("class_name", [])
    image_pair_lr_paths = batch.get("image_pair_lr_paths", [])
    image_pair_gr_paths = batch.get("image_pair_gr_paths", [])
    flo_lr_paths = batch.get("flo_lr_paths", [])
    flo_gr_paths = batch.get("flo_gr_paths", [])

    logger.info(f"[Debug] aligned sample count: {len(sample_keys)}")
    for index, sample_key in enumerate(sample_keys):
        class_name = class_names[index] if index < len(class_names) else "UNKNOWN"
        image_pair_lr_path = image_pair_lr_paths[index] if index < len(image_pair_lr_paths) else "N/A"
        image_pair_gr_path = image_pair_gr_paths[index] if index < len(image_pair_gr_paths) else "N/A"
        flo_lr_path = flo_lr_paths[index] if index < len(flo_lr_paths) else "N/A"
        flo_gr_path = flo_gr_paths[index] if index < len(flo_gr_paths) else "N/A"

        logger.info(f"[Debug] sample[{index}] key={sample_key}, class={class_name}")
        logger.info(f"  image_pair lr_paths: {image_pair_lr_path}")
        logger.info(f"  image_pair gr_paths: {image_pair_gr_path}")
        logger.info(f"  flo lr_paths: {flo_lr_path}")
        logger.info(f"  flo gr_paths: {flo_gr_path}")

    logger.info("=" * 80)


"""
loader 序列化存储和读取 start

# 1) 单独保存 train_loader
save_loader_paths(train_loader, "splits.json", "train")

# 2) 批量保存 train + test
save_loaders_paths("splits.json", train_loader=train_loader, test_loader=test_loader)

# 3) 单独读取 validate_loader
validate_loader = load_loader_paths(
    split_json_path="splits.json",
    split_name="validate",
    class_to_idx=class_to_idx,
    image_transform=image_transform,
    flow_transform=flow_transform,
    batch_size=16,
)

# 4) 批量读取 train + test
loaded = load_loaders_paths(
    split_json_path="splits.json",
    class_to_idx=class_to_idx,
    image_transform=image_transform,
    flow_transform=flow_transform,
    splits=("train", "test"),
    batch_size=16,
    train_shuffle=True,
)
train_loader = loaded["train"]
test_loader = loaded["test"]
"""
def _serialize_samples(samples: list[dict]) -> list[dict[str, Any]]:
    """
    将样本列表转换为可 JSON 序列化结构（Path -> str）。

    参数:
    - samples: SRPairedDataset.samples 列表

    返回:
    - list[dict]: 可直接写入 JSON 的样本信息
    """
    data = []
    for s in samples:
        data.append({
            "class_name": s["class_name"],
            "sample_key": s["sample_key"],
            "image_pair": {
                "gr_paths": [str(p) for p in s["image_pair"]["gr_paths"]],
                "lr_paths": [str(p) for p in s["image_pair"]["lr_paths"]],
            },
            "flo": {
                "gr_paths": [str(p) for p in s["flo"]["gr_paths"]],
                "lr_paths": [str(p) for p in s["flo"]["lr_paths"]],
            },
        })
    return data


def _deserialize_samples(data: list[dict[str, Any]]) -> list[dict]:
    """
    将 JSON 结构恢复为样本列表（str -> Path）。

    参数:
    - data: 从 JSON 读取出的样本结构

    返回:
    - list[dict]: 可用于 SRPairedDataset 的 samples
    """
    samples = []
    for s in data:
        samples.append({
            "class_name": s["class_name"],
            "sample_key": s["sample_key"],
            "image_pair": {
                "gr_paths": [Path(x) for x in s["image_pair"]["gr_paths"]],
                "lr_paths": [Path(x) for x in s["image_pair"]["lr_paths"]],
            },
            "flo": {
                "gr_paths": [Path(x) for x in s["flo"]["gr_paths"]],
                "lr_paths": [Path(x) for x in s["flo"]["lr_paths"]],
            },
        })
    return samples


def _build_loader_from_samples(
    samples: list[dict],
    class_to_idx: dict[str, int],
    image_transform,
    flow_transform,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    verbose: bool = False,
):
    """
    根据样本列表构建 DataLoader。

    参数:
    - samples: 样本列表
    - class_to_idx: 类别映射
    - image_transform: 图像变换
    - flow_transform: 光流变换
    - batch_size: 批大小
    - shuffle: 是否打乱
    - num_workers: 子进程数
    - verbose: 数据集日志开关

    返回:
    - DataLoader
    """
    dataset = SRPairedDataset(
        samples=samples,
        class_to_idx=class_to_idx,
        image_transform=image_transform,
        flow_transform=flow_transform,
        verbose=verbose,
        cache_file_reads=True,
        cache_transformed_samples=False,
        tensor_cache_root=None,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=sr_paired_collate_fn,
    )


def save_loader_paths(loader, out_path: str, split_name: str):
    """
    单独保存某一个 loader 的样本路径。

    参数:
    - loader: train/validate/test 任一 DataLoader
    - out_path: 输出 JSON 文件路径
    - split_name: split 名称（如 "train"/"validate"/"test"）
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {}
    if out.exists():
        payload = json.loads(out.read_text(encoding="utf-8"))

    payload[split_name] = _serialize_samples(loader.dataset.samples) if loader is not None else []
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_loaders_paths(
    out_path: str,
    train_loader=None,
    validate_loader=None,
    test_loader=None,
):
    """
    批量保存多个 loader（可任意组合）。

    参数:
    - out_path: 输出 JSON 文件路径
    - train_loader/validate_loader/test_loader: 传入哪个就保存哪个
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {}
    if train_loader is not None:
        payload["train"] = _serialize_samples(train_loader.dataset.samples)
    if validate_loader is not None:
        payload["validate"] = _serialize_samples(validate_loader.dataset.samples)
    if test_loader is not None:
        payload["test"] = _serialize_samples(test_loader.dataset.samples)

    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_loader_paths(
    split_json_path: str,
    split_name: str,
    class_to_idx: dict[str, int],
    image_transform,
    flow_transform,
    batch_size: int = 4,
    num_workers: int = 0,
    shuffle: bool = False,
    verbose: bool = False,
):
    """
    单独读取某个 split 并重建一个 loader。

    参数:
    - split_json_path: 保存路径 JSON 文件
    - split_name: "train"/"validate"/"test"
    - class_to_idx: 类别映射
    - image_transform: 图像变换
    - flow_transform: 光流变换
    - batch_size: 批大小
    - num_workers: 子进程数
    - shuffle: 是否打乱
    - verbose: 数据集日志开关

    返回:
    - DataLoader | None: split 不存在或为空时返回 None
    """
    p = Path(split_json_path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    samples_data = payload.get(split_name, [])
    if not samples_data:
        return None

    samples = _deserialize_samples(samples_data)
    return _build_loader_from_samples(
        samples=samples,
        class_to_idx=class_to_idx,
        image_transform=image_transform,
        flow_transform=flow_transform,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        verbose=verbose,
    )


def load_loaders_paths(
    split_json_path: str,
    class_to_idx: dict[str, int],
    image_transform,
    flow_transform,
    batch_size: int = 4,
    num_workers: int = 0,
    train_shuffle: bool = True,
    verbose: bool = False,
    splits: tuple[str, ...] = ("train", "validate", "test"),
):
    """
    批量读取多个 split 并重建 loader（可任意组合）。

    参数:
    - split_json_path: 保存路径 JSON 文件
    - class_to_idx: 类别映射
    - image_transform: 图像变换
    - flow_transform: 光流变换
    - batch_size: 批大小
    - num_workers: 子进程数
    - train_shuffle: train 是否打乱
    - verbose: 数据集日志开关
    - splits: 要读取的 split 名称元组，如 ("train","test")

    返回:
    - dict[str, DataLoader | None]:
      例如 {"train": train_loader, "validate": None, "test": test_loader}
    """
    p = Path(split_json_path)
    payload = json.loads(p.read_text(encoding="utf-8"))

    result: dict[str, DataLoader | None] = {}
    for split_name in splits:
        samples_data = payload.get(split_name, [])
        if not samples_data:
            result[split_name] = None
            continue

        samples = _deserialize_samples(samples_data)
        split_shuffle = train_shuffle if split_name == "train" else False
        result[split_name] = _build_loader_from_samples(
            samples=samples,
            class_to_idx=class_to_idx,
            image_transform=image_transform,
            flow_transform=flow_transform,
            batch_size=batch_size,
            shuffle=split_shuffle,
            num_workers=num_workers,
            verbose=verbose,
        )
    return result
"""
loader 序列化存储和读取 end
"""

# ==============================
# 主加载接口
# ==============================

def load_data(
    gr_data_root_dir: str,
    lr_data_root_dir: str,
    lr_data_variant: str = "default",
    selected_classes: str | list[str] | tuple[str, ...] | None = None,
    batch_size: int = 4,
    num_workers: int = 24,
    shuffle: bool = True,
    target_size: tuple[int, int] | None = None,
    train_nums_rate: float = 0.8,
    validate_nums_rate: float | None = None,
    test_nums_rate: float = 0.0,
    random_seed: int = 42,
    verbose: bool = False,
    use_metadata_cache: bool = True,
    cache_file_reads: bool = True,
    cache_transformed_samples: bool = False,
    use_disk_tensor_cache: bool = False,
    return_test_loader: bool = False,
):
    """
    加载超分辨率 GR/LR 成对数据集。

    保留的全部需求：
    1. 两个根目录：GR / LR
    2. 自动获取类别名，支持按指定类别加载
    3. 每个类别内分成 `image_pair` 和 `flo`
    4. image_pair 和 flo 在样本级别严格一一对应
    5. image_pair 保持 RGB，不转灰度
    6. flo 输出 3 通道：u / v / magnitude
    7. 训练集与验证集按类别分别划分（selected_classes=None 时全局随机三划分）
    8. batch 输出结构中同时包含：
       - batch["image_pair"]["previous"]["lr_data"], batch["image_pair"]["previous"]["gr_data"]
       - batch["image_pair"]["next"]["lr_data"], batch["image_pair"]["next"]["gr_data"]
       - batch["flo"]["lr_data"], batch["flo"]["gr_data"]
    9. 提供详细日志和 batch 调试函数

    参数说明（超参数）:
        - gr_data_root_dir/lr_data_root_dir: GR/LR 数据根目录
        - selected_classes: 只加载指定类别，None 表示全部类别
        - batch_size: DataLoader 批大小
        - num_workers: DataLoader 并行加载进程数
        - shuffle: 是否打乱训练集
        - target_size: 统一 resize 尺寸 (H, W)，None 表示保持原尺寸
        - train_nums_rate: 训练集比例（按类内划分或全局划分）
        - validate_nums_rate: 验证集比例，None 时自动=1-train_nums_rate-test_nums_rate
        - test_nums_rate: 测试集比例
        - random_seed: 划分随机种子
        - verbose: 是否输出详细逐文件日志
        - use_metadata_cache: 是否启用样本元数据缓存
        - cache_file_reads: 是否启用文件读取级缓存（lru）
        - cache_transformed_samples: 是否缓存变换后的样本到内存
        - use_disk_tensor_cache: 是否将样本张量缓存到磁盘
        - return_test_loader: 是否在返回值中包含 test_loader
    """
    gr_root = ensure_valid_root_dir(gr_data_root_dir, "gr_data_root_dir")
    resolved_lr_data_root_dir = resolve_lr_data_root_dir(lr_data_root_dir, lr_data_variant)
    lr_root = ensure_valid_root_dir(resolved_lr_data_root_dir, "lr_data_root_dir")

    is_global_mix_split = selected_classes is None

    logger.info("=" * 80)
    logger.info("[Start] Begin loading SR dataset")
    logger.info(f"[Start] GR root: {gr_root}")
    logger.info(f"[Start] LR root: {lr_root} (variant={lr_data_variant})")
    logger.info(f"[Start] batch_size={batch_size}, num_workers={num_workers}, shuffle={shuffle}")
    logger.info(f"[Start] target_size={target_size}, random_seed={random_seed}")
    logger.info(f"[Start] selected_classes={selected_classes}")
    logger.info(f"[Start] split_mode={'global_mix' if is_global_mix_split else 'per_class'}")
    logger.info(f"[Start] verbose={verbose}")
    logger.info(
        f"[Start] use_metadata_cache={use_metadata_cache}, "
        f"cache_file_reads={cache_file_reads}, "
        f"cache_transformed_samples={cache_transformed_samples}, "
        f"use_disk_tensor_cache={use_disk_tensor_cache}"
    )
    logger.info("=" * 80)

    available_class_names = get_class_names(gr_data_root_dir)
    class_names = normalize_selected_classes(available_class_names, selected_classes)

    logger.info(f"[Info] Selected {len(class_names)} classes: {class_names}")
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    logger.info(f"[Info] class_to_idx mapping: {class_to_idx}")

    image_transform = transforms.Compose(
        [transforms.ToTensor()] + ([transforms.Resize(target_size)] if target_size else [])
    )
    flow_transform = transforms.Compose(
        ([FlowResize(target_size)] if target_size else []) + [FloToTensor()]
    )

    logger.info(f"[Transform] Image transform: {image_transform}")
    logger.info(f"[Transform] Flow transform: {flow_transform}")

    cache_dir = Path(__file__).resolve().parent / ".sr_cache"
    cache_path = build_metadata_cache_path(cache_dir, gr_root, lr_root, class_names)
    tensor_cache_root = build_tensor_cache_root(
        cache_dir,
        gr_root,
        lr_root,
        class_names,
        target_size,
    ) if use_disk_tensor_cache else None

    samples: list[dict]
    if use_metadata_cache and cache_path.exists():
        logger.info(f"[Cache] Loading paired sample metadata from: {cache_path}")
        samples = load_samples_cache(cache_path)
    else:
        samples = []
        for class_name in progress_iter(class_names, desc="Collect SR samples", total=len(class_names)):
            gr_class_dir = gr_root / class_name / class_name
            lr_class_dir = lr_root / class_name / class_name
            class_samples = pair_sr_class_samples(
                class_name,
                gr_class_dir,
                lr_class_dir,
                verbose=verbose,
            )
            samples.extend(class_samples)
            log_info(f"[Summary] Class '{class_name}' contributes {len(class_samples)} paired samples", verbose)

        if use_metadata_cache:
            dump_samples_cache(cache_path, samples)
            logger.info(f"[Cache] Saved paired sample metadata to: {cache_path}")

    if not samples:
        logger.error('No paired GR/LR samples found.')
        raise ValueError("No paired GR/LR samples found.")

    if validate_nums_rate is None:
        validate_nums_rate = 1 - train_nums_rate - test_nums_rate

    if not 0 < train_nums_rate < 1:
        logger.error(f'train_nums_rate must be between 0 and 1, got {train_nums_rate}')
        raise ValueError(f"train_nums_rate must be between 0 and 1, got {train_nums_rate}")
    if not 0 <= validate_nums_rate < 1:
        logger.error(f'validate_nums_rate must be between 0 and 1, got {validate_nums_rate}')
        raise ValueError(f"validate_nums_rate must be between 0 and 1, got {validate_nums_rate}")
    if not 0 <= test_nums_rate < 1:
        logger.error(f'test_nums_rate must be between 0 and 1, got {test_nums_rate}')
        raise ValueError(f"test_nums_rate must be between 0 and 1, got {test_nums_rate}")
    if abs((train_nums_rate + validate_nums_rate + test_nums_rate) - 1.0) > 1e-8:
        logger.error(f'train_nums_rate + validate_nums_rate + test_nums_rate must equal 1.0, got {train_nums_rate + validate_nums_rate + test_nums_rate}')
        raise ValueError(
            "train_nums_rate + validate_nums_rate + test_nums_rate must equal 1.0, "
            f"got {train_nums_rate + validate_nums_rate + test_nums_rate}"
        )

    # 关键修改：
    # 1) selected_classes is None -> 全局混合三划分
    # 2) selected_classes is not None -> 按类别三划分（你要求的新逻辑）
    if is_global_mix_split:
        train_samples, validate_samples, test_samples = split_samples_global(
            samples=samples,
            train_nums_rate=train_nums_rate,
            validate_nums_rate=validate_nums_rate,
            test_nums_rate=test_nums_rate,
            random_seed=random_seed,
        )
        validate_grouped = group_samples_by_class(validate_samples, available_class_names, other_name="other")
        test_grouped = group_samples_by_class(test_samples, available_class_names, other_name="other")
        # 这里再展平成 list，不是多此一举：
        # Dataset 仍然吃扁平样本列表，但顺序已经按类别整理好，同时 metadata 里也保留了真正分组结果。
        validate_samples = [sample for bucket_samples in validate_grouped.values() for sample in bucket_samples]
        test_samples = [sample for bucket_samples in test_grouped.values() for sample in bucket_samples]
        # 用统计方式生成每类汇总，保持打印信息完整
        train_counter = Counter([s["class_name"] for s in train_samples])
        val_counter = Counter([s["class_name"] for s in validate_samples])
        test_counter = Counter([s["class_name"] for s in test_samples])
        total_counter = Counter([s["class_name"] for s in samples])

        split_summary = {
            c: {
                "total": total_counter.get(c, 0),
                "train": train_counter.get(c, 0),
                "val": val_counter.get(c, 0),
                "test": test_counter.get(c, 0),
            }
            for c in class_names
        }
    else:
        train_samples, validate_samples, test_samples, split_summary = split_samples_by_class_three_way(
            samples=samples,
            class_names=class_names,
            train_nums_rate=train_nums_rate,
            validate_nums_rate=validate_nums_rate,
            test_nums_rate=test_nums_rate,
            random_seed=random_seed,
        )

    train_size = len(train_samples)
    val_size = len(validate_samples)
    test_size = len(test_samples)
    total_samples = len(samples)

    if train_size == 0:
        logger.error('Split produced an empty training set.')
        raise ValueError("Split produced an empty training set.")
    if val_size == 0:
        logger.warning("[Warn] Split produced an empty validation set.")
    if test_nums_rate > 0 and test_size == 0:
        logger.warning("[Warn] test_nums_rate > 0 but split produced an empty test set.")

    train_dataset = SRPairedDataset(
        samples=train_samples,
        class_to_idx=class_to_idx,
        image_transform=image_transform,
        flow_transform=flow_transform,
        verbose=verbose,
        cache_file_reads=cache_file_reads,
        cache_transformed_samples=cache_transformed_samples,
        tensor_cache_root=tensor_cache_root,
    )
    validate_dataset = SRPairedDataset(
        samples=validate_samples,
        class_to_idx=class_to_idx,
        image_transform=image_transform,
        flow_transform=flow_transform,
        verbose=verbose,
        cache_file_reads=cache_file_reads,
        cache_transformed_samples=cache_transformed_samples,
        tensor_cache_root=tensor_cache_root,
    )

    test_dataset = SRPairedDataset(
        samples=test_samples,
        class_to_idx=class_to_idx,
        image_transform=image_transform,
        flow_transform=flow_transform,
        verbose=verbose,
        cache_file_reads=cache_file_reads,
        cache_transformed_samples=cache_transformed_samples,
        tensor_cache_root=tensor_cache_root,
    ) if test_size > 0 else None

    attach_grouped_class_metadata(validate_dataset, available_class_names, other_name="other")
    if test_dataset is not None:
        attach_grouped_class_metadata(test_dataset, available_class_names, other_name="other")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=sr_paired_collate_fn,
    )
    validate_dataloader = DataLoader(
        validate_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=sr_paired_collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=sr_paired_collate_fn,
    ) if test_dataset is not None else None

    logger.info("=" * 80)
    logger.info(f"[Done] Available classes: {available_class_names}")
    logger.info(f"[Done] Loaded classes: {class_names}")
    logger.info(f"[Done] Total paired samples: {total_samples}")
    logger.info(
        f"[Done] Split dataset with train_nums_rate={train_nums_rate}, "
        f"validate_nums_rate={validate_nums_rate}, test_nums_rate={test_nums_rate}"
    )
    logger.info(f"[Done] Train samples: {train_size}")
    logger.info(f"[Done] Validate samples: {val_size}")
    logger.info(f"[Done] Test samples: {test_size}")

    if is_global_mix_split:
        for bucket_name, bucket_samples in validate_dataset.grouped_samples_by_class.items():
            logger.info(f"[Done] Validate bucket '{bucket_name}': {len(bucket_samples)}")
        if test_dataset is not None:
            for bucket_name, bucket_samples in test_dataset.grouped_samples_by_class.items():
                logger.info(f"[Done] Test bucket '{bucket_name}': {len(bucket_samples)}")

    for class_name in class_names:
        class_summary = split_summary.get(class_name, {"total": 0, "train": 0, "val": 0, "test": 0})
        logger.info(
            f"[Done] Class '{class_name}': "
            f"total={class_summary['total']}, "
            f"train={class_summary['train']}, "
            f"val={class_summary['val']}, "
            f"test={class_summary['test']}"
        )
    logger.info("=" * 80)

    if return_test_loader:
        return train_dataloader, validate_dataloader, test_dataloader, class_names, samples
    return train_dataloader, validate_dataloader, class_names, samples


# ==============================
# 示例入口
# ==============================

if __name__ == "__main__":
    available_class_names = get_class_names(GR_DATA_ROOT_DIR, LR_DATA_ROOT_DIR)
    logger.info(f"available_class_names: {available_class_names}")

    # 示例1：selected_classes=None -> 全类别混合三划分
    train_loader, validate_loader, test_loader, class_names, samples = load_data(
        gr_data_root_dir=GR_DATA_ROOT_DIR,
        lr_data_root_dir=LR_DATA_ROOT_DIR,
        selected_classes=None,
        train_nums_rate=0.8,
        validate_nums_rate=0.2,
        test_nums_rate=0.0,
        return_test_loader=True,
    )

    # 示例2：selected_classes!=None -> 按类别三划分（新逻辑）
    # train_loader, validate_loader, test_loader, class_names, samples = load_data(
    #     gr_data_root_dir=GR_DATA_ROOT_DIR,
    #     lr_data_root_dir=LR_DATA_ROOT_DIR,
    #     selected_classes=available_class_names[:1],
    #     train_nums_rate=0.7,
    #     validate_nums_rate=0.2,
    #     test_nums_rate=0.1,
    #     return_test_loader=True,
    # )

    logger.info(f"class_names: {class_names}")
    logger.info(f"num_samples: {len(samples)}")

    first_train_batch = next(iter(train_loader))
    print_batch_debug_info(first_train_batch, batch_name="train_batch")
    logger.info(f"train image_pair previous lr_data shape: {first_train_batch['image_pair']['previous']['lr_data'].shape}")
    logger.info(f"train image_pair previous gr_data shape: {first_train_batch['image_pair']['previous']['gr_data'].shape}")
    logger.info(f"train image_pair next lr_data shape: {first_train_batch['image_pair']['next']['lr_data'].shape}")
    logger.info(f"train image_pair next gr_data shape: {first_train_batch['image_pair']['next']['gr_data'].shape}")
    logger.info(f"train flo lr_data shape: {first_train_batch['flo']['lr_data'].shape}")
    logger.info(f"train flo gr_data shape: {first_train_batch['flo']['gr_data'].shape}")

    if len(validate_loader.dataset) > 0:
        first_validate_batch = next(iter(validate_loader))
        print_batch_debug_info(first_validate_batch, batch_name="validate_batch")
        logger.info(f"validate image_pair previous lr_data shape: {first_validate_batch['image_pair']['previous']['lr_data'].shape}")
        logger.info(f"validate image_pair previous gr_data shape: {first_validate_batch['image_pair']['previous']['gr_data'].shape}")
        logger.info(f"validate image_pair next lr_data shape: {first_validate_batch['image_pair']['next']['lr_data'].shape}")
        logger.info(f"validate image_pair next gr_data shape: {first_validate_batch['image_pair']['next']['gr_data'].shape}")
        logger.info(f"validate flo lr_data shape: {first_validate_batch['flo']['lr_data'].shape}")
        logger.info(f"validate flo gr_data shape: {first_validate_batch['flo']['gr_data'].shape}")

    if test_loader is not None and len(test_loader.dataset) > 0:
        first_test_batch = next(iter(test_loader))
        print_batch_debug_info(first_test_batch, batch_name="test_batch")
