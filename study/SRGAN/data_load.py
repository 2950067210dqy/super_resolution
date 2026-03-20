#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import pickle
import re
from functools import lru_cache
from collections import defaultdict
from pathlib import Path

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

PAIR_KEY_PATTERNS = (
    re.compile(r"^(.*?)(?:[_-]?ti[_-]?\d+)$"),
    re.compile(r"^(.*?)(?:[_-]?t[_-]?i[_-]?\d+)$"),
    re.compile(r"^(.*?)(?:[_-]?(?:img|image|frame)[_-]?\d+)$"),
    re.compile(r"^(.*?)(?:[_-]?\d+)$"),
)
LR_SUFFIX_PATTERN = re.compile(r"([_-]lr)+$")
FLOW_LR_SUFFIX_PATTERN = re.compile(r"([_-]flow)+([_-]lr)+$")
FLOW_SUFFIX_PATTERN = re.compile(r"([_-]flow)+$")


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

    print(f"[Progress] {desc}")
    return iterable


def log_info(message: str, verbose: bool) -> None:
    """
    按需打印日志，避免大数据集下逐文件输出拖慢速度。
    """
    if verbose:
        print(message)


def _normalize_image_array(image: np.ndarray, image_path: str) -> np.ndarray:
    """
    把读取出的图像统一整理成 H x W x 3 的 numpy 数组。
    """
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    elif image.ndim == 3 and image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4):
        image = np.moveaxis(image, 0, -1)

    if image.ndim != 3:
        raise ValueError(f"Unsupported image shape: {image.shape}, file: {image_path}")

    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] >= 4:
        image = image[..., :3]

    return np.ascontiguousarray(image)


@lru_cache(maxsize=512)
def _read_rgb_image_cached(path_str: str) -> np.ndarray:
    return _normalize_image_array(tifffile.imread(path_str), path_str)


@lru_cache(maxsize=512)
def _read_flo_cached(path_str: str) -> np.ndarray:
    path = Path(path_str)
    header = np.fromfile(path, dtype=np.float32, count=1)
    if header.size != 1:
        raise ValueError(f"Incomplete .flo file header: {path}")

    magic = float(header[0])
    if abs(magic - FLO_MAGIC) > 1e-4:
        raise ValueError(f"Invalid .flo magic number: {path}")

    dims = np.fromfile(path, dtype=np.int32, count=2, offset=4)
    if dims.size != 2:
        raise ValueError(f"Failed to read width/height from .flo file: {path}")

    width, height = int(dims[0]), int(dims[1])
    flow_uv = np.fromfile(path, dtype=np.float32, offset=12)
    expected_size = 2 * width * height
    if flow_uv.size != expected_size:
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
    为当前数据配置生成稳定的元数据缓存文件路径。
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
    为当前数据配置生成张量缓存目录。
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
        raise ValueError(f"Please pass {root_name}.")

    root_path = Path(root_dir).expanduser().resolve()
    if not root_path.exists() or not root_path.is_dir():
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
    print(f"[Scan] Discovering class folders under: {data_root}")

    for candidate in progress_iter(sorted(data_root.iterdir()), desc="Discover classes"):
        if candidate.is_dir() and (candidate / candidate.name).is_dir():
            class_names.append(candidate.name)
            print(f"[Scan] Found class: {candidate.name}")

    return class_names


def get_class_names(gr_data_root_dir: str, lr_data_root_dir: str) -> list[str]:
    """
    对外公开的类别名获取函数。

    返回 GR 和 LR 根目录共同拥有的类别名交集。
    """
    gr_root = ensure_valid_root_dir(gr_data_root_dir, "gr_data_root_dir")
    lr_root = ensure_valid_root_dir(lr_data_root_dir, "lr_data_root_dir")

    gr_class_names = discover_class_names(gr_root)
    lr_class_names = discover_class_names(lr_root)
    class_names = sorted(set(gr_class_names) & set(lr_class_names))

    if not class_names:
        raise ValueError(
            f"No shared class folders found.\n"
            f"GR classes: {gr_class_names}\n"
            f"LR classes: {lr_class_names}"
        )

    return class_names


def normalize_selected_classes(
    available_class_names: list[str],
    selected_classes: str | list[str] | tuple[str, ...] | None,
) -> list[str]:
    """
    标准化外部传入的类别筛选参数。

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
        raise ValueError("selected_classes cannot be empty.")

    invalid_names = [name for name in normalized if name not in available_class_names]
    if invalid_names:
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
    统一配对时使用的样本键。

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
    ????????????? sample_key ???????????
    - 2 ????img1 / img2
    - 1 ? flo?flow
    """
    image_groups: dict[str, list[Path]] = defaultdict(list)
    flo_map: dict[str, Path] = {}

    if not class_dir.exists() or not class_dir.is_dir():
        raise ValueError(f"{domain_name} class directory does not exist: {class_dir}")

    class_files = sorted(class_dir.iterdir())
    print(f"[Scan] Reading {domain_name} class '{class_name}' from: {class_dir}")
    print(f"[Scan] Found {len(class_files)} entries in {domain_name} class '{class_name}'")

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
        print(f"[Debug] Incomplete {domain_name} samples in class '{class_name}':")
        for sample_key, reason in incomplete_keys[:50]:
            print(f"  key={sample_key}, reason={reason}")
        if len(incomplete_keys) > 50:
            print(f"[Debug] ... and {len(incomplete_keys) - 50} more incomplete keys")

    print(f"[Done] {domain_name} class '{class_name}' collected {len(samples)} complete samples")
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
    """
    ?? GR ? LR ???????????? sample_key ? GR/LR ???
    """
    gr_samples = collect_root_class_samples(class_name, gr_class_dir, "GR", verbose=verbose)
    lr_samples = collect_root_class_samples(class_name, lr_class_dir, "LR", verbose=verbose)

    gr_map = {sample["sample_key"]: sample for sample in gr_samples}
    lr_map = {sample["sample_key"]: sample for sample in lr_samples}

    common_keys = sorted(set(gr_map.keys()) & set(lr_map.keys()))
    gr_only_keys = sorted(set(gr_map.keys()) - set(lr_map.keys()))
    lr_only_keys = sorted(set(lr_map.keys()) - set(gr_map.keys()))

    if gr_only_keys:
        print(f"[Warn] GR-only sample keys in class '{class_name}': {gr_only_keys[:20]}")
        if len(gr_only_keys) > 20:
            print(f"[Warn] ... and {len(gr_only_keys) - 20} more")
    if lr_only_keys:
        print(f"[Warn] LR-only sample keys in class '{class_name}': {lr_only_keys[:20]}")
        if len(lr_only_keys) > 20:
            print(f"[Warn] ... and {len(lr_only_keys) - 20} more")

    if not common_keys:
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

    print(f"[Done] Class '{class_name}' paired {len(paired_samples)} aligned SR samples")
    return paired_samples



# ==============================
# 按类别划分 train / val
# ==============================

def split_samples_by_class(
    samples: list[dict],
    class_names: list[str],
    train_nums_rate: float,
    random_seed: int,
) -> tuple[list[dict], list[dict], dict[str, dict[str, int]]]:
    """
    按类别分别划分训练集和验证集。
    """
    train_samples: list[dict] = []
    validate_samples: list[dict] = []
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
        class_val_size = class_total - class_train_size

        if class_total >= 2:
            if class_train_size == 0:
                class_train_size = 1
                class_val_size = class_total - class_train_size
            if class_val_size == 0:
                class_val_size = 1
                class_train_size = class_total - class_val_size
        elif class_total == 1:
            class_train_size = 1
            class_val_size = 0

        train_samples.extend(shuffled_samples[:class_train_size])
        validate_samples.extend(shuffled_samples[class_train_size:])
        split_summary[class_name] = {
            "total": class_total,
            "train": class_train_size,
            "val": class_val_size,
        }

    return train_samples, validate_samples, split_summary


# ==============================
# Dataset / collate
# ==============================

class SRPairedDataset(Dataset):
    """
    超分辨率成对数据集。

    每个样本包含两种模态：
    - image_pair
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

    def _load_image_pair(self, paths: list[Path]) -> torch.Tensor:
        """
        读取两张 RGB 图像，并沿通道维拼接。

        每张图是 `3 x H x W`，两张拼接后变成 `6 x H x W`。
        """
        log_info(f"[Read] Loading image pair: {paths[0].name}, {paths[1].name}", self.verbose)
        image_tensors = []
        for image_path in paths:
            if self.cache_file_reads:
                image = _read_rgb_image_cached(str(image_path))
            else:
                image = _normalize_image_array(tifffile.imread(str(image_path)), str(image_path))

            image_tensors.append(self.image_transform(image))
        return torch.cat(image_tensors, dim=0)

    def __getitem__(self, index: int) -> dict:
        if self.cache_transformed_samples and index in self._sample_cache:
            cached = self._sample_cache[index]
            return {
                "image_pair": {
                    "lr_data": cached["image_pair"]["lr_data"].clone(),
                    "gr_data": cached["image_pair"]["gr_data"].clone(),
                },
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
                item = {
                    "image_pair": {
                        "lr_data": cached["image_pair"]["lr_data"],
                        "gr_data": cached["image_pair"]["gr_data"],
                    },
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
                        "image_pair": {
                            "lr_data": item["image_pair"]["lr_data"].clone(),
                            "gr_data": item["image_pair"]["gr_data"].clone(),
                        },
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

        item = {
            "image_pair": {
                "lr_data": self._load_image_pair(sample["image_pair"]["lr_paths"]),
                "gr_data": self._load_image_pair(sample["image_pair"]["gr_paths"]),
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
                    "image_pair": {
                        "lr_data": item["image_pair"]["lr_data"],
                        "gr_data": item["image_pair"]["gr_data"],
                    },
                    "flo": {
                        "lr_data": item["flo"]["lr_data"],
                        "gr_data": item["flo"]["gr_data"],
                    },
                },
                tensor_cache_path,
            )

        if self.cache_transformed_samples:
            self._sample_cache[index] = {
                "image_pair": {
                    "lr_data": item["image_pair"]["lr_data"].clone(),
                    "gr_data": item["image_pair"]["gr_data"].clone(),
                },
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
    把一组已经一一对应的多模态样本整理成 batch。
    """
    if not samples:
        return None

    return {
        "image_pair": {
            "lr_data": torch.stack([sample["image_pair"]["lr_data"] for sample in samples], dim=0),
            "gr_data": torch.stack([sample["image_pair"]["gr_data"] for sample in samples], dim=0),
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
    collate 后返回一个对齐 batch：

    batch["image_pair"]["lr_data"]
    batch["image_pair"]["gr_data"]
    batch["flo"]["lr_data"]
    batch["flo"]["gr_data"]

    对于同一个 batch 下的第 i 个样本，
    `image_pair` 和 `flo` 一定对应同一个 `sample_key`。
    """
    return build_aligned_batch(batch)


# ==============================
# 调试工具
# ==============================

def print_batch_debug_info(batch: dict | None, batch_name: str = "batch") -> None:
    """
    打印一个 batch 的完整结构，方便调试。
    """
    print("=" * 80)
    print(f"[Debug] {batch_name} structure")
    if batch is None:
        print("[Debug] batch is None")
        print("=" * 80)
        return

    print(f"[Debug] top-level keys: {list(batch.keys())}")

    image_pair = batch.get("image_pair")
    flo = batch.get("flo")

    if image_pair is not None:
        print("[Debug] image_pair sub-batch:")
        print(f"  lr_data shape: {tuple(image_pair['lr_data'].shape)}")
        print(f"  gr_data shape: {tuple(image_pair['gr_data'].shape)}")
    else:
        print("[Debug] image_pair sub-batch: None")

    if flo is not None:
        print("[Debug] flo sub-batch:")
        print(f"  lr_data shape: {tuple(flo['lr_data'].shape)}")
        print(f"  gr_data shape: {tuple(flo['gr_data'].shape)}")
    else:
        print("[Debug] flo sub-batch: None")

    sample_keys = batch.get("sample_key", [])
    class_names = batch.get("class_name", [])
    image_pair_lr_paths = batch.get("image_pair_lr_paths", [])
    image_pair_gr_paths = batch.get("image_pair_gr_paths", [])
    flo_lr_paths = batch.get("flo_lr_paths", [])
    flo_gr_paths = batch.get("flo_gr_paths", [])

    print(f"[Debug] aligned sample count: {len(sample_keys)}")
    for index, sample_key in enumerate(sample_keys):
        class_name = class_names[index] if index < len(class_names) else "UNKNOWN"
        image_pair_lr_path = image_pair_lr_paths[index] if index < len(image_pair_lr_paths) else "N/A"
        image_pair_gr_path = image_pair_gr_paths[index] if index < len(image_pair_gr_paths) else "N/A"
        flo_lr_path = flo_lr_paths[index] if index < len(flo_lr_paths) else "N/A"
        flo_gr_path = flo_gr_paths[index] if index < len(flo_gr_paths) else "N/A"

        print(f"[Debug] sample[{index}] key={sample_key}, class={class_name}")
        print(f"  image_pair lr_paths: {image_pair_lr_path}")
        print(f"  image_pair gr_paths: {image_pair_gr_path}")
        print(f"  flo lr_paths: {flo_lr_path}")
        print(f"  flo gr_paths: {flo_gr_path}")

    print("=" * 80)


# ==============================
# 主加载接口
# ==============================

def load_data(
    gr_data_root_dir: str,
    lr_data_root_dir: str,
    selected_classes: str | list[str] | tuple[str, ...] | None = None,
    batch_size: int = 4,
    num_workers: int = 12,
    shuffle: bool = True,
    target_size: tuple[int, int] | None = None,
    train_nums_rate: float = 0.8,
    validate_nums_rate: float | None = None,
    random_seed: int = 42,
    verbose: bool = False,
    use_metadata_cache: bool = True,
    cache_file_reads: bool = True,
    cache_transformed_samples: bool = False,
    use_disk_tensor_cache: bool = False,
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
    7. 训练集与验证集按类别分别划分
    8. batch 输出结构中同时包含：
       - batch["image_pair"]["lr_data"], batch["image_pair"]["gr_data"]
       - batch["flo"]["lr_data"], batch["flo"]["gr_data"]
    9. 提供详细日志和 batch 调试函数
    """
    gr_root = ensure_valid_root_dir(gr_data_root_dir, "gr_data_root_dir")
    lr_root = ensure_valid_root_dir(lr_data_root_dir, "lr_data_root_dir")

    print("=" * 80)
    print("[Start] Begin loading SR dataset")
    print(f"[Start] GR root: {gr_root}")
    print(f"[Start] LR root: {lr_root}")
    print(f"[Start] batch_size={batch_size}, num_workers={num_workers}, shuffle={shuffle}")
    print(f"[Start] target_size={target_size}, random_seed={random_seed}")
    print(f"[Start] verbose={verbose}")
    print(
        f"[Start] use_metadata_cache={use_metadata_cache}, "
        f"cache_file_reads={cache_file_reads}, "
        f"cache_transformed_samples={cache_transformed_samples}, "
        f"use_disk_tensor_cache={use_disk_tensor_cache}"
    )
    print("=" * 80)

    available_class_names = get_class_names(gr_data_root_dir, lr_data_root_dir)
    class_names = normalize_selected_classes(available_class_names, selected_classes)

    print(f"[Info] Selected {len(class_names)} classes: {class_names}")
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    print(f"[Info] class_to_idx mapping: {class_to_idx}")

    image_transform = transforms.Compose(
        [transforms.ToTensor()] + ([transforms.Resize(target_size)] if target_size else [])
    )
    flow_transform = transforms.Compose(
        ([FlowResize(target_size)] if target_size else []) + [FloToTensor()]
    )

    print(f"[Transform] Image transform: {image_transform}")
    print(f"[Transform] Flow transform: {flow_transform}")

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
        print(f"[Cache] Loading paired sample metadata from: {cache_path}")
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
            print(f"[Cache] Saved paired sample metadata to: {cache_path}")

    if not samples:
        raise ValueError("No paired GR/LR samples found.")

    if validate_nums_rate is None:
        validate_nums_rate = 1 - train_nums_rate

    if not 0 < train_nums_rate < 1:
        raise ValueError(f"train_nums_rate must be between 0 and 1, got {train_nums_rate}")
    if not 0 <= validate_nums_rate < 1:
        raise ValueError(f"validate_nums_rate must be between 0 and 1, got {validate_nums_rate}")
    if abs((train_nums_rate + validate_nums_rate) - 1.0) > 1e-8:
        raise ValueError(
            "train_nums_rate + validate_nums_rate must equal 1.0, "
            f"got {train_nums_rate + validate_nums_rate}"
        )

    train_samples, validate_samples, split_summary = split_samples_by_class(
        samples=samples,
        class_names=class_names,
        train_nums_rate=train_nums_rate,
        random_seed=random_seed,
    )

    train_size = len(train_samples)
    val_size = len(validate_samples)
    total_samples = len(samples)
    if train_size == 0:
        raise ValueError("Per-class split produced an empty training set.")
    if val_size == 0:
        print("[Warn] Per-class split produced an empty validation set. This usually means each class has only one sample.")

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

    print("=" * 80)
    print(f"[Done] Available classes: {available_class_names}")
    print(f"[Done] Loaded classes: {class_names}")
    print(f"[Done] Total paired samples: {total_samples}")
    print(
        f"[Done] Split dataset with train_nums_rate={train_nums_rate}, "
        f"validate_nums_rate={validate_nums_rate}"
    )
    print(f"[Done] Train samples: {train_size}")
    print(f"[Done] Validate samples: {val_size}")
    for class_name in class_names:
        class_summary = split_summary.get(class_name, {"total": 0, "train": 0, "val": 0})
        print(
            f"[Done] Class '{class_name}': "
            f"total={class_summary['total']}, "
            f"train={class_summary['train']}, "
            f"val={class_summary['val']}"
        )
    print("=" * 80)

    return train_dataloader, validate_dataloader, class_names, samples


# ==============================
# 示例入口
# ==============================

if __name__ == "__main__":
    available_class_names = get_class_names(GR_DATA_ROOT_DIR, LR_DATA_ROOT_DIR)
    print(f"available_class_names: {available_class_names}")

    train_loader, validate_loader, class_names, samples = load_data(
        gr_data_root_dir=GR_DATA_ROOT_DIR,
        lr_data_root_dir=LR_DATA_ROOT_DIR,
        selected_classes=available_class_names[:1] if available_class_names else None,
    )
    print(f"class_names: {class_names}")
    print(f"num_samples: {len(samples)}")

    first_train_batch = next(iter(train_loader))
    print_batch_debug_info(first_train_batch, batch_name="train_batch")
    print(f"train image_pair lr_data shape: {first_train_batch['image_pair']['lr_data'].shape}")
    print(f"train image_pair gr_data shape: {first_train_batch['image_pair']['gr_data'].shape}")
    print(f"train flo lr_data shape: {first_train_batch['flo']['lr_data'].shape}")
    print(f"train flo gr_data shape: {first_train_batch['flo']['gr_data'].shape}")

    if len(validate_loader.dataset) > 0:
        first_validate_batch = next(iter(validate_loader))
        print_batch_debug_info(first_validate_batch, batch_name="validate_batch")
        print(f"validate image_pair lr_data shape: {first_validate_batch['image_pair']['lr_data'].shape}")
        print(f"validate image_pair gr_data shape: {first_validate_batch['image_pair']['gr_data'].shape}")
        print(f"validate flo lr_data shape: {first_validate_batch['flo']['lr_data'].shape}")
        print(f"validate flo gr_data shape: {first_validate_batch['flo']['gr_data'].shape}")
