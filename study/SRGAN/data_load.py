#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
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
训练时的 batch 使用方式：

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

# Middlebury .flo 文件头标识
FLO_MAGIC = 202021.25

# 支持的图像扩展名
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


# ==============================
# 通用工具
# ==============================

def progress_iter(iterable, desc: str, total: int | None = None):
    """
    统一包装进度显示。
    """
    if tqdm is not None:
        return tqdm(iterable, desc=desc, total=total, leave=False)

    print(f"[Progress] {desc}")
    return iterable


def ensure_valid_root_dir(root_dir: str, root_name: str) -> Path:
    """
    校验根目录是否合法，并返回绝对路径。
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
    使用 numpy 直接读取 .flo 光流文件，返回 H x W x 3。

    3 个通道分别是：
    - u
    - v
    - magnitude = sqrt(u^2 + v^2)
    """
    print(f"[Read] Loading flow file: {path}")

    # 先按 float32 读取第一个数，用于校验 magic
    header_float = np.fromfile(path, dtype=np.float32, count=1)
    if header_float.size != 1:
        raise ValueError(f"Incomplete .flo file header: {path}")

    magic = float(header_float[0])
    if abs(magic - FLO_MAGIC) > 1e-4:
        raise ValueError(f"Invalid .flo magic number: {path}")

    # 宽高位于偏移 4 字节之后，按 int32 读取
    dims = np.fromfile(path, dtype=np.int32, count=2, offset=4)
    if dims.size != 2:
        raise ValueError(f"Failed to read width/height from .flo file: {path}")

    width, height = int(dims[0]), int(dims[1])

    # 从偏移 12 字节后读取真正的光流数据
    flow_uv = np.fromfile(path, dtype=np.float32, offset=12)
    expected_size = 2 * width * height
    if flow_uv.size != expected_size:
        raise ValueError(
            f"Incomplete .flo data: {path}, expected {expected_size}, got {flow_uv.size}"
        )

    flow_uv = flow_uv.reshape(height, width, 2)
    flow_magnitude = np.linalg.norm(flow_uv, axis=2, keepdims=True)
    return np.concatenate([flow_uv, flow_magnitude], axis=2)


class FloToTensor:
    """
    把 H x W x 3 的 flo 数组转成 3 x H x W 张量。
    """

    def __call__(self, flow: np.ndarray) -> torch.Tensor:
        if flow.ndim != 3 or flow.shape[-1] != 3:
            raise ValueError(f"Unexpected flow shape: {flow.shape}")

        flow = np.transpose(flow.astype(np.float32, copy=False), (2, 0, 1))
        return torch.from_numpy(flow)


class FlowResize:
    """
    对 flo 数据做 resize。
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
    获取 GR 和 LR 根目录共同拥有的类别名。
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
# 文件名到样本键的归一化
# ==============================

def infer_image_pair_key(file_stem: str) -> str:
    """
    从图像文件名里提取样本主键。

    例如：
    - backstep_Re800_00001_img1 -> backstep_re800_00001
    - backstep_Re800_00001_img2_lr -> backstep_re800_00001
    """
    stem = file_stem.lower()
    stem = re.sub(r"([_-]lr)+$", "", stem)

    patterns = [
        r"^(.*?)(?:[_-]?img[_-]?\d+)$",
        r"^(.*?)(?:[_-]?image[_-]?\d+)$",
        r"^(.*?)(?:[_-]?frame[_-]?\d+)$",
        r"^(.*?)(?:[_-]?ti[_-]?\d+)$",
        r"^(.*?)(?:[_-]?\d+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, stem)
        if match and match.group(1):
            return match.group(1)
    return stem


def normalize_pair_key(sample_key: str) -> str:
    """
    统一样本键，去掉末尾的 _lr 和 _flow。
    """
    normalized_key = sample_key.lower()
    normalized_key = re.sub(r"([_-]lr)+$", "", normalized_key)
    normalized_key = re.sub(r"([_-]flow)+$", "", normalized_key)
    return normalized_key


# ==============================
# 单个根目录内部样本打包
# ==============================

def collect_root_class_samples(class_name: str, class_dir: Path, domain_name: str) -> list[dict]:
    """
    在单个根目录内部先把同一个 sample_key 的 3 个文件打包成完整样本：
    - img1
    - img2
    - flow
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
            flo_map[sample_key] = file_path
            print(f"[Collect] {domain_name}:{class_name} flo -> {file_path.name}, key={sample_key}")
            continue

        if suffix in IMAGE_EXTENSIONS:
            sample_key = normalize_pair_key(infer_image_pair_key(file_path.stem))
            image_groups[sample_key].append(file_path)
            print(f"[Collect] {domain_name}:{class_name} image -> {file_path.name}, key={sample_key}")

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

        print(
            f"[Pack] {domain_name}:{class_name} sample_key={sample_key} "
            f"-> images={[path.name for path in image_paths]}, flo={flo_path.name}"
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
# GR / LR 成对配对
# ==============================

def pair_sr_class_samples(class_name: str, gr_class_dir: Path, lr_class_dir: Path) -> list[dict]:
    """
    先在 GR 和 LR 内部分别组完整样本，再按 sample_key 做 GR/LR 配对。
    """
    gr_samples = collect_root_class_samples(class_name, gr_class_dir, "GR")
    lr_samples = collect_root_class_samples(class_name, lr_class_dir, "LR")

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
# Dataset
# ==============================

class SRPairedDataset(Dataset):
    """
    超分辨率成对数据集。

    每个样本包含：
    - image_pair: 两张 RGB 图像，拼接后是 6 x H x W
    - flo: 3 通道光流，3 x H x W
    """

    def __init__(
        self,
        samples: list[dict],
        class_to_idx: dict[str, int],
        image_transform: transforms.Compose,
        flow_transform: transforms.Compose,
    ) -> None:
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.image_transform = image_transform
        self.flow_transform = flow_transform

    def __len__(self) -> int:
        return len(self.samples)

    def _read_tif_as_rgb(self, path: Path) -> np.ndarray:
        """
        使用 tifffile 读取 tif，并整理成 H x W x 3。
        """
        image = tifffile.imread(str(path))

        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.ndim == 3 and image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4):
            image = np.moveaxis(image, 0, -1)

        if image.ndim != 3:
            raise ValueError(f"Unsupported tif shape: {image.shape}, file: {path}")

        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        elif image.shape[-1] >= 4:
            image = image[..., :3]

        return image

    def _load_image_pair(self, paths: list[Path]) -> torch.Tensor:
        """
        读取两张 RGB 图像并沿通道维拼接。
        """
        print(f"[Read] Loading image pair: {paths[0].name}, {paths[1].name}")
        image_tensors = []
        for image_path in paths:
            image = self._read_tif_as_rgb(image_path)
            image_tensors.append(self.image_transform(image))
        return torch.cat(image_tensors, dim=0)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        label = self.class_to_idx[sample["class_name"]]

        return {
            "image_pair": {
                "lr_data": self._load_image_pair(sample["image_pair"]["lr_paths"]),
                "gr_data": self._load_image_pair(sample["image_pair"]["gr_paths"]),
            },
            "flo": {
                "lr_data": self.flow_transform(read_flo(sample["flo"]["lr_paths"][0])),
                "gr_data": self.flow_transform(read_flo(sample["flo"]["gr_paths"][0])),
            },
            "label": label,
            "class_name": sample["class_name"],
            "sample_key": sample["sample_key"],
            "image_pair_lr_paths": " | ".join(str(path) for path in sample["image_pair"]["lr_paths"]),
            "image_pair_gr_paths": " | ".join(str(path) for path in sample["image_pair"]["gr_paths"]),
            "flo_lr_paths": " | ".join(str(path) for path in sample["flo"]["lr_paths"]),
            "flo_gr_paths": " | ".join(str(path) for path in sample["flo"]["gr_paths"]),
        }


# ==============================
# collate_fn
# ==============================

def build_aligned_batch(samples: list[dict]) -> dict | None:
    """
    把一组样本整理成对齐 batch。
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
    返回对齐后的 batch 结构。
    """
    return build_aligned_batch(batch)


# ==============================
# 调试工具
# ==============================

def print_batch_debug_info(batch: dict | None, batch_name: str = "batch") -> None:
    """
    打印一个 batch 的结构、shape、sample_key 和路径信息。
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

    if flo is not None:
        print("[Debug] flo sub-batch:")
        print(f"  lr_data shape: {tuple(flo['lr_data'].shape)}")
        print(f"  gr_data shape: {tuple(flo['gr_data'].shape)}")

    sample_keys = batch.get("sample_key", [])
    class_names = batch.get("class_name", [])
    image_pair_lr_paths = batch.get("image_pair_lr_paths", [])
    image_pair_gr_paths = batch.get("image_pair_gr_paths", [])
    flo_lr_paths = batch.get("flo_lr_paths", [])
    flo_gr_paths = batch.get("flo_gr_paths", [])

    print(f"[Debug] aligned sample count: {len(sample_keys)}")
    for index, sample_key in enumerate(sample_keys):
        print(f"[Debug] sample[{index}] key={sample_key}, class={class_names[index]}")
        print(f"  image_pair lr_paths: {image_pair_lr_paths[index]}")
        print(f"  image_pair gr_paths: {image_pair_gr_paths[index]}")
        print(f"  flo lr_paths: {flo_lr_paths[index]}")
        print(f"  flo gr_paths: {flo_gr_paths[index]}")

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
):
    """
    加载超分辨率 GR/LR 成对数据集。
    """
    gr_root = ensure_valid_root_dir(gr_data_root_dir, "gr_data_root_dir")
    lr_root = ensure_valid_root_dir(lr_data_root_dir, "lr_data_root_dir")

    print("=" * 80)
    print("[Start] Begin loading SR dataset")
    print(f"[Start] GR root: {gr_root}")
    print(f"[Start] LR root: {lr_root}")
    print(f"[Start] batch_size={batch_size}, num_workers={num_workers}, shuffle={shuffle}")
    print(f"[Start] target_size={target_size}, random_seed={random_seed}")
    print("=" * 80)

    available_class_names = get_class_names(gr_data_root_dir, lr_data_root_dir)
    class_names = normalize_selected_classes(available_class_names, selected_classes)

    print(f"[Info] Selected {len(class_names)} classes: {class_names}")
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    print(f"[Info] class_to_idx mapping: {class_to_idx}")

    image_transform = transforms.Compose(
        [transforms.ToTensor()]
        + ([transforms.Resize(target_size)] if target_size else [])
    )
    flow_transform = transforms.Compose(
        ([FlowResize(target_size)] if target_size else []) + [FloToTensor()]
    )

    print(f"[Transform] Image transform: {image_transform}")
    print(f"[Transform] Flow transform: {flow_transform}")

    samples: list[dict] = []
    for class_name in progress_iter(class_names, desc="Collect SR samples", total=len(class_names)):
        gr_class_dir = gr_root / class_name / class_name
        lr_class_dir = lr_root / class_name / class_name
        class_samples = pair_sr_class_samples(class_name, gr_class_dir, lr_class_dir)
        samples.extend(class_samples)
        print(f"[Summary] Class '{class_name}' contributes {len(class_samples)} paired samples")

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
    )
    validate_dataset = SRPairedDataset(
        samples=validate_samples,
        class_to_idx=class_to_idx,
        image_transform=image_transform,
        flow_transform=flow_transform,
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
