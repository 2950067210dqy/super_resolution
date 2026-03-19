#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


# 真实数据根路径
#真实数据根路径
GR_DATA_ROOT_DIR = rf"/study_datas/sr_dataset/class_1/data"
#低分辨率数据根地址
LR_DATA_ROOT_DIR = rf"/study_datas/sr_dataset/class_1_lr/x{4}/data"

FLO_MAGIC = 202021.25
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def progress_iter(iterable, desc: str, total: int | None = None):
    if tqdm is not None:
        return tqdm(iterable, desc=desc, total=total, leave=False)

    print(f"[Progress] {desc}")
    return iterable


def read_flo(path: Path) -> np.ndarray:
    print(f"[Read] Loading flow file: {path}")

    with path.open("rb") as file:
        magic = struct.unpack("f", file.read(4))[0]
        if abs(magic - FLO_MAGIC) > 1e-4:
            raise ValueError(f"Invalid .flo magic number: {path}")

        width = struct.unpack("i", file.read(4))[0]
        height = struct.unpack("i", file.read(4))[0]
        data = np.fromfile(file, np.float32, count=2 * width * height)

    if data.size != 2 * width * height:
        raise ValueError(f"Incomplete .flo data: {path}")

    return data.reshape(height, width, 2)


class FloToTensor:
    def __call__(self, flow: np.ndarray) -> torch.Tensor:
        if flow.ndim != 3 or flow.shape[-1] != 2:
            raise ValueError(f"Unexpected flow shape: {flow.shape}")

        flow = np.transpose(flow.astype(np.float32, copy=False), (2, 0, 1))
        return torch.from_numpy(flow)


class FlowResize:
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


def discover_class_names(data_root: Path) -> list[str]:
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

    返回 GR 和 LR 两个根目录里都存在的类别名交集，
    确保后续做超分辨率配对时每个类别都能同时找到真实数据和低分辨率数据。
    """
    if not gr_data_root_dir or not lr_data_root_dir:
        raise ValueError("Please pass both gr_data_root_dir and lr_data_root_dir.")

    gr_root = Path(gr_data_root_dir).expanduser().resolve()
    lr_root = Path(lr_data_root_dir).expanduser().resolve()
    if not gr_root.exists() or not gr_root.is_dir():
        raise ValueError(f"Invalid GR_DATA_ROOT_DIR: {gr_root}")
    if not lr_root.exists() or not lr_root.is_dir():
        raise ValueError(f"Invalid LR_DATA_ROOT_DIR: {lr_root}")

    gr_class_names = discover_class_names(gr_root)
    lr_class_names = discover_class_names(lr_root)
    class_names = sorted(set(gr_class_names) & set(lr_class_names))
    if not class_names:
        raise ValueError(
            f"No shared class folders found.\nGR classes: {gr_class_names}\nLR classes: {lr_class_names}"
        )

    return class_names


def normalize_selected_classes(
    available_class_names: list[str],
    selected_classes: str | list[str] | tuple[str, ...] | None,
) -> list[str]:
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


def infer_ti_group_key(file_stem: str) -> str:
    stem = file_stem.lower()
    patterns = [
        r"^(.*?)(?:[_-]?ti[_-]?\d+)$",
        r"^(.*?)(?:[_-]?t[_-]?i[_-]?\d+)$",
        r"^(.*?)(?:[_-]?(?:img|image|frame)[_-]?\d+)$",
        r"^(.*?)(?:[_-]?\d+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, stem)
        if match and match.group(1):
            return match.group(1)
    return stem


def normalize_pair_key(sample_key: str, domain_name: str) -> str:
    """
    统一 GR/LR 配对时使用的样本键。

    低分辨率数据常见命名形式会在文件名末尾额外带 `_lr`，
    这里在 LR 域中把这个后缀去掉，便于和 GR 文件名对齐。
    """
    normalized_key = sample_key.lower()
    if domain_name.upper() == "LR":
        normalized_key = re.sub(r"([_-]lr)+$", "", normalized_key)
    return normalized_key


def collect_root_class_samples(class_name: str, class_dir: Path, domain_name: str) -> list[dict]:
    """
    扫描单个根目录下某个类别的样本。

    这里仍沿用原来的归类规则：
    - `.flo` 作为单独样本
    - 名称中包含 `ti` 的图像按文件名归成 2 张一组
    """
    samples: list[dict] = []
    ti_groups: dict[str, list[Path]] = defaultdict(list)

    if not class_dir.exists() or not class_dir.is_dir():
        raise ValueError(f"{domain_name} class directory does not exist: {class_dir}")

    class_files = sorted(class_dir.iterdir())
    print(f"[Scan] Reading {domain_name} class '{class_name}' from: {class_dir}")
    print(f"[Scan] Found {len(class_files)} entries in {domain_name} class '{class_name}'")

    for file_path in progress_iter(class_files, desc=f"Scan {domain_name}:{class_name}", total=len(class_files)):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()
        stem = file_path.stem.lower()

        if suffix == ".flo":
            print(f"[Collect] {domain_name}:{class_name} flow sample -> {file_path.name}")
            samples.append(
                {
                    "sample_type": "flo",
                    "sample_key": normalize_pair_key(file_path.stem, domain_name),
                    "paths": [file_path],
                }
            )
            continue

        if suffix in IMAGE_EXTENSIONS and "ti" in stem:
            group_key = normalize_pair_key(infer_ti_group_key(file_path.stem), domain_name)
            ti_groups[group_key].append(file_path)
            print(f"[Collect] {domain_name}:{class_name} ti image -> {file_path.name}, group={group_key}")

    for group_key, image_paths in sorted(ti_groups.items()):
        image_paths = sorted(image_paths)
        if len(image_paths) != 2:
            raise ValueError(
                f"{domain_name} category '{class_name}' has invalid ti pair '{group_key}': {image_paths}"
            )

        print(
            f"[Pair] {domain_name}:{class_name} built ti_pair '{group_key}' with "
            f"{image_paths[0].name} and {image_paths[1].name}"
        )
        samples.append(
            {
                "sample_type": "ti_pair",
                "sample_key": group_key,
                "paths": image_paths,
            }
        )

    print(f"[Done] {domain_name} class '{class_name}' collected {len(samples)} samples")
    return samples


def pair_sr_class_samples(class_name: str, gr_class_dir: Path, lr_class_dir: Path) -> list[dict]:
    """
    将同一类别下的 GR 和 LR 样本按 sample_type + sample_key 进行配对。
    """
    gr_samples = collect_root_class_samples(class_name, gr_class_dir, "GR")
    lr_samples = collect_root_class_samples(class_name, lr_class_dir, "LR")

    gr_map = {(sample["sample_type"], sample["sample_key"]): sample for sample in gr_samples}
    lr_map = {(sample["sample_type"], sample["sample_key"]): sample for sample in lr_samples}

    common_keys = sorted(set(gr_map) & set(lr_map))
    gr_only_keys = sorted(set(gr_map) - set(lr_map))
    lr_only_keys = sorted(set(lr_map) - set(gr_map))

    if gr_only_keys:
        print(f"[Warn] GR-only samples in class '{class_name}': {gr_only_keys}")
    if lr_only_keys:
        print(f"[Warn] LR-only samples in class '{class_name}': {lr_only_keys}")
    if not common_keys:
        raise ValueError(f"No paired GR/LR samples found for class '{class_name}'")

    paired_samples: list[dict] = []
    for sample_type, sample_key in common_keys:
        paired_samples.append(
            {
                "class_name": class_name,
                "sample_type": sample_type,
                "sample_key": sample_key,
                "gr_paths": gr_map[(sample_type, sample_key)]["paths"],
                "lr_paths": lr_map[(sample_type, sample_key)]["paths"],
            }
        )

    print(f"[Done] Class '{class_name}' paired {len(paired_samples)} GR/LR samples")
    return paired_samples


def split_samples_by_class(
    samples: list[dict],
    class_names: list[str],
    train_nums_rate: float,
    random_seed: int,
) -> tuple[list[dict], list[dict], dict[str, dict[str, int]]]:
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


class SRPairedDataset(Dataset):
    """
    超分辨率成对数据集。

    每个样本都包含：
    - lr_data: 低分辨率数据
    - gr_data: 真实高分辨率数据
    - label: 类别索引
    - class_name / sample_type / sample_key
    - lr_paths / gr_paths: 原始文件路径
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

    def _load_image_pair(self, paths: list[Path]) -> torch.Tensor:
        print(f"[Read] Loading image pair: {paths[0].name}, {paths[1].name}")
        image_tensors = []
        for image_path in paths:
            with Image.open(image_path) as image:
                image_tensors.append(self.image_transform(image.convert("L")))
        return torch.cat(image_tensors, dim=0)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        label = self.class_to_idx[sample["class_name"]]

        if sample["sample_type"] == "flo":
            lr_data = self.flow_transform(read_flo(sample["lr_paths"][0]))
            gr_data = self.flow_transform(read_flo(sample["gr_paths"][0]))
        else:
            lr_data = self._load_image_pair(sample["lr_paths"])
            gr_data = self._load_image_pair(sample["gr_paths"])

        return {
            "lr_data": lr_data,
            "gr_data": gr_data,
            "label": label,
            "class_name": sample["class_name"],
            "sample_type": sample["sample_type"],
            "sample_key": sample["sample_key"],
            "lr_paths": " | ".join(str(path) for path in sample["lr_paths"]),
            "gr_paths": " | ".join(str(path) for path in sample["gr_paths"]),
        }


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

    训练/验证集划分方式保持不变：仍然是按类别分别划分。
    改动点只是每个样本现在同时包含 GR 和 LR 两份数据。
    """
    if not gr_data_root_dir or not lr_data_root_dir:
        raise ValueError("Please pass both gr_data_root_dir and lr_data_root_dir.")

    gr_root = Path(gr_data_root_dir).expanduser().resolve()
    lr_root = Path(lr_data_root_dir).expanduser().resolve()
    if not gr_root.exists() or not gr_root.is_dir():
        raise ValueError(f"Invalid GR_DATA_ROOT_DIR: {gr_root}")
    if not lr_root.exists() or not lr_root.is_dir():
        raise ValueError(f"Invalid LR_DATA_ROOT_DIR: {lr_root}")

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
        ([transforms.Resize(target_size)] if target_size else []) + [transforms.ToTensor()]
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
    )
    validate_dataloader = DataLoader(
        validate_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
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
    print(f"train lr_data shape: {first_train_batch['lr_data'].shape}")
    print(f"train gr_data shape: {first_train_batch['gr_data'].shape}")
    if len(validate_loader.dataset) > 0:
        first_validate_batch = next(iter(validate_loader))
        print(f"validate lr_data shape: {first_validate_batch['lr_data'].shape}")
        print(f"validate gr_data shape: {first_validate_batch['gr_data'].shape}")
