from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from SRGAN.experiment_handle.global_class import global_data
except ModuleNotFoundError:
    from global_class import global_data


def natural_sort_key(path: Path) -> list[Any]:
    parts = global_data.NATURAL_SORT_PATTERN.split(path.name.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path))


def to_gray_float(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32, copy=False)
    if image.ndim == 3 and image.shape[2] >= 3:
        rgb = image[..., :3].astype(np.float32, copy=False)
        r_weight, g_weight, b_weight = global_data.GRAY_RGB_WEIGHTS
        return r_weight * rgb[..., 0] + g_weight * rgb[..., 1] + b_weight * rgb[..., 2]
    raise ValueError(f"Unsupported image shape: {image.shape}")


def subtract_background(image: np.ndarray, background: np.ndarray) -> np.ndarray:
    if image.shape != background.shape:
        raise ValueError(
            f"Image/background shape mismatch: image={image.shape}, background={background.shape}"
        )

    result = image.astype(np.int32) - background.astype(np.int32)
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        result = np.clip(result, info.min, info.max).astype(image.dtype)
    else:
        result = result.astype(image.dtype, copy=False)
    return result


def save_bmp(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def write_flo(path: Path, flow: np.ndarray) -> None:
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"flow must be HxWx2, got {flow.shape}")

    h, w, _ = flow.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("f", global_data.FLO_MAGIC))
        f.write(struct.pack("i", w))
        f.write(struct.pack("i", h))
        flow.astype(np.float32, copy=False).tofile(f)


def save_component_png(path: Path, component: np.ndarray) -> None:
    arr = np.asarray(component, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        image = np.zeros(arr.shape, dtype=np.uint8)
    else:
        max_abs = float(np.max(np.abs(finite)))
        if max_abs <= global_data.COMPONENT_PREVIEW_EPS:
            image = np.full(arr.shape, global_data.COMPONENT_PREVIEW_CENTER, dtype=np.uint8)
        else:
            scaled = (
                (np.nan_to_num(arr, nan=0.0) / max_abs + 1.0)
                * global_data.COMPONENT_PREVIEW_SCALE
            )
            image = np.clip(scaled, 0, 255).astype(np.uint8)

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="L").save(path)


def resize_sparse_flow_to_image(
    u: np.ndarray,
    v: np.ndarray,
    x: np.ndarray | None,
    y: np.ndarray | None,
    image_shape: tuple[int, int],
) -> np.ndarray:
    h, w = image_shape
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    if u.shape != v.shape:
        raise ValueError(f"u/v shape mismatch: u={u.shape}, v={v.shape}")

    if u.shape == (h, w):
        return np.stack([u, v], axis=-1).astype(np.float32, copy=False)

    if u.ndim != 2:
        raise ValueError(f"Expected 2D u/v displacement maps, got {u.shape}")

    if x is None or y is None:
        src_x = np.linspace(0.0, w - 1.0, u.shape[1], dtype=np.float32)
        src_y = np.linspace(0.0, h - 1.0, u.shape[0], dtype=np.float32)
    else:
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        src_x = x[0] if x.ndim == 2 else x
        src_y = y[:, 0] if y.ndim == 2 else y

    dst_x = np.arange(w, dtype=np.float32)
    dst_y = np.arange(h, dtype=np.float32)

    def interp2(field: np.ndarray) -> np.ndarray:
        rows = np.vstack([np.interp(dst_x, src_x, row) for row in field])
        cols = np.vstack([np.interp(dst_y, src_y, rows[:, col]) for col in range(w)]).T
        return cols.astype(np.float32, copy=False)

    return np.stack([interp2(u), interp2(v)], axis=-1)


def call_openpiv(previous: np.ndarray, next_image: np.ndarray) -> np.ndarray:
    module = __import__(
        global_data.OPENPIV_PACKAGE_NAME,
        fromlist=[global_data.OPENPIV_PROCESS_MODULE_NAME],
    )
    pyprocess = getattr(module, global_data.OPENPIV_PROCESS_MODULE_NAME)
    process_func = getattr(pyprocess, global_data.OPENPIV_PROCESS_FUNCTION_NAME)

    result = process_func(
        previous,
        next_image,
        window_size=global_data.OPENPIV_WINDOW_SIZE,
        overlap=global_data.OPENPIV_OVERLAP,
        dt=global_data.OPENPIV_DT,
        search_area_size=global_data.OPENPIV_SEARCH_AREA_SIZE,
        sig2noise_method=global_data.OPENPIV_SIG2NOISE_METHOD,
    )
    u, v = result[:2]
    x = y = None
    coordinate_func = getattr(pyprocess, global_data.OPENPIV_COORDINATE_FUNCTION_NAME, None)
    if global_data.OPENPIV_USE_COORDINATES and callable(coordinate_func):
        x, y = coordinate_func(
            image_size=previous.shape,
            search_area_size=global_data.OPENPIV_SEARCH_AREA_SIZE,
            overlap=global_data.OPENPIV_OVERLAP,
        )
    return resize_sparse_flow_to_image(u, v, x, y, previous.shape)


def compute_flow(previous: np.ndarray, next_image: np.ndarray) -> np.ndarray:
    prev_gray = to_gray_float(previous)
    next_gray = to_gray_float(next_image)
    if prev_gray.shape != next_gray.shape:
        raise ValueError(
            f"previous/next image shape mismatch: previous={prev_gray.shape}, next={next_gray.shape}"
        )

    try:
        return call_openpiv(prev_gray, next_gray)
    except ImportError as exc:
        raise ImportError(global_data.OPENPIV_IMPORT_ERROR_MESSAGE) from exc


def preprocess(
    input_dir: Path = global_data.INPUT_DIR,
    output_dir: Path = global_data.OUTPUT_DIR,
) -> None:
    background_path = input_dir / global_data.BACKGROUND_NAME
    if not background_path.exists():
        raise FileNotFoundError(f"Background image not found: {background_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    background = load_image(background_path)
    bmp_paths = sorted(
        (
            path
            for path in input_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in global_data.IMAGE_SUFFIXES
            and path.name.lower() != global_data.BACKGROUND_NAME.lower()
        ),
        key=natural_sort_key,
    )

    pair_count = len(bmp_paths) // 2
    if pair_count == 0:
        print("No complete image pairs found.")
        return

    discarded = len(bmp_paths) - pair_count * 2
    if discarded:
        print(f"Discarding last unpaired image: {bmp_paths[-1].name}")

    for pair_index in range(pair_count):
        group_name = global_data.GROUP_NAME_TEMPLATE.format(index=pair_index + 1)
        previous_path = bmp_paths[pair_index * 2]
        next_path = bmp_paths[pair_index * 2 + 1]

        previous = subtract_background(load_image(previous_path), background)
        next_image = subtract_background(load_image(next_path), background)

        save_bmp(output_dir / f"{group_name}{global_data.IMG1_SUFFIX}", previous)
        save_bmp(output_dir / f"{group_name}{global_data.IMG2_SUFFIX}", next_image)

        flow = compute_flow(previous, next_image)
        write_flo(output_dir / f"{group_name}{global_data.FLOW_SUFFIX}", flow)
        save_component_png(output_dir / f"{group_name}{global_data.FLOW_U_SUFFIX}", flow[..., 0])
        save_component_png(output_dir / f"{group_name}{global_data.FLOW_V_SUFFIX}", flow[..., 1])

        print(
            f"{group_name}: {previous_path.name} -> {next_path.name}, "
            f"flow_shape={flow.shape}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subtract bj.bmp from BMP image pairs and generate PIV flow files."
    )
    parser.add_argument("--input-dir", type=Path, default=global_data.INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=global_data.OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
