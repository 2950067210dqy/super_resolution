from loguru import logger
import torch
import numpy as np
from matplotlib import cm

from SRGAN.model.esrgan_update.global_class import global_data


def _to_gray( x: torch.Tensor,to_gray_one_channel = True) -> torch.Tensor:
    # 如果输入不是 [0,1]，而更像 [0,255]，先缩放到 [0,1]
    # if x.detach().max() > 1.0:
    #     x = x / 255.0
    if x.shape[1] == 1:
        return x
    if to_gray_one_channel:
        if global_data.esrgan.SAVE_AS_GRAY:
            return _select_metric_or_save_channels(x, data_type="image_pair", save_as_gray=global_data.esrgan.SAVE_AS_GRAY)
        else:
            return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    else:
        return x
def _select_metric_or_save_channels(x: torch.Tensor, data_type: str, save_as_gray: bool) -> torch.Tensor:
    """
    统一通道选择策略：
    - image_pair 且 SAVE_AS_GRAY=True: 仅使用第一个通道
    - 其他情况: 保持原通道
    """
    if data_type == "image_pair" and save_as_gray:
        if x.shape[1] < 1:
            logger.error(f'Expected at least 1 channel for image_pair, got {x.shape[1]}')
            raise ValueError(f"Expected at least 1 channel for image_pair, got {x.shape[1]}")
        return x[:, 0:1, :, :]
    return x
def add_vertical_separator(tensor, sep_width=8, value=1.0):
    """生成竖向白色分隔条，用于拼图。"""
    b, c, h, _ = tensor.shape
    return torch.full((b, c, h, sep_width), value, device=tensor.device, dtype=tensor.dtype)


def add_horizontal_separator(width, channels=3, sep_height=8, value=1.0, device="cpu", dtype=torch.float32):
    """生成横向白色分隔条，用于拼图。"""
    return torch.full((1, channels, sep_height, width), value, device=device, dtype=dtype)


def build_triplet_row(lr, fake, hr, sep_width=6):
    """拼接一行对比图：LR | Fake | HR。"""
    sep = add_vertical_separator(lr, sep_width=sep_width, value=1.0)
    return torch.cat([lr, sep, fake, sep, hr], dim=3)


def build_pair_row(left, right, sep_width=6):
    """拼接一行对比图：Left | Right。"""
    sep = add_vertical_separator(left, sep_width=sep_width, value=1.0)
    return torch.cat([left, sep, right], dim=3)


def to_gray_3ch(x):
    """
    把 3 通道图压成灰度后再复制为 3 通道，便于统一可视化接口。
    x: [B, 3, H, W]
    先压成灰度 [B,1,H,W]，再复制成 [B,3,H,W] 方便可视化拼接
    """
    gray = x.mean(dim=1, keepdim=True)
    return gray.repeat(1, 3, 1, 1)
def convert_fake_for_display(fake, fake_mode="rgb"):
    """
    根据显示模式返回 RGB 或灰度可视化版本。
    fake: [B, 3, H, W]
    fake_mode:
        - rgb: 原图显示
        - gray: 压成灰度显示
    """
    if fake_mode == "rgb":
        return fake
    if fake_mode == "gray":
        return to_gray_3ch(fake)
    logger.error(f'Unsupported fake_mode: {fake_mode}')
    raise ValueError(f"Unsupported fake_mode: {fake_mode}")
def _hsv_to_rgb_torch(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """将 HSV 张量转换为 RGB 张量。"""

    i = torch.floor(h * 6.0).to(torch.int64)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6

    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)

    m = i == 0
    r[m], g[m], b[m] = v[m], t[m], p[m]
    m = i == 1
    r[m], g[m], b[m] = q[m], v[m], p[m]
    m = i == 2
    r[m], g[m], b[m] = p[m], v[m], t[m]
    m = i == 3
    r[m], g[m], b[m] = p[m], q[m], v[m]
    m = i == 4
    r[m], g[m], b[m] = t[m], p[m], v[m]
    m = i == 5
    r[m], g[m], b[m] = v[m], p[m], q[m]

    return torch.stack([r, g, b], dim=1)  # [B,3,H,W]


def flow_to_color_tensor(flow: torch.Tensor, ref_max_rad: float | None = None) -> tuple[torch.Tensor, float]:
    """
    将光流 uv 通道映射为与 flo U/V/S 面板一致风格的伪彩 RGB 图。
    flow: [B,C,H,W], C>=2（可为2或3，若3则第3通道会被忽略）
    返回: rgb [B,3,H,W] in [0,1], max_rad

    说明：
        这里不再使用传统光流色轮 HSV 编码，
        而是改为和 fake_uvw_panel 一致的 jet 伪彩风格。
        具体做法是：
        1. 先由 uv 计算 magnitude
        2. 再将 magnitude 归一化到 [0,1]
        3. 最后用 jet colormap 映射到 RGB
        4. 最终返回标准可视化常用范围 [0,1]
    """
    if flow.ndim != 4 or flow.shape[1] < 2:
        logger.error(f'flow shape must be [B,C,H,W] and C>=2, got {tuple(flow.shape)}')
        raise ValueError(f"flow shape must be [B,C,H,W] and C>=2, got {tuple(flow.shape)}")

    # 只使用 uv，忽略可能存在的 magnitude 第三通道
    u = flow[:, 0]
    v = flow[:, 1]

    mag = torch.nan_to_num(torch.sqrt(u * u + v * v), nan=0.0, posinf=0.0, neginf=0.0)

    if ref_max_rad is None:
        max_rad = torch.quantile(mag.flatten(), 0.99).item()
    else:
        max_rad = float(ref_max_rad)
    if not np.isfinite(max_rad):
        max_rad = 1e-6
    max_rad = max(max_rad, 1e-6)

    # 用 magnitude 做归一化，并复用 fake_uvw_panel 同风格的 jet 伪彩映射。
    val = torch.clamp(mag / max_rad, 0.0, 1.0).unsqueeze(1)  # [B,1,H,W] in [0,1]
    rgb = scalar_to_jet(val)  # [B,3,H,W] in [0,1]
    return rgb, max_rad
def scalar_to_jet(x01: torch.Tensor) -> torch.Tensor:
    """
    将单通道标量场映射为 jet 伪彩色图。
    x01: [B,1,H,W] in [0,1]
    return: [B,3,H,W] in [0,1], jet colormap
    """
    x = torch.nan_to_num(x01, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
    x_np = x.detach().cpu().numpy()  # [B,1,H,W]
    rgba = cm.get_cmap("jet")(x_np[:, 0])  # [B,H,W,4]
    rgb = torch.from_numpy(rgba[..., :3]).to(x.device, dtype=x.dtype)  # [B,H,W,3]
    return rgb.permute(0, 3, 1, 2).contiguous()  # [B,3,H,W]

