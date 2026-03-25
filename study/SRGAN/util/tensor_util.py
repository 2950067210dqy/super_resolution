import numpy as np
import torch


def _to_np_2d(x: torch.Tensor) -> np.ndarray:
    """将 torch 2D/3D(单通道)张量转为 numpy 2D。"""
    # x: [H,W] or [1,H,W]
    if x.ndim == 3:
        x = x.squeeze(0)
    return x.detach().float().cpu().numpy()