import copy
import math
import os
import time
from datetime import datetime
from os import mkdir
from os.path import exists
from pathlib import Path

import matplotlib
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from matplotlib import pyplot as plt, cm
from matplotlib.animation import FuncAnimation, PillowWriter
from torchvision.models import vgg19
from tqdm import tqdm
from torch.nn import BatchNorm1d
from torchvision.utils import save_image
from d2l import torch as d2l   # 或 from d2l import mxnet as d2l
import wandb
from study.SRGAN.data_load import get_class_names, load_data
"""
分类别去训练
"""
"""
超参数 start
"""
name = "v3"
DESCRIPTION = ""
device = torch.device("cuda")
#是否加载之前的模型
IS_LOAD_EXISTS_MODEL = False

SAVE_AS_GRAY = True  # True: 保存为灰度图(1通道)；False: 按原通道保存 只影响图片对，不影响flo文件,同时处理相关损失函数也会按照这个

#轮次
EPOCH_NUMS = 20
#批量大小
BATCH_SIZE = 16
#是否打乱训练集
SHUFFLE = True
#将数据的图像和光流统一到该尺寸 tuple[int, int]
TARGET_SIZE=None
#随机划分时的随机种子，保证结果复现
RANDOM_SEED = 42
#上采样系数 SCALE^2
SCALE = 2

#生成器感知损失里面的系数 其实就是对抗损失的参数
LAMBDA_PERCEPTION =5e-4
#生成器正则损失的系数 ！弃用
LAMBDA_regularization_loss=2e-8
#生成器像素损失的系数
LAMBDA_loss_pixel =1

# 组合像素损失参数
LAMBDA_PIXEL_L1 = 1e-2
LAMBDA_PIXEL_MSE = 1e-3
PIXEL_WHITE_ALPHA = 1.0       # 白点区域权重
LAMBDA_GRAY_CONS = 1e-2       # 灰度复制RGB时通道一致性，别太


#正则项
weight_decay=0
#优化器 betas
g_optimizer_betas = (0.5,0.999)
d_optimizer_betas = (0.5,0.999)
#学习率
G_LR = 0.0001
D_LR = 0.0001

#训练数据集和验证集合比例 测试集  比例
Train_nums_rate=0.8
Test_nums_rate=0.0
Validate_nums_rate=1-Train_nums_rate-Test_nums_rate


#真实数据根路径
GR_DATA_ROOT_DIR = rf"/study_datas/sr_dataset/class_1/data"
#低分辨率数据根地址
LR_DATA_ROOT_DIR = rf"/study_datas/sr_dataset/class_1_lr/x{SCALE*SCALE}/data"

#如果路径不存在则创建路径
OUT_PUT_DIR = f"./train_data/{name}"
LOSS_DIR = "/train_loss"
MODEL_DIR = "/train_model"
PREDICT_DIR = "/predict"
use_gpu = torch.cuda.is_available()
Path(OUT_PUT_DIR).mkdir(parents=True, exist_ok=True)

# DATA_TYPES =['image_pair','flo']
DATA_TYPES =['flo']
IMAGE_PAIR_TYPES = ['previous','next']
"""
超参数 end
"""
loss_label = ['g_loss', 'g_perceptual_loss', "g_content_loss",
              "g_adversarial_loss", 'g_regularization_loss', 'g_loss_pixel',
                "g_loss_pixel_l1", "g_loss_pixel_mse",
              'd_loss', 'd_real_loss', 'd_fake_loss']
validate_label = ['Validation_Loss', 'Avg_PSNR']
# 使用wandb可视化训练过程
# 初始化 WandB
wandb.login(key="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")
# Start a new wandb run to track this script.
wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="2950067210-usst",
    # Set the wandb project where this run will be logged.
    project="srgnn",
    name=f"{name}_{DESCRIPTION}",
    # Track hyperparameters and run metadata.
    config={
        "epochs": EPOCH_NUMS,
        "batch_size": BATCH_SIZE,
        "lr_G": G_LR,
        "lr_D": D_LR,
        "RANDOM_SEED":RANDOM_SEED,
        "SCALE":SCALE,
        "SHUFFLE":SHUFFLE,
        "LAMBDA_PERCEPTION":LAMBDA_PERCEPTION,
        "LAMBDA_regularization_loss":LAMBDA_regularization_loss,
        "LAMBDA_loss_pixel":LAMBDA_loss_pixel,


        "LAMBDA_PIXEL_L1":LAMBDA_PIXEL_L1,
        "LAMBDA_PIXEL_MSE" :LAMBDA_PIXEL_MSE,
        "PIXEL_WHITE_ALPHA" :PIXEL_WHITE_ALPHA,       # 白点区域权重
        "LAMBDA_GRAY_CONS" : LAMBDA_GRAY_CONS  ,    # 灰度复制RGB时通道一致性，别太
        "SAVE_AS_GRAY" :SAVE_AS_GRAY,
        "weight_decay":weight_decay,
        "g_optimizer_betas":g_optimizer_betas,
        "d_optimizer_betas":d_optimizer_betas,
        "Train_nums_rate":Train_nums_rate
    },
)
# 配置超参数
wandb.config = {

}


"""
工具 start
"""
def save_hyper_parameters_txt(file_path="hyper_parameter.txt"):
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"# created_at = {created_at}",
        f'name = "{name}"',
        f'DESCRIPTION = "{DESCRIPTION}"',
        f'device = torch.device("{device.type}")',
        f"IS_LOAD_EXISTS_MODEL = {IS_LOAD_EXISTS_MODEL}",
        "",
        f"SAVE_AS_GRAY = {SAVE_AS_GRAY}",
        "",
        f"EPOCH_NUMS = {EPOCH_NUMS}",
        f"BATCH_SIZE = {BATCH_SIZE}",
        f"SHUFFLE = {SHUFFLE}",
        f"TARGET_SIZE = {TARGET_SIZE}",
        f"RANDOM_SEED = {RANDOM_SEED}",
        f"SCALE = {SCALE}",
        "",
        f"LAMBDA_PERCEPTION = {LAMBDA_PERCEPTION}",
        f"LAMBDA_regularization_loss = {LAMBDA_regularization_loss}",
        f"LAMBDA_loss_pixel = {LAMBDA_loss_pixel}",
        f"PIXEL_WHITE_ALPHA:{PIXEL_WHITE_ALPHA}",
        f"LAMBDA_GRAY_CONS:{LAMBDA_GRAY_CONS}",
        f"LAMBDA_PIXEL_L1 = {LAMBDA_PIXEL_L1}",
        f"LAMBDA_PIXEL_MSE = {LAMBDA_PIXEL_MSE}",


        f"weight_decay = {weight_decay}",
        f"g_optimizer_betas = {g_optimizer_betas}",
        f"d_optimizer_betas = {d_optimizer_betas}",
        f"G_LR = {G_LR}",
        f"D_LR = {D_LR}",
        "",
        f"Train_nums_rate = {Train_nums_rate}",
        f"Test_nums_rate = {Test_nums_rate}",
        f"Validate_nums_rate = {Validate_nums_rate}",
        "",
        f'GR_DATA_ROOT_DIR = r"{GR_DATA_ROOT_DIR}"',
        f'LR_DATA_ROOT_DIR = r"{LR_DATA_ROOT_DIR}"',
        "",
        f'OUT_PUT_DIR = r"{OUT_PUT_DIR}"',
        f'LOSS_DIR = r"{LOSS_DIR}"',
        f'MODEL_DIR = r"{MODEL_DIR}"',
        f'PREDICT_DIR = r"{PREDICT_DIR}"',
        f"use_gpu = {use_gpu}",
        f"DATA_TYPES = {DATA_TYPES}",
        f"IMAGE_PAIR_TYPES = {IMAGE_PAIR_TYPES}",
    ]

    Path(file_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"hyper_parameter Saved to {file_path}")
class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
def _in_notebook():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return False
        return ip.__class__.__name__ in ("ZMQInteractiveShell", "Shell")
    except Exception:
        return False


class Animator:
    def __init__(
        self,
        xlabel=None,
        ylabel=None,
        legend=None,
        xlim=None,
        ylim=None,
        xscale="linear",
        yscale="linear",
        fmts=None,
        figsize=(6, 4),
    ):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.legend = legend or []
        self.xlim = xlim
        self.ylim = ylim
        self.xscale = xscale
        self.yscale = yscale
        self.base_figsize = figsize

        self.X = None
        self.Y = None
        self.frames = []

        self.fmts = fmts if fmts is not None else self._build_auto_fmts()

        self.fig = None
        self.axes = None

    def _build_auto_fmts(self):
        colors = ["b", "g", "r", "c", "m", "y", "k"]
        linestyles = ["-", "--", "-.", ":"]
        markers = ["", "o", "s", "d", "^", "v", "x", "*"]

        n = len(self.legend) if self.legend else 8
        fmts = []

        for linestyle in linestyles:
            for marker in markers:
                for color in colors:
                    fmts.append(f"{color}{linestyle}{marker}")
                    if len(fmts) >= n:
                        return fmts

        return fmts

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        y = list(y)

        n = len(self.legend) if self.legend else len(y)

        if not hasattr(x, "__len__"):
            x = [x] * n
        else:
            x = list(x)
            if len(x) < n:
                x = x + [x[-1] if x else None] * (n - len(x))
            else:
                x = x[:n]

        if len(y) < n:
            y = y + [None] * (n - len(y))
        else:
            y = y[:n]

        if self.X is None:
            self.X = [[] for _ in range(n)]
        if self.Y is None:
            self.Y = [[] for _ in range(n)]

        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)

        self.frames.append(([row[:] for row in self.X], [row[:] for row in self.Y]))

    def _filter_series(self, Xf, Yf, exclude_legends=None):
        exclude_legends = set(exclude_legends or [])
        filtered = []

        n = max(len(self.legend), len(Xf), len(Yf))
        for i in range(n):
            xx = Xf[i] if i < len(Xf) else []
            yy = Yf[i] if i < len(Yf) else []
            name = self.legend[i] if i < len(self.legend) else f"series_{i}"
            if name in exclude_legends:
                continue
            filtered.append({
                "idx": i,
                "name": name,
                "x": xx,
                "y": yy,
                "fmt": self.fmts[i % len(self.fmts)],
            })
        return filtered

    def _series_scale(self, y):
        if not y:
            return 0.0
        return max(abs(v) for v in y)

    def _group_series_by_scale(self, series_list, split_ratio=8.0):
        non_empty = []
        empty = []

        for s in series_list:
            scale = self._series_scale(s["y"])
            item = {**s, "scale": scale}
            if scale == 0:
                empty.append(item)
            else:
                non_empty.append(item)

        non_empty.sort(key=lambda s: s["scale"])

        groups = []
        current_group = []

        for s in non_empty:
            if not current_group:
                current_group.append(s)
                continue

            current_max = max(item["scale"] for item in current_group)
            if current_max == 0 or s["scale"] / current_max <= split_ratio:
                current_group.append(s)
            else:
                groups.append(current_group)
                current_group = [s]

        if current_group:
            groups.append(current_group)

        if empty:
            if groups:
                groups[0].extend(empty)
            else:
                groups = [empty]

        return groups if groups else [[]]

    def _apply_fixed_groups(self, series_list, fixed_groups=None, split_ratio=8.0):
        """
        fixed_groups 支持:
        - 按名字: "g_loss"
        - 按名字+序号: "g_loss#0" / "g_loss#1"（同名序列时可精确指定）
        - 按索引: 0, 1, 2 ...
        """
        fixed_groups = fixed_groups or []

        def clone_series(src):
            return {
                "idx": src["idx"],
                "name": src["name"],
                "x": list(src["x"]),
                "y": list(src["y"]),
                "fmt": src["fmt"],
            }

        # name -> [series...]
        by_name = {}
        for s in series_list:
            by_name.setdefault(s["name"], []).append(s)

        final_groups = []
        used_positions = set()

        for group in fixed_groups:
            cur = []
            for item in group:
                src = None

                # 1) 整数索引
                if isinstance(item, int):
                    if 0 <= item < len(series_list):
                        src = series_list[item]
                        used_positions.add(item)

                # 2) 字符串：name 或 name#k
                elif isinstance(item, str):
                    if "#" in item:
                        name, k = item.rsplit("#", 1)
                        if k.isdigit():
                            k = int(k)
                            candidates = by_name.get(name, [])
                            if candidates:
                                src = candidates[min(k, len(candidates) - 1)]
                    else:
                        candidates = by_name.get(item, [])
                        if candidates:
                            src = candidates[0]

                    if src is not None:
                        for pos, s in enumerate(series_list):
                            if s is src:
                                used_positions.add(pos)
                                break

                if src is not None:
                    cur.append(clone_series(src))

            if cur:
                final_groups.append(cur)

        # 剩余未固定的再动态分组
        remaining = [s for i, s in enumerate(series_list) if i not in used_positions]
        dynamic_groups = self._group_series_by_scale(remaining, split_ratio=split_ratio)
        final_groups.extend([g for g in dynamic_groups if g])

        return final_groups if final_groups else [[]]

    def _build_figure(self, n_subplots):
        if self.fig is not None:
            plt.close(self.fig)

        width, height = self.base_figsize
        self.fig, axes = plt.subplots(
            n_subplots,
            1,
            figsize=(width, height * n_subplots),
            squeeze=False
        )
        self.axes = [ax[0] for ax in axes]

    def _config_axis(self, ax, series_group):
        ax.set_xlabel(self.xlabel or "")
        ax.set_ylabel(self.ylabel or "")

        if self.xlim is not None:
            ax.set_xlim(self.xlim)

        y_values = [v for s in series_group for v in s["y"]]
        if y_values:
            data_min = min(y_values)
            data_max = max(y_values)

            if self.ylim is not None:
                ymin, ymax = self.ylim
                ymax = max(ymax, data_max)
                ymin = min(ymin, data_min)
            else:
                ymin, ymax = data_min, data_max

            if ymin == ymax:
                pad = 1.0 if ymin == 0 else abs(ymin) * 0.05
                ymin -= pad
                ymax += pad

            ax.set_ylim((ymin, ymax+0.2))

        ax.set_xscale(self.xscale)
        ax.set_yscale(self.yscale)

        if series_group:
            handles = ax.lines
            labels = [s["name"] for s in series_group]
            ax.legend(handles, labels, loc="upper right", fontsize=9)

        ax.grid(True, alpha=0.3)

    def _draw_frame(self, frame_idx, exclude_legends=None, split_ratio=8.0, fixed_groups=None):
        Xf, Yf = self.frames[frame_idx]
        filtered_series = self._filter_series(Xf, Yf, exclude_legends=exclude_legends)
        groups = self._apply_fixed_groups(
            filtered_series,
            fixed_groups=fixed_groups,
            split_ratio=split_ratio
        )

        self._build_figure(len(groups))

        for ax, group in zip(self.axes, groups):
            ax.cla()
            for s in group:
                ax.plot(s["x"], s["y"], s["fmt"])
            self._config_axis(ax, group)

        self.fig.tight_layout()

    def save(self, gif_path="train.gif", fps=20, exclude_legends=None, split_ratio=8.0, fixed_groups=None):
        if not self.frames:
            raise ValueError("没有可保存的帧，请先调用 add().")

        self._draw_frame(
            0,
            exclude_legends=exclude_legends,
            split_ratio=split_ratio,
            fixed_groups=fixed_groups,
        )

        def update(frame_idx):
            self._draw_frame(
                frame_idx,
                exclude_legends=exclude_legends,
                split_ratio=split_ratio,
                fixed_groups=fixed_groups,
            )
            lines = []
            for ax in self.axes:
                lines.extend(ax.lines)
            return lines

        ani = FuncAnimation(
            self.fig,
            update,
            frames=len(self.frames),
            interval=1000 / fps,
            blit=False
        )
        ani.save(gif_path, writer=PillowWriter(fps=fps))
        plt.close(self.fig)
        self.fig = None
        self.axes = None

    def save_png(self, png_path="train.png", exclude_legends=None, split_ratio=8.0, fixed_groups=None):
        if not self.frames:
            raise ValueError("没有可保存的帧，请先调用 add().")

        self._draw_frame(
            len(self.frames) - 1,
            exclude_legends=exclude_legends,
            split_ratio=split_ratio,
            fixed_groups=fixed_groups,
        )
        self.fig.savefig(png_path, dpi=200, bbox_inches="tight")
def add_vertical_separator(tensor, sep_width=8, value=1.0):
    b, c, h, _ = tensor.shape
    return torch.full((b, c, h, sep_width), value, device=tensor.device, dtype=tensor.dtype)


def add_horizontal_separator(width, channels=3, sep_height=8, value=1.0, device="cpu", dtype=torch.float32):
    return torch.full((1, channels, sep_height, width), value, device=device, dtype=dtype)


def build_triplet_row(lr, fake, hr, sep_width=6):
    sep = add_vertical_separator(lr, sep_width=sep_width, value=1.0)
    return torch.cat([lr, sep, fake, sep, hr], dim=3)


def to_gray_3ch(x):
    """
    x: [B, 3, H, W]
    先压成灰度 [B,1,H,W]，再复制成 [B,3,H,W] 方便可视化拼接
    """
    gray = x.mean(dim=1, keepdim=True)
    return gray.repeat(1, 3, 1, 1)


def convert_fake_for_display(fake, fake_mode="rgb"):
    """
    fake: [B, 3, H, W]
    fake_mode:
        - rgb: 原图显示
        - gray: 压成灰度显示
    """
    if fake_mode == "rgb":
        return fake
    if fake_mode == "gray":
        return to_gray_3ch(fake)
    raise ValueError(f"Unsupported fake_mode: {fake_mode}")
def _hsv_to_rgb_torch(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
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
    flow: [B,C,H,W], C>=2（可为2或3，若3则第3通道会被忽略）
    返回: rgb [B,3,H,W] in [0,1], max_rad
    """
    if flow.ndim != 4 or flow.shape[1] < 2:
        raise ValueError(f"flow shape must be [B,C,H,W] and C>=2, got {tuple(flow.shape)}")

    # 只使用 uv，忽略可能存在的 magnitude 第三通道
    u = flow[:, 0]
    v = flow[:, 1]

    mag = torch.sqrt(u * u + v * v)
    ang = torch.atan2(v, u)  # [-pi, pi]

    h = (ang + torch.pi) / (2.0 * torch.pi)
    s = torch.ones_like(h)

    if ref_max_rad is None:
        max_rad = torch.quantile(mag.flatten(), 0.99).item()
    else:
        max_rad = float(ref_max_rad)
    max_rad = max(max_rad, 1e-6)

    val = torch.clamp(mag / max_rad, 0.0, 1.0)
    rgb = _hsv_to_rgb_torch(h, s, val)
    return rgb, max_rad
def scalar_to_jet(x01: torch.Tensor) -> torch.Tensor:
    """
    x01: [B,1,H,W] in [0,1]
    return: [B,3,H,W] in [0,1], jet colormap
    """
    x = x01.clamp(0, 1)
    x_np = x.detach().cpu().numpy()  # [B,1,H,W]
    rgba = cm.get_cmap("jet")(x_np[:, 0])  # [B,H,W,4]
    rgb = torch.from_numpy(rgba[..., :3]).to(x.device, dtype=x.dtype)  # [B,H,W,3]
    return rgb.permute(0, 3, 1, 2).contiguous()  # [B,3,H,W]
def build_flo_uvw_fake_panel(fake_bchw, col_sep=8):
    """
    只显示 fake，按三列排列：
    U* | V* | S*
    """
    if fake_bchw.ndim != 4 or fake_bchw.shape[1] < 3:
        raise ValueError(f"Need [B,>=3,H,W], got {tuple(fake_bchw.shape)}")

    cmin = fake_bchw[:, :3].amin(dim=(0, 2, 3), keepdim=True)
    cmax = fake_bchw[:, :3].amax(dim=(0, 2, 3), keepdim=True)
    den = (cmax - cmin).clamp_min(1e-8)
    x = (fake_bchw[:, :3] - cmin) / den  # [B,3,H,W]

    col_u = scalar_to_jet(x[:, 0:1])# [B,3,H,W]
    col_v = scalar_to_jet(x[:, 1:2])
    col_s = scalar_to_jet(x[:, 2:3])

    B, C, H, _ = col_u.shape
    v_sep = torch.full((B, C, H, col_sep), 1.0, device=x.device, dtype=x.dtype)  # 竖向分隔条

    # 关键：dim=3 左右拼接 -> 三列
    out = torch.cat([col_u, v_sep, col_v, v_sep, col_s], dim=3)
    return out.clamp(0, 1)


def build_flo_uvw_compare_panel(lr_bchw, fake_bchw, hr_bchw, sep_width=6, row_sep=8, sample_sep=10):
    """
    validate用：每个样本三列
    U*: LR|Fake|HR
    V*: LR|Fake|HR
    S*: LR|Fake|HR
    """
    for t in (lr_bchw, fake_bchw, hr_bchw):
        if t.shape[1] < 3:
            raise ValueError("Need 3 channels (U,V,S).")

    # 统一用 HR 做每通道 min-max，保证可比
    cmin = hr_bchw[:, :3].amin(dim=(0, 2, 3), keepdim=True)
    cmax = hr_bchw[:, :3].amax(dim=(0, 2, 3), keepdim=True)
    den = (cmax - cmin).clamp_min(1e-8)

    lr_n = (lr_bchw[:, :3] - cmin) / den
    fk_n = (fake_bchw[:, :3] - cmin) / den
    hr_n = (hr_bchw[:, :3] - cmin) / den

    sample_rows = []
    for i in range(lr_n.size(0)):
        ch_rows = []
        for ch in range(3):  # U,V,S
            lr_ch = scalar_to_jet(lr_n[i:i+1, ch:ch+1])
            fk_ch = scalar_to_jet(fk_n[i:i+1, ch:ch+1])
            hr_ch = scalar_to_jet(hr_n[i:i+1, ch:ch+1])
            # lr_ch = lr_n[i:i+1, ch:ch+1].repeat(1, 3, 1, 1)
            # fk_ch = fk_n[i:i+1, ch:ch+1].repeat(1, 3, 1, 1)
            # hr_ch = hr_n[i:i+1, ch:ch+1].repeat(1, 3, 1, 1)
            ch_rows.append(build_triplet_row(lr_ch, fk_ch, hr_ch, sep_width=sep_width))

        one = ch_rows[0]
        for r in ch_rows[1:]:
            h_sep = add_horizontal_separator(
                width=one.shape[3], channels=one.shape[1], sep_height=row_sep,
                value=1.0, device=one.device, dtype=one.dtype
            )
            one = torch.cat([one, h_sep, r], dim=2)
        sample_rows.append(one)

    out = sample_rows[0]
    for r in sample_rows[1:]:
        h_sep = add_horizontal_separator(
            width=out.shape[3], channels=out.shape[1], sep_height=sample_sep,
            value=1.0, device=out.device, dtype=out.dtype
        )
        out = torch.cat([out, h_sep, r], dim=2)
    return out.clamp(0, 1)


#瞬时涡流速度场 start
def _to_np_2d(x: torch.Tensor) -> np.ndarray:
    # x: [H,W] or [1,H,W]
    if x.ndim == 3:
        x = x.squeeze(0)
    return x.detach().float().cpu().numpy()

def _omega_star_from_uv(u: np.ndarray, v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    # omega = dv/dx - du/dy
    dv_dy, dv_dx = np.gradient(v)
    du_dy, du_dx = np.gradient(u)
    omega = dv_dx - du_dy

    omin = omega.min()
    omax = omega.max()
    omega01 = (omega - omin) / (omax - omin + eps)   # [0,1]
    omega_star = omega01 * 4.0 - 2.0                 # [-2,2]
    return omega_star

def save_vorticity_quiver_single(fake_bchw: torch.Tensor, save_path: str, stride: int = 8):
    """
    batch_train: 只看生成图效果（fake）
    fake_bchw: [B,3,H,W], channel0=u, channel1=v, channel2=s
    """
    u = _to_np_2d(fake_bchw[0, 0])
    v = _to_np_2d(fake_bchw[0, 1])
    omega_star = _omega_star_from_uv(u, v)

    H, W = u.shape
    yy, xx = np.mgrid[0:H, 0:W]

    fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.4), dpi=160)
    vmin = float(omega_star.min())
    vmax = float(omega_star.max())

    im = ax.imshow(omega_star, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.quiver(
        xx[::stride, ::stride], yy[::stride, ::stride],
        u[::stride, ::stride], v[::stride, ::stride],
        color="k",
        pivot="mid",
        angles="xy",
        scale_units="xy",
        scale=0.25,  # 原来1.8太短，越小越长
        width=0.004,  # 箭杆加粗

    )
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(r"$\omega^\ast$")

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_ticks(np.linspace(vmin, vmax, 5))

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

def save_vorticity_quiver_compare(
    lr_bchw: torch.Tensor,
    fake_bchw: torch.Tensor,
    hr_bchw: torch.Tensor,
    save_path: str,
    stride: int = 8
):
    """
    把一个 batch 画成一张图：
    每个样本一行，三列 LR | Fake | HR
    输入: [B,3,H,W]
    """
    B = min(lr_bchw.shape[0], fake_bchw.shape[0], hr_bchw.shape[0])
    titles = ["LR", "Fake", "HR"]

    fig, axes = plt.subplots(B, 3, figsize=(11.0, 3.2 * B), dpi=160, squeeze=False)

    for i in range(B):
        triplet = [lr_bchw[i:i+1], fake_bchw[i:i+1], hr_bchw[i:i+1]]

        # 同一行共用色标范围，方便比较
        row_omegas = []
        for t in triplet:
            u = _to_np_2d(t[0, 0])
            v = _to_np_2d(t[0, 1])
            row_omegas.append(_omega_star_from_uv(u, v))
        row_vmin = min(float(w.min()) for w in row_omegas)
        row_vmax = max(float(w.max()) for w in row_omegas)

        for j, (title, t, omega_star) in enumerate(zip(titles, triplet, row_omegas)):
            ax = axes[i, j]
            u = _to_np_2d(t[0, 0])
            v = _to_np_2d(t[0, 1])

            H, W = u.shape
            yy, xx = np.mgrid[0:H, 0:W]

            im = ax.imshow(
                omega_star,
                origin="lower",
                cmap="RdBu_r",
                vmin=row_vmin,
                vmax=row_vmax
            )
            ax.quiver(
                xx[::stride, ::stride], yy[::stride, ::stride],
                u[::stride, ::stride], v[::stride, ::stride],
                color="k",
                pivot="mid",
                angles="xy",
                scale_units="xy",
                scale=0.25,
                width=0.004,
            )
            if i == 0:
                ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])

        # 每行只放一个 colorbar（挂在 HR 子图右边）
        plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.02)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
#瞬时涡流速度场 end
"""
工具 end
"""

"""
模型 start
"""
def icnr_(tensor: torch.Tensor, scale: int = 2, initializer=nn.init.kaiming_normal_) -> torch.Tensor:
    """
    ICNR init for sub-pixel convolution weights.
    tensor shape: [out_channels, in_channels, kH, kW]
    """
    out_channels, in_channels, kH, kW = tensor.shape
    if out_channels % (scale ** 2) != 0:
        raise ValueError(f"out_channels({out_channels}) must be divisible by scale^2({scale**2})")

    subkernel = torch.zeros(
        [out_channels // (scale ** 2), in_channels, kH, kW],
        device=tensor.device,
        dtype=tensor.dtype,
    )
    initializer(subkernel)
    subkernel = subkernel.repeat_interleave(scale ** 2, dim=0)

    with torch.no_grad():
        tensor.copy_(subkernel)
    return tensor
class ResidualBlock(nn.Module):
    """
    残差块

    设计原则：
    1. 残差连接要求输入和输出形状完全一致，才能做 x + F(x)
    2. 所以这里通道数保持不变：64 -> 64
    3. 卷积使用 kernel_size=3, stride=1, padding=1，这样高宽不变

    输入:
        x: [B, 64, H, W]

    输出:
        out: [B, 64, H, W]
    """
    def __init__(self, channels=64):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            # 第1个卷积:
            # in_channel = 64
            # out_channel = 64
            # kernel_size = 3
            # stride = 1
            # padding = 1
            # 输出尺寸不变: H x W -> H x W
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),

            # 第2个卷积:
            # 仍然保持 64 -> 64，方便和输入直接相加
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        # 残差连接
        return x + self.block(x)


class Generator(nn.Module):
    """
    SRGAN 生成器

    当前版本: 4x 超分
    输入:
        x: [B, 3, 64, 64]

    输出:
        out: [B, 3, 256, 256]

    整体结构:
    1. 浅层特征提取
    2. 16个残差块
    3. 全局残差连接
    4. 两次 2x 上采样，总共 4x
    5. 输出 RGB 图像
    """
    def __init__(self,inner_chanel=3, num_residual_blocks=16, scale=2):
        super(Generator, self).__init__()


        self.scale = scale

        # 第一层卷积:
        # 输入是 RGB 图像，所以 in_channel=3
        # 输出 64 个特征图，所以 out_channel=64
        # kernel_size=9 是 SRGAN 经典设计，感受野更大
        # stride=1 不下采样
        # padding=4 保证尺寸不变
        #
        # 尺寸计算公式:
        # out = floor((W + 2P - K) / S) + 1
        # 对 64x64 来说:
        # (64 + 2*4 - 9) / 1 + 1 = 64
        #
        # 所以:
        # [B, 3, 64, 64] -> [B, 64, 64, 64]
        self.conv1 = nn.Sequential(
            nn.Conv2d(inner_chanel, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        # 残差块堆叠
        # 每个残差块都保持 [B, 64, H, W] 不变
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_residual_blocks)]
        )

        # 残差块之后的融合卷积
        # 这里仍然保持 64 -> 64
        # 目的是把残差块提取到的特征再融合一下
        #
        # [B, 64, 64, 64] -> [B, 64, 64, 64]
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

        # 上采样模块
        # PixelShuffle(2) 的作用:
        # 把通道数除以 4，同时把高宽各扩大 2 倍
        #
        # 所以如果想上采样 2 倍:
        # 先把通道升到 64 * 2^2 = 256
        # 再 PixelShuffle(2) -> 回到 64 通道，尺寸扩大 2 倍

        self.upsample = nn.Sequential(
            # 64 -> 256
            # 为什么是 256?
            # 因为 PixelShuffle(2) 需要输出通道数能被 2^2=4 整除
            # 并且 PixelShuffle 后希望仍然得到 64 通道
            # 所以卷积输出通道 = 64 * 4 = 256
            nn.Conv2d(64, 64 * self.scale * self.scale, kernel_size=3, stride=1, padding=1),

            # [B, 256, H, W] -> [B, 64, 2H, 2W]
            nn.PixelShuffle(self.scale),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(),

            nn.Conv2d(64, 64 * self.scale * self.scale, kernel_size=3, stride=1, padding=1),
            # [B, 256, H, W] -> [B, 64, 2H, 2W]
            nn.PixelShuffle(self.scale),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(),
        )

        # 最后一层输出 RGB 图像
        # 64 -> 3
        # kernel_size=9, padding=4 保持尺寸不变
        #
        # 若 scale=4:
        # [B, 64, 256, 256] -> [B, 3, 256, 256]
        self.conv_out = nn.Sequential(
            nn.Conv2d(64, inner_chanel, kernel_size=9, stride=1, padding=4),

        )

        #ICNR 初始化
        self._init_subpixel_weights()

    def _init_subpixel_weights(self):
        #对 PixelShuffle 前的 Conv2d(64, 64*scale*scale, ...) 做了 ICNR，能明显减轻棋盘纹。
        for m in self.upsample:
            if isinstance(m, nn.Conv2d) and m.out_channels == 64 * self.scale * self.scale:
                icnr_(m.weight, scale=self.scale, initializer=nn.init.kaiming_normal_)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def forward(self, x):
        # x: [B, 3, 64, 64]

        x1 = self.conv1(x)
        # x1: [B, 64, 64, 64]

        x2 = self.residual_blocks(x1)
        # x2: [B, 64, 64, 64]

        x3 = self.conv2(x2)
        # x3: [B, 64, 64, 64]

        # 全局残差连接
        x4 = x1 + x3
        # x4: [B, 64, 64, 64]

        x5 = self.upsample(x4)
        # 如果 scale=2:
        # x5: [B, 64, 128, 128]
        #
        # 如果 scale=4:
        # x5: [B, 64, 256, 256]

        out = self.conv_out(x5)
        # scale=2 时: [B, 3, 128, 128]
        # scale=4 时: [B, 3, 256, 256]

        return out


class DownSample(nn.Module):
    """
    判别器中的下采样块

    常见设计:
    Conv -> BN -> LeakyReLU

    这里通过 stride 控制是否下采样:
    - stride=1: 高宽不变
    - stride=2: 高宽减半
    """
    def __init__(self, input_channel, output_channel, stride, kernel_size=3, padding=1):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class Discriminator(nn.Module):
    """
    SRGAN 判别器

    如果生成器做的是 4x 超分:
        输入图像尺寸通常是 [B, 3, 256, 256]

    判别器任务:
        判断输入图像是真实高分辨率图像，还是生成器生成的图像
    """
    def __init__(self,inner_chanel=3):
        super(Discriminator, self).__init__()

        # 第一层通常不加 BN，这是 GAN 里较常见的写法
        # [B, 3, 256, 256] -> [B, 64, 256, 256]
        self.conv1 = nn.Sequential(
            nn.Conv2d(inner_chanel, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 尺寸变化过程（假设输入是 256x256）:
        #
        # 1. 64 -> 64, stride=2:   256 -> 128
        # 2. 64 -> 128, stride=1:  128 -> 128
        # 3. 128 -> 128, stride=2: 128 -> 64
        # 4. 128 -> 256, stride=1: 64 -> 64
        # 5. 256 -> 256, stride=2: 64 -> 32
        # 6. 256 -> 512, stride=1: 32 -> 32
        # 7. 512 -> 512, stride=2: 32 -> 16
        self.down = nn.Sequential(
            DownSample(64, 64, stride=2, kernel_size=3, padding=1),
            DownSample(64, 128, stride=1, kernel_size=3, padding=1),
            DownSample(128, 128, stride=2, kernel_size=3, padding=1),
            DownSample(128, 256, stride=1, kernel_size=3, padding=1),
            DownSample(256, 256, stride=2, kernel_size=3, padding=1),
            DownSample(256, 512, stride=1, kernel_size=3, padding=1),
            DownSample(512, 512, stride=2, kernel_size=3, padding=1),
        )

        # 最后把空间信息压缩到 1x1，再输出真假概率
        #
        # AdaptiveAvgPool2d(1):
        # [B, 512, 16, 16] -> [B, 512, 1, 1]
        #
        # 1x1 Conv 相当于全连接层的卷积写法
        self.dense = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.down(x)
        x = self.dense(x)
        return x

"""
模型 end
"""

"""
损失函数 start
"""
# class VGG(nn.Module):
#     def __init__(self, device):
#         super(VGG, self).__init__()
#         vgg = models.vgg19(True)
#         for pa in vgg.parameters():
#             pa.requires_grad = False
#         self.vgg = vgg.features[:16]
#         self.vgg = self.vgg.to(device)
#
#     def forward(self, x):
#         out = self.vgg(x)
#         return out

class ContentLoss(nn.Module):
    def __init__(self, vgg):
        super().__init__()
        self.vgg = vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.criterion = nn.MSELoss()

    def forward(self, fake, real):
        # 给 G 传梯度：fake 不 detach；real 可以 detach
        fake_features = self.vgg(fake)
        real_features = self.vgg(real).detach()
        return self.criterion(fake_features, real_features)
class AdversarialLoss(nn.Module):
    """
    对抗损失
    """
    def __init__(self):
        super(AdversarialLoss,self).__init__()

    def forward(self, x):
        loss = torch.sum(-torch.log(x))
        return loss
class PerceptualLoss(nn.Module):
    """
    感知损失 = 内容损失+1e-3 * 对抗损失
    也可以加正则化损失和像素损失
    """
    def __init__(self, vgg):
        super(PerceptualLoss,self).__init__()
        self.vgg_loss = ContentLoss(vgg)
        self.adversarial = AdversarialLoss()

    def forward(self, fake, real, x):
        vgg_loss = self.vgg_loss(fake, real)
        adversarial_loss = self.adversarial(x)

        return vgg_loss +LAMBDA_PERCEPTION*adversarial_loss,vgg_loss,adversarial_loss
class RegularizationLoss(nn.Module):
    """
    正则化损失
    """
    def __init__(self):
        super(RegularizationLoss,self).__init__()

    def forward(self, x):
        a = torch.square(
            x[:, :, :x.shape[2]-1, :x.shape[3]-1] - x[:, :, 1:x.shape[2], :x.shape[3]-1]
        )
        b = torch.square(
            x[:, :, :x.shape[2]-1, :x.shape[3]-1] - x[:, :, :x.shape[2]-1, 1:x.shape[3]]
        )
        loss = torch.sum(torch.pow(a+b, 1.25))
        return loss



class CombinedPixelLoss(nn.Module):
    """
    SAVE_AS_GRAY=True 且 image_pair 时:
      - 用 target 亮度做加权，提升白点召回
      - 加轻量通道一致性约束（防伪彩）
    其余情况:
      - 普通 RGB L1 + MSE
    返回: total, l1_term, mse_term
    """
    def __init__(self, lambda_l1=2e-2, lambda_mse=1e-3, white_alpha=4.0, lambda_cons=1e-2):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_mse = lambda_mse
        self.white_alpha = white_alpha
        self.lambda_cons = lambda_cons
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, pred, target, gray_triplet=False):
        if gray_triplet:
            pred_gray = pred[:, 0:1]
            target_gray = target[:, 0:1]

            # 白点加权
            weight = 1.0 + self.white_alpha * target_gray
            l1_term = (weight * (pred_gray - target_gray).abs()).mean()
            mse_term = self.mse(pred_gray, target_gray)

            # 灰度复制RGB一致性（不要太大）
            cons = (
                self.l1(pred[:, 0:1], pred[:, 1:2]) +
                self.l1(pred[:, 1:2], pred[:, 2:3]) +
                self.l1(pred[:, 0:1], pred[:, 2:3])
            ) / 3.0

            total = self.lambda_l1 * l1_term + self.lambda_mse * mse_term + self.lambda_cons * cons
            return total, l1_term, mse_term

        l1_term = self.l1(pred, target)
        mse_term = self.mse(pred, target)
        total = self.lambda_l1 * l1_term + self.lambda_mse * mse_term
        return total, l1_term, mse_term
"""
损失函数 end
"""

# 定义像素损失函数
pixel_loss = CombinedPixelLoss(
    lambda_l1=LAMBDA_PIXEL_L1,
    lambda_mse=LAMBDA_PIXEL_MSE,
    white_alpha=PIXEL_WHITE_ALPHA,
    lambda_cons=LAMBDA_GRAY_CONS,
).to(device)
# 这里vgg是针对三通道RGB图的
vgg = vgg19(pretrained=True).features[:16].eval()  # 提取 VGG 特征
# vgg模型预测模式
vgg = vgg.to(device).eval()

# 感知损失
perceptual_loss = PerceptualLoss(vgg=vgg)
# 判别器的损失函数
loss_d = nn.BCELoss()
# 归一化损失 正则化损失
regularization_loss = RegularizationLoss()
"""
验证函数 start
"""
def validate_and_save(result_dir, generator, val_dataloader, device, epoch, data_type):
    """
    flo:
        LR | Fake | HR

    image_pair:
        (previous: LR|Fake|HR) || (next: LR|Fake|HR)


    """

    def _convert_fake_for_display_by_hparam(fake_tensor: torch.Tensor) -> torch.Tensor:
        if not SAVE_AS_GRAY:
            return fake_tensor
        if fake_tensor.shape[1] != 3:
            raise ValueError(f"SAVE_AS_GRAY=True requires 3 channels, got {fake_tensor.shape[1]}")
        gray = fake_tensor[:, 0:1, :, :]
        return gray.repeat(1, 3, 1, 1)

    generator.eval()
    os.makedirs(result_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            if data_type == "flo":
                lr_images = batch[data_type]["lr_data"].to(device)   # [B,3,H,W] (u,v,mag)
                hr_images = batch[data_type]["gr_data"].to(device)   # [B,3,H,W]
                fake_images = generator(lr_images)                   # [B,3,H,W] (去掉Sigmoid后可超出[0,1])
                H, W = hr_images.shape[2:]
                h, w = lr_images.shape[-2],lr_images.shape[-1]
                # 如果是整数倍，直接像素复制，最“原汁原味”
                if H % h == 0 and W % w == 0:
                    sh, sw = H // h, W // w
                    resize_lr_images= lr_images.repeat_interleave(sh, dim=2).repeat_interleave(sw, dim=3)
                else:
                    resize_lr_images = F.interpolate(
                        lr_images,
                        size=hr_images.shape[2:],
                        mode="nearest",# linear | bilinear | bicubic | trilinear
                        # align_corners=False,
                    )

                if lr_images.shape[1] < 2 or fake_images.shape[1] < 2 or hr_images.shape[1] < 2:
                    raise ValueError("flo 可视化至少需要前两通道(u,v)")

                # # 仅打印一次数值差异
                # print(
                #     "flo diff:",
                #     "lr-hr", (resize_lr_images[:, :2] - hr_images[:, :2]).abs().mean().item(),
                #     "fake-hr", (fake_images[:, :2] - hr_images[:, :2]).abs().mean().item(),
                #     "fake-lr", (fake_images[:, :2] - resize_lr_images[:, :2]).abs().mean().item(),
                # )

                # 统一颜色尺度：用 HR 的 uv 计算 ref_max_rad
                hr_u = hr_images[:, 0]
                hr_v = hr_images[:, 1]
                hr_mag_uv = torch.sqrt(hr_u * hr_u + hr_v * hr_v)
                ref_max_rad = max(torch.quantile(hr_mag_uv.flatten(), 0.99).item(), 1e-6)

                # flo 统一用 uv 转彩色可视化（不直接 save 原3通道）
                lr_color, _ = flow_to_color_tensor(resize_lr_images[:, :2], ref_max_rad=ref_max_rad)
                fake_color, _ = flow_to_color_tensor(fake_images[:, :2], ref_max_rad=ref_max_rad)
                hr_color, _ = flow_to_color_tensor(hr_images[:, :2], ref_max_rad=ref_max_rad)

                sample_rows = []
                for i in range(lr_images.size(0)):
                    row = build_triplet_row(
                        lr_color[i].unsqueeze(0),
                        fake_color[i].unsqueeze(0),
                        hr_color[i].unsqueeze(0),
                        sep_width=6
                    )
                    sample_rows.append(row)
                uvs_compare_panel = build_flo_uvw_compare_panel(
                    resize_lr_images, fake_images, hr_images
                )


            elif data_type == "image_pair":
                lr_prev = batch["image_pair"]["previous"]["lr_data"].to(device)
                hr_prev = batch["image_pair"]["previous"]["gr_data"].to(device)
                lr_next = batch["image_pair"]["next"]["lr_data"].to(device)
                hr_next = batch["image_pair"]["next"]["gr_data"].to(device)

                fake_prev = generator(lr_prev)
                fake_next = generator(lr_next)

                resize_lr_prev = F.interpolate(
                    lr_prev,
                    size=hr_prev.shape[2:],
                    mode="nearest",  # linear | bilinear | bicubic | trilinear
                    # align_corners=False,
                )
                resize_lr_next = F.interpolate(
                    lr_next,
                    size=hr_next.shape[2:],
                    mode="nearest",  # linear | bilinear | bicubic | trilinear
                    # align_corners=False,
                )

                if hr_prev.shape[1] != 3:
                    raise ValueError(f"Unsupported previous channel count: {hr_prev.shape[1]}")
                if hr_next.shape[1] != 3:
                    raise ValueError(f"Unsupported next channel count: {hr_next.shape[1]}")

                sample_rows = []
                for i in range(lr_prev.size(0)):
                    single_lr_prev = resize_lr_prev[i].unsqueeze(0)
                    single_fake_prev = fake_prev[i].unsqueeze(0)
                    single_hr_prev = hr_prev[i].unsqueeze(0)

                    single_lr_next = resize_lr_next[i].unsqueeze(0)
                    single_fake_next = fake_next[i].unsqueeze(0)
                    single_hr_next = hr_next[i].unsqueeze(0)

                    # 去掉Sigmoid后，显示前先裁剪，避免可视化异常
                    display_fake_prev = _convert_fake_for_display_by_hparam(single_fake_prev.clamp(0, 1))
                    display_fake_next = _convert_fake_for_display_by_hparam(single_fake_next.clamp(0, 1))

                    left_group = build_triplet_row(single_lr_prev, display_fake_prev, single_hr_prev, sep_width=6)
                    right_group = build_triplet_row(single_lr_next, display_fake_next, single_hr_next, sep_width=6)

                    group_sep = add_vertical_separator(left_group, sep_width=16, value=1.0)
                    row = torch.cat([left_group, group_sep, right_group], dim=3)
                    sample_rows.append(row)

            else:
                raise ValueError(f"Unsupported data_type: {data_type}")

            batch_combined = sample_rows[0]
            for row in sample_rows[1:]:
                h_sep = add_horizontal_separator(
                    width=batch_combined.shape[3],
                    channels=batch_combined.shape[1],
                    sep_height=10,
                    value=1.0,
                    device=batch_combined.device,
                    dtype=batch_combined.dtype,
                )
                batch_combined = torch.cat([batch_combined, h_sep, row], dim=2)

            save_path = os.path.join(
                result_dir,
                f"epoch_{epoch + 1}_batch_{batch_idx}_results.png"
            )
            if data_type == "flo":
                #在保存一张u v s通道的图
                save_image(
                    uvs_compare_panel,
                    os.path.join(result_dir, f"epoch_{epoch + 1}_batch_{batch_idx}_results_uvs.png"),
                    normalize=False
                )
                #瞬时涡流速度场
                save_vorticity_quiver_compare(
                    resize_lr_images, fake_images, hr_images,
                    os.path.join(result_dir, f"epoch_{epoch + 1}_batch_{batch_idx}_vorticity_quiver.png"),
                    stride=8
                )
            save_image(batch_combined.clamp(0, 1), save_path, normalize=False)
            print(f"Saved validation image: {save_path}")
            break
# 计算 PSNR 函数
def calculate_psnr(fake_image, hr_image):
    mse = torch.mean((fake_image - hr_image) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(1.0 / math.sqrt(mse))
    return psnr

# 验证函数  flo文件
def validate_flow(generator, dataloader, device):
    #设置模型为评估模式
    generator.eval()
    val_loss = 0
    total_psnr = 0
    num_images = 0
    with torch.no_grad():
        for batch in dataloader:
            # 低分辨率图像
            lr_images = batch["flo"]['lr_data'].to(device)
            # 真实图像
            hr_images = batch["flo"]['gr_data'].to(device)
            lr_images, hr_images = lr_images.to(device), hr_images.to(device)
            fake_images = generator(lr_images)
            pixel_total, _, _ = pixel_loss(fake_images, hr_images, False)
            val_loss += pixel_total.item()

            for fake_image, hr_image in zip(fake_images, hr_images):
                total_psnr += calculate_psnr(fake_image, hr_image)
                num_images += 1

    val_loss /= len(dataloader)
    avg_psnr = total_psnr / num_images
    return val_loss, avg_psnr
# 验证函数 图像对
def validate_image_pair(generator, dataloader, device):
    # 设置模型为评估模式
    generator.eval()
    val_loss = 0
    total_psnr = 0
    num_images = 0
    with torch.no_grad():
        for batch in dataloader:
            for image_pair_type in IMAGE_PAIR_TYPES:
                # 低分辨率图像
                lr_images = batch["image_pair"][image_pair_type]['lr_data'].to(device)
                # 真实图像
                hr_images = batch["image_pair"][image_pair_type]['gr_data'].to(device)

                lr_images, hr_images = lr_images.to(device), hr_images.to(device)
                fake_images = generator(lr_images)

                pixel_total, _, _ = pixel_loss(fake_images, hr_images, SAVE_AS_GRAY)
                val_loss += pixel_total.item()
                for fake_image, hr_image in zip(fake_images, hr_images):
                    total_psnr += calculate_psnr(fake_image, hr_image)
                    num_images += 1

    val_loss /= (len(dataloader)*2)
    avg_psnr = total_psnr / num_images
    return val_loss, avg_psnr
"""
验证函数 end
"""
def train():
    """
    训练
    :return:
    """
    # 1.拿上数据
    pass
def image_pair_train(batch,i, data_type, device, generator, discriminator,
                g_optimizer, d_optimizer,
                train_progress_bar,
                metric,class_name):
    """
    图片对训练 ，因为是有两张图片所以训练两次
    :param batch: batch数据块
    :param i:  第几个batch
    :param data_type: 数据类型 data_tyoes:[image_pair,flo]
    :param device:cuda或者cpu
    :param generator:生成器
    :param discriminator:判别器
    :param g_optimizer:优化函数——生成器
    :param d_optimizer:优化函数——判别器
    :param train_progress_bar:训练进度条
    :param metric:loss等数据累加器
    :param class_name 类型名
    :return:
    """
    for image_pair_type in IMAGE_PAIR_TYPES:
        # 低分辨率图像
        lr_images = batch[data_type][image_pair_type]['lr_data'].to(device)
        # 真实图像
        gr_images = batch[data_type][image_pair_type]['gr_data'].to(device)
        batch_train(lr_images=lr_images, gr_images=gr_images, i=i, g_optimizer=g_optimizer,
                    d_optimizer=d_optimizer, generator=generator,
                    discriminator=discriminator, train_progress_bar=train_progress_bar,
                    metric=metric, data_type=data_type, device=device, class_name=class_name,image_pair_type = image_pair_type)
    pass
def flow_train(batch,i, data_type, device, generator, discriminator,
                g_optimizer, d_optimizer,
                train_progress_bar,
                metric,class_name):
    """
    flo数据训练
    :param batch: batch数据块
    :param i:  第几个batch
    :param data_type: 数据类型 data_tyoes:[image_pair,flo]
    :param device:cuda或者cpu
    :param generator:生成器
    :param discriminator:判别器
    :param g_optimizer:优化函数——生成器
    :param d_optimizer:优化函数——判别器
    :param train_progress_bar:训练进度条
    :param metric:loss等数据累加器
    :param class_name 类型名
    :return:
    """
    # 低分辨率图像
    lr_images = batch[data_type]['lr_data'].to(device)
    # 真实图像
    gr_images = batch[data_type]['gr_data'].to(device)
    batch_train(lr_images=lr_images, gr_images=gr_images, i=i, g_optimizer=g_optimizer,
                d_optimizer=d_optimizer, generator=generator,
                discriminator=discriminator, train_progress_bar=train_progress_bar,
                metric=metric, data_type=data_type, device=device, class_name=class_name)
    pass
def batch_train(lr_images,gr_images, i, data_type, device, generator, discriminator,
                g_optimizer, d_optimizer,
                train_progress_bar,
                metric,class_name,image_pair_type=None) -> None:
    """
    每一个batch的训练过程
    :param lr_images: 低分辨率图像
    :param gr_images: 真实图像
    :param i:  第几个batch
    :param data_type: 数据类型 data_tyoes:[image_pair,flo]
    :param device:cuda或者cpu
    :param generator:生成器
    :param discriminator:判别器
    :param g_optimizer:优化函数——生成器
    :param d_optimizer:优化函数——判别器
    :param train_progress_bar:训练进度条
    :param metric:loss等数据累加器
    :param class_name: 类型名
    :param image_pair_type : 图像对类别 previous next  如果是flo文件则为None
    :return:
    """

    """
    给真实标签做一点平滑
    不要让判别器太容易自信到极致，比如：
    real label: 1.0 -> 0.9
    """
    # real_labels_out = torch.ones((len(lr_images), 1, 1, 1)).to(device)
    # real_labels = torch.full_like(real_labels_out, 0.9).to(device)
    real_labels = torch.ones((len(lr_images), 1, 1, 1)).to(device)
    fake_labels = torch.zeros((len(lr_images), 1, 1, 1)).to(device)

    # 生成器生成图像
    pred_images = generator(lr_images)
    # print(f"pred_images:min,max,mean:{pred_images.min().data,pred_images.max().data,pred_images.mean().data} | lr_images:min,max,mean:{lr_images.min().data,lr_images.max().data,lr_images.mean().data} | gr_images:min,max,mean:{gr_images.min().data,gr_images.max().data,gr_images.mean().data} | ")
    # 判别器判别图像
    probability = discriminator(pred_images)

    """生成器训练 start"""
    # 感知损失
    perceptual_loss_value,content_loss,adversarial_loss = perceptual_loss(pred_images, gr_images, probability)
    # 像素损失（灰白数据可开加权，flo 默认不开）
    gray_triplet = (SAVE_AS_GRAY and data_type == "image_pair")  # 仅 image_pair 且设置为灰度复制模式
    g_loss_pixel, g_loss_l1, g_loss_mse = pixel_loss(pred_images, gr_images, gray_triplet=gray_triplet)
    # 正则损失
    """
    这是很常见的现象，这个 RegularizationLoss 本质上是在惩罚图像相邻像素差，也就是一种平滑约束。
    它一开始很大、随后迅速掉到很小，通常不代表代码错了，更多说明生成器输出很快变“更平滑”了。
    也就是说，图像越抖、越噪、局部变化越剧烈，这个值越大；图像越平滑，这个值越小
    """
    regularization_loss_value = regularization_loss(pred_images)
    # 生成器总损失
    # g_loss = perceptual_loss_value + LAMBDA_regularization_loss *regularization_loss_value +LAMBDA_loss_pixel*g_loss_pixel  # 这里的percuptual_loss包含了vgg_loss和对抗损失
    # g_loss = perceptual_loss_value
    g_loss = perceptual_loss_value+LAMBDA_loss_pixel*g_loss_pixel

    # 优化生成器
    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()
    """生成器训练 end"""

    # #因为判别器太强了，让它弱一点，每两次训练一次
    # if i % 2 == 0:
    """判别器训练 start"""
    # 判别器判别真实图片之后将概率结果放入损失函数并且优化生成器
    real_loss = loss_d(discriminator(gr_images), copy.deepcopy(real_labels))
    fake_loss = loss_d(discriminator(pred_images.detach()), fake_labels)
    d_loss = (real_loss + fake_loss)
    # 优化判别器
    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()
    """判别器训练 end"""
    # 在进度条上显示损失
    train_progress_bar.set_postfix({
        "class": class_name,
        "D Loss": d_loss.item(),
        "G Loss": g_loss.item()
    })

    # 需要和loss_label对应
    metric.add(g_loss.item(), perceptual_loss_value.item(),content_loss.item(),adversarial_loss.item(), regularization_loss_value.item(),
               g_loss_pixel.item(),g_loss_l1.item(),g_loss_mse.item(),
               d_loss.item(), real_loss.item(), fake_loss.item())
    # end if i % 2 == 0:
    if i % 20 == 0:
        image = pred_images.detach()
        save_dir = f"{OUT_PUT_DIR}/{class_name}/{data_type}/scale_{SCALE * SCALE}"
        os.makedirs(save_dir, exist_ok=True)

        save_prefix = f"{save_dir}/image_{len(train_loader) * epoch + i}_{name}"

        if image.dim() == 3:
            image = image.unsqueeze(0)  # [C,H,W] -> [1,C,H,W]

        if image.shape[1] >= 2 and data_type == "flo":
            # flo: 先把 uv 转成可视化彩色图，再保存
            # 用当前 batch 的 GT 作为统一尺度，避免每张图颜色漂移
            hr_u = gr_images[:, 0]
            hr_v = gr_images[:, 1]
            hr_mag_uv = torch.sqrt(hr_u * hr_u + hr_v * hr_v)
            ref_max_rad = max(torch.quantile(hr_mag_uv.flatten(), 0.99).item(), 1e-6)

            pred_color, _ = flow_to_color_tensor(image[:, :2], ref_max_rad=ref_max_rad)  # [N,3,H,W] in [0,1]
            torchvision.utils.save_image(
                pred_color.clamp(0, 1),
                f"{save_prefix}.png",
                nrow=4,
                normalize=False
            )

            #u v s 通道图
            fake_uvw_panel = build_flo_uvw_fake_panel(image)  # image=pred_images.detach()
            torchvision.utils.save_image(
                fake_uvw_panel,
                f"{save_prefix}_uvs.png",
                nrow=1,
                normalize=False
            )
            #瞬时涡流速度场
            save_vorticity_quiver_single(
                image,  # pred_images.detach()
                f"{save_prefix}_vorticity_quiver.png",
                stride=8
            )
        elif image.shape[1] == 3:
            # image_pair: 若去掉了 Sigmoid，保存前裁剪到 [0,1]
            image_to_save = image
            if SAVE_AS_GRAY and data_type != "flo":
                image_to_save = image[:, 0:1, :, :]  # [N,1,H,W]

            torchvision.utils.save_image(
                image_to_save.clamp(0, 1),
                f"{save_prefix}_{image_pair_type}.png" if image_pair_type else f"{save_prefix}.png",
                nrow=4,
                normalize=False
            )
def evaluate(epoch,class_name,data_type,device,
             generator,discriminator,animator,
             validate_loader,loss_label,validate_label):
    """

    :param epoch: 轮次
    :param class_name:类别
    :param data_type: 数据类型 data_tyoes:[image_pair,flo]
    :param device: cuda或者cpu
    :param generator: 生成器
    :param discriminator: 判别器
    :param animator: 图表动画
    :param validate_loader: 验证集数据加载器
    :param loss_label: 损失函数描述label
    :param validate_label:  验证参数的label
    :return:
    """
    # 每轮训练结束后进行验证

    val_loss,avg_psnr = 0,0

    if data_type =="image_pair":
        val_loss, avg_psnr = validate_image_pair(generator, validate_loader, device)
    elif data_type =="flo":
        val_loss, avg_psnr = validate_flow(generator, validate_loader, device)
    wandb.log({
        "classname": class_name,
        "data_type": data_type,
        "Validation Loss": val_loss,
        "avg_psnr": avg_psnr,
        "Epoch": epoch,
        **{
            loss_label[index]: metric[index] / len(train_loader)
            for index in range(len(loss_label))
        }
    })
    current_time = time.time()
    print(
        f"Epoch [{epoch + 1}/{EPOCH_NUMS}] |{class_name} {data_type} |running time:{int(current_time - start_time)}s | "
        f"Val Loss: {val_loss:.4f} | Avg PSNR: {avg_psnr:.2f}", end=""
    )
    loss_str = "".join([loss_label[index] + ':' + str(metric[index] / len(train_loader)) + "," for index in
                        range(len(loss_label))])
    print(loss_str)

    # 每轮训练结束后进行验证，并保存最后一批图像
    validate_and_save(f"{OUT_PUT_DIR}/{class_name}/{data_type}/scale_{SCALE * SCALE}/{PREDICT_DIR}", generator,
                      validate_loader, device, epoch, data_type=data_type)
    # 保存模型
    generator_save_path = f"{OUT_PUT_DIR}/{class_name}/{data_type}/scale_{SCALE * SCALE}/{MODEL_DIR}/discriminator_{name}.pth"
    discriminator_save_path = f"{OUT_PUT_DIR}/{class_name}/{data_type}/scale_{SCALE * SCALE}/{MODEL_DIR}/generator_{name}.pth"
    torch.save(discriminator.state_dict(),discriminator_save_path )
    torch.save(generator.state_dict(), generator_save_path)
    print(
        f"{class_name} {data_type} |Models saved: Generator -> {generator_save_path}, Discriminator -> {discriminator_save_path}")

    # 保存每一epoch的损失
    animator.add(epoch + 1,
                 [metric[index] / len(train_loader) for index in range(len(loss_label))] + [val_loss, avg_psnr])
    animator.save_png(
        f"{OUT_PUT_DIR}/{class_name}/{data_type}/scale_{SCALE * SCALE}/{LOSS_DIR}/train_loss_epoch_{epoch + 1}_{name}.png",
        fixed_groups=[
            [loss_label[0], loss_label[8], validate_label[0]],
            [loss_label[1], loss_label[2], loss_label[3]],
            [loss_label[1], loss_label[4], loss_label[5]],
            [loss_label[5], loss_label[6], loss_label[7]],
            [loss_label[8],loss_label[9], loss_label[10]],
            [validate_label[1]]
        ])
    pass
if __name__ =="__main__":
    #保存超参数
    save_hyper_parameters_txt(f"{OUT_PUT_DIR}/hyper_parameters.txt")
    #获取类别名
    #获取数据 自动根据类别划分数据集并读取，每个类别都安装比例划分训练集和验证集
    available_class_names = get_class_names(GR_DATA_ROOT_DIR, LR_DATA_ROOT_DIR)

    print(f"一共{len(available_class_names)}个类别：{available_class_names}")
    #每个类别读取数据并且训练验证和保存模型
    for class_name in available_class_names:


        #根据类别读取数据
        train_loader, validate_loader, class_names, samples = load_data(
            gr_data_root_dir=GR_DATA_ROOT_DIR,
            lr_data_root_dir=LR_DATA_ROOT_DIR,
            batch_size=BATCH_SIZE,
            shuffle=SHUFFLE,
            target_size=TARGET_SIZE,
            train_nums_rate=Train_nums_rate,
            validate_nums_rate=Validate_nums_rate,
            random_seed=RANDOM_SEED,
            selected_classes=available_class_names[:1] if available_class_names else None,
        )
        # 每个类别的图像对和flo文件分别训练验证和保存模型
        for data_type in DATA_TYPES:
            # 创建文件夹
            Path(f"{OUT_PUT_DIR}/{class_name}/{data_type}/scale_{SCALE * SCALE}/{LOSS_DIR}").mkdir(parents=True, exist_ok=True)
            Path(f"{OUT_PUT_DIR}/{class_name}/{data_type}/scale_{SCALE * SCALE}/{MODEL_DIR}").mkdir(parents=True, exist_ok=True)
            Path(f"{OUT_PUT_DIR}/{class_name}/{data_type}/scale_{SCALE * SCALE}/{PREDICT_DIR}").mkdir(parents=True, exist_ok=True)

            animator = Animator(xlabel='epoch', xlim=[1, EPOCH_NUMS], ylim=[0, 0.5],
                                legend=loss_label + validate_label)
            # 实例化generator
            generator = Generator(inner_chanel=3).to(device)
            # 实例化Discriminator
            discriminator = Discriminator(inner_chanel=3).to(device)
            # 加载预训练模型
            if IS_LOAD_EXISTS_MODEL:
                generator_save_path = f"{OUT_PUT_DIR}/{class_name}/{data_type}/scale_{SCALE * SCALE}/{MODEL_DIR}/discriminator_{name}.pth"
                if os.path.exists(generator_save_path):
                    generator.load_state_dict(torch.load(generator_save_path, map_location=device))
                    print(f"Loaded pretrained model generator from {generator_save_path}")
                else:
                    print("No pretrained model generator found. Starting training from scratch.")
                discriminator_save_path = f"{OUT_PUT_DIR}/{class_name}/{data_type}/scale_{SCALE * SCALE}/{MODEL_DIR}/generator_{name}.pth"
                if os.path.exists(discriminator_save_path):
                    discriminator.load_state_dict(torch.load(discriminator_save_path, map_location=device))
                    print(f"Loaded pretrained model discriminator from {discriminator_save_path}")
                else:
                    print("No pretrained model discriminator found. Starting training from scratch.")
            # 优化器
            g_optimizer = torch.optim.Adam(generator.parameters(), lr=G_LR, betas=g_optimizer_betas,
                                           weight_decay=weight_decay)
            d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=D_LR, betas=d_optimizer_betas,
                                           weight_decay=weight_decay)

            start_time = time.time()
            #轮数
            for epoch in range(EPOCH_NUMS):
                generator.train()# 确保生成器在训练模式
                discriminator.train()# 确保判别器在训练模式
                # 算每轮epoch的总体loss
                metric = Accumulator(len(loss_label))
                # 拿batch——size数据
                train_progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{EPOCH_NUMS}] {class_name} {data_type} Training", unit="batch")
                #1600/32 =50
                for i,batch in enumerate(train_progress_bar):
                    """ 图片对训练"""
                    if data_type == "image_pair":
                        image_pair_train(
                            batch=batch, i=i, g_optimizer=g_optimizer,
                            d_optimizer=d_optimizer, generator=generator,
                            discriminator=discriminator, train_progress_bar=train_progress_bar,
                            metric=metric, data_type=data_type, device=device, class_name=class_name
                        )
                    elif data_type == "flo":
                        """flo文件训练"""
                        flow_train( batch=batch, i=i, g_optimizer=g_optimizer,
                            d_optimizer=d_optimizer, generator=generator,
                            discriminator=discriminator, train_progress_bar=train_progress_bar,
                            metric=metric, data_type=data_type, device=device, class_name=class_name)
                #每轮结束后评价一次
                evaluate(epoch, class_name, data_type, device, generator, discriminator, animator, validate_loader, loss_label, validate_label)

    # # 测试生成器
    # lr = torch.randn(2, 3, 64, 64)
    #
    # # 4x 超分: 64x64 -> 256x256
    # G = Generator(scale=2)
    # sr = G(lr)
    # print("Generator output shape:", sr.shape)
    #
    # # 判别器输入应该和 HR / SR 图像尺寸一致
    # D = Discriminator()
    # out = D(sr)
    # print("Discriminator output shape:", out.shape)
    pass