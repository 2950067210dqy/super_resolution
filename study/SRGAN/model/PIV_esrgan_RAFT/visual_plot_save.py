"""
可视化 保存 start
"""
from pathlib import Path
import torch.nn.functional as F
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.utils import save_image
from PIL import Image, ImageDraw, ImageFont

from study.SRGAN.model.PIV_esrgan_RAFT.visual_plot_init import _omega_star_from_uv
from study.SRGAN.util.image_util import build_triplet_row, build_pair_row
from study.SRGAN.util.tensor_util import _to_np_2d


def _tensor_to_rgb_pil(tensor: torch.Tensor) -> Image.Image:
    """将 [1,C,H,W] 或 [C,H,W] 张量转成 RGB PIL 图像。"""
    if tensor.dim() == 4:
        tensor = tensor[0]
    tensor = tensor.detach().cpu().clamp(0, 1)
    if tensor.shape[0] == 1:
        arr = (tensor[0].numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(arr, mode="L").convert("RGB")
    arr = (tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _pil_rgb_to_tensor01(image: Image.Image, device, dtype) -> torch.Tensor:
    """将 RGB PIL 图像转回 [1,3,H,W] 的 [0,1] 张量。"""
    arr = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)


def _add_headers_to_panel(
    panel: torch.Tensor,
    headers: list[str],
    column_widths: list[int],
    separator_widths: list[int],
    header_height: int = 22,
) -> torch.Tensor:
    """在拼图顶部加列标题。"""
    base = _tensor_to_rgb_pil(panel)
    canvas = Image.new("RGB", (base.width, base.height + header_height), color=(255, 255, 255))
    canvas.paste(base, (0, header_height))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    x = 0
    for idx, (title, width) in enumerate(zip(headers, column_widths)):
        bbox = draw.textbbox((0, 0), title, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        tx = int(x + max((width - text_w) * 0.5, 0))
        ty = int(max((header_height - text_h) * 0.5, 0))
        draw.text((tx, ty), title, fill=(0, 0, 0), font=font)
        x += width
        if idx < len(separator_widths):
            x += separator_widths[idx]

    return _pil_rgb_to_tensor01(canvas, panel.device, panel.dtype)


def _save_energy_spectrum_plot(pred_curve: np.ndarray, gt_curve: np.ndarray, out_png: Path, title: str) -> None:
    """保存能量谱对比图（log-log）。"""
    k = np.arange(1, len(pred_curve) + 1)
    plt.figure(figsize=(6, 4), dpi=160)
    plt.loglog(k, np.maximum(gt_curve, 1e-12), label="GT", linewidth=2)
    plt.loglog(k, np.maximum(pred_curve, 1e-12), label="Pred", linewidth=2, linestyle="--")
    plt.xlabel("Wavenumber k")
    plt.ylabel("E(k)")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def _save_triplet(lr_1chw: torch.Tensor, fake_1chw: torch.Tensor, hr_1chw: torch.Tensor, out_png: Path) -> None:
    """保存三联图 LR|Fake|HR。"""
    trip = build_triplet_row(lr_1chw, fake_1chw, hr_1chw, sep_width=6)
    col_w = int(lr_1chw.shape[-1])
    trip = _add_headers_to_panel(
        trip,
        headers=["LR", "Fake", "HR"],
        column_widths=[col_w, col_w, col_w],
        separator_widths=[6, 6],
    )
    save_image(trip.clamp(0, 1), str(out_png), normalize=False)


def _save_pair(left_1chw: torch.Tensor, right_1chw: torch.Tensor, out_png: Path) -> None:
    """保存双联图 Left|Right。"""
    pair = build_pair_row(left_1chw, right_1chw, sep_width=6)
    col_w = int(left_1chw.shape[-1])
    pair = _add_headers_to_panel(
        pair,
        headers=["Pred", "HR"],
        column_widths=[col_w, col_w],
        separator_widths=[6],
    )
    save_image(pair.clamp(0, 1), str(out_png), normalize=False)



def save_vorticity_quiver_single(fake_bchw: torch.Tensor, save_path: str, stride: int = 8):
    """
    保存单样本 fake 的涡量背景 + 速度矢量图。
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
    pred_bchw: torch.Tensor,
    hr_bchw: torch.Tensor,
    save_path: str,
    stride: int = 8
):
    """
    保存 batch 级对比图：每行一个样本，两列 Pred/HR 的涡量+矢量。
    把一个 batch 画成一张图：
    每个样本一行，两列 Pred | HR
    输入: [B,3,H,W]
    自动处理两者分辨率不一致
    """
    B = min(pred_bchw.shape[0], hr_bchw.shape[0])
    titles = ["Pred", "HR"]

    def _resize_np_nearest(arr, out_h, out_w):
        # arr: [h,w]
        t = torch.from_numpy(arr).float()[None, None]  # [1,1,h,w]
        t = F.interpolate(t, size=(out_h, out_w), mode="nearest")
        return t[0, 0].numpy()

    # 额外给每一行预留一个独立的 colorbar 轴，确保色条始终贴在整张图最右侧。
    fig, axes = plt.subplots(
        B, 3,
        figsize=(8.3, 3.2 * B),
        dpi=160,
        squeeze=False,
        gridspec_kw={"width_ratios": [1.0, 1.0, 0.055]},
    )

    for i in range(B):
        triplet = [pred_bchw[i:i + 1], hr_bchw[i:i + 1]]

        # 当前样本三图目标画布尺寸：取最大
        hs = [int(t.shape[-2]) for t in triplet]
        ws = [int(t.shape[-1]) for t in triplet]
        Ht, Wt = max(hs), max(ws)

        row_items = []
        for t in triplet:
            u = _to_np_2d(t[0, 0])  # [h,w]
            v = _to_np_2d(t[0, 1])  # [h,w]
            omega_star = _omega_star_from_uv(u, v)  # 原分辨率算涡量
            row_items.append((u, v, omega_star))

        # 同一行共用色标范围
        row_vmin = min(float(w.min()) for _, _, w in row_items)
        row_vmax = max(float(w.max()) for _, _, w in row_items)

        for j, (title, (u, v, omega_star)) in enumerate(zip(titles, row_items)):
            ax = axes[i, j]
            h, w = u.shape

            # 背景统一到目标尺寸，便于三列视觉一致
            if (h, w) != (Ht, Wt):
                omega_show = _resize_np_nearest(omega_star, Ht, Wt)
            else:
                omega_show = omega_star

            im = ax.imshow(
                omega_show,
                origin="lower",
                cmap="RdBu_r",
                vmin=row_vmin,
                vmax=row_vmax
            )

            # 矢量坐标从原网格映射到目标画布（保持“原始稀疏度”）
            yy, xx = np.mgrid[0:h, 0:w]
            sx = Wt / float(w)
            sy = Ht / float(h)

            cur_stride = stride
            ax.quiver(
                (xx * sx)[::cur_stride, ::cur_stride],
                (yy * sy)[::cur_stride, ::cur_stride],
                u[::cur_stride, ::cur_stride],
                v[::cur_stride, ::cur_stride],
                color="k",
                pivot="mid",
                angles="xy",
                scale_units="xy",
                scale=0.25,
                width=0.004,
            )

            ax.set_xlim(0, Wt - 1)
            ax.set_ylim(0, Ht - 1)
            if i == 0:
                ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])

        # colorbar 专用轴不参与图像绘制，只保留右侧色标。
        axes[i, 2].set_frame_on(True)

        # 当前已经改成两列 Pred / HR，因此把色条固定画到这一行最右边的专用轴上。
        cax = axes[i, 2]
        fig.colorbar(im, cax=cax)
        cax.yaxis.set_ticks_position("right")

    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
#瞬时涡流速度场 end
"""
可视化保存 end
"""
