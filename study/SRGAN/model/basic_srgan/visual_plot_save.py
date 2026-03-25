"""
可视化 保存 start
"""
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
    save_image(trip.clamp(0, 1), str(out_png), normalize=False)



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
    lr_bchw: torch.Tensor,
    fake_bchw: torch.Tensor,
    hr_bchw: torch.Tensor,
    save_path: str,
    stride: int = 8
):
    """
    保存 batch 级对比图：每行一个样本，三列 LR/Fake/HR 的涡量+矢量。
    把一个 batch 画成一张图：
    每个样本一行，三列 LR（原始 未扩充大小） | Fake | HR
    输入: [B,3,H,W]
    自动处理三者分辨率不一致
    """
    B = min(lr_bchw.shape[0], fake_bchw.shape[0], hr_bchw.shape[0])
    titles = ["LR", "Fake", "HR"]

    def _resize_np_nearest(arr, out_h, out_w):
        # arr: [h,w]
        t = torch.from_numpy(arr).float()[None, None]  # [1,1,h,w]
        t = F.interpolate(t, size=(out_h, out_w), mode="nearest")
        return t[0, 0].numpy()

    fig, axes = plt.subplots(B, 3, figsize=(11.0, 3.2 * B), dpi=160, squeeze=False)

    for i in range(B):
        triplet = [lr_bchw[i:i + 1], fake_bchw[i:i + 1], hr_bchw[i:i + 1]]

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

        plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.02)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
#瞬时涡流速度场 end
"""
可视化保存 end
"""