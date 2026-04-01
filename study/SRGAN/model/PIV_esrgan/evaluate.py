from loguru import logger
import os
import time
from datetime import datetime
from pathlib import Path
import csv
import math
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt

from SRGAN.model.PIV_esrgan.Module.loss import pixel_loss
from study.SRGAN.model.PIV_esrgan.global_class import global_data
from study.SRGAN.model.PIV_esrgan.judge_delicators import _to_np_chw, _mse, _psnr_from_mse, _energy_spectrum_mse, \
    _r2_score, _ssim_score, _tke_reconstruction_accuracy, _nrmse, _energy_spectrum_curves


from study.SRGAN.model.PIV_esrgan.visual_plot_init import build_flo_uvw_compare_panel
from study.SRGAN.model.PIV_esrgan.visual_plot_save import save_vorticity_quiver_compare, _save_triplet, \
    _save_energy_spectrum_plot
from study.SRGAN.util.image_util import flow_to_color_tensor, build_triplet_row, add_vertical_separator, \
    add_horizontal_separator, _select_metric_or_save_channels

"""
验证函数 start
"""


def validate_and_save(result_dir, generator, val_dataloader, device, epoch, data_type, SAVE_AS_GRAY=None):
    """
    每轮验证时保存主对比图，并在 flo 模态下额外保存 U/V/S 与涡量矢量图。 只验证保存loader的第一个batch的图
    flo:
        LR | Fake | HR

    image_pair:
        (previous: LR|Fake|HR) || (next: LR|Fake|HR)


    """

    if SAVE_AS_GRAY is None:
        SAVE_AS_GRAY = global_data.esrgan.SAVE_AS_GRAY

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
                    logger.error('flo 可视化至少需要前两通道(u,v)')
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

                if hasattr(generator, "forward_pair"):
                    fake_prev, fake_next = generator.forward_pair(lr_prev, lr_next)
                else:
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

                if SAVE_AS_GRAY:
                    if hr_prev.shape[1] < 1:
                        logger.error(f'Unsupported previous channel count: {hr_prev.shape[1]}')
                        raise ValueError(f"Unsupported previous channel count: {hr_prev.shape[1]}")
                    if hr_next.shape[1] < 1:
                        logger.error(f'Unsupported next channel count: {hr_next.shape[1]}')
                        raise ValueError(f"Unsupported next channel count: {hr_next.shape[1]}")
                else:
                    if hr_prev.shape[1] != 3:
                        logger.error(f'Unsupported previous channel count: {hr_prev.shape[1]}')
                        raise ValueError(f"Unsupported previous channel count: {hr_prev.shape[1]}")
                    if hr_next.shape[1] != 3:
                        logger.error(f'Unsupported next channel count: {hr_next.shape[1]}')
                        raise ValueError(f"Unsupported next channel count: {hr_next.shape[1]}")

                sample_rows = []
                for i in range(lr_prev.size(0)):
                    single_lr_prev = _select_metric_or_save_channels(
                        resize_lr_prev[i].unsqueeze(0), "image_pair", SAVE_AS_GRAY
                    ).clamp(0, 1)
                    single_fake_prev = _select_metric_or_save_channels(
                        fake_prev[i].unsqueeze(0), "image_pair", SAVE_AS_GRAY
                    ).clamp(0, 1)
                    single_hr_prev = _select_metric_or_save_channels(
                        hr_prev[i].unsqueeze(0), "image_pair", SAVE_AS_GRAY
                    ).clamp(0, 1)

                    single_lr_next = _select_metric_or_save_channels(
                        resize_lr_next[i].unsqueeze(0), "image_pair", SAVE_AS_GRAY
                    ).clamp(0, 1)
                    single_fake_next = _select_metric_or_save_channels(
                        fake_next[i].unsqueeze(0), "image_pair", SAVE_AS_GRAY
                    ).clamp(0, 1)
                    single_hr_next = _select_metric_or_save_channels(
                        hr_next[i].unsqueeze(0), "image_pair", SAVE_AS_GRAY
                    ).clamp(0, 1)

                    left_group = build_triplet_row(single_lr_prev, single_fake_prev, single_hr_prev, sep_width=6)
                    right_group = build_triplet_row(single_lr_next, single_fake_next, single_hr_next, sep_width=6)

                    group_sep = add_vertical_separator(left_group, sep_width=16, value=1.0)
                    row = torch.cat([left_group, group_sep, right_group], dim=3)
                    sample_rows.append(row)

            else:
                logger.error(f'Unsupported data_type: {data_type}')
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
                    lr_images, fake_images, hr_images,
                    os.path.join(result_dir, f"epoch_{epoch + 1}_batch_{batch_idx}_vorticity_quiver.png"),
                    stride=6
                )
            save_image(batch_combined.clamp(0, 1), save_path, normalize=False)
            logger.info(f"Saved validation image: {save_path}")
            break
# 计算 PSNR 函数
def calculate_psnr(fake_image, hr_image):
    """计算单张图 PSNR。"""

    mse = torch.mean((fake_image - hr_image) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(1.0 / math.sqrt(mse))
    return psnr

# 验证函数  flo文件
def validate_flow(generator, dataloader, device):
    """在 flo 验证集上计算平均像素损失与平均 PSNR。 验证所有的验证集"""
    #设置模型为评估模式
    generator.eval()
    total_val_ssim_loss = 0.0
    total_val_mse_loss = 0.0
    total_psnr = 0.0
    loss_count = 0
    num_images = 0
    with torch.no_grad():
        for batch in dataloader:
            # 低分辨率图像
            lr_images = batch["flo"]['lr_data'].to(device)
            # 真实图像
            hr_images = batch["flo"]['gr_data'].to(device)
            lr_images, hr_images = lr_images.to(device), hr_images.to(device)
            fake_images = generator(lr_images)
            _, _, mse_total, ssim_total, _ = pixel_loss(fake_images, hr_images, False)
            total_val_ssim_loss += ssim_total.item()
            total_val_mse_loss += mse_total.item()
            loss_count += 1

            for fake_image, hr_image in zip(fake_images, hr_images):
                total_psnr += calculate_psnr(fake_image, hr_image)
                num_images += 1
            # 一个batch 就行了 因为训练中的验证只需要1次batch验证
            break
    avg_val_ssim_loss = total_val_ssim_loss / max(loss_count, 1)
    avg_val_mse_loss = total_val_mse_loss / max(loss_count, 1)
    avg_psnr = total_psnr / max(num_images, 1)
    return avg_val_ssim_loss,avg_val_mse_loss, avg_psnr
# 验证函数 图像对
def validate_image_pair(generator, dataloader, device):
    generator.eval()
    total_val_ssim_loss = 0.0
    total_val_mse_loss = 0.0

    total_psnr = 0.0
    loss_count = 0
    num_images = 0

    with torch.no_grad():
        for batch in dataloader:
            lr_prev = batch["image_pair"]["previous"]["lr_data"].to(device)
            hr_prev = batch["image_pair"]["previous"]["gr_data"].to(device)
            lr_next = batch["image_pair"]["next"]["lr_data"].to(device)
            hr_next = batch["image_pair"]["next"]["gr_data"].to(device)

            if hasattr(generator, "forward_pair"):
                fake_prev, fake_next = generator.forward_pair(lr_prev, lr_next)
            else:
                fake_prev = generator(lr_prev)
                fake_next = generator(lr_next)

            for fake_images, hr_images in ((fake_prev, hr_prev), (fake_next, hr_next)):
                _, _, mse_total, ssim_total, _ = pixel_loss(fake_images, hr_images, global_data.esrgan.SAVE_AS_GRAY)
                total_val_ssim_loss += ssim_total.item()
                total_val_mse_loss += mse_total.item()
                loss_count += 1

                fake_images_for_metric = _select_metric_or_save_channels(
                    fake_images, "image_pair", global_data.esrgan.SAVE_AS_GRAY
                )
                hr_images_for_metric = _select_metric_or_save_channels(
                    hr_images, "image_pair", global_data.esrgan.SAVE_AS_GRAY
                )

                for fake_image, hr_image in zip(fake_images_for_metric, hr_images_for_metric):
                    total_psnr += calculate_psnr(fake_image, hr_image)
                    num_images += 1
            #一个batch 就行了 因为训练中的验证只需要1次batch验证
            break
    avg_val_ssim_loss = total_val_ssim_loss / max(loss_count, 1)
    avg_val_mse_loss = total_val_mse_loss / max(loss_count, 1)
    avg_psnr = total_psnr / max(num_images, 1)
    return avg_val_ssim_loss, avg_val_mse_loss, avg_psnr

"""
验证函数 end
"""
def evaluate(epoch,class_name,data_type,device,
             generator,discriminator,animator,
             validate_loader,loss_label,validate_label,SCALE,csvOperator,metric,train_loader_lens=1):
    """
   每轮结束后执行验证、记录日志、保存模型与损失曲线。
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
    :param SCALE:上采样因子 具体放大平方倍
    :param csvOperator:loss等数据 存储csv
    :param metric:累加器
    :param train_loader_lens:训练数据长度
    :return:
    """
    # 每轮训练结束后进行验证

    avg_val_ssim_loss,avg_val_mse_loss,avg_psnr = 0,0,0

    if data_type =="image_pair":
        avg_val_ssim_loss, avg_val_mse_loss, avg_psnr = validate_image_pair(generator, validate_loader, device)
    elif data_type =="flo":
        avg_val_ssim_loss, avg_val_mse_loss, avg_psnr = validate_flow(generator, validate_loader, device)
    wandb.log({
        "classname": class_name,
        "data_type": data_type,
        "VAL_AVG_MSE_LOSS": avg_val_mse_loss ,
        "VAL_AVG_SSIM_LOSS": avg_val_ssim_loss ,
        "avg_psnr": avg_psnr,
        "Epoch": epoch,
        **{
            loss_label[index]: metric[index] / (train_loader_lens)
            for index in range(len(loss_label))
        }
    })
    current_time = time.time()
    logger.info(
        f"Epoch [{epoch + 1}/{global_data.esrgan.EPOCH_NUMS}] |{class_name} {data_type} |running time:{int(current_time - global_data.esrgan.START_TIME )}s | "
        f"VAL_AVG_MSE_LOSS: {avg_val_mse_loss} | VAL_AVG_SSIM_LOSS: {avg_val_ssim_loss} | Avg PSNR: {avg_psnr:.2f}"
    )
    loss_str = "".join([loss_label[index] + ':' + str(metric[index] / train_loader_lens) + "," for index in
                        range(len(loss_label))])
    logger.info(loss_str)

    # 每轮训练结束后进行验证，并保存最后一批图像
    validate_and_save(f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.PREDICT_DIR}", generator,
                      validate_loader, device, epoch, data_type=data_type)
    # 保存模型
    generator_save_path = f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.MODEL_DIR}/discriminator_{global_data.esrgan.name}.pth"
    discriminator_save_path = f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.MODEL_DIR}/generator_{global_data.esrgan.name}.pth"
    torch.save(discriminator.state_dict(),discriminator_save_path )
    torch.save(generator.state_dict(), generator_save_path)
    logger.info(
        f"{class_name} {data_type} |Models saved: Generator -> {generator_save_path}, Discriminator -> {discriminator_save_path}")

    # 保存每一epoch的损失 image_pair 是计算了前图和后图两次 所以要多除2
    all_loss_and_val_Datas = [metric[index] / (train_loader_lens)  for index in range(len(loss_label))] + [avg_val_mse_loss,avg_val_ssim_loss, avg_psnr]
    animator.add(epoch + 1,all_loss_and_val_Datas )
    # 保存到csv文件中
    csvOperator.create(dict(zip(global_data.esrgan.CSV_COLUMNS,[epoch + 1]+all_loss_and_val_Datas+[datetime.now().strftime("%Y-%m-%d %H:%M:%S")])))
    animator.save_png(
        f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.LOSS_DIR}/train_loss_epoch_{epoch + 1}_{global_data.esrgan.name}.png",
        fixed_groups=[
            ["g_loss", "d_loss"],
            ["g_perceptual_loss", "g_content_loss", "g_adversarial_loss"],
            ["g_loss_pixel", "g_loss_pixel_l1", "g_loss_pixel_mse", "g_loss_ssim", "g_loss_fft"],
            ["g_pair_temporal_loss", "g_pair_delta_loss", "g_pair_gradient_loss"],
            ["g_particle_loss", "g_physic_loss", "g_structure_loss"],
            ["d_loss", "d_real_loss", "d_fake_loss"],
            ["g_CHARBONNIER_loss", "g_edge_loss", "g_BRIGHT_MASK_loss", "g_MASS_loss", "g_peak_loss", "g_SEPARATION_loss"],
            ["g_PARTICLE_COUNT_loss", "g_PARTICLE_DENSITY_loss"],
            [validate_label[0], validate_label[1], validate_label[2]]
        ])
    pass

def evaluate_all(
    generator,
    data_loader,
    class_name: str,
    data_type: str,
    SCALE: float,
    output_root: str | Path,
    metrics_csv_path: str | Path | None = None,
    stride: int = 6,
):
    """
    验证整个 data_loader，计算并保存每个样本指标与可视化结果。
    evaluate_all(
        generator=generator,
        data_loader=validate_loader,
        class_name=class_name,
        data_type=data_type,
        SCALE=SCALE,
        output_root=f"{OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE*SCALE)}/{PREDICT_ALL_DIR}",
        metrics_csv_path=f"{OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE*SCALE)}/metrics_all.csv",
        stride=6,
    )
    指标:
    - MSE
    - PSNR
    - Energy Spectrum MSE
    - R2
    - SSIM
    - TKE 重建精度
    - NRMSE

    保存内容:
    - 每个样本独立文件夹
    - image_pair: 保存 LR/Fake/HR 自身图 + 差异图 + 三联图 + 能量谱曲线(.npy/.png)
    - flo: 保存 flo 数组(.npy) + 颜色流场三联图 + U/V/S 对比图 + 涡量矢量图 + 能量谱曲线(.npy/.png)
    - CSV: 每样本指标 + 均值行
    - 全局均值能量谱曲线(.npy/.png)
    """
    # evaluate_all 做的是“全量验证集/测试集统计”，不是训练中的单 batch 快速验证。
    # 所以这里除了保存每个样本结果，还会：
    # 1. 按类别写子目录
    # 2. 按类别写 metrics.csv
    # 3. 额外汇总 전체 metrics_all.csv
    device = next(generator.parameters()).device
    generator.eval()

    if data_type == "image_pair":
        logger.info(f"[evaluate_all] SAVE_AS_GRAY={global_data.esrgan.SAVE_AS_GRAY}")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if metrics_csv_path is None:
        metrics_csv_path = output_root / f"metrics_{class_name}_{data_type}_x{int(SCALE * SCALE)}.csv"
    else:
        metrics_csv_path = Path(metrics_csv_path)
        metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)

    csv_fields = [
        "class_name", "data_type", "scale", "sample_id", "pair_type",
        "mse", "psnr", "energy_spectrum_mse", "r2", "ssim", "tke_acc", "nrmse"
    ]

    # 这些属性是在 load_data 里给 validate/test dataset 动态挂上的。
    # mixed 模式下它们用于把样本重新按真实类别归档。
    dataset = getattr(data_loader, "dataset", None)
    known_class_names = list(getattr(dataset, "known_class_names", []))
    other_name = getattr(dataset, "other_class_name", "other")

    def bucket_class_name(sample_class_name: str | None) -> str:
        # 优先用已知类别；没有命中就归到 other，避免评估阶段因脏类别名直接丢样本。
        if sample_class_name in known_class_names:
            return str(sample_class_name)
        if sample_class_name is None or str(sample_class_name).strip() == "":
            return other_name
        return str(sample_class_name) if not known_class_names else other_name

    rows = []
    rows_by_class: dict[str, list[dict]] = {}
    curves_by_class: dict[str, dict[str, list[np.ndarray]]] = {}
    all_pred_curves = []
    all_gt_curves = []

    def register_curve(bucket_name: str, pred_curve: np.ndarray, gt_curve: np.ndarray) -> None:
        # 频谱曲线既要进全局统计，也要进分类别统计，所以这里同时登记两份。
        all_pred_curves.append(pred_curve)
        all_gt_curves.append(gt_curve)
        if bucket_name not in curves_by_class:
            curves_by_class[bucket_name] = {"pred": [], "gt": []}
        curves_by_class[bucket_name]["pred"].append(pred_curve)
        curves_by_class[bucket_name]["gt"].append(gt_curve)

    def append_row(row: dict) -> None:
        # 同一条样本记录：
        # - rows 用于总表
        # - rows_by_class 用于类别子表
        rows.append(row)
        bucket = row["class_name"]
        rows_by_class.setdefault(bucket, []).append(row)

    def save_mean_spectrum(pred_curves: list[np.ndarray], gt_curves: list[np.ndarray], out_dir: Path, title: str) -> None:
        if not pred_curves or not gt_curves:
            return
        # 不同样本曲线长度可能略有不同，所以先截到共同最短长度再平均。
        min_len = min(min(len(x) for x in pred_curves), min(len(x) for x in gt_curves))
        pred_mean = np.mean(np.stack([x[:min_len] for x in pred_curves], axis=0), axis=0)
        gt_mean = np.mean(np.stack([x[:min_len] for x in gt_curves], axis=0), axis=0)
        np.save(out_dir / "energy_spectrum_pred_mean.npy", pred_mean.astype(np.float32))
        np.save(out_dir / "energy_spectrum_gt_mean.npy", gt_mean.astype(np.float32))
        _save_energy_spectrum_plot(pred_mean, gt_mean, out_dir / "energy_spectrum_mean_compare.png", title=title)

    def build_mean_row(target_rows: list[dict], bucket_name: str) -> dict:
        def _mean_of(key: str) -> float:
            vals = [float(r[key]) for r in target_rows if np.isfinite(float(r[key]))]
            return float(np.mean(vals)) if vals else float("nan")

        return {
            "class_name": bucket_name,
            "data_type": data_type,
            "scale": int(SCALE * SCALE),
            "sample_id": "MEAN",
            "pair_type": "all",
            "mse": _mean_of("mse"),
            "psnr": _mean_of("psnr"),
            "energy_spectrum_mse": _mean_of("energy_spectrum_mse"),
            "r2": _mean_of("r2"),
            "ssim": _mean_of("ssim"),
            "tke_acc": _mean_of("tke_acc"),
            "nrmse": _mean_of("nrmse"),
        }

    with torch.no_grad():
        pbar = tqdm(
            data_loader,
            desc=f"{class_name} {data_type} scale_{int(SCALE * SCALE)} Validating(all)",
            unit="batch", dynamic_ncols=True,
            ascii=True,
            leave=True,
        )

        for batch_idx, batch in enumerate(pbar):
            batch_class_names = batch.get("class_name", [])
            # batch['class_name'] 来自 data_load 的 collate_fn，是当前 batch 每个样本的真实类别名列表。

            if data_type == "image_pair":
                lr_prev = batch["image_pair"]["previous"]["lr_data"].to(device)
                hr_prev = batch["image_pair"]["previous"]["gr_data"].to(device)
                lr_next = batch["image_pair"]["next"]["lr_data"].to(device)
                hr_next = batch["image_pair"]["next"]["gr_data"].to(device)

                if hasattr(generator, "forward_pair"):
                    fake_prev, fake_next = generator.forward_pair(lr_prev, lr_next)
                else:
                    fake_prev = generator(lr_prev)
                    fake_next = generator(lr_next)

                lr_prev_up = F.interpolate(lr_prev, size=hr_prev.shape[2:], mode="nearest")
                lr_next_up = F.interpolate(lr_next, size=hr_next.shape[2:], mode="nearest")

                B = lr_prev.shape[0]
                for i in range(B):
                    sample_bucket = bucket_class_name(batch_class_names[i] if i < len(batch_class_names) else None)
                    # 所有属于同一真实类别的样本都写到同一个类别目录下，便于后处理和人工检查。
                    class_root = output_root / sample_bucket
                    class_root.mkdir(parents=True, exist_ok=True)

                    for pair_type, lr1_up, fk1, hr1 in [
                        ("previous", lr_prev_up[i:i+1], fake_prev[i:i+1], hr_prev[i:i+1]),
                        ("next",     lr_next_up[i:i+1], fake_next[i:i+1], hr_next[i:i+1]),
                    ]:
                        sid = f"batch_{batch_idx}_idx_{i}_fid_{batch_idx}_{pair_type}"
                        one_dir = class_root / sid
                        one_dir.mkdir(parents=True, exist_ok=True)

                        lr_eval = _select_metric_or_save_channels(lr1_up, "image_pair", global_data.esrgan.SAVE_AS_GRAY)
                        fk_eval = _select_metric_or_save_channels(fk1, "image_pair", global_data.esrgan.SAVE_AS_GRAY)
                        hr_eval = _select_metric_or_save_channels(hr1, "image_pair", global_data.esrgan.SAVE_AS_GRAY)

                        lr_save = lr_eval.clamp(0, 1)
                        fk_save = fk_eval.clamp(0, 1)
                        hr_save = hr_eval.clamp(0, 1)

                        save_image(lr_save, str(one_dir / "lr.png"), normalize=False)
                        save_image(fk_save, str(one_dir / "fake.png"), normalize=False)
                        save_image(hr_save, str(one_dir / "hr.png"), normalize=False)

                        diff = (fk_save - hr_save).abs()
                        diff_gray = diff if diff.shape[1] == 1 else diff.mean(dim=1, keepdim=True)
                        save_image(diff, str(one_dir / "diff_abs.png"), normalize=False)
                        save_image(diff_gray, str(one_dir / "diff_abs_gray.png"), normalize=False)
                        save_image(diff_gray / (diff_gray.max() + 1e-8), str(one_dir / "diff_abs_gray_norm.png"), normalize=False)
                        _save_triplet(lr_save, fk_save, hr_save, one_dir / "image_triplet.png")

                        p = _to_np_chw(fk_eval[0])
                        g = _to_np_chw(hr_eval[0])

                        mse = _mse(p, g)
                        psnr = _psnr_from_mse(mse)
                        es_mse = _energy_spectrum_mse(p, g)
                        r2 = _r2_score(p, g)
                        ssim = _ssim_score(p, g)
                        tke = _tke_reconstruction_accuracy(p, g)
                        nrmse = _nrmse(p, g)

                        pred_curve, gt_curve = _energy_spectrum_curves(p, g)
                        np.save(one_dir / "energy_spectrum_pred.npy", pred_curve.astype(np.float32))
                        np.save(one_dir / "energy_spectrum_gt.npy", gt_curve.astype(np.float32))
                        _save_energy_spectrum_plot(pred_curve, gt_curve, one_dir / "energy_spectrum_compare.png", title=f"{sid} Energy Spectrum")
                        register_curve(sample_bucket, pred_curve, gt_curve)

                        append_row({
                            "class_name": sample_bucket,
                            "data_type": data_type,
                            "scale": int(SCALE * SCALE),
                            "sample_id": sid,
                            "pair_type": pair_type,
                            "mse": mse,
                            "psnr": psnr,
                            "energy_spectrum_mse": es_mse,
                            "r2": r2,
                            "ssim": ssim,
                            "tke_acc": tke,
                            "nrmse": nrmse,
                        })

            elif data_type == "flo":
                lr = batch["flo"]["lr_data"].to(device)
                hr = batch["flo"]["gr_data"].to(device)
                fake = generator(lr)
                lr_up = F.interpolate(lr, size=hr.shape[2:], mode="nearest")
                B = lr.shape[0]

                for i in range(B):
                    sample_bucket = bucket_class_name(batch_class_names[i] if i < len(batch_class_names) else None)
                    class_root = output_root / sample_bucket
                    class_root.mkdir(parents=True, exist_ok=True)

                    sid = f"batch_{batch_idx}_idx_{i}_fid_{batch_idx}"
                    one_dir = class_root / sid
                    one_dir.mkdir(parents=True, exist_ok=True)

                    lr1 = lr[i:i+1]
                    lr_up1 = lr_up[i:i+1]
                    fk1 = fake[i:i+1]
                    hr1 = hr[i:i+1]

                    np.save(one_dir / "lr_flo.npy", _to_np_chw(lr1[0]).transpose(1, 2, 0))
                    np.save(one_dir / "fake_flo.npy", _to_np_chw(fk1[0]).transpose(1, 2, 0))
                    np.save(one_dir / "hr_flo.npy", _to_np_chw(hr1[0]).transpose(1, 2, 0))

                    hr_u = hr1[:, 0]
                    hr_v = hr1[:, 1]
                    hr_mag_uv = torch.sqrt(hr_u * hr_u + hr_v * hr_v)
                    ref_max_rad = max(torch.quantile(hr_mag_uv.flatten(), 0.99).item(), 1e-6)

                    lr_color, _ = flow_to_color_tensor(lr_up1[:, :2], ref_max_rad=ref_max_rad)
                    fk_color, _ = flow_to_color_tensor(fk1[:, :2], ref_max_rad=ref_max_rad)
                    hr_color, _ = flow_to_color_tensor(hr1[:, :2], ref_max_rad=ref_max_rad)
                    _save_triplet(lr_color, fk_color, hr_color, one_dir / "flow_triplet.png")

                    uvs_panel = build_flo_uvw_compare_panel(lr_up1, fk1, hr1)
                    save_image(uvs_panel.clamp(0, 1), str(one_dir / "uvs_compare.png"), normalize=False)
                    save_vorticity_quiver_compare(lr1, fk1, hr1, str(one_dir / "vorticity_quiver.png"), stride=stride)

                    p = _to_np_chw(fk1[0])
                    g = _to_np_chw(hr1[0])

                    mse = _mse(p, g)
                    psnr = _psnr_from_mse(mse)
                    es_mse = _energy_spectrum_mse(p, g)
                    r2 = _r2_score(p, g)
                    ssim = _ssim_score(p, g)
                    tke = _tke_reconstruction_accuracy(p, g)
                    nrmse = _nrmse(p, g)

                    pred_curve, gt_curve = _energy_spectrum_curves(p, g)
                    np.save(one_dir / "energy_spectrum_pred.npy", pred_curve.astype(np.float32))
                    np.save(one_dir / "energy_spectrum_gt.npy", gt_curve.astype(np.float32))
                    _save_energy_spectrum_plot(pred_curve, gt_curve, one_dir / "energy_spectrum_compare.png", title=f"{sid} Energy Spectrum")
                    register_curve(sample_bucket, pred_curve, gt_curve)

                    append_row({
                        "class_name": sample_bucket,
                        "data_type": data_type,
                        "scale": int(SCALE * SCALE),
                        "sample_id": sid,
                        "pair_type": "flo",
                        "mse": mse,
                        "psnr": psnr,
                        "energy_spectrum_mse": es_mse,
                        "r2": r2,
                        "ssim": ssim,
                        "tke_acc": tke,
                        "nrmse": nrmse,
                    })
            else:
                logger.error(f'Unsupported data_type: {data_type}')
                raise ValueError(f"Unsupported data_type: {data_type}")

    mean_row = build_mean_row(rows, class_name)
    all_rows_with_mean = rows + [mean_row]

    # 根目录下的均值频谱图表示“整个 validate/test loader”的总体表现。
    save_mean_spectrum(
        all_pred_curves,
        all_gt_curves,
        output_root,
        title=f"{class_name}-{data_type}-x{int(SCALE*SCALE)} Mean Energy Spectrum",
    )

    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(all_rows_with_mean)

    for bucket_name, class_rows in rows_by_class.items():
        class_root = output_root / bucket_name
        class_root.mkdir(parents=True, exist_ok=True)
        class_mean_row = build_mean_row(class_rows, bucket_name)
        # 每个类别目录都有自己的 metrics.csv，末尾附一行该类别的均值结果。
        with open(class_root / "metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(class_rows + [class_mean_row])
        bucket_curves = curves_by_class.get(bucket_name, {"pred": [], "gt": []})
        # 每个类别也各自输出一张平均能量谱对比图，方便直接做类间比较。
        save_mean_spectrum(
            bucket_curves["pred"],
            bucket_curves["gt"],
            class_root,
            title=f"{bucket_name}-{data_type}-x{int(SCALE*SCALE)} Mean Energy Spectrum",
        )

    logger.info(f"[evaluate_all] metrics csv: {metrics_csv_path}")
    logger.info(f"[evaluate_all] sample outputs: {output_root}")
    return mean_row


