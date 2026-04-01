import os

import torch
import torchvision

from study.SRGAN.model.PIV_esrgan.global_class import global_data
from study.SRGAN.model.PIV_esrgan.Module.loss import (
    descriminator_loss,
    image_pair_temporal_loss,
    particle_loss,
    perceptual_loss,
    pixel_loss,
)
from study.SRGAN.model.PIV_esrgan.visual_plot_init import build_flo_uvw_fake_panel
from study.SRGAN.model.PIV_esrgan.visual_plot_save import save_vorticity_quiver_single
from study.SRGAN.util.image_util import flow_to_color_tensor


def _compute_generator_terms(epoch, data_type, discriminator, pred_images, gr_images):
    """
    统一计算“单帧生成器侧”的所有损失项。

    这样拆的原因：
    1. image_pair 的 previous / next 两帧要共享同一套损失定义；
    2. flo 和 image_pair 的单帧训练也能复用；
    3. 后续如果继续往里加物理损失，不需要把训练主流程写得越来越乱。
    """
    probability_pred_images = discriminator(pred_images)
    with torch.no_grad():
        probability_gr_images = discriminator(gr_images)

    use_adversarial = epoch >= global_data.esrgan.PRE_TRIAN_G_EPOCH - 1
    perceptual_loss_value, content_loss, adversarial_loss = perceptual_loss(
        pred_images, gr_images, probability_pred_images, probability_gr_images, use_adversarial
    )
    gray_triplet = global_data.esrgan.SAVE_AS_GRAY and data_type == "image_pair"
    # pixel_loss 会返回多个子项，后面日志、csv、可视化会分别记录，便于做消融。
    g_loss_pixel, g_loss_l1, g_loss_mse, g_loss_ssim, g_loss_fft = pixel_loss(
        pred_images, gr_images, gray_triplet=gray_triplet
    )
    p_loss, p_loss_struct = particle_loss(pred_images, gr_images)

    return {
        "perceptual_loss": perceptual_loss_value,
        "content_loss": content_loss,
        "adversarial_loss": adversarial_loss,
        "pixel_total": g_loss_pixel,
        "pixel_l1": g_loss_l1,
        "pixel_mse": g_loss_mse,
        "pixel_ssim": g_loss_ssim,
        "pixel_fft": g_loss_fft,
        "particle_total": p_loss,
        "particle_dict": p_loss_struct,
    }


def _mean_terms(term_list):
    # 对 previous / next 两帧的单帧损失做平均，避免图像对训练时某一帧在总损失里权重更大。
    keys = [k for k in term_list[0].keys() if k != "particle_dict"]
    mean_terms = {k: sum(item[k] for item in term_list) / len(term_list) for k in keys}

    particle_keys = term_list[0]["particle_dict"].keys()
    mean_terms["particle_dict"] = {
        k: sum(item["particle_dict"][k] for item in term_list) / len(term_list)
        for k in particle_keys
    }
    return mean_terms


def _save_training_preview(
    epoch,
    i,
    train_progress_bar,
    class_name,
    data_type,
    image,
    gr_images,
    image_pair_type=None,
    SCALE=2,
):
    # 训练中间图只做轻量保存，不参与反向传播；目的是快速观察颗粒是否在朝正确方向恢复。
    if i % global_data.esrgan.TRAIN_DATA_SAVING_STEP != 0:
        return

    save_dir = (
        f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/"
        f"scale_{int(SCALE * SCALE)}/{global_data.esrgan.TRAINING_DIR}"
    )
    os.makedirs(save_dir, exist_ok=True)
    save_prefix = f"{save_dir}/image_{len(train_progress_bar) * epoch + i}_{global_data.esrgan.name}"

    if image.dim() == 3:
        image = image.unsqueeze(0)

    if image.shape[1] >= 2 and data_type == "flo":
        # flo 可视化统一用 GT 的 99 分位速度幅值做颜色参考，减少不同样本之间色标漂移。
        hr_u = gr_images[:, 0]
        hr_v = gr_images[:, 1]
        hr_mag_uv = torch.sqrt(hr_u * hr_u + hr_v * hr_v)
        ref_max_rad = max(torch.quantile(hr_mag_uv.flatten(), 0.99).item(), 1e-6)

        pred_color, _ = flow_to_color_tensor(image[:, :2], ref_max_rad=ref_max_rad)
        torchvision.utils.save_image(
            pred_color.clamp(0, 1),
            f"{save_prefix}.png",
            nrow=4,
            normalize=False,
        )

        fake_uvw_panel = build_flo_uvw_fake_panel(image)
        torchvision.utils.save_image(
            fake_uvw_panel,
            f"{save_prefix}_uvs.png",
            nrow=1,
            normalize=False,
        )
        save_vorticity_quiver_single(
            image,
            f"{save_prefix}_vorticity_quiver.png",
            stride=6,
        )
        return

    if image.shape[1] == 3:
        image_to_save = image[:, 0:1, :, :] if global_data.esrgan.SAVE_AS_GRAY and data_type != "flo" else image
        torchvision.utils.save_image(
            image_to_save.clamp(0, 1),
            f"{save_prefix}_{image_pair_type}.png" if image_pair_type else f"{save_prefix}.png",
            nrow=4,
            normalize=False,
        )


def image_pair_train(
    epoch,
    batch,
    i,
    data_type,
    device,
    generator,
    discriminator,
    g_optimizer,
    d_optimizer,
    train_progress_bar,
    metric,
    class_name,
    SCALE,
):
    """
    图像对联合训练。

    和旧逻辑的关键区别：
    - 旧逻辑：previous / next 各当一张单图单独训练
    - 新逻辑：两帧一起 forward，单帧损失求平均，再叠加 pair temporal loss

    这样模型学到的不只是“每张图清晰”，还会学“前后帧变化要合理”。
    """
    lr_prev = batch[data_type]["previous"]["lr_data"].to(device)
    hr_prev = batch[data_type]["previous"]["gr_data"].to(device)
    lr_next = batch[data_type]["next"]["lr_data"].to(device)
    hr_next = batch[data_type]["next"]["gr_data"].to(device)

    if hasattr(generator, "forward_pair"):
        # 如果生成器支持双帧联合前向，就优先走这条路径。
        pred_prev, pred_next = generator.forward_pair(lr_prev, lr_next)
    else:
        pred_prev = generator(lr_prev)
        pred_next = generator(lr_next)

    for p in discriminator.parameters():
        p.requires_grad = False

    prev_terms = _compute_generator_terms(epoch, data_type, discriminator, pred_prev, hr_prev)
    next_terms = _compute_generator_terms(epoch, data_type, discriminator, pred_next, hr_next)
    mean_terms = _mean_terms([prev_terms, next_terms])

    # temporal loss 显式约束前后帧差分图的一致性，这是 PIV 图像对任务和普通超分最不同的地方。
    pair_temporal_total, pair_temporal_dict = image_pair_temporal_loss(pred_prev, pred_next, hr_prev, hr_next)
    g_loss = mean_terms["perceptual_loss"] + mean_terms["pixel_total"] + mean_terms["particle_total"] + pair_temporal_total

    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    for p in discriminator.parameters():
        p.requires_grad = True

    pred_fake_prev_d = discriminator(pred_prev.detach())
    pred_real_prev_d = discriminator(hr_prev)
    d_prev, fake_prev, real_prev = descriminator_loss(pred_fake_prev_d, pred_real_prev_d)

    pred_fake_next_d = discriminator(pred_next.detach())
    pred_real_next_d = discriminator(hr_next)
    d_next, fake_next, real_next = descriminator_loss(pred_fake_next_d, pred_real_next_d)

    # 判别器对两帧分别判别，再取平均，保持 previous / next 在对抗学习上的地位一致。
    d_loss = 0.5 * (d_prev + d_next)
    fake_loss = 0.5 * (fake_prev + fake_next)
    real_loss = 0.5 * (real_prev + real_next)

    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()

    train_progress_bar.set_postfix({
        "class": class_name,
        "D Loss": d_loss.item(),
        "G Loss": g_loss.item(),
        "Pair": pair_temporal_total.item(),
    })

    metric.add(
        # 注意这里的记录顺序必须和 global_class.loss_label 完全一致，否则 csv/plot 会错位。
        g_loss.item(),
        mean_terms["perceptual_loss"].item(),
        mean_terms["content_loss"].item(),
        mean_terms["adversarial_loss"].item(),
        mean_terms["pixel_total"].item(),
        mean_terms["particle_total"].item(),
        mean_terms["particle_dict"]["weighted_particle_physical_loss"].item(),
        mean_terms["particle_dict"]["weighted_particle_structure_loss"].item(),
        mean_terms["pixel_l1"].item(),
        mean_terms["pixel_mse"].item(),
        mean_terms["pixel_ssim"].item(),
        mean_terms["pixel_fft"].item(),
        pair_temporal_total.item(),
        pair_temporal_dict["pair_delta_loss"].item(),
        pair_temporal_dict["pair_gradient_loss"].item(),
        d_loss.item(),
        real_loss.item(),
        fake_loss.item(),
        mean_terms["particle_dict"]["charbonnier_loss"].item(),
        mean_terms["particle_dict"]["edge_loss"].item(),
        mean_terms["particle_dict"]["bright_mask_loss"].item(),
        mean_terms["particle_dict"]["mass_loss"].item(),
        mean_terms["particle_dict"]["peak_loss"].item(),
        mean_terms["particle_dict"]["separation_loss"].item(),
        mean_terms["particle_dict"]["particle_count_loss"].item(),
        mean_terms["particle_dict"]["particle_density_loss"].item(),
    )

    _save_training_preview(epoch, i, train_progress_bar, class_name, data_type, pred_prev.detach(), hr_prev, "previous", SCALE)
    _save_training_preview(epoch, i, train_progress_bar, class_name, data_type, pred_next.detach(), hr_next, "next", SCALE)


def flow_train(
    epoch,
    batch,
    i,
    data_type,
    device,
    generator,
    discriminator,
    g_optimizer,
    d_optimizer,
    train_progress_bar,
    metric,
    class_name,
    SCALE,
):
    # flo 保持单帧路径，不引入 pair temporal loss，避免把图像对假设硬套到流场上。
    lr_images = batch[data_type]["lr_data"].to(device)
    gr_images = batch[data_type]["gr_data"].to(device)
    batch_train(
        epoch=epoch,
        lr_images=lr_images,
        gr_images=gr_images,
        i=i,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        generator=generator,
        discriminator=discriminator,
        train_progress_bar=train_progress_bar,
        metric=metric,
        data_type=data_type,
        device=device,
        class_name=class_name,
        SCALE=SCALE,
    )


def batch_train(
    epoch,
    lr_images,
    gr_images,
    i,
    data_type,
    device,
    generator,
    discriminator,
    g_optimizer,
    d_optimizer,
    train_progress_bar,
    metric,
    class_name,
    image_pair_type=None,
    SCALE=2,
) -> None:
    """
    单帧训练路径。

    这个函数主要服务两种情况：
    1. flo 训练
    2. 不走 forward_pair 的单图像训练
    """
    pred_images = generator(lr_images)

    for p in discriminator.parameters():
        p.requires_grad = False

    terms = _compute_generator_terms(epoch, data_type, discriminator, pred_images, gr_images)
    # 单帧训练没有 pair temporal loss，对应日志项统一记 0，保证 csv 列结构不变。
    zero_pair = torch.zeros((), device=device)
    g_loss = terms["perceptual_loss"] + terms["pixel_total"] + terms["particle_total"]

    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    for p in discriminator.parameters():
        p.requires_grad = True

    pred_fake_d = discriminator(pred_images.detach())
    pred_real_d = discriminator(gr_images)
    d_loss, fake_loss, real_loss = descriminator_loss(pred_fake_d, pred_real_d)

    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()

    train_progress_bar.set_postfix({
        "class": class_name,
        "D Loss": d_loss.item(),
        "G Loss": g_loss.item(),
    })

    metric.add(
        g_loss.item(),
        terms["perceptual_loss"].item(),
        terms["content_loss"].item(),
        terms["adversarial_loss"].item(),
        terms["pixel_total"].item(),
        terms["particle_total"].item(),
        terms["particle_dict"]["weighted_particle_physical_loss"].item(),
        terms["particle_dict"]["weighted_particle_structure_loss"].item(),
        terms["pixel_l1"].item(),
        terms["pixel_mse"].item(),
        terms["pixel_ssim"].item(),
        terms["pixel_fft"].item(),
        zero_pair.item(),
        zero_pair.item(),
        zero_pair.item(),
        d_loss.item(),
        real_loss.item(),
        fake_loss.item(),
        terms["particle_dict"]["charbonnier_loss"].item(),
        terms["particle_dict"]["edge_loss"].item(),
        terms["particle_dict"]["bright_mask_loss"].item(),
        terms["particle_dict"]["mass_loss"].item(),
        terms["particle_dict"]["peak_loss"].item(),
        terms["particle_dict"]["separation_loss"].item(),
        terms["particle_dict"]["particle_count_loss"].item(),
        terms["particle_dict"]["particle_density_loss"].item(),
    )

    _save_training_preview(epoch, i, train_progress_bar, class_name, data_type, pred_images.detach(), gr_images, image_pair_type, SCALE)
