import os

import torch
import torchvision
from torch.cuda.amp import GradScaler
from study.SRGAN.model.PIV_esrgan_RAFT.Module.PIV_ESRGAN_RAFT_Model import PIV_ESRGAN_RAFT
from study.SRGAN.model.PIV_esrgan_RAFT.global_class import global_data
from study.SRGAN.model.PIV_esrgan_RAFT.Module.loss import (
    descriminator_loss,
    image_pair_temporal_loss,
    particle_loss,
    perceptual_loss,
    pixel_loss,
)
from study.SRGAN.model.PIV_esrgan_RAFT.visual_plot_init import build_flo_uvw_fake_panel
from study.SRGAN.model.PIV_esrgan_RAFT.visual_plot_save import save_vorticity_quiver_single
from study.SRGAN.util.image_util import flow_to_color_tensor





def _save_training_preview(
    epoch,
    i,
    train_progress_bar,
    class_name,
    data_type,
    present_type,
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

    if image.shape[1] >= 2 and present_type == "flo":
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
        image_to_save = image[:, 0:1, :, :] if global_data.esrgan.SAVE_AS_GRAY and present_type != "flo" else image
        torchvision.utils.save_image(
            image_to_save.clamp(0, 1),
            f"{save_prefix}_{image_pair_type}.png" if image_pair_type else f"{save_prefix}.png",
            nrow=4,
            normalize=False,
        )


def esrgan_union_RAFT_train(
    epoch,
    batch,
    i,
    data_type,
    device,
    RAFT_optimizer:torch.optim.Adam,
    scaler:GradScaler,
    generator_scaler:GradScaler,
    g_optimizer:torch.optim.Adam,
    d_optimizer:torch.optim.Adam,
    train_progress_bar,
    metric,
    class_name,
    SCALE,
    model: PIV_ESRGAN_RAFT,
):
    """
    超分辨率与RAFT联合训练。
    """

    lr_prev = batch['image_pair']["previous"]["lr_data"].to(device)
    hr_prev = batch['image_pair']["previous"]["gr_data"].to(device)
    lr_next = batch['image_pair']["next"]["lr_data"].to(device)
    hr_next = batch['image_pair']["next"]["gr_data"].to(device)
    flow_hr = batch['flo']["gr_data"].to(device)

    #更新梯度
    pred_prev,pred_next,flow_predictions,loss_dict = model.train_step(
        input_lr_prev=lr_prev,
        input_lr_next=lr_next,
        input_gr_prev=hr_prev,
        input_gr_next=hr_next,
        flowl0=flow_hr,
        generator_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        raft_optimizer=RAFT_optimizer,
        scaler=scaler,
    )
    metric.add(
        # 注意这里的记录顺序必须和 global_class.loss_label 完全一致，否则 csv/plot 会错位。
        loss_dict['sr_loss'].item(),
        loss_dict["perceptual_loss"].item(),
        loss_dict["content_loss"].item(),
        loss_dict["adversarial_loss"].item(),
        loss_dict["pixel_total"].item(),
        loss_dict["pixel_l1"].item(),
        loss_dict["pixel_mse"].item(),
        loss_dict["pixel_ssim"].item(),
        loss_dict["pixel_fft"].item(),
        loss_dict['pair_delta_loss'].item(),
        loss_dict["pair_delta_loss"].item(),
        loss_dict["pair_gradient_loss"].item(),
        loss_dict["discriminator_loss"].item(),
        loss_dict["d_real_loss"].item(),
        loss_dict["d_fake_loss"].item(),
        loss_dict["raft_loss"].item(),
        loss_dict["raft_epe"].item(),
        loss_dict["raft_1px"].item(),
        loss_dict["raft_3px"].item(),
        loss_dict["raft_5px"].item(),


    )

    _save_training_preview(epoch, i, train_progress_bar, class_name, data_type,"image_pair", pred_prev.detach(), hr_prev, "previous", SCALE)
    _save_training_preview(epoch, i, train_progress_bar, class_name, data_type,"image_pair", pred_next.detach(), hr_next, "next", SCALE)
    _save_training_preview(epoch, i, train_progress_bar, class_name, data_type,"flo", flow_predictions.detach(), hr_next, "", SCALE)



