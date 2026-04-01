import copy
import os

import torch
import torchvision


from study.SRGAN.model.esrgan_update.global_class import global_data
from study.SRGAN.model.esrgan_update.Module.loss import perceptual_loss, pixel_loss,  \
    descriminator_loss,particle_loss
from study.SRGAN.model.esrgan_update.visual_plot_init import build_flo_uvw_fake_panel
from study.SRGAN.model.esrgan_update.visual_plot_save import save_vorticity_quiver_single
from study.SRGAN.util.image_util import flow_to_color_tensor


def image_pair_train(epoch,batch,i, data_type, device, generator, discriminator,
                g_optimizer, d_optimizer,
                train_progress_bar,
                metric,class_name,SCALE):
    """
    执行 image_pair 的一个 batch 训练（previous/next 各训练一次）。
    图片对训练 ，因为是有两张图片所以训练两次
    :param epoch:训练轮次
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
    :param SCALE:上采样因子 具体放大平方倍
    :return:
    """
    for image_pair_type in global_data.esrgan.IMAGE_PAIR_TYPES:
        # 低分辨率图像
        lr_images = batch[data_type][image_pair_type]['lr_data'].to(device)
        # 真实图像
        gr_images = batch[data_type][image_pair_type]['gr_data'].to(device)
        batch_train(epoch=epoch,lr_images=lr_images, gr_images=gr_images, i=i, g_optimizer=g_optimizer,
                    d_optimizer=d_optimizer, generator=generator,
                    discriminator=discriminator, train_progress_bar=train_progress_bar,
                    metric=metric, data_type=data_type, device=device, class_name=class_name,image_pair_type = image_pair_type,SCALE=SCALE)
    pass
def flow_train(epoch,batch,i, data_type, device, generator, discriminator,
                g_optimizer, d_optimizer,
                train_progress_bar,
                metric,class_name,SCALE):
    """
    执行 flo 的一个 batch 训练。
    flo数据训练
    :param epoch:训练轮次
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
    :param SCALE:上采样因子 具体放大平方倍
    :return:
    """
    # 低分辨率图像
    lr_images = batch[data_type]['lr_data'].to(device)
    # 真实图像
    gr_images = batch[data_type]['gr_data'].to(device)
    batch_train(epoch=epoch,lr_images=lr_images, gr_images=gr_images, i=i, g_optimizer=g_optimizer,
                d_optimizer=d_optimizer, generator=generator,
                discriminator=discriminator, train_progress_bar=train_progress_bar,
                metric=metric, data_type=data_type, device=device, class_name=class_name,SCALE=SCALE)
    pass
def batch_train(epoch,lr_images,gr_images, i, data_type, device, generator, discriminator,
                g_optimizer, d_optimizer,
                train_progress_bar,
                metric,class_name,image_pair_type=None,SCALE=2) -> None:
    """
    单 batch 的 G/D 训练、损失统计与中间可视化保存。
    每一个batch的训练过程
    :param epoch:训练轮次
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
    :param SCALE:上采样因子 具体放大平方倍
    :return:
    """

    """
    给真实标签做一点平滑
    不要让判别器太容易自信到极致，比如：
    real label: 1.0 -> 0.9
    """
    # real_labels_out = torch.ones((len(lr_images), 1, 1, 1)).to(device)
    # real_labels = torch.full_like(real_labels_out, 0.9).to(device)


    # 生成器生成图像
    pred_images = generator(lr_images)
    # 冻结判别器参数 避免梯度更新
    for p in discriminator.parameters():
        p.requires_grad = False
    # print(f"pred_images:min,max,mean:{pred_images.min().data,pred_images.max().data,pred_images.mean().data} | lr_images:min,max,mean:{lr_images.min().data,lr_images.max().data,lr_images.mean().data} | gr_images:min,max,mean:{gr_images.min().data,gr_images.max().data,gr_images.mean().data} | ")
    # 判别器判别生成图像
    probability_pred_images = discriminator(pred_images)
    #判别器判别真实图像
    with torch.no_grad():
        probability_gr_images = discriminator(gr_images)
    """生成器训练 start"""
    # 感知损失 第无论才开启对抗损失 先与训练生成器
    if epoch>=global_data.esrgan.PRE_TRIAN_G_EPOCH-1:
        perceptual_loss_value,content_loss,adversarial_loss = perceptual_loss(pred_images, gr_images, probability_pred_images,probability_gr_images,True)
    else :
        perceptual_loss_value,content_loss,adversarial_loss = perceptual_loss(pred_images, gr_images, probability_pred_images,probability_gr_images,False)
    # 像素损失（灰白数据可开加权，flo 默认不开）
    gray_triplet = (global_data.esrgan.SAVE_AS_GRAY and data_type == "image_pair")  # 仅 image_pair 且设置为灰度复制模式
    g_loss_pixel, g_loss_l1, g_loss_mse,g_loss_ssim = pixel_loss(pred_images, gr_images, gray_triplet=gray_triplet)
    # 正则损失
    """
    这是很常见的现象，这个 RegularizationLoss 本质上是在惩罚图像相邻像素差，也就是一种平滑约束。
    它一开始很大、随后迅速掉到很小，通常不代表代码错了，更多说明生成器输出很快变“更平滑”了。
    也就是说，图像越抖、越噪、局部变化越剧烈，这个值越大；图像越平滑，这个值越小
    """

    # 生成器总损失
    # g_loss = perceptual_loss_value + LAMBDA_regularization_loss *regularization_loss_value +LAMBDA_loss_pixel*g_loss_pixel  # 这里的percuptual_loss包含了vgg_loss和对抗损失
    # g_loss = perceptual_loss_value #最原始的esrgan
    #没有用混合像素损失 而是直接根据esrgan 用了L1损失
    p_loss,p_loss_struct = particle_loss(pred_images, gr_images)
    g_loss = perceptual_loss_value+p_loss+global_data.esrgan.LAMBDA_PIXEL_L1*g_loss_l1+global_data.esrgan.LAMBDA_SSIM*g_loss_ssim

    # 优化生成器
    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()
    """生成器训练 end"""

    # #因为判别器太强了，让它弱一点，每两次训练一次
    # if i % 2 == 0:
    """判别器训练 start"""
    # 判别器判别真实图片之后将概率结果放入损失函数并且优化生成器
    #启用判别器梯度
    for p in discriminator.parameters():
        p.requires_grad = True
    # 重新判别
    pred_fake_d = discriminator(pred_images.detach())
    pred_real_d = discriminator(gr_images)
    """
    这里的real_loss 代表real data More realistic than fake ？
        fake_loss 代表fake Less realistic than real data? ？
        #因为之前已经用了probability_pred_images去更新生成器的梯度了，经过了一次反向传播，所以在用它要detach
    """
    d_loss,fake_loss,real_loss =descriminator_loss(pred_fake_d,pred_real_d)
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
    metric.add(g_loss.item(), perceptual_loss_value.item(),content_loss.item(),
               adversarial_loss.item(), g_loss_pixel.item(),
               p_loss,p_loss_struct['weighted_particle_physical_loss'].item(),p_loss_struct['weighted_particle_structure_loss'].item(),
               g_loss_l1.item(),g_loss_mse.item(),g_loss_ssim.item(),
               p_loss_struct['charbonnier_loss'],p_loss_struct["edge_loss"],p_loss_struct["bright_mask_loss"],p_loss_struct["mass_loss"],p_loss_struct["peak_loss"],p_loss_struct["separation_loss"],
               p_loss_struct["particle_count_loss"],p_loss_struct["particle_density_loss"],
               d_loss.item(), real_loss.item(), fake_loss.item())
    # end if i % 2 == 0:
    if i % global_data.esrgan.TRAIN_DATA_SAVING_STEP == 0:
        image = pred_images.detach()
        save_dir =f"{global_data.esrgan.OUT_PUT_DIR}/{class_name}/{data_type}/scale_{int(SCALE * SCALE)}/{global_data.esrgan.TRAINING_DIR}"
        os.makedirs(save_dir, exist_ok=True)

        save_prefix = f"{save_dir}/image_{len(train_progress_bar) * epoch + i}_{global_data.esrgan.name}"

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
                stride=6
            )
        elif image.shape[1] == 3:
            # image_pair: 保存前裁剪到 [0,1] 且取第一个通道
            image_to_save = image
            if global_data.esrgan.SAVE_AS_GRAY and data_type != "flo":
                image_to_save = image[:, 0:1, :, :]  # [N,1,H,W]

            torchvision.utils.save_image(
                image_to_save.clamp(0, 1),
                f"{save_prefix}_{image_pair_type}.png" if image_pair_type else f"{save_prefix}.png",
                nrow=4,
                normalize=False
            )
