"""
损失函数 start
"""
import torch
from torch import nn
from torchvision import models
from torchvision.models import vgg19
import torch.nn.functional as F

from study.SRGAN.util.image_util import _select_metric_or_save_channels, _to_gray
from study.SRGAN.model.PIV_esrgan_RAFT.global_class import global_data


# class VGG(nn.Module):
#     def __init__(self, device):
#         super(VGG, self).__init__()
#         vgg = models.vgg19(True)
#         for pa in vgg.parameters():
#             pa.requires_grad = False
#         self.vgg = vgg.features[:16]
#         self.vgg = self.vgg.to(device, non_blocking=True)
#
#     def forward(self, x):
#         out = self.vgg(x)
#         return out
class CharbonnierLoss(nn.Module):
    """
    Charbonnier 重建损失

    功能：
    1. 作为基础像素重建项，保证 SR 与 HR 在整体灰度分布上保持一致。
    2. 相比 MSE 对异常亮点更鲁棒，相比普通 L1 在零点附近更平滑。
    3. 适合微米颗粒图像中稀疏高亮颗粒的稳定恢复。

    数学形式：
        L_char = mean( sqrt((pred - target)^2 + eps^2) )
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = _to_gray(pred) - _to_gray(target)
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
class SobelEdgeLoss(nn.Module):
    """
    Sobel 边缘损失

    功能：
    1. 约束预测图像与真值图像在边缘梯度上的一致性。
    2. 强化颗粒边界与轮廓恢复。
    3. 减少颗粒边缘被平滑掉的问题。

    实现方式：
    1. 先将输入转为灰度图。
    2. 使用 Sobel 卷积核计算水平梯度与垂直梯度。
    3. 计算梯度幅值图。
    4. 对预测图和真值图的梯度幅值图做 L1 约束。
    """

    def __init__(self):
        super().__init__()

        sobel_x = torch.tensor(
            [[1, 0, -1],
             [2, 0, -2],
             [1, 0, -1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[1, 2, 1],
             [0, 0, 0],
             [-1, -2, -1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)



    def _grad_mag(self, x: torch.Tensor) -> torch.Tensor:
        x = _to_gray(x)
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_edge = self._grad_mag(pred)
        target_edge = self._grad_mag(target)
        return F.l1_loss(pred_edge, target_edge)
class GANLoss(nn.Module):
    """
    根据esrgan 的公式 生成器的对抗损失和判别器的损失都要用到这个计算
    """
    def __init__(self):
        super().__init__()
        # RaGAN / ESRGAN 这里应该吃“logits”，不是已经过 sigmoid 的概率。
        # 用 BCEWithLogitsLoss 后，输入不需要限制在 [0,1]，数值稳定性也更好。
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target_is_real):
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return self.bce(pred, target)

class ContentLoss(nn.Module):
    """
        内容损失（VGG特征空间L1 ）：约束生成图在感知特征上接近真值图。
        """

    def __init__(self, vgg,resize=False):
        """冻结 VGG 特征提取器并初始化 L1 损失。"""
        super().__init__()
        self.vgg = vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.criterion = nn.L1Loss()
        self.resize = resize


    def forward(self, fake, real):
        """计算 fake 与 real 的 VGG 特征 L1 。"""
        # 标准化 因为VGG需要【0，1】的图像且需要3通道
        fake = _to_gray(fake,False)
        real = _to_gray(real,False)

        # 给 G 传梯度：fake 不 detach；real 可以 detach
        fake_features = self.vgg(fake)
        real_features = self.vgg(real).detach()
        return self.criterion(fake_features, real_features)

class AdversarialLoss(nn.Module):
    """
    对抗损失：鼓励生成图被判别器判为真实。
    """
    def __init__(self):
        super(AdversarialLoss,self).__init__()
        self.gan_criterion = GANLoss()
    def forward(self, pred_fake,pred_real):
        # Relativistic average GAN for generator
        g_real = pred_real.detach()
        g_fake = pred_fake

        #注意这里和判别器的损失不同，这里是REAL 对 False 就是 和0比较 而判别器是对1比较
        loss_g_real = self.gan_criterion(g_real - torch.mean(g_fake), False)
        loss_g_fake = self.gan_criterion(g_fake - torch.mean(g_real), True)
        # adversarial_loss = (loss_g_real + loss_g_fake) / 2
        adversarial_loss = loss_g_real + loss_g_fake
        return adversarial_loss
class PerceptualLoss(nn.Module):
    """
    感知损失 = 内容损失+1e-3 * 对抗损失
    """
    def __init__(self, vgg):
        super(PerceptualLoss,self).__init__()
        self.vgg_loss = ContentLoss(vgg)
        # self.content_loss = ParticleTotalLoss(
        #     lambda_structure=global_data.esrgan.LAMBDA_STRUCTURE,
        #     lambda_physical=global_data.esrgan.LAMBDA_PHYSICAL
        # )
        self.adversarial = AdversarialLoss()

    def forward(self, fake, real, pred_fake,pred_real,is_adversarial=False):
        content_loss = self.vgg_loss(fake, real)

        # content_loss,_ = self.content_loss(fake, real)
        adversarial_loss = self.adversarial(pred_fake, pred_real)


        #是否启用对抗损失 就是是否预训练生成器
        if is_adversarial:
            # return vgg_loss +global_data.esrgan.LAMBDA_ADVERSARIAL*adversarial_loss,vgg_loss,adversarial_loss
            return global_data.esrgan.LAMBDA_CONTENT*content_loss +global_data.esrgan.LAMBDA_ADVERSARIAL*adversarial_loss,content_loss,adversarial_loss
        else:
            return global_data.esrgan.LAMBDA_CONTENT*content_loss,content_loss,adversarial_loss
# class RegularizationLoss(nn.Module):
#     """
#     图像平滑正则：惩罚相邻像素突变，抑制高频噪声。
#     """
#
#     def __init__(self):
#         super(RegularizationLoss,self).__init__()
#
#     def forward(self, x):
#         """计算基于局部梯度的平滑正则损失。"""
#         a = torch.square(
#             x[:, :, :x.shape[2]-1, :x.shape[3]-1] - x[:, :, 1:x.shape[2], :x.shape[3]-1]
#         )
#         b = torch.square(
#             x[:, :, :x.shape[2]-1, :x.shape[3]-1] - x[:, :, :x.shape[2]-1, 1:x.shape[3]]
#         )
#         loss = torch.sum(torch.pow(a+b, 1.25))
#         return loss

class SSIMLoss(nn.Module):
    """
    结构相似性损失（SSIM Loss）

    功能：
    1. 约束预测图像与目标图像在局部结构上的一致性。
    2. 同时考虑亮度、对比度和结构信息。
    3. 相比纯像素损失，更强调局部区域的结构保真。

    返回：
        L_ssim = 1 - SSIM(pred, target)
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        channels: int = 1,
        size_average: bool = True,
        data_range: float = 1.0,
        k1: float = 0.01,
        k2: float = 0.03,
    ):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channels = channels
        self.size_average = size_average
        self.data_range = data_range
        self.k1 = k1
        self.k2 = k2

        window = self._create_gaussian_window(window_size, sigma, channels)
        self.register_buffer("window", window)

    def _gaussian_1d(self, window_size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def _create_gaussian_window(self, window_size: int, sigma: float, channels: int) -> torch.Tensor:
        g1d = self._gaussian_1d(window_size, sigma).unsqueeze(1)
        g2d = g1d @ g1d.t()
        g2d = g2d.unsqueeze(0).unsqueeze(0)
        window = g2d.expand(channels, 1, window_size, window_size).contiguous()
        return window


    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = _to_gray(pred)
        target = _to_gray(target)

        c1 = (self.k1 * self.data_range) ** 2
        c2 = (self.k2 * self.data_range) ** 2

        window = self.window
        if window.shape[0] != pred.shape[1]:
            window = self._create_gaussian_window(
                self.window_size,
                self.sigma,
                pred.shape[1]
            ).to(pred.device, non_blocking=True)

        mu_pred = F.conv2d(pred, window, padding=self.window_size // 2, groups=pred.shape[1])
        mu_target = F.conv2d(target, window, padding=self.window_size // 2, groups=target.shape[1])

        mu_pred_sq = mu_pred.pow(2)
        mu_target_sq = mu_target.pow(2)
        mu_pred_target = mu_pred * mu_target

        sigma_pred_sq = F.conv2d(pred * pred, window, padding=self.window_size // 2, groups=pred.shape[1]) - mu_pred_sq
        sigma_target_sq = F.conv2d(target * target, window, padding=self.window_size // 2, groups=target.shape[1]) - mu_target_sq
        sigma_pred_target = F.conv2d(pred * target, window, padding=self.window_size // 2, groups=pred.shape[1]) - mu_pred_target

        ssim_map = (
            (2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)
        ) / (
            (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
        )

        if self.size_average:
            ssim_value = ssim_map.mean()
        else:
            ssim_value = ssim_map.mean(dim=(1, 2, 3))

        return 1.0 - ssim_value


class FrequencyReconstructionLoss(nn.Module):
    """
    频域幅值约束。

    对 PIV 颗粒图像来说，频域分布对应颗粒尺度与空间能量分布，
    能补足纯像素损失对高频纹理和颗粒间距约束不足的问题。
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_g = _to_gray(pred)
        target_g = _to_gray(target)

        # 对灰度图做频域约束即可，因为这里关心的是颗粒尺度/间距对应的能量分布，而不是颜色。
        pred_fft = torch.fft.fft2(pred_g, norm="ortho")
        target_fft = torch.fft.fft2(target_g, norm="ortho")

        # 不直接对 complex tensor 用 torch.abs 做反传，
        # 某些环境/版本下这条复数梯度链会在 backward 里触发
        # “imag is not implemented for tensors with non-complex dtypes”。
        # 这里改成显式的实部/虚部幅值计算，保持数学等价但更稳定。
        pred_fft_ri = torch.view_as_real(pred_fft)
        target_fft_ri = torch.view_as_real(target_fft)

        pred_amp = torch.log1p(
            torch.sqrt(pred_fft_ri[..., 0] * pred_fft_ri[..., 0] + pred_fft_ri[..., 1] * pred_fft_ri[..., 1] + 1e-12)
        )
        target_amp = torch.log1p(
            torch.sqrt(
                target_fft_ri[..., 0] * target_fft_ri[..., 0] +
                target_fft_ri[..., 1] * target_fft_ri[..., 1] + 1e-12
            )
        )
        return F.l1_loss(pred_amp, target_amp)

class CombinedPixelLoss(nn.Module):
    """
    组合像素损失：
    - 常规模式：L1 + MSE
    - 灰度增强模式：白点加权 + 通道一致性约束               !!!!!!!!!!!优化前就是L1
    SAVE_AS_GRAY=True 且 image_pair 时:
      - 用 target 亮度做加权，提升白点召回
      - 加轻量通道一致性约束（防伪彩）
    其余情况:
      - 普通 RGB L1 + MSE
    返回: total, l1_term, mse_term
    """
    def __init__(self, lambda_l1=2e-2, lambda_mse=1e-3, white_alpha=4.0, lambda_cons=1e-2, lambda_fft=1e-2):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_mse = lambda_mse
        self.white_alpha = white_alpha
        self.lambda_cons = lambda_cons
        self.lambda_fft = lambda_fft
        self.l1 = nn.SmoothL1Loss()
        self.mse = nn.MSELoss()
        self.fft = FrequencyReconstructionLoss()
        self.ssim = SSIMLoss(
            window_size=global_data.esrgan.SSIM_WINDOW_SIZE,
            sigma=global_data.esrgan.SSIM_SIGMA,
            channels=1,
            size_average=True,
            data_range=global_data.esrgan.SSIM_DATA_RANGE,
            k1=global_data.esrgan.SSIM_K1,
            k2=global_data.esrgan.SSIM_K2,
        )

    def forward(self, pred, target, gray_triplet=False):
        # 这里当前统一走“普通多项组合像素损失”。
        # `gray_triplet` 先保留接口不删，是为了兼容你之前灰度图像对训练时的逻辑接口。
        #优化前
        # if gray_triplet:
        #     pred_gray = pred[:, 0:1]
        #     target_gray = target[:, 0:1]
        #
        #     # 白点加权
        #     weight = 1.0 + self.white_alpha * target_gray
        #     l1_term = (weight * (pred_gray - target_gray).abs()).mean()
        #     mse_term = self.mse(pred_gray, target_gray)
        #
        #     # 灰度复制RGB一致性（不要太大）
        #     cons = (
        #         self.l1(pred[:, 0:1], pred[:, 1:2]) +
        #         self.l1(pred[:, 1:2], pred[:, 2:3]) +
        #         self.l1(pred[:, 0:1], pred[:, 2:3])
        #     ) / 3.0
        #
        #     total = self.lambda_l1 * l1_term + self.lambda_mse * mse_term + self.lambda_cons * cons
        #     return total, l1_term, mse_term

        l1_term = self.l1(pred, target)
        mse_term = self.mse(pred, target)
        SSIM_term = self.ssim(pred, target)
        fft_term = self.fft(pred, target)
        # total 里四项的角色：
        # - L1: 主重建项，稳
        # - MSE: 补充大误差惩罚
        # - SSIM: 保局部结构
        # - FFT: 保颗粒尺度与高频能量分布
        total = (
            self.lambda_l1 * l1_term +
            self.lambda_mse * mse_term +
            global_data.esrgan.LAMBDA_SSIM * SSIM_term +
            self.lambda_fft * fft_term
        )
        return total, l1_term, mse_term, SSIM_term, fft_term


class FlowWarpConsistencyLoss(nn.Module):
    """
    GT flow 引导的 SR 图像对一致性损失。

    previous 和 next 两张 PIV 粒子图像不能在同一个像素坐标上直接比较，
    因为颗粒在两次曝光之间会发生位移。这个损失使用 previous->next 的真实光流，
    将 SR next 采样回 SR previous 的坐标系，然后约束 SR previous 与对齐后的 SR next 在灰度上保持一致。
    """

    def __init__(self, flow_warp_weight: float = 1.0, eps: float = 1e-3):
        super().__init__()
        self.flow_warp_weight = flow_warp_weight
        self.eps = eps

    @staticmethod
    def _resize_flow_to_image(flow: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        将 flow 调整到图像分辨率，并同步缩放像素位移量。

        flow 的数值单位是像素位移。如果空间尺寸发生变化，u 需要按宽度比例缩放，
        v 需要按高度比例缩放；否则 grid_sample 虽然拿到了正确尺寸的 flow，
        但使用的物理位移幅值会是错的。
        """
        if flow.dim() != 4:
            raise ValueError(f"Expected flow to be 4D [B, C, H, W], got shape={tuple(flow.shape)}")
        if flow.size(1) < 2:
            raise ValueError(f"Expected flow to have at least 2 channels, got shape={tuple(flow.shape)}")

        flow_uv = flow[:, :2, :, :]
        image_h, image_w = image.shape[-2:]
        flow_h, flow_w = flow_uv.shape[-2:]
        if (flow_h, flow_w) == (image_h, image_w):
            return flow_uv

        resized_flow = F.interpolate(flow_uv, size=(image_h, image_w), mode="bilinear", align_corners=True)
        resized_flow[:, 0:1, :, :] = resized_flow[:, 0:1, :, :] * (image_w / max(flow_w, 1))
        resized_flow[:, 1:2, :, :] = resized_flow[:, 1:2, :, :] * (image_h / max(flow_h, 1))
        return resized_flow

    @staticmethod
    def _warp_next_to_prev(next_image: torch.Tensor, flow_prev_to_next: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        将 next_image 反向采样到 previous 帧的坐标系中。

        对 previous 帧坐标 (x, y)，previous->next 光流给出它在 next 帧中的对应坐标：
        (x + u, y + v)。因此采样网格就是 base_grid + flow。
        同时返回 valid_mask，用来屏蔽越界采样区域，避免边界外像素影响一致性损失。
        """
        if next_image.dim() != 4:
            raise ValueError(f"Expected next_image to be 4D [B, C, H, W], got shape={tuple(next_image.shape)}")

        batch, _, height, width = next_image.shape
        device = next_image.device
        dtype = next_image.dtype

        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        base_grid = torch.stack((x_coords, y_coords), dim=0).unsqueeze(0).repeat(batch, 1, 1, 1)
        sample_grid = base_grid + flow_prev_to_next.to(device=device, dtype=dtype)

        valid_x = (sample_grid[:, 0:1, :, :] >= 0) & (sample_grid[:, 0:1, :, :] <= width - 1)
        valid_y = (sample_grid[:, 1:2, :, :] >= 0) & (sample_grid[:, 1:2, :, :] <= height - 1)
        valid_mask = (valid_x & valid_y).to(dtype=dtype)

        # grid_sample 要求坐标归一化到 [-1, 1]，并且最后一维顺序必须是 [x, y]。
        norm_x = 2.0 * sample_grid[:, 0, :, :] / max(width - 1, 1) - 1.0
        norm_y = 2.0 * sample_grid[:, 1, :, :] / max(height - 1, 1) - 1.0
        normalized_grid = torch.stack((norm_x, norm_y), dim=-1)
        warped_next = F.grid_sample(
            next_image,
            normalized_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return warped_next, valid_mask

    def forward(
        self,
        pred_prev: torch.Tensor,
        pred_next: torch.Tensor,
        flow_prev_to_next: torch.Tensor,
    ):
        pred_prev_gray = _to_gray(pred_prev)
        pred_next_gray = _to_gray(pred_next)
        flow_uv = self._resize_flow_to_image(flow_prev_to_next, pred_prev_gray)
        warped_next_gray, valid_mask = self._warp_next_to_prev(pred_next_gray, flow_uv)

        # 只统计有效采样区域：先计算 Charbonnier map，再乘 mask，避免无效边界区域贡献 eps 常数项。
        diff = pred_prev_gray - warped_next_gray
        valid_count = valid_mask.sum().clamp_min(1.0)
        loss_map = torch.sqrt(diff * diff + self.eps * self.eps) * valid_mask
        flow_warp_consistency_loss = loss_map.sum() / valid_count
        flow_warp_consistency_weighted_loss = self.flow_warp_weight * flow_warp_consistency_loss

        return flow_warp_consistency_weighted_loss, {
            "flow_warp_consistency_loss": flow_warp_consistency_loss,
            "flow_warp_consistency_weighted_loss": flow_warp_consistency_weighted_loss,
        }
class Discriminator_loss(nn.Module):
    """
    判别器损失
    """
    def __init__(self):
        super().__init__()
        self.gan_criterion = GANLoss()
    def forward(self, pred_fake, pred_real):
        # Relativistic average GAN for discriminator
        # 注意这里和生成器的对抗损失不同，这里是REAL 对 True 就是 和1比较 而生成器的对抗损失对0比较
        loss_real = self.gan_criterion(pred_real - torch.mean(pred_fake.detach()), True)
        loss_fake = self.gan_criterion(pred_fake - torch.mean(pred_real.detach()), False)
        # d_loss = (loss_real + loss_fake) / 2
        d_loss = loss_real + loss_fake
        return d_loss,loss_fake,loss_real

"""
损失函数 end
"""
# 实例化loss
# 定义像素损失函数
pixel_loss = CombinedPixelLoss(
    lambda_l1=global_data.esrgan.LAMBDA_PIXEL_L1,
    lambda_mse=global_data.esrgan.LAMBDA_PIXEL_MSE,
    white_alpha=global_data.esrgan.PIXEL_WHITE_ALPHA,
    lambda_cons=global_data.esrgan.LAMBDA_GRAY_CONS,
    lambda_fft=global_data.esrgan.LAMBDA_PIXEL_FFT,
).to(global_data.esrgan.device, non_blocking=True)
# 这里vgg是针对三通道RGB图的
# vgg = vgg19(pretrained=True).features[:15].eval()  # 提取 VGG 特征 srgan 用的vgg
vgg =vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:15]
# vgg模型预测模式
vgg = vgg.to(global_data.esrgan.device, non_blocking=True).eval()

# 感知损失
perceptual_loss = PerceptualLoss(vgg=vgg).to(global_data.esrgan.device, non_blocking=True)
# 归一化损失 正则化损失
# regularization_loss = RegularizationLoss().to(global_data.esrgan.device, non_blocking=True)

# 这个 loss 只在 image_pair / RAFT 联合训练时使用；它用 GT flow 将 SR next 对齐回 SR previous。
flow_warp_consistency_loss = FlowWarpConsistencyLoss(
    flow_warp_weight=global_data.esrgan.LAMBDA_FLOW_WARP_CONSISTENCY,
).to(global_data.esrgan.device, non_blocking=True)
#判别器损失
descriminator_loss = Discriminator_loss().to(global_data.esrgan.device, non_blocking=True)
