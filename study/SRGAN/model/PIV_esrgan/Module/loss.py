"""
损失函数 start
"""
import torch
from torch import nn
from torchvision import models
from torchvision.models import vgg19
import torch.nn.functional as F

from SRGAN.util.image_util import _select_metric_or_save_channels, _to_gray
from study.SRGAN.model.PIV_esrgan.global_class import global_data


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
class GANLoss(nn.Module):
    """
    根据esrgan 的公式 生成器的对抗损失和判别器的损失都要用到这个计算
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

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


class BrightMaskLoss(nn.Module):
    """
    高亮颗粒区域加权损失

    功能：
    1. 对 HR 图像中的高亮颗粒区域赋予更高权重。
    2. 避免背景像素数量过多导致损失被黑背景主导。
    3. 提高颗粒区域的恢复优先级，减少颗粒漏检。

    参数：
        threshold:
            当 use_soft_mask=False 时，使用二值阈值生成高亮颗粒 mask。
        use_soft_mask:
            True: 使用 target 灰度本身作为软权重。
            False: 使用 target > threshold 的二值 mask。
        base_weight:
            背景区域基础权重。
        peak_weight:
            颗粒区域附加权重。
    """

    def __init__(
        self,
        threshold: float = 0.5,
        use_soft_mask: bool = True,
        base_weight: float = 1.0,
        peak_weight: float = 1.0,
    ):
        super().__init__()
        self.threshold = threshold
        self.use_soft_mask = use_soft_mask
        self.base_weight = base_weight
        self.peak_weight = peak_weight


    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_g = _to_gray(pred)
        target_g = _to_gray(target)

        if self.use_soft_mask:
            mask = target_g.detach()
        else:
            mask = (target_g > self.threshold).float().detach()

        weight = self.base_weight + self.peak_weight * mask
        return torch.mean(weight * torch.abs(pred_g - target_g))


class MassConservationLoss(nn.Module):
    """
    颗粒总亮度守恒损失

    功能：
    1. 约束 SR 与 HR 的整体亮度总量一致。
    2. 防止颗粒整体偏暗导致粒子消失。
    3. 防止整体偏亮导致伪颗粒增加。

    数学形式：
        L_mass = mean( |sum(pred) - sum(target)| / (sum(target) + eps) )
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_g = _to_gray(pred)
        target_g = _to_gray(target)

        pred_mass = pred_g.sum(dim=(1, 2, 3))
        target_mass = target_g.sum(dim=(1, 2, 3))

        rel_err = torch.abs(pred_mass - target_mass) / (target_mass + self.eps)
        return rel_err.mean()
class PeakStructureLoss(nn.Module):
    """
    局部峰值结构损失

    功能：
    1. 强化颗粒局部峰值的恢复。
    2. 使每个颗粒更接近独立亮点，而不是模糊团块。
    3. 抑制颗粒峰值被抹平的问题。

    实现方式：
    1. 将图像转为灰度图。
    2. 通过局部最大池化估计邻域内最大响应。
    3. 计算像素与局部均值的差，突出局部峰值。
    4. 仅在接近局部最大值的位置保留峰响应。
    5. 对预测图与真值图的峰值图做 L1 约束。
    """

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2



    def _local_peak_map(self, x: torch.Tensor) -> torch.Tensor:
        x = _to_gray(x)

        # 提取局部最大值，用于定位潜在颗粒中心。
        local_max = F.max_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
        )

        # 计算局部均值，用于衡量当前位置相对邻域的突出程度。
        local_mean = F.avg_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
        )

        # 局部峰值强度 = 当前像素 - 局部均值，若小于0则截断。
        peak_strength = torch.relu(x - local_mean)

        # 只保留局部最大位置附近的峰，抑制平坦亮区。
        peak_mask = (x >= local_max - 1e-6).float()

        return peak_strength * peak_mask

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_peak = self._local_peak_map(pred)
        target_peak = self._local_peak_map(target)
        return F.l1_loss(pred_peak, target_peak)
class ParticleSeparationLoss(nn.Module):
    """
    颗粒分离损失

    功能：
    1. 增强颗粒中心与邻域背景之间的局部对比度。
    2. 减少颗粒之间的桥接和粘连。
    3. 恢复颗粒之间应有的暗间隔。

    实现方式：
    1. 将图像转为灰度图。
    2. 计算每个像素与其局部邻域均值之间的差值。
    3. 将这种“局部中心-邻域对比度图”在 SR 与 HR 之间做 L1 约束。

    参数：
        kernel_size:
            局部邻域大小。值越大，分离约束越偏向大尺度。
    """

    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2



    def _contrast_map(self, x: torch.Tensor) -> torch.Tensor:
        x = _to_gray(x)

        # 计算局部邻域均值，表示当前位置周围背景水平。
        local_mean = F.avg_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
        )

        # 局部对比度 = 当前像素 - 邻域均值。
        return x - local_mean

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_contrast = self._contrast_map(pred)
        target_contrast = self._contrast_map(target)
        return F.l1_loss(pred_contrast, target_contrast)
class ParticleStructurePreservingLoss(nn.Module):
    """
    颗粒结构保真损失

    组成：
    1. Charbonnier 重建损失
    2. 边缘损失
    3. 高亮颗粒区域加权损失
    4. 颗粒总亮度守恒损失
    5. 局部峰值结构损失
    6. 颗粒分离损失
    """

    def __init__(
        self,
        charbonnier_weight: float = 1.0,
        edge_weight: float = 0.15,
        bright_mask_weight: float = 0.35,
        mass_weight: float = 0.10,
        peak_weight: float = 0.35,
        separation_weight: float = 0.25,
        charbonnier_eps: float = 1e-3,
        bright_threshold: float = 0.5,
        use_soft_mask: bool = True,
        bright_base_weight: float = 1.0,
        bright_peak_weight: float = 1.0,
        peak_kernel_size: int = 3,
        separation_kernel_size: int = 5,
    ):
        super().__init__()

        self.charbonnier_weight = charbonnier_weight
        self.edge_weight = edge_weight
        self.bright_mask_weight = bright_mask_weight
        self.mass_weight = mass_weight
        self.peak_weight = peak_weight
        self.separation_weight = separation_weight

        self.charbonnier = CharbonnierLoss(eps=charbonnier_eps)
        self.edge = SobelEdgeLoss()
        self.bright_mask = BrightMaskLoss(
            threshold=bright_threshold,
            use_soft_mask=use_soft_mask,
            base_weight=bright_base_weight,
            peak_weight=bright_peak_weight,
        )
        self.mass = MassConservationLoss()
        self.peak = PeakStructureLoss(kernel_size=peak_kernel_size)
        self.separation = ParticleSeparationLoss(kernel_size=separation_kernel_size)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        charbonnier_loss = self.charbonnier(pred, target)
        edge_loss = self.edge(pred, target)
        bright_mask_loss = self.bright_mask(pred, target)
        mass_loss = self.mass(pred, target)
        peak_loss = self.peak(pred, target)
        separation_loss = self.separation(pred, target)

        total_loss = (
            self.charbonnier_weight * charbonnier_loss +
            self.edge_weight * edge_loss +
            self.bright_mask_weight * bright_mask_loss +
            self.mass_weight * mass_loss +
            self.peak_weight * peak_loss +
            self.separation_weight * separation_loss
        )

        loss_dict = {
            "charbonnier_loss": charbonnier_loss,
            "edge_loss": edge_loss,
            "bright_mask_loss": bright_mask_loss,
            "mass_loss": mass_loss,
            "peak_loss": peak_loss,
            "separation_loss": separation_loss,
            "particle_structure_loss": total_loss,
        }

        return total_loss, loss_dict




class ParticleCountLoss(nn.Module):
    """
    颗粒数量损失

    功能：
    1. 约束 SR 图像与 HR 图像中的颗粒总数量保持一致。
    2. 抑制颗粒漏检和伪颗粒增生。
    3. 通过软阈值方式构建可导的颗粒占据图，从而可直接参与反向传播。

    数学思想：
    1. 先将图像转为灰度图。
    2. 使用 sigmoid(sharpness * (x - threshold)) 构造软颗粒占据概率图。
    3. 对整张图的占据概率求和，近似表示颗粒总量。
    4. 计算 SR 与 HR 颗粒总量之间的相对误差。

    参数：
        threshold:
            颗粒阈值，决定哪些区域更可能被视为颗粒。
        sharpness:
            Sigmoid 的斜率。越大越接近硬阈值，越小越平滑。
        eps:
            防止分母为零的数值稳定项。
    """

    def __init__(
        self,
        threshold: float = 0.5,
        sharpness: float = 20.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.threshold = threshold
        self.sharpness = sharpness
        self.eps = eps


    def _soft_occupancy(self, x: torch.Tensor) -> torch.Tensor:
        x = _to_gray(x)
        return torch.sigmoid(self.sharpness * (x - self.threshold))

    def _hard_occupancy(self, x: torch.Tensor) -> torch.Tensor:
        x = _to_gray(x)
        return (x > self.threshold).float()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_occ = self._soft_occupancy(pred)
        target_occ = self._hard_occupancy(target).detach()

        pred_count = pred_occ.sum(dim=(1, 2, 3))
        target_count = target_occ.sum(dim=(1, 2, 3))

        rel_err = torch.abs(pred_count - target_count) / (target_count + self.eps)
        return rel_err.mean()


class ParticleDensityLoss(nn.Module):
    """
    颗粒局部密度分布损失

    功能：
    1. 约束 SR 与 HR 图像中的局部颗粒空间密度分布一致。
    2. 防止颗粒在局部区域中过度聚集或过度稀疏。
    3. 从空间统计角度约束颗粒分布，而不只是约束总数量。

    数学思想：
    1. 先构造颗粒占据图（软占据图用于 pred，硬占据图用于 target）。
    2. 再使用高斯核或均值核对占据图做局部平滑，得到局部密度图。
    3. 最后对 SR 与 HR 的密度图进行 L1 约束。

    参数：
        threshold:
            构造颗粒占据图时的阈值。
        sharpness:
            软阈值 Sigmoid 的斜率。
        kernel_size:
            局部密度估计卷积核大小。
        sigma:
            若 use_gaussian=True，则用于生成高斯核的标准差。
        use_gaussian:
            True 时使用高斯核估计局部密度；
            False 时使用均值池化近似局部密度。
    """

    def __init__(
        self,
        threshold: float = 0.5,
        sharpness: float = 20.0,
        kernel_size: int = 9,
        sigma: float = 2.0,
        use_gaussian: bool = True,
    ):
        super().__init__()
        self.threshold = threshold
        self.sharpness = sharpness
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.use_gaussian = use_gaussian

        if use_gaussian:
            kernel = self._build_gaussian_kernel(kernel_size, sigma)
            self.register_buffer("density_kernel", kernel)



    def _soft_occupancy(self, x: torch.Tensor) -> torch.Tensor:
        x = _to_gray(x)
        return torch.sigmoid(self.sharpness * (x - self.threshold))

    def _hard_occupancy(self, x: torch.Tensor) -> torch.Tensor:
        x = _to_gray(x)
        return (x > self.threshold).float()

    def _build_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, kernel_size, kernel_size)

    def _density_map(self, occ: torch.Tensor) -> torch.Tensor:
        if self.use_gaussian:
            return F.conv2d(
                occ,
                self.density_kernel,
                padding=self.kernel_size // 2,
            )
        return F.avg_pool2d(
            occ,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_occ = self._soft_occupancy(pred)
        target_occ = self._hard_occupancy(target).detach()

        pred_density = self._density_map(pred_occ)
        target_density = self._density_map(target_occ)

        return F.l1_loss(pred_density, target_density)

class ParticlePhysicalDistributionLoss(nn.Module):
    """
    颗粒物理分布损失

    组成：
    1. 颗粒数量损失
    2. 颗粒局部密度分布损失
    """

    def __init__(
        self,
        count_weight: float = 0.40,
        density_weight: float = 0.40,
        threshold: float = 0.5,
        sharpness: float = 20.0,
        count_eps: float = 1e-6,
        density_kernel_size: int = 9,
        density_sigma: float = 2.0,
        density_use_gaussian: bool = True,
    ):
        super().__init__()

        self.count_weight = count_weight
        self.density_weight = density_weight

        self.count_loss = ParticleCountLoss(
            threshold=threshold,
            sharpness=sharpness,
            eps=count_eps,
        )
        self.density_loss = ParticleDensityLoss(
            threshold=threshold,
            sharpness=sharpness,
            kernel_size=density_kernel_size,
            sigma=density_sigma,
            use_gaussian=density_use_gaussian,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        count_loss = self.count_loss(pred, target)
        density_loss = self.density_loss(pred, target)

        total_loss = (
            self.count_weight * count_loss +
            self.density_weight * density_loss
        )

        loss_dict = {
            "particle_count_loss": count_loss,
            "particle_density_loss": density_loss,
            "particle_physical_distribution_loss": total_loss,
        }

        return total_loss, loss_dict


class ParticleTotalLoss(nn.Module):
    """
    最终总损失

    总体思路：
    1. 结构保真损失：约束颗粒亮度、边界、峰值与分离性
    2. 物理分布损失：约束颗粒数量与局部空间密度分布
    3. 两者加权求和形成最终训练目标

    总损失形式：
        L_total = lambda_structure * L_structure + lambda_physical * L_physical
    """

    def __init__(
        self,
        lambda_structure: float = 1.0,
        lambda_physical: float = 0.6,

        charbonnier_weight: float = 1.0,
        edge_weight: float = 0.15,
        bright_mask_weight: float = 0.35,
        mass_weight: float = 0.10,
        peak_weight: float = 0.35,
        separation_weight: float = 0.25,
        charbonnier_eps: float = 1e-3,
        bright_threshold: float = 0.5,
        use_soft_mask: bool = True,
        bright_base_weight: float = 1.0,
        bright_peak_weight: float = 1.0,
        peak_kernel_size: int = 3,
        separation_kernel_size: int = 5,

        count_weight: float = 0.40,
        density_weight: float = 0.40,
        physical_threshold: float = 0.5,
        physical_sharpness: float = 20.0,
        count_eps: float = 1e-6,
        density_kernel_size: int = 9,
        density_sigma: float = 2.0,
        density_use_gaussian: bool = True,
    ):
        super().__init__()

        self.lambda_structure = lambda_structure
        self.lambda_physical = lambda_physical

        self.structure_loss = ParticleStructurePreservingLoss(
            charbonnier_weight=global_data.esrgan.LAMBDA_CHARBONNIER,
            edge_weight=global_data.esrgan.LAMBDA_EDGE,
            bright_mask_weight=global_data.esrgan.LAMBDA_BRIGHT_MASK,
            mass_weight=global_data.esrgan.LAMBDA_MASS,
            peak_weight=global_data.esrgan.LAMBDA_PEAK,
            separation_weight=global_data.esrgan.LAMBDA_SEPARATION,
            charbonnier_eps=global_data.esrgan.CHARBONNIER_EPS,
            bright_threshold=global_data.esrgan.BRIGHT_THRESHOLD,
            use_soft_mask=global_data.esrgan.USE_SOFT_MASK,
            bright_base_weight=global_data.esrgan.BRIGHT_BASE_WEIGHT,
            bright_peak_weight=global_data.esrgan.BRIGHT_PEAK_WEIGHT,
            peak_kernel_size=global_data.esrgan.PEAK_KERNEL_SIZE,
            separation_kernel_size=global_data.esrgan.SEPARATION_KERNEL_SIZE,
        )

        self.physical_loss = ParticlePhysicalDistributionLoss(
            count_weight=global_data.esrgan.LAMBDA_PARTICLE_COUNT,
            density_weight=global_data.esrgan.LAMBDA_PARTICLE_DENSITY,
            threshold=global_data.esrgan.PARTICLE_THRESHOLD,
            sharpness=global_data.esrgan.PARTICLE_SHARPNESS,
            count_eps=global_data.esrgan.PARTICLE_COUNT_EPS,
            density_kernel_size=global_data.esrgan.PARTICLE_DENSITY_KERNEL_SIZE,
            density_sigma=global_data.esrgan.PARTICLE_DENSITY_SIGMA,
            density_use_gaussian=global_data.esrgan.PARTICLE_DENSITY_USE_GAUSSIAN,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        structure_total, structure_dict = self.structure_loss(pred, target)
        physical_total, physical_dict = self.physical_loss(pred, target)

        total_loss = (
            self.lambda_structure * structure_total +
            self.lambda_physical * physical_total
        )

        loss_dict = {
            **structure_dict,
            **physical_dict,
            "weighted_particle_structure_loss": self.lambda_structure * structure_total,
            "weighted_particle_physical_loss": self.lambda_physical * physical_total,
            "particle_total_loss": total_loss,
        }

        return total_loss, loss_dict
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
            ).to(pred.device)

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

        pred_fft = torch.fft.fft2(pred_g, norm="ortho")
        target_fft = torch.fft.fft2(target_g, norm="ortho")

        pred_amp = torch.log1p(torch.abs(pred_fft))
        target_amp = torch.log1p(torch.abs(target_fft))
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
        total = (
            self.lambda_l1 * l1_term +
            self.lambda_mse * mse_term +
            global_data.esrgan.LAMBDA_SSIM * SSIM_term +
            self.lambda_fft * fft_term
        )
        return total, l1_term, mse_term, SSIM_term, fft_term


class ImagePairTemporalConsistencyLoss(nn.Module):
    """
    图像对时间一致性损失。

    核心思想：
    1. previous/next 的差分应与 HR 差分一致。
    2. 差分图的梯度结构也应一致，避免位移颗粒被生成成“糊块”。
    """

    def __init__(self, delta_weight: float = 1.0, gradient_weight: float = 0.5, eps: float = 1e-3):
        super().__init__()
        self.delta_weight = delta_weight
        self.gradient_weight = gradient_weight
        self.charbonnier = CharbonnierLoss(eps=eps)
        self.edge = SobelEdgeLoss()

    def forward(
        self,
        pred_prev: torch.Tensor,
        pred_next: torch.Tensor,
        target_prev: torch.Tensor,
        target_next: torch.Tensor,
    ):
        pred_delta = _to_gray(pred_next) - _to_gray(pred_prev)
        target_delta = _to_gray(target_next) - _to_gray(target_prev)

        delta_loss = self.charbonnier(pred_delta, target_delta)
        gradient_loss = self.edge(pred_delta, target_delta)
        total_loss = self.delta_weight * delta_loss + self.gradient_weight * gradient_loss

        return total_loss, {
            "pair_delta_loss": delta_loss,
            "pair_gradient_loss": gradient_loss,
            "pair_temporal_loss": total_loss,
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
).to(global_data.esrgan.device)
# 这里vgg是针对三通道RGB图的
# vgg = vgg19(pretrained=True).features[:15].eval()  # 提取 VGG 特征 srgan 用的vgg
vgg =vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:15]
# vgg模型预测模式
vgg = vgg.to(global_data.esrgan.device).eval()

# 感知损失
perceptual_loss = PerceptualLoss(vgg=vgg).to(global_data.esrgan.device)
# 归一化损失 正则化损失
# regularization_loss = RegularizationLoss().to(global_data.esrgan.device)
particle_loss = ParticleTotalLoss(
    lambda_structure=global_data.esrgan.LAMBDA_STRUCTURE,
    lambda_physical=global_data.esrgan.LAMBDA_PHYSICAL
).to(global_data.esrgan.device)
image_pair_temporal_loss = ImagePairTemporalConsistencyLoss(
    delta_weight=global_data.esrgan.LAMBDA_PAIR_DELTA,
    gradient_weight=global_data.esrgan.LAMBDA_PAIR_GRADIENT,
).to(global_data.esrgan.device)
#判别器损失
descriminator_loss = Discriminator_loss().to(global_data.esrgan.device)
