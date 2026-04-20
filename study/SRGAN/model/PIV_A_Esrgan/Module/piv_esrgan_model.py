"""
模型 start
"""
from loguru import logger
import torch
from torch import nn
import torch.nn.functional as F







def icnr_(tensor: torch.Tensor, scale: int = 2, initializer=nn.init.kaiming_normal_) -> torch.Tensor:
    """
    对 PixelShuffle 前卷积层执行 ICNR 初始化，减轻棋盘格伪影。
    ICNR init for sub-pixel convolution weights.
    tensor shape: [out_channels, in_channels, kH, kW]
    """
    out_channels, in_channels, kH, kW = tensor.shape
    if out_channels % (scale ** 2) != 0:
        logger.error(f'out_channels({out_channels}) must be divisible by scale^2({scale ** 2})')
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
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1,padding_mode="reflect"),

            nn.PReLU(),

            # 第2个卷积:
            # 仍然保持 64 -> 64，方便和输入直接相加
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1,padding_mode="reflect"),

        )

    def forward(self, x):
        # 残差连接
        return x + self.block(x)
class DenseResidualBlock(nn.Module):
    def __init__(self, channels=64, growth_channels=32, res_scale=0.2):
        super().__init__()
        self.res_scale = res_scale

        self.conv1 = nn.Conv2d(channels, growth_channels, 3, 1, 1,padding_mode="reflect")
        self.conv2 = nn.Conv2d(channels + growth_channels, growth_channels, 3, 1, 1,padding_mode="reflect")
        self.conv3 = nn.Conv2d(channels + growth_channels * 2, growth_channels, 3, 1, 1,padding_mode="reflect")
        self.conv4 = nn.Conv2d(channels + growth_channels * 3, growth_channels, 3, 1, 1,padding_mode="reflect")
        self.conv5 = nn.Conv2d(channels + growth_channels * 4, channels, 3, 1, 1,padding_mode="reflect")

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x + x5 * self.res_scale

class RRDB(nn.Module):
    def __init__(self, channels, growth_channels=32, res_scale=0.2):
        super().__init__()
        self.rdb1 = DenseResidualBlock(channels, growth_channels, res_scale)
        self.rdb2 = DenseResidualBlock(channels, growth_channels, res_scale)
        self.rdb3 = DenseResidualBlock(channels, growth_channels, res_scale)
        self.res_scale = res_scale

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + out * self.res_scale


class PairAwareFeatureFusion(nn.Module):
    """
    PIV 图像对特征交互模块。

    设计动机：
    1. `previous` 和 `next` 两帧不是独立样本，而是同一粒子场在短时间内的两次观测。
    2. 单帧超分只能学到“像 HR”，但不一定能学到“前后帧位移关系”。
    3. 这里在编码特征层面引入跨帧消息传递，让网络在恢复当前帧时参考另一帧的颗粒变化。

    结构说明：
    - `message` 分支负责从 partner 帧和差分特征中提取可迁移信息。
    - `gate` 分支负责决定这些信息对当前帧到底该注入多少，避免无脑相加带来伪影。
    """
    def __init__(self, channels=64):
        super().__init__()
        self.message = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1,padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1,padding_mode="reflect"),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=3, stride=1, padding=1,padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def _fuse_one(self, anchor: torch.Tensor, partner: torch.Tensor) -> torch.Tensor:
        # diff 强调两帧真正发生变化的区域；这些区域往往正是 PIV 位移最敏感的位置。
        diff = torch.abs(anchor - partner)
        # message 学“另一帧能给当前帧补什么细节”。
        message = self.message(torch.cat([partner, diff], dim=1))
        # gate 学“该补多少”，让融合是自适应的，而不是固定权重混合。
        gate = self.gate(torch.cat([anchor, partner, diff], dim=1))
        return anchor + gate * message

    def forward(self, feat_prev: torch.Tensor, feat_next: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._fuse_one(feat_prev, feat_next), self._fuse_one(feat_next, feat_prev)


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
    2. 23个RRDB  11
    3. 全局残差连接
    4. 两次 2x 上采样，总共 4x
    5. 输出 RGB 图像
    """
    def __init__(self,inner_chanel=3, num_residual_blocks=8, scale=2):
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
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(inner_chanel, 64, kernel_size=9, stride=1, padding=4),
        #     nn.LeakyReLU(0.2, inplace=True)
        # )
        """
        改成并联：
        3x3 更擅长小颗粒、细边缘、局部纹理
        5x5 适合中等颗粒和局部结构
        7x7 或 9x9 更适合大一点的亮斑和上下文
        """
        self.stem3 = nn.Sequential(
            nn.Conv2d(inner_chanel, 32, kernel_size=3, stride=1, padding=1,padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.stem5 = nn.Sequential(
            nn.Conv2d(inner_chanel, 32, kernel_size=5, stride=1, padding=2,padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.stem7 = nn.Sequential(
            nn.Conv2d(inner_chanel, 32, kernel_size=7, stride=1, padding=3,padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv1_fuse = nn.Sequential(
            nn.Conv2d(32*3, 64, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 残差块堆叠
        # 每个残差块都保持 [B, 64, H, W] 不变
        self.residual_blocks = nn.Sequential(
            *[RRDB(64,growth_channels=32) for _ in range(num_residual_blocks)]
        )

        # 残差块之后的融合卷积
        # 这里仍然保持 64 -> 64
        # 目的是把残差块提取到的特征再融合一下
        #
        # [B, 64, 64, 64] -> [B, 64, 64, 64]
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,padding_mode="reflect"),
        )
        # 只在双帧前向中启用；单帧 forward 不经过这里，保证向后兼容。
        self.pair_fusion = PairAwareFeatureFusion(64)

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
            # nn.Conv2d(64, 64 * self.scale * self.scale, kernel_size=3, stride=1, padding=1),

            # [B, 256, H, W] -> [B, 64, 2H, 2W]
            # nn.PixelShuffle(self.scale),
            nn.Upsample(scale_factor=self.scale, mode='nearest'),
            nn.Conv2d(64, 64, 3, 1, 1,padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
            #后面接一个RB让上采样更清晰
            ResidualBlock(64),
            # nn.Conv2d(64, 64 * self.scale * self.scale, kernel_size=3, stride=1, padding=1),
            # [B, 256, H, W] -> [B, 64, 2H, 2W]
            # nn.PixelShuffle(self.scale),
            nn.Upsample(scale_factor=self.scale, mode='nearest'),
            nn.Conv2d(64, 64, 3, 1, 1,padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
            # 后面接一个RB让上采样更清晰
            ResidualBlock(64),
        )

        # 最后一层输出 RGB 图像
        # 64 -> 3
        # kernel_size=3, padding=1 保持尺寸不变
        #
        # 若 scale=4:
        # [B, 64, 256, 256] -> [B, 3, 256, 256]
        self.conv_out = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, inner_chanel, kernel_size=3, stride=1, padding=1,padding_mode="reflect"),

        )

        #ICNR 初始化 esrgan不应该有这个
        # self._init_subpixel_weights()

    def _init_subpixel_weights(self):
        #对 PixelShuffle 前的 Conv2d(64, 64*scale*scale, ...) 做了 ICNR，能明显减轻棋盘纹。
        for m in self.upsample:
            if isinstance(m, nn.Conv2d) and m.out_channels == 64 * self.scale * self.scale:
                icnr_(m.weight, scale=self.scale, initializer=nn.init.kaiming_normal_)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        # 三个不同感受野的 stem 分支分别提取：
        # - 小核：微小颗粒、细边界
        # - 中核：局部结构
        # - 大核：较大亮斑和周围上下文
        x1_3 = self.stem3(x)
        x1_5 = self.stem5(x)
        x1_7 = self.stem7(x)

        x1 = self.conv1_fuse(torch.cat([x1_3, x1_5, x1_7], dim=1))
        # 主干仍然是 RRDB 堆叠，负责单帧内部的高频重建。
        x2 = self.residual_blocks(x1)
        x3 = self.conv2(x2)
        # 返回浅层特征 + 深层残差特征的融合结果，供单帧解码或双帧交互使用。
        return x1 + x3

    def _decode(self, features: torch.Tensor) -> torch.Tensor:
        # 解码阶段保持原来的上采样策略，避免这次改动把双帧交互和解码逻辑耦合得太紧。
        return self.conv_out(self.upsample(features))

    def forward_pair(self, prev: torch.Tensor, next_frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 先分别编码，再做跨帧特征融合，最后各自解码。
        # 这样既保留每一帧自己的外观信息，又显式利用前后帧的运动相关性。
        prev_feat = self._encode(prev)
        next_feat = self._encode(next_frame)
        prev_feat, next_feat = self.pair_fusion(prev_feat, next_feat)
        return self._decode(prev_feat), self._decode(next_feat)

    def forward(self, x):
        return self._decode(self._encode(x))


def _maybe_spectral_norm(layer: nn.Module, enabled: bool) -> nn.Module:
    """
    按需给卷积层加 spectral normalization。

    对 GAN 判别器来说，spectral normalization 会限制每层卷积的 Lipschitz 常数，
    能明显降低“小 batch + 强判别器”时 D 过快变得极端自信的问题。A-ESRGAN 论文中
    也使用了 spectral normalization 来稳定 attention U-Net 判别器训练。
    """
    return nn.utils.spectral_norm(layer) if enabled else layer


class DiscriminatorConvBlock(nn.Module):
    """
    Attention U-Net 判别器使用的基础卷积块。

    这里故意不使用 BatchNorm：
    1. 你的训练 batch size 通常比较小，BatchNorm 的 batch 统计容易抖动；
    2. 判别器已经使用 spectral normalization 稳定参数尺度；
    3. PIV 颗粒图像对更依赖亮度/对比度的细微差异，BatchNorm 可能会抹掉一部分
       真实强度分布信息。
    """

    def __init__(self, in_channels: int, out_channels: int, spectral_norm: bool = True):
        super().__init__()
        self.block = nn.Sequential(
            _maybe_spectral_norm(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                spectral_norm,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            _maybe_spectral_norm(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                spectral_norm,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate2D(nn.Module):
    """
    2D Attention U-Net 的 skip attention gate。

    参数含义：
    - skip: 编码器同尺度特征，保留颗粒的精确空间位置；
    - gating: 解码器更深层特征，携带更大感受野的真假判断上下文。

    输出是 skip * attention_map。这样 U-Net 在回传局部真假反馈时，会更关注颗粒边缘、
    亮点、帧间变化等“真正需要判别”的区域，而不是平均地惩罚整张背景。
    """

    def __init__(self, skip_channels: int, gating_channels: int, inter_channels: int, spectral_norm: bool = True):
        super().__init__()
        self.skip_project = _maybe_spectral_norm(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            spectral_norm,
        )
        self.gating_project = _maybe_spectral_norm(
            nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            spectral_norm,
        )
        self.attention_score = _maybe_spectral_norm(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0),
            spectral_norm,
        )
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, skip: torch.Tensor, gating: torch.Tensor) -> torch.Tensor:
        # 解码器 gating 特征有时因为奇偶尺寸下采样/上采样产生 1 像素差异，
        # 这里统一插值到 skip 的空间尺寸，保证 attention map 与 skip 一一对应。
        if gating.shape[-2:] != skip.shape[-2:]:
            gating = F.interpolate(gating, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        attention = self.activation(self.skip_project(skip) + self.gating_project(gating))
        attention = self.sigmoid(self.attention_score(attention))
        return skip * attention


class AttentionUNetDiscriminator(nn.Module):
    """
    单尺度 Attention U-Net 判别器。

    输入语义不是普通 RGB，而是 PIV 时序三通道：
    [prev_gray, next_gray, abs(next_gray - prev_gray)]。

    输出为 [B, 1, H, W] 的 logit map：
    - 每个位置都是局部真假判断，不再像旧 SRGAN 判别器那样被全局池化成 1 个数；
    - 这样生成器能收到更细的梯度，知道哪些颗粒边缘、背景噪声、帧间变化区域不像 HR。
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32, spectral_norm: bool = True):
        super().__init__()
        c1 = int(base_channels)
        c2 = c1 * 2
        c3 = c1 * 4
        c4 = c1 * 8

        # 编码器：逐级扩大感受野，学习从微小颗粒边缘到较大颗粒团簇的真假线索。
        self.enc0 = DiscriminatorConvBlock(in_channels, c1, spectral_norm=spectral_norm)
        self.down1 = nn.Sequential(
            _maybe_spectral_norm(nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1), spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc1 = DiscriminatorConvBlock(c2, c2, spectral_norm=spectral_norm)
        self.down2 = nn.Sequential(
            _maybe_spectral_norm(nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1), spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc2 = DiscriminatorConvBlock(c3, c3, spectral_norm=spectral_norm)
        self.down3 = nn.Sequential(
            _maybe_spectral_norm(nn.Conv2d(c3, c4, kernel_size=4, stride=2, padding=1), spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.bottleneck = DiscriminatorConvBlock(c4, c4, spectral_norm=spectral_norm)

        # 解码器：把深层真假上下文还原到像素级位置，并用 attention gate 选择性接入 skip 特征。
        self.att2 = AttentionGate2D(c3, c4, c2, spectral_norm=spectral_norm)
        self.dec2 = DiscriminatorConvBlock(c4 + c3, c3, spectral_norm=spectral_norm)
        self.att1 = AttentionGate2D(c2, c3, c1, spectral_norm=spectral_norm)
        self.dec1 = DiscriminatorConvBlock(c3 + c2, c2, spectral_norm=spectral_norm)
        self.att0 = AttentionGate2D(c1, c2, max(c1 // 2, 8), spectral_norm=spectral_norm)
        self.dec0 = DiscriminatorConvBlock(c2 + c1, c1, spectral_norm=spectral_norm)

        # 最后一层输出 logits，不加 sigmoid；后续 BCEWithLogitsLoss 会负责数值稳定的 sigmoid+BCE。
        self.out_logits = _maybe_spectral_norm(
            nn.Conv2d(c1, 1, kernel_size=3, stride=1, padding=1),
            spectral_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(x)
        e1 = self.enc1(self.down1(e0))
        e2 = self.enc2(self.down2(e1))
        b = self.bottleneck(self.down3(e2))

        u2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        s2 = self.att2(e2, b)
        d2 = self.dec2(torch.cat([u2, s2], dim=1))

        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        s1 = self.att1(e1, d2)
        d1 = self.dec1(torch.cat([u1, s1], dim=1))

        u0 = F.interpolate(d1, size=e0.shape[-2:], mode="bilinear", align_corners=False)
        s0 = self.att0(e0, d1)
        d0 = self.dec0(torch.cat([u0, s0], dim=1))
        return self.out_logits(d0)


class Discriminator(nn.Module):
    """
    PIV_A_Esrgan 的时序 Attention U-Net 判别器包装类。

    保持类名 `Discriminator` 是为了兼容训练代码中的
    `self.piv_esrgan_discriminator = Discriminator(...)`。

    可选多尺度逻辑：
    - use_multiscale=False: 只用原尺度 U-Net，适合作为第一阶段稳定消融；
    - use_multiscale=True: 额外启用 2x 下采样 U-Net，模仿 A-ESRGAN 的 multi-scale D，
      一个分支看颗粒边缘/亮点，另一个分支看更大范围的颗粒密度和团簇纹理。
    """

    def __init__(
        self,
        inner_chanel: int = 3,
        base_channels: int = 32,
        use_multiscale: bool = False,
        spectral_norm: bool = True,
    ):
        super().__init__()
        self.use_multiscale = bool(use_multiscale)
        self.normal_scale = AttentionUNetDiscriminator(
            in_channels=inner_chanel,
            base_channels=base_channels,
            spectral_norm=spectral_norm,
        )
        self.downsampled_scale = None
        if self.use_multiscale:
            self.downsampled_scale = AttentionUNetDiscriminator(
                in_channels=inner_chanel,
                base_channels=base_channels,
                spectral_norm=spectral_norm,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normal_logits = self.normal_scale(x)
        if self.downsampled_scale is None:
            return normal_logits

        # 多尺度分支先看 2x 下采样后的图像对，再把输出插值回原尺度。
        # 这里返回 [B, 2, H, W]，BCEWithLogitsLoss 会把两个尺度的每个位置都作为判别反馈。
        x_down = F.avg_pool2d(x, kernel_size=2, stride=2, ceil_mode=False)
        down_logits = self.downsampled_scale(x_down)
        down_logits = F.interpolate(down_logits, size=normal_logits.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([normal_logits, down_logits], dim=1)

"""
模型 end
"""
