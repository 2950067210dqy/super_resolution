import copy
import time
from os import mkdir
from os.path import exists

import matplotlib

import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from torch.nn import BatchNorm1d
from torchvision.utils import save_image
from d2l import torch as d2l   # 或 from d2l import mxnet as d2l
epoch_num = 50
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
    def __init__(self, num_residual_blocks=16, scale=2):
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
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
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
                nn.Conv2d(64, 64*self.scale*self.scale, kernel_size=3, stride=1, padding=1),

                # [B, 256, H, W] -> [B, 64, 2H, 2W]
                nn.PixelShuffle(self.scale),
                nn.PReLU(),

                nn.Conv2d(64, 64*self.scale*self.scale, kernel_size=3, stride=1, padding=1),
                # [B, 256, H, W] -> [B, 64, 2H, 2W]
                nn.PixelShuffle(self.scale),
                nn.PReLU()
        )

        # 最后一层输出 RGB 图像
        # 64 -> 3
        # kernel_size=9, padding=4 保持尺寸不变
        #
        # 若 scale=4:
        # [B, 64, 256, 256] -> [B, 3, 256, 256]
        self.conv_out = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

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
    def __init__(self):
        super(Discriminator, self).__init__()

        # 第一层通常不加 BN，这是 GAN 里较常见的写法
        # [B, 3, 256, 256] -> [B, 64, 256, 256]
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
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


def train():
    """
    训练
    :return:
    """
    # 1.拿上数据
    pass
def evaluate():
    """
    测试 评价
    :return:
    """
    pass
if __name__ =="__main__":
    # 测试生成器
    lr = torch.randn(2, 3, 64, 64)

    # 4x 超分: 64x64 -> 256x256
    G = Generator(scale=2)
    sr = G(lr)
    print("Generator output shape:", sr.shape)

    # 判别器输入应该和 HR / SR 图像尺寸一致
    D = Discriminator()
    out = D(sr)
    print("Discriminator output shape:", out.shape)
    pass