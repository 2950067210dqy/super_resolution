import copy
import math
import os
import time
from os import mkdir
from os.path import exists
from pathlib import Path

import matplotlib
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from torchvision.models import vgg19
from tqdm import tqdm
from torch.nn import BatchNorm1d
from torchvision.utils import save_image
from d2l import torch as d2l   # 或 from d2l import mxnet as d2l
import wandb
from study.SRGAN.data_load import get_class_names, load_data
name = "v1"
device = torch.device("cuda")
#轮次
EPOCH_NUMS = 2
#批量大小
BATCH_SIZE = 32
#是否打乱训练集
SHUFFLE = True
#将数据的图像和光流统一到该尺寸 tuple[int, int]
TARGET_SIZE=None
#随机划分时的随机种子，保证结果复现
RANDOM_SEED = 42
#上采样系数 SCALE^2
SCALE = 2

#生成器感知损失里面的系数
LAMBDA_PERCEPTION =1e-3
#生成器正则损失的系数
LAMBDA_regularization_loss=2e-8
#生成器像素损失的系数
LAMBDA_loss_pixel =0.0001
#正则项
weight_decay=0.0001
#优化器 betas
g_optimizer_betas = (0.5,0.999)
d_optimizer_betas = (0.5,0.999)
#学习率
g_lr = 0.001
d_lr = 0.0001

#训练数据集和验证集合比例
Train_nums_rate=0.8
Validate_nums_rate=1-Train_nums_rate

# 使用wandb可视化训练过程
# 初始化 WandB
wandb.login(key="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")
# Start a new wandb run to track this script.
wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="2950067210-usst",
    # Set the wandb project where this run will be logged.
    project="srgnn",
    name=name,
    # Track hyperparameters and run metadata.
    config={
        "epochs": EPOCH_NUMS,
        "batch_size": BATCH_SIZE,
        "lr_G": g_lr,
        "lr_D": d_lr,
        "RANDOM_SEED":RANDOM_SEED,
        "SCALE":SCALE,
        "SHUFFLE":SHUFFLE,
        "LAMBDA_PERCEPTION":LAMBDA_PERCEPTION,
        "LAMBDA_regularization_loss":LAMBDA_regularization_loss,
        "LAMBDA_loss_pixel":LAMBDA_loss_pixel,
        "weight_decay":weight_decay,
        "g_optimizer_betas":g_optimizer_betas,
        "d_optimizer_betas":d_optimizer_betas,
        "Train_nums_rate":Train_nums_rate
    },
)
wandb.init(project="SRGAN",)
# 配置超参数
wandb.config = {

}

#真实数据根路径
GR_DATA_ROOT_DIR = rf"/study_datas/sr_dataset/class_1/data"
#低分辨率数据根地址
LR_DATA_ROOT_DIR = rf"/study_datas/sr_dataset/class_1_lr/x{SCALE*SCALE}/data"

#如果路径不存在则创建路径
out_put_dir = f"./train_data/{name}"
loss_dir = "/train_loss"
model_dir = "/train_model"
predict_dir = "/predict"
use_gpu = torch.cuda.is_available()
Path(out_put_dir).mkdir(parents=True, exist_ok=True)


"""
工具 start
"""
class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
def _in_notebook():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return False
        return ip.__class__.__name__ in ("ZMQInteractiveShell", "Shell")
    except Exception:
        return False


class Animator:
    """Docker/.py 可用：记录数据，最后导出 GIF"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None,
                 xscale="linear", yscale="linear",
                 fmts=("-", "m--", "g-.", "r:"), figsize=(6, 4)):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.legend = legend or []
        self.xlim = xlim
        self.ylim = ylim
        self.xscale = xscale
        self.yscale = yscale
        self.fmts = fmts

        self.X = None
        self.Y = None
        self.frames = []  # 每一帧保存一次快照

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)

        if not hasattr(x, "__len__"):
            x = [x] * n

        if self.X is None:
            self.X = [[] for _ in range(n)]
        if self.Y is None:
            self.Y = [[] for _ in range(n)]

        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)

        # 记录当前帧数据（深拷贝）
        self.frames.append(([row[:] for row in self.X], [row[:] for row in self.Y]))

    def _config_axes(self, current_y=None):
        self.ax.set_xlabel(self.xlabel if self.xlabel else "")
        self.ax.set_ylabel(self.ylabel if self.ylabel else "")
        if self.xlim is not None:
            self.ax.set_xlim(self.xlim)

        if self.ylim is not None:
            ymin, ymax = self.ylim[0],self.ylim[1]
            if current_y:
                data_max = max(
                    max(series) for series in current_y if series
                )
                ymax = max(ymax, data_max)  # 数据更大时，自动扩展上限
            self.ax.set_ylim((ymin, ymax))

        self.ax.set_xscale(self.xscale)
        self.ax.set_yscale(self.yscale)
        if self.legend:
            self.ax.legend(self.legend, loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=9)

    def _draw_frame(self,frame_idx):
        self.ax.cla()
        Xf, Yf = self.frames[frame_idx]
        for i, (xx, yy) in enumerate(zip(Xf, Yf)):
            fmt = self.fmts[i % len(self.fmts)]
            self.ax.plot(xx, yy, fmt)
        self._config_axes(current_y=Yf)
    def save(self, gif_path="train.gif", png_path="train.png", fps=20):
        if not self.frames:
            raise ValueError("没有可保存的帧，请先调用 add().")



        def update(frame_idx):
            self._draw_frame(frame_idx)
            return self.ax.lines

        # 1) 保存 GIF
        ani = FuncAnimation(
            self.fig, update, frames=len(self.frames), interval=1000 / fps, blit=False
        )
        ani.save(gif_path, writer=PillowWriter(fps=fps))



        plt.close(self.fig)
    def save_png(self, png_path="train.png"):
        # 2) 保存 PNG（最后一帧）
        self._draw_frame(len(self.frames) - 1)
        self.fig.savefig(png_path, dpi=200, bbox_inches="tight")
def validate_and_save(result_dir, generator, val_dataloader, device, epoch):
    generator.eval()  # 设置生成器为评估模式
    with torch.no_grad():
        for batch_idx, (lr_images, hr_images) in enumerate(val_dataloader):
            lr_images, hr_images = lr_images.to(device), hr_images.to(device)

            # 生成超分辨率图像
            fake_images = generator(lr_images)

            # 使用双三次插值将低分辨率图像调整到与高分辨率图像相同的大小
            resized_lr_images = F.interpolate(
                lr_images,
                size=hr_images.shape[2:],  # 调整到高分辨率图像的大小
                mode='bicubic',           # 使用双三次插值
                align_corners=False       # 推荐设置为 False，避免插值误差
            )

            # 按行拼接一批图像
            batch_combined = []
            for i in range(lr_images.size(0)):  # 遍历 batch 中的每张图片
                # 获取单张图像
                single_lr = resized_lr_images[i].unsqueeze(0)  # 调整后的低分辨率图像
                single_fake = fake_images[i].unsqueeze(0)      # 生成的高分辨率图像
                single_hr = hr_images[i].unsqueeze(0)          # 高分辨率图像

                # 按列拼接单张图片的三种结果
                combined = torch.cat([single_lr, single_fake, single_hr], dim=3)  # 按列（宽度方向）拼接
                batch_combined.append(combined)

            # 将一批图片按行拼接
            batch_combined = torch.cat(batch_combined, dim=2)  # 按高度方向（行）拼接整个 batch

            # 保存结果
            save_path = os.path.join(result_dir, f"epoch_{epoch+1}_batch_{batch_idx}_results.png")
            save_image(batch_combined, save_path)
            print(f"Saved validation image: {save_path}")

            break  # 只保存一个 batch 的结果

"""
工具 end
"""

"""
模型 start
"""
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
    def __init__(self,inner_chanel=3, num_residual_blocks=16, scale=2):
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
            nn.Conv2d(inner_chanel, 64, kernel_size=9, stride=1, padding=4),
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
        self.conv_out = nn.Conv2d(64, inner_chanel, kernel_size=9, stride=1, padding=4)

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
    def __init__(self,inner_chanel=3):
        super(Discriminator, self).__init__()

        # 第一层通常不加 BN，这是 GAN 里较常见的写法
        # [B, 3, 256, 256] -> [B, 64, 256, 256]
        self.conv1 = nn.Sequential(
            nn.Conv2d(inner_chanel, 64, kernel_size=3, stride=1, padding=1),
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

"""
模型 end
"""

"""
损失函数 start
"""
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

class ContentLoss(nn.Module):
    """

    内容损失=vgg-19模块提取特征的损失
    """
    def __init__(self, vgg):
        super(ContentLoss,self).__init__()
        self.vgg = vgg
        #L2损失
        self.criterion = nn.MSELoss()

    def forward(self, fake, real):
        # 使用 VGG 提取高层特征
        #通道为3则默认 通道为6则拆开
        channels = fake.shape[1]
        if channels ==6:
            real1, real2 = real[:, :3, :, :], real[:, 3:, :, :]
            fake1, fake2 = fake[:, :3, :, :], fake[:, 3:, :, :]
            feat_real1 = vgg(real1)
            feat_fake1 = vgg(fake1).detach()
            feat_real2 = vgg(real2)
            feat_fake2 = vgg(fake2).detach()

            return (self.criterion( feat_real1, feat_fake1) + self.criterion(feat_real2, feat_fake2))/2

        else:
            fake_features = self.vgg(fake).detach()
            real_features = self.vgg(real)
            return self.criterion(fake_features, real_features)
class AdversarialLoss(nn.Module):
    """
    对抗损失
    """
    def __init__(self):
        super(AdversarialLoss,self).__init__()

    def forward(self, x):
        loss = torch.sum(-torch.log(x))
        return loss
class PerceptualLoss(nn.Module):
    """
    感知损失 = 内容损失+1e-3 * 对抗损失
    也可以加正则化损失和像素损失
    """
    def __init__(self, vgg):
        super(PerceptualLoss,self).__init__()
        self.vgg_loss = ContentLoss(vgg)
        self.adversarial = AdversarialLoss()

    def forward(self, fake, real, x):
        vgg_loss = self.vgg_loss(fake, real)
        adversarial_loss = self.adversarial(x)
        return vgg_loss +LAMBDA_PERCEPTION*adversarial_loss
class RegularizationLoss(nn.Module):
    """
    正则化损失
    """
    def __init__(self):
        super(RegularizationLoss,self).__init__()

    def forward(self, x):
        a = torch.square(
            x[:, :, :x.shape[2]-1, :x.shape[3]-1] - x[:, :, 1:x.shape[2], :x.shape[3]-1]
        )
        b = torch.square(
            x[:, :, :x.shape[2]-1, :x.shape[3]-1] - x[:, :, :x.shape[2]-1, 1:x.shape[3]]
        )
        loss = torch.sum(torch.pow(a+b, 1.25))
        return loss

"""
损失函数 end
"""

# 定义像素损失函数
pixel_loss = nn.MSELoss()
# 这里vgg是针对三通道RGB图的
vgg = vgg19(pretrained=True).features[:16].eval()  # 提取 VGG 特征
# vgg模型预测模式
vgg = vgg.to(device).eval()

# 感知损失
perceptual_loss = PerceptualLoss(vgg=vgg)
# 判别器的损失函数
loss_d = nn.BCELoss()
# 归一化损失 正则化损失
regularization_loss = RegularizationLoss()
"""
验证函数 start
"""
# 计算 PSNR 函数
def calculate_psnr(fake_image, hr_image):
    mse = torch.mean((fake_image - hr_image) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(1.0 / math.sqrt(mse))
    return psnr

# 验证函数
def validate(generator, dataloader, device):
    #设置模型为评估模式
    generator.eval()
    val_loss = 0
    total_psnr = 0
    num_images = 0
    with torch.no_grad():
        for lr_images, hr_images in dataloader:
            lr_images, hr_images = lr_images.to(device), hr_images.to(device)
            fake_images = generator(lr_images)
            val_loss += pixel_loss(fake_images, hr_images).item()

            for fake_image, hr_image in zip(fake_images, hr_images):
                total_psnr += calculate_psnr(fake_image, hr_image)
                num_images += 1

    val_loss /= len(dataloader)
    avg_psnr = total_psnr / num_images
    return val_loss, avg_psnr

"""
验证函数 end
"""
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

    #获取类别名
    #获取数据 自动根据类别划分数据集并读取，每个类别都安装比例划分训练集和验证集
    available_class_names = get_class_names(GR_DATA_ROOT_DIR, LR_DATA_ROOT_DIR)
    loss_label = ['g_loss','g_perceptual_loss','g_regularization_loss','g_loss_pixel', 'd_loss', 'd_real_loss', 'd_fake_loss']
    validate_label = ['Validation_Loss', 'Avg_PSNR']
    print(f"一共{len(available_class_names)}个类别：{available_class_names}")
    #每个类别读取数据并且训练验证和保存模型
    for class_name in available_class_names:


        #根据类别读取数据
        train_loader, validate_loader, class_names, samples = load_data(
            gr_data_root_dir=GR_DATA_ROOT_DIR,
            lr_data_root_dir=LR_DATA_ROOT_DIR,
            batch_size=BATCH_SIZE,
            shuffle=SHUFFLE,
            target_size=TARGET_SIZE,
            train_nums_rate=Train_nums_rate,
            validate_nums_rate=Validate_nums_rate,
            random_seed=RANDOM_SEED,
            selected_classes=available_class_names[:1] if available_class_names else None,
        )
        # 每个类别的图像对和flo文件分别训练验证和保存模型
        for data_type in ['image_pair','flo']:
            # 创建文件夹
            Path(f"{out_put_dir}/{class_name}/{data_type}/scale_{SCALE * SCALE}/{loss_dir}").mkdir(parents=True, exist_ok=True)
            Path(f"{out_put_dir}/{class_name}/{data_type}/scale_{SCALE * SCALE}/{model_dir}").mkdir(parents=True, exist_ok=True)
            Path(f"{out_put_dir}/{class_name}/{data_type}/scale_{SCALE * SCALE}/{predict_dir}").mkdir(parents=True, exist_ok=True)

            animator = Animator(xlabel='ste', xlim=[1, EPOCH_NUMS], ylim=[0, 2],
                                legend=loss_label + validate_label)
            # 实例化generator
            generator = Generator(inner_chanel=3 if data_type=='flo' else 6).to(device)
            # 实例化Discriminator
            discriminator = Discriminator(inner_chanel=3 if data_type=='flo' else 6).to(device)
            # 加载预训练模型
            generator_save_path = f"{out_put_dir}/{class_name}/{data_type}/scale_{SCALE * SCALE}/{model_dir}/discriminator_{name}.pth"
            if os.path.exists(generator_save_path):
                generator.load_state_dict(torch.load(generator_save_path, map_location=device))
                print(f"Loaded pretrained model generator from {generator_save_path},{discriminator_save_path}")
            else:
                print("No pretrained model generator found. Starting training from scratch.")
            discriminator_save_path = f"{out_put_dir}/{class_name}/{data_type}/scale_{SCALE * SCALE}/{model_dir}/generator_{name}.pth"
            if os.path.exists(discriminator_save_path):
                discriminator.load_state_dict(torch.load(discriminator_save_path, map_location=device))
                print(f"Loaded pretrained model discriminator from {generator_save_path},{discriminator_save_path}")
            else:
                print("No pretrained model discriminator found. Starting training from scratch.")
            # 优化器
            g_optimizer = torch.optim.Adam(generator.parameters(), lr=g_lr, betas=g_optimizer_betas,
                                           weight_decay=weight_decay)
            d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=d_lr, betas=d_optimizer_betas,
                                           weight_decay=weight_decay)

            start_time = time.time()
            #轮数
            for epoch in range(EPOCH_NUMS):
                generator.train()# 确保生成器在训练模式
                discriminator.train()# 确保判别器在训练模式
                # 算每轮epoch的总体loss
                metric = Accumulator(len(loss_label))
                # 拿batch——size数据
                train_progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{EPOCH_NUMS}] {class_name} {data_type} Training", unit="batch")
                for i,batch in enumerate(train_progress_bar):
                    # 低分辨率图像
                    lr_images = batch[data_type]['lr_data'].to(device)
                    # 真实图像
                    gr_images=batch[data_type]['gr_data'].to(device)

                    real_labels = torch.ones((len(lr_images), 1, 1, 1)).to(device)
                    fake_labels = torch.zeros((len(lr_images), 1, 1, 1)).to(device)

                    # 生成器生成图像
                    pred_images = generator(lr_images)
                    #判别器判别图像 pred_images.detach()是因为这时候pred_image不需要计算梯度，所以让它从计算图中分离出来
                    probability = discriminator(pred_images.detach())

                    """生成器训练 start"""
                    # 感知损失
                    perceptual_loss_value = perceptual_loss(pred_images, gr_images, probability)
                    # 像素损失
                    g_loss_pixel = pixel_loss(pred_images, gr_images)
                    #正则损失
                    regularization_loss_value= regularization_loss(pred_images)
                    #生成器总损失
                    g_loss = perceptual_loss_value + LAMBDA_regularization_loss *regularization_loss_value +LAMBDA_loss_pixel*g_loss_pixel  # 这里的percuptual_loss包含了vgg_loss和对抗损失
                    # 优化生成器
                    g_optimizer.zero_grad()
                    g_loss.backward()
                    g_optimizer.step()
                    """生成器训练 end"""


                    #因为判别器太强了，让它弱一点，每两次训练一次
                    if i % 2 == 0:
                        """判别器训练 start"""
                        # 判别器判别真实图片之后将概率结果放入损失函数并且优化生成器   pred_images.detach()是因为这时候pred_image不需要计算梯度，所以让它从计算图中分离出来
                        real_loss = loss_d(discriminator(gr_images), copy.deepcopy(real_labels))
                        fake_loss = loss_d(discriminator(pred_images.detach()), copy.deepcopy(fake_labels))
                        d_loss = (real_loss + fake_loss)
                        # 优化判别器
                        d_optimizer.zero_grad()
                        d_loss.backward()
                        d_optimizer.step()
                        """判别器训练 end"""
                        # 在进度条上显示损失
                        train_progress_bar.set_postfix({
                            "class":class_name,
                            "D Loss": d_loss.item(),
                            "G Loss": g_loss.item()
                        })

                        # 需要和loss_label对应
                        metric.add(g_loss.item(),perceptual_loss_value.item(),regularization_loss_value.item(),g_loss_pixel.item() ,d_loss.item(), real_loss.item(), fake_loss.item())
                    if i % 400 == 0:
                        image = pred_images.detach()
                        # save_image中的normalize设置成True，目的是将像素值min-max自动归一到【0,1】范围内，如果已经预测了【0,1】之间，则可以不用设置True
                        save_dir = f"{out_put_dir}/{class_name}/{data_type}/scale_{SCALE * SCALE}"
                        os.makedirs(save_dir, exist_ok=True)

                        save_prefix = f"{save_dir}/image_{len(train_loader) * epoch + i}_{name}"

                        if image.dim() == 3:
                            image = image.unsqueeze(0)  # [C,H,W] -> [1,C,H,W]

                        channels = image.shape[1]

                        # save_image 中 normalize=True 会把像素值自动归一化到 [0, 1]
                        if channels == 3:
                            torchvision.utils.save_image(
                                image,
                                f"{save_prefix}.png",
                                nrow=4,
                                normalize=True
                            )
                        elif channels == 6:
                            torchvision.utils.save_image(
                                image[:, :3, :, :],
                                f"{save_prefix}_img1.png",
                                nrow=4,
                                normalize=True
                            )
                            torchvision.utils.save_image(
                                image[:, 3:, :, :],
                                f"{save_prefix}_img2.png",
                                nrow=4,
                                normalize=True
                            )
                        else:
                            raise ValueError(f"Unsupported image channels: {channels}")
                # 每轮训练结束后进行验证
                val_loss, avg_psnr = validate(generator, validate_loader, device)
                wandb.log({"classname":class_name,"data_type":data_type,"Validation Loss": val_loss, "avg_psnr":avg_psnr,"Epoch": epoch}+{loss_label[index] : str(metric[index] / len(train_loader)) + "," for index in
                                    range(len(loss_label))})
                current_time = time.time()
                print(
                    f"Epoch [{epoch + 1}/{EPOCH_NUMS}] |{class_name} {data_type} |running time:{int(current_time - start_time)}s | "
                    f"Val Loss: {val_loss:.4f} | Avg PSNR: {avg_psnr:.2f}",end=""
                )
                loss_str = "".join([loss_label[index] + ':' + str(metric[index] / len(train_loader)) + "," for index in
                                    range(len(loss_label))])
                print(loss_str)

                # 每轮训练结束后进行验证，并保存最后一批图像
                validate_and_save(f"{out_put_dir}/{class_name}/{data_type}/scale_{SCALE*SCALE}/{predict_dir}", generator, validate_loader, device, epoch)
                # 保存模型
                generator_save_path=f"{out_put_dir}/{class_name}/{data_type}/scale_{SCALE*SCALE}/{model_dir}/discriminator_{name}.pth"
                discriminator_save_path=f"{out_put_dir}/{class_name}/{data_type}/scale_{SCALE*SCALE}/{model_dir}/generator_{name}.pth"
                torch.save(discriminator.state_dict(),generator_save_path )
                torch.save(generator.state_dict(), discriminator_save_path)
                print(f"{class_name} {data_type} |Models saved: Generator -> {generator_save_path}, Discriminator -> {discriminator_save_path}")

                #保存每一epoch的损失
                animator.add(epoch + 1, [metric[index] / len(train_loader) for index in range(len(loss_label))]+[val_loss,avg_psnr])
                animator.save_png(f"{out_put_dir}/{class_name}/{data_type}/scale_{SCALE*SCALE}/{loss_dir}/train_loss_epoch_{epoch + 1}_{name}.png")

    # # 测试生成器
    # lr = torch.randn(2, 3, 64, 64)
    #
    # # 4x 超分: 64x64 -> 256x256
    # G = Generator(scale=2)
    # sr = G(lr)
    # print("Generator output shape:", sr.shape)
    #
    # # 判别器输入应该和 HR / SR 图像尺寸一致
    # D = Discriminator()
    # out = D(sr)
    # print("Discriminator output shape:", out.shape)
    pass