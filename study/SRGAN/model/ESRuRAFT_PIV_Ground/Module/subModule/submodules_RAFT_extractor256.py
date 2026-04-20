'''
Copyright (c) 2020-2021, Christian Lagemann
'''
'''
Portions of this code copyright 2020, princeton-vl 
In the framework of:
Teed, Zachary, and Jia Deng. "Raft: Recurrent all-pairs field transforms for optical flow." European Conference on Computer Vision. Springer, Cham, 2020.
URL: https://github.com/princeton-vl/RAFT
'''

import torch  # 导入 PyTorch 主库
import torch.nn as nn  # 导入神经网络模块，简写为 nn
import torch.nn.functional as F  # 导入函数式接口，简写为 F


class ResidualBlock(nn.Module):
    """
    标准残差块。

    作用：
        通过两层 3x3 卷积提取特征，并与输入通过 shortcut 分支相加。

    输入：
        x: [B, in_planes, H, W]

    输出：
        out: [B, planes, H/stride, W/stride]

    说明：
        1. 当 stride=1 时，通常只改变通道数，不改变空间尺寸
        2. 当 stride=2 时，会做下采样，空间尺寸减半
        3. shortcut 分支用于保证残差相加时尺寸一致
    """

    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        """
        初始化残差块。

        参数：
            in_planes: 输入通道数
            planes: 输出通道数
            norm_fn: 归一化方式，可选 'group' / 'batch' / 'instance' / 'none'
            stride: 第一个卷积层步长
        """
        super(ResidualBlock, self).__init__()  # 调用父类初始化

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        # 第一层 3x3 卷积
        # 输入:  [B, in_planes, H, W]
        # 输出:  [B, planes, H/stride, W/stride]

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        # 第二层 3x3 卷积
        # 输入:  [B, planes, H/stride, W/stride]
        # 输出:  [B, planes, H/stride, W/stride]

        self.relu = nn.ReLU(inplace=True)
        # ReLU 激活函数，不改变张量尺寸

        num_groups = planes // 8
        # GroupNorm 的组数，一般每 8 个通道分一组

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            # conv1 后做 GroupNorm，尺寸不变

            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            # conv2 后做 GroupNorm，尺寸不变

            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
                # shortcut 分支可能使用的归一化层

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            # 空层，占位，不做任何处理

            self.norm2 = nn.Sequential()
            # 空层，占位

            if not stride == 1:
                self.norm3 = nn.Sequential()
                # shortcut 分支空层占位

        self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride))
        # shortcut 分支的 1x1 卷积
        # 输入:  [B, in_planes, H, W]
        # 输出:  [B, planes, H/stride, W/stride]
        # 用于匹配主分支输出的通道数和空间尺寸

    def forward(self, x):
        """
        前向传播。

        参数：
            x: 输入特征图 [B, in_planes, H, W]

        返回：
            输出特征图 [B, planes, H/stride, W/stride]
        """
        y = x
        # 保存输入作为主分支起点
        # y: [B, in_planes, H, W]

        y = self.relu(self.norm1(self.conv1(y)))
        # conv1:
        # [B, in_planes, H, W] -> [B, planes, H/stride, W/stride]
        # norm1 / relu 后尺寸不变

        y = self.relu(self.norm2(self.conv2(y)))
        # conv2:
        # [B, planes, H/stride, W/stride] -> [B, planes, H/stride, W/stride]
        # norm2 / relu 后尺寸不变

        if self.downsample is not None:
            x = self.downsample(x)
            # shortcut 分支调整输入尺寸
            # [B, in_planes, H, W] -> [B, planes, H/stride, W/stride]

        return self.relu(x+y)
        # 残差相加
        # x 与 y 尺寸相同: [B, planes, H/stride, W/stride]
        # 最终输出: [B, planes, H/stride, W/stride]



class BottleneckBlock(nn.Module):
    """
    瓶颈残差块。

    作用：
        使用 1x1 -> 3x3 -> 1x1 的卷积结构，
        先降维、再提取空间特征、再升维，以减少参数量和计算量。

    输入：
        x: [B, in_planes, H, W]

    输出：
        out: [B, planes, H/stride, W/stride]
    """

    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        """
        初始化瓶颈残差块。

        参数：
            in_planes: 输入通道数
            planes: 输出通道数
            norm_fn: 归一化方式
            stride: 3x3 卷积的步长
        """
        super(BottleneckBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes//4, kernel_size=1, padding=0)
        # 第一个 1x1 卷积，先降维
        # 输入:  [B, in_planes, H, W]
        # 输出:  [B, planes//4, H, W]

        self.conv2 = nn.Conv2d(planes//4, planes//4, kernel_size=3, padding=1, stride=stride)
        # 第二个 3x3 卷积，提取空间特征
        # 输入:  [B, planes//4, H, W]
        # 输出:  [B, planes//4, H/stride, W/stride]

        self.conv3 = nn.Conv2d(planes//4, planes, kernel_size=1, padding=0)
        # 第三个 1x1 卷积，升维恢复通道
        # 输入:  [B, planes//4, H/stride, W/stride]
        # 输出:  [B, planes, H/stride, W/stride]

        self.relu = nn.ReLU(inplace=True)
        # ReLU 激活函数

        num_groups = planes // 8
        # GroupNorm 的组数

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            # conv1 后归一化

            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            # conv2 后归一化

            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            # conv3 后归一化

            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
                # shortcut 分支归一化

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes//4)
            self.norm2 = nn.BatchNorm2d(planes//4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes//4)
            self.norm2 = nn.InstanceNorm2d(planes//4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample = None
            # stride=1 时，不做 shortcut 下采样

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)
            # shortcut 分支
            # 输入:  [B, in_planes, H, W]
            # 输出:  [B, planes, H/stride, W/stride]

    def forward(self, x):
        """
        前向传播。

        参数：
            x: 输入特征图 [B, in_planes, H, W]

        返回：
            输出特征图 [B, planes, H/stride, W/stride]
        """
        y = x
        # y: [B, in_planes, H, W]

        y = self.relu(self.norm1(self.conv1(y)))
        # conv1:
        # [B, in_planes, H, W] -> [B, planes//4, H, W]

        y = self.relu(self.norm2(self.conv2(y)))
        # conv2:
        # [B, planes//4, H, W] -> [B, planes//4, H/stride, W/stride]

        y = self.relu(self.norm3(self.conv3(y)))
        # conv3:
        # [B, planes//4, H/stride, W/stride] -> [B, planes, H/stride, W/stride]

        if self.downsample is not None:
            x = self.downsample(x)
            # shortcut:
            # [B, in_planes, H, W] -> [B, planes, H/stride, W/stride]

        return self.relu(x+y)
        # 残差相加
        # 输出: [B, planes, H/stride, W/stride]


class BasicEncoder256(nn.Module):
    """
    基础编码器 256 版。

    作用：
        对单通道输入图像提取特征，逐层下采样，最终输出较深层特征图。

    输入：
        单输入时:
            x: [B, 1, H, W]
        列表输入时:
            [x1, x2]，每个都是 [B, 1, H, W]

    输出：
        单输入时:
            [B, output_dim, H/8, W/8]
        列表输入时:
            两个张量，每个 [B, output_dim, H/8, W/8]

    说明：
        由于 layer1/layer2/layer3 的 stride 都是 2，
        整体总下采样倍率为 8。
    """

    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        """
        初始化编码器。

        参数：
            output_dim: 最终输出通道数
            norm_fn: 归一化方式
            dropout: dropout 概率
        """
        super(BasicEncoder256, self).__init__()
        self.norm_fn = norm_fn
        # 保存归一化方式

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            # 第一层卷积后的 GroupNorm

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
            # 第一层卷积后的 BatchNorm

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)
            # 第一层卷积后的 InstanceNorm

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()
            # 不使用归一化时的占位空层

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3)
        # 第一层卷积
        # 输入:  [B, 1, H, W]
        # 输出:  [B, 64, H, W]
        # 这里 stride=1，所以空间尺寸不变

        self.relu1 = nn.ReLU(inplace=True)
        # ReLU 激活

        self.in_planes = 64
        # 当前输入通道数记录为 64

        self.layer1 = self._make_layer(64,  stride=2)
        # layer1:
        # 输入:  [B, 64, H, W]
        # 输出:  [B, 64, H/2, W/2]

        self.layer2 = self._make_layer(96, stride=2)
        # layer2:
        # 输入:  [B, 64, H/2, W/2]
        # 输出:  [B, 96, H/4, W/4]

        self.layer3 = self._make_layer(128, stride=2)
        # layer3:
        # 输入:  [B, 96, H/4, W/4]
        # 输出:  [B, 128, H/8, W/8]

        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)
        # 输出卷积层
        # 输入:  [B, 128, H/8, W/8]
        # 输出:  [B, output_dim, H/8, W/8]

        self.dropout = None
        # 默认不使用 dropout
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
            # Dropout2d 不改变尺寸

        for m in self.modules():
            # 遍历所有子模块并初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # 卷积层使用 Kaiming 初始化

            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                # 归一化层初始化
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                    # 权重初始化为 1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    # 偏置初始化为 0

    def _make_layer(self, dim, stride=1):
        """
        构建一个由两个 ResidualBlock 组成的层。

        参数：
            dim: 当前层输出通道数
            stride: 第一个残差块的步长

        返回：
            一个顺序容器 nn.Sequential
        """
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        # 第一个残差块
        # 输入:  [B, self.in_planes, H, W]
        # 输出:  [B, dim, H/stride, W/stride]

        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        # 第二个残差块
        # 输入:  [B, dim, H/stride, W/stride]
        # 输出:  [B, dim, H/stride, W/stride]

        layers = (layer1, layer2)
        # 组成一个两层残差结构

        self.in_planes = dim
        # 更新当前通道数，供下一层使用

        return nn.Sequential(*layers)
        # 返回顺序容器

    def forward(self, x):
        """
        前向传播。

        参数：
            x:
                1. 单个张量 [B, 1, H, W]
                2. list/tuple，包含两个 [B, 1, H, W]

        返回：
            1. 单输入时返回 [B, output_dim, H/8, W/8]
            2. 列表输入时返回两个 [B, output_dim, H/8, W/8]
        """

        is_list = isinstance(x, tuple) or isinstance(x, list)
        # 判断输入是不是 list 或 tuple

        if is_list:
            batch_dim = x[0].shape[0]
            # 记录单个输入的 batch 大小 B

            x = torch.cat(x, dim=0)
            # 例如两个输入分别是 [B,1,H,W]
            # 拼接后变成 [2B,1,H,W]

        x = self.conv1(x)
        # [B,1,H,W] -> [B,64,H,W]
        # 或 [2B,1,H,W] -> [2B,64,H,W]

        x = self.norm1(x)
        # 尺寸不变

        x = self.relu1(x)
        # 尺寸不变

        x = self.layer1(x)
        # -> [B,64,H/2,W/2]
        # 或 [2B,64,H/2,W/2]

        x = self.layer2(x)
        # -> [B,96,H/4,W/4]
        # 或 [2B,96,H/4,W/4]

        x = self.layer3(x)
        # -> [B,128,H/8,W/8]
        # 或 [2B,128,H/8,W/8]

        x = self.conv2(x)
        # -> [B,output_dim,H/8,W/8]
        # 或 [2B,output_dim,H/8,W/8]

        if self.training and self.dropout is not None:
            x = self.dropout(x)
            # 训练时才启用 dropout，尺寸不变

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)
            # [2B,output_dim,H/8,W/8]
            # 拆分成两个 [B,output_dim,H/8,W/8]

        return x
        # 返回编码结果


class SmallEncoder(nn.Module):
    """
    轻量编码器。

    作用：
        使用 BottleneckBlock 提取更轻量的图像特征。

    输入：
        单输入时:
            x: [B, 3, H, W]
        列表输入时:
            [x1, x2]，每个都是 [B, 3, H, W]

    输出：
        单输入时:
            [B, output_dim, H/8, W/8]
        列表输入时:
            两个张量，每个 [B, output_dim, H/8, W/8]

    说明：
        conv1、layer2、layer3 都进行了下采样，
        所以总下采样倍率是 8。
    """

    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        """
        初始化轻量编码器。

        参数：
            output_dim: 输出通道数
            norm_fn: 归一化方式
            dropout: dropout 概率
        """
        super(SmallEncoder, self).__init__()
        self.norm_fn = norm_fn
        # 保存归一化方式

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)
            # 第一层卷积后的 GroupNorm

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(32)
            # 第一层卷积后的 BatchNorm

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(32)
            # 第一层卷积后的 InstanceNorm

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()
            # 空层占位

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        # 第一层卷积
        # 输入:  [B, 3, H, W]
        # 输出:  [B, 32, H/2, W/2]

        self.relu1 = nn.ReLU(inplace=True)
        # ReLU 激活

        self.in_planes = 32
        # 当前输入通道数记录为 32

        self.layer1 = self._make_layer(32,  stride=1)
        # layer1:
        # 输入:  [B,32,H/2,W/2]
        # 输出:  [B,32,H/2,W/2]

        self.layer2 = self._make_layer(64, stride=2)
        # layer2:
        # 输入:  [B,32,H/2,W/2]
        # 输出:  [B,64,H/4,W/4]

        self.layer3 = self._make_layer(96, stride=2)
        # layer3:
        # 输入:  [B,64,H/4,W/4]
        # 输出:  [B,96,H/8,W/8]

        self.dropout = None
        # 默认不使用 dropout
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
            # Dropout2d 不改变尺寸

        self.conv2 = nn.Conv2d(96, output_dim, kernel_size=1)
        # 输出卷积
        # 输入:  [B,96,H/8,W/8]
        # 输出:  [B,output_dim,H/8,W/8]

        for m in self.modules():
            # 遍历所有子模块做初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # 卷积层 Kaiming 初始化

            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                # 归一化层初始化
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        """
        构建一个由两个 BottleneckBlock 组成的层。

        参数：
            dim: 当前层输出通道数
            stride: 第一个瓶颈块的步长

        返回：
            一个顺序容器 nn.Sequential
        """
        layer1 = BottleneckBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        # 第一个瓶颈块
        # 输入:  [B, self.in_planes, H, W]
        # 输出:  [B, dim, H/stride, W/stride]

        layer2 = BottleneckBlock(dim, dim, self.norm_fn, stride=1)
        # 第二个瓶颈块
        # 输入:  [B, dim, H/stride, W/stride]
        # 输出:  [B, dim, H/stride, W/stride]

        layers = (layer1, layer2)
        # 组成两层瓶颈结构

        self.in_planes = dim
        # 更新当前通道数

        return nn.Sequential(*layers)
        # 返回顺序容器

    def forward(self, x):
        """
        前向传播。

        参数：
            x:
                1. 单个张量 [B, 3, H, W]
                2. list/tuple，包含两个 [B, 3, H, W]

        返回：
            1. 单输入时返回 [B, output_dim, H/8, W/8]
            2. 列表输入时返回两个 [B, output_dim, H/8, W/8]
        """

        is_list = isinstance(x, tuple) or isinstance(x, list)
        # 判断输入是否为 list/tuple

        if is_list:
            batch_dim = x[0].shape[0]
            # 记录单个输入的 batch 大小 B

            x = torch.cat(x, dim=0)
            # 两个 [B,3,H,W] 拼接后变成 [2B,3,H,W]

        x = self.conv1(x)
        # -> [B,32,H/2,W/2]
        # 或 [2B,32,H/2,W/2]

        x = self.norm1(x)
        # 尺寸不变

        x = self.relu1(x)
        # 尺寸不变

        x = self.layer1(x)
        # -> [B,32,H/2,W/2]
        # 或 [2B,32,H/2,W/2]

        x = self.layer2(x)
        # -> [B,64,H/4,W/4]
        # 或 [2B,64,H/4,W/4]

        x = self.layer3(x)
        # -> [B,96,H/8,W/8]
        # 或 [2B,96,H/8,W/8]

        x = self.conv2(x)
        # -> [B,output_dim,H/8,W/8]
        # 或 [2B,output_dim,H/8,W/8]

        if self.training and self.dropout is not None:
            x = self.dropout(x)
            # 训练时启用 dropout，尺寸不变

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)
            # [2B,output_dim,H/8,W/8]
            # 拆成两个 [B,output_dim,H/8,W/8]

        return x
        # 返回最终输出
