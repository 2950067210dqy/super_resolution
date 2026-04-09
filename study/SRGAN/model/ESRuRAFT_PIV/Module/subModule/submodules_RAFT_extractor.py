import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    标准残差块。

    作用：
        通过两层卷积提取特征，并将输入通过 shortcut 分支与主分支结果相加。

    输入：
        x: [B, in_planes, H, W]

    输出：
        out: [B, planes, H/stride, W/stride]

    说明：
        1. 当 stride=1 时，空间尺寸通常不变
        2. 当 stride!=1 时，空间尺寸会缩小
        3. shortcut 分支用于匹配通道数和尺寸
    """

    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        """
        初始化残差块。

        参数：
            in_planes: 输入通道数
            planes: 输出通道数
            norm_fn: 归一化方式，可选 'group' / 'batch' / 'instance' / 'none'
            stride: 卷积步长
        """
        super(ResidualBlock, self).__init__()  # 调用父类初始化

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        # 第一个 3x3 卷积
        # [B, in_planes, H, W] -> [B, planes, H/stride, W/stride]

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        # 第二个 3x3 卷积
        # [B, planes, H/stride, W/stride] -> [B, planes, H/stride, W/stride]

        self.relu = nn.ReLU(inplace=True)
        # ReLU 激活函数，尺寸不变

        num_groups = planes // 8
        # GroupNorm 的分组数

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            # conv1 后归一化，尺寸不变

            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            # conv2 后归一化，尺寸不变

            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
                # shortcut 分支归一化，尺寸不变

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
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        self.downsample = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride)
        )
        # shortcut 分支的 1x1 卷积
        # [B, in_planes, H, W] -> [B, planes, H/stride, W/stride]

    def forward(self, x):
        """
        前向传播。

        参数：
            x: 输入特征图，尺寸为 [B, in_planes, H, W]

        返回：
            输出特征图，尺寸为 [B, planes, H/stride, W/stride]
        """
        y = x
        # 保存输入作为主分支起点
        # y: [B, in_planes, H, W]

        y = self.relu(self.norm1(self.conv1(y)))
        # conv1 -> norm1 -> relu
        # [B, in_planes, H, W] -> [B, planes, H/stride, W/stride]

        y = self.relu(self.norm2(self.conv2(y)))
        # conv2 -> norm2 -> relu
        # [B, planes, H/stride, W/stride] -> [B, planes, H/stride, W/stride]

        if self.downsample is not None:
            x = self.downsample(x)
            # shortcut 分支调整输入
            # [B, in_planes, H, W] -> [B, planes, H/stride, W/stride]

        return self.relu(x + y)
        # 残差相加后激活
        # 输出: [B, planes, H/stride, W/stride]


class BottleneckBlock(nn.Module):
    """
    瓶颈残差块。

    作用：
        使用 1x1 -> 3x3 -> 1x1 的结构先降维、再提特征、再升维，
        以减少参数量和计算量。

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
            stride: 步长
        """
        super(BottleneckBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes // 4, kernel_size=1, padding=0)
        # 1x1 卷积降维
        # [B, in_planes, H, W] -> [B, planes//4, H, W]

        self.conv2 = nn.Conv2d(planes // 4, planes // 4, kernel_size=3, padding=1, stride=stride)
        # 3x3 卷积提取空间特征
        # [B, planes//4, H, W] -> [B, planes//4, H/stride, W/stride]

        self.conv3 = nn.Conv2d(planes // 4, planes, kernel_size=1, padding=0)
        # 1x1 卷积升维
        # [B, planes//4, H/stride, W/stride] -> [B, planes, H/stride, W/stride]

        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes // 4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes // 4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes // 4)
            self.norm2 = nn.BatchNorm2d(planes // 4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes // 4)
            self.norm2 = nn.InstanceNorm2d(planes // 4)
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
            # 不下采样时不走 shortcut 调整
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
                self.norm4
            )
            # shortcut 分支
            # [B, in_planes, H, W] -> [B, planes, H/stride, W/stride]

    def forward(self, x):
        """
        前向传播。

        参数：
            x: 输入特征图 [B, in_planes, H, W]

        返回：
            输出特征图 [B, planes, H/stride, W/stride]
        """
        y = x
        # 主分支起点

        y = self.relu(self.norm1(self.conv1(y)))
        # [B, in_planes, H, W] -> [B, planes//4, H, W]

        y = self.relu(self.norm2(self.conv2(y)))
        # [B, planes//4, H, W] -> [B, planes//4, H/stride, W/stride]

        y = self.relu(self.norm3(self.conv3(y)))
        # [B, planes//4, H/stride, W/stride] -> [B, planes, H/stride, W/stride]

        if self.downsample is not None:
            x = self.downsample(x)
            # [B, in_planes, H, W] -> [B, planes, H/stride, W/stride]

        return self.relu(x + y)
        # 输出: [B, planes, H/stride, W/stride]


class BasicEncoder(nn.Module):
    """
    基础编码器。

    作用：
        对输入图像提取特征，常用于 RAFT 中的特征编码和上下文编码。

    输入：
        单输入时:
            x: [B, 1, H, W]
        列表输入时:
            [x1, x2]，每个都是 [B, 1, H, W]

    输出：
        单输入时:
            [B, output_dim, H, W]
        列表输入时:
            两个张量，每个 [B, output_dim, H, W]

    特点：
        整个网络基本不改变空间分辨率，只改变通道数。
    """

    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        """
        初始化编码器。

        参数：
            output_dim: 最终输出通道数
            norm_fn: 归一化方式
            dropout: dropout 概率
        """
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3)
        # 输入: [B, 1, H, W]
        # 输出: [B, 64, H, W]

        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64

        self.layer1 = self._make_layer(64, stride=1)
        # [B, 64, H, W] -> [B, 64, H, W]

        self.layer2 = self._make_layer(96, stride=1)
        # [B, 64, H, W] -> [B, 96, H, W]

        self.layer3 = self._make_layer(128, stride=1)
        # [B, 96, H, W] -> [B, 128, H, W]

        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)
        # [B, 128, H, W] -> [B, output_dim, H, W]

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
            # 尺寸不变

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        """
        构建一个由两个 ResidualBlock 组成的层。

        参数：
            dim: 该层输出通道数
            stride: 第一个残差块的步长

        返回：
            一个 nn.Sequential 模块
        """
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        # [B, self.in_planes, H, W] -> [B, dim, H/stride, W/stride]

        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        # [B, dim, H/stride, W/stride] -> [B, dim, H/stride, W/stride]

        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播。

        参数：
            x:
                1. 单个张量 [B, 1, H, W]
                2. list/tuple，包含两个 [B, 1, H, W]

        返回：
            1. 单输入时返回 [B, output_dim, H, W]
            2. 列表输入时返回两个 [B, output_dim, H, W]
        """
        is_list = isinstance(x, tuple) or isinstance(x, list)
        # 判断是不是双输入列表

        if is_list:
            batch_dim = x[0].shape[0]
            # 记录原始 batch 大小 B

            x = torch.cat(x, dim=0)
            # [B,1,H,W] + [B,1,H,W] -> [2B,1,H,W]

        x = self.conv1(x)
        # -> [B,64,H,W] 或 [2B,64,H,W]

        x = self.norm1(x)
        # 尺寸不变

        x = self.relu1(x)
        # 尺寸不变

        x = self.layer1(x)
        # -> [B,64,H,W] 或 [2B,64,H,W]

        x = self.layer2(x)
        # -> [B,96,H,W] 或 [2B,96,H,W]

        x = self.layer3(x)
        # -> [B,128,H,W] 或 [2B,128,H,W]

        x = self.conv2(x)
        # -> [B,output_dim,H,W] 或 [2B,output_dim,H,W]

        if self.training and self.dropout is not None:
            x = self.dropout(x)
            # 尺寸不变

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)
            # [2B,output_dim,H,W] -> 两个 [B,output_dim,H,W]

        return x


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

    特点：
        整体会进行 8 倍下采样。
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

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(32)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(32)
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        # [B, 3, H, W] -> [B, 32, H/2, W/2]

        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 32

        self.layer1 = self._make_layer(32, stride=1)
        # [B,32,H/2,W/2] -> [B,32,H/2,W/2]

        self.layer2 = self._make_layer(64, stride=2)
        # [B,32,H/2,W/2] -> [B,64,H/4,W/4]

        self.layer3 = self._make_layer(96, stride=2)
        # [B,64,H/4,W/4] -> [B,96,H/8,W/8]

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        self.conv2 = nn.Conv2d(96, output_dim, kernel_size=1)
        # [B,96,H/8,W/8] -> [B,output_dim,H/8,W/8]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        """
        构建一个由两个 BottleneckBlock 组成的层。

        参数：
            dim: 输出通道数
            stride: 第一个 bottleneck 的步长

        返回：
            一个顺序模块
        """
        layer1 = BottleneckBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = BottleneckBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

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

        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
            # [B,3,H,W] + [B,3,H,W] -> [2B,3,H,W]

        x = self.conv1(x)
        # -> [B,32,H/2,W/2] 或 [2B,32,H/2,W/2]

        x = self.norm1(x)
        # 尺寸不变

        x = self.relu1(x)
        # 尺寸不变

        x = self.layer1(x)
        # -> [B,32,H/2,W/2] 或 [2B,32,H/2,W/2]

        x = self.layer2(x)
        # -> [B,64,H/4,W/4] 或 [2B,64,H/4,W/4]

        x = self.layer3(x)
        # -> [B,96,H/8,W/8] 或 [2B,96,H/8,W/8]

        x = self.conv2(x)
        # -> [B,output_dim,H/8,W/8] 或 [2B,output_dim,H/8,W/8]

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)
            # [2B,output_dim,H/8,W/8] -> 两个 [B,output_dim,H/8,W/8]

        return x
