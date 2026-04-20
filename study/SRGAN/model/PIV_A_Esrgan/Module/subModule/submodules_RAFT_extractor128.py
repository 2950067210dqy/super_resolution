import torch
import torch.nn as nn
import torch.nn.functional as F

from study.SRGAN.model.PIV_A_Esrgan.Module.subModule.submodules_RAFT_extractor256 import ResidualBlock


class BasicEncoder128(nn.Module):
    """
    RAFT128 使用的基础编码器。

    和 RAFT256 的 BasicEncoder256 相比，这里只下采样到 1/4：
    - layer1 stride=2: H,W -> H/2,W/2
    - layer2 stride=2: H/2,W/2 -> H/4,W/4
    - layer3 stride=1: 保持 H/4,W/4

    这样在 256x256 输入下，RAFT 内部相关体和 GRU 工作在 64x64，
    介于 RAFT32 的全分辨率和 RAFT256 的 32x32 之间。
    """

    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(BasicEncoder128, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()
        else:
            raise ValueError(f"Unsupported norm_fn: {self.norm_fn}")

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.in_planes = 64

        # RAFT128 只做两次空间下采样，总倍率为 4。
        self.layer1 = self._make_layer(64, stride=2)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=1)
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        self.in_planes = dim
        return nn.Sequential(layer1, layer2)

    def forward(self, x):
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)
        return x
