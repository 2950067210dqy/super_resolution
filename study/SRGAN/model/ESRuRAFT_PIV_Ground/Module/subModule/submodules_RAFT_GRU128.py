import torch
import torch.nn as nn
import torch.nn.functional as F

from study.SRGAN.model.ESRuRAFT_PIV_Ground.Module.subModule.submodules_RAFT_GRU256 import (
    BasicMotionEncoder,
    FlowHead,
    SepConvGRU,
)


class BasicUpdateBlock128(nn.Module):
    """
    RAFT128 使用的更新块。

    主体沿用 RAFT256 的 BasicUpdateBlock256，但 convex 上采样倍率从 8 改成 4，
    因此 mask 通道数从 8*8*9=576 改为 4*4*9=144。
    """

    def __init__(self, corr_levels=4, corr_radius=4, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock128, self).__init__()
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.encoder = BasicMotionEncoder(self.corr_levels, self.corr_radius)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + input_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        # RAFT128 内部流场是 1/4 分辨率，convex 上采样需要 4x4 个子像素权重。
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4 * 4 * 9, 1, padding=0),
        )

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # 保持和 RAFT256 一致的 mask 梯度缩放策略。
        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow
