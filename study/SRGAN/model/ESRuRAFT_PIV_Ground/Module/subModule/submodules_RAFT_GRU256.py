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
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入函数式接口


class FlowHead(nn.Module):
    """
    光流预测头。

    作用：
        将隐藏状态特征映射成二维光流增量 delta_flow。

    输入：
        x: [B, input_dim, H, W]

    输出：
        flow: [B, 2, H, W]

    说明：
        输出的两个通道通常表示光流在 x、y 两个方向上的位移增量。
    """

    def __init__(self, input_dim=128, hidden_dim=256):
        """
        初始化光流预测头。

        参数：
            input_dim: 输入特征通道数
            hidden_dim: 中间隐藏层通道数
        """
        super(FlowHead, self).__init__()  # 调用父类初始化

        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        # 第一个 3x3 卷积
        # 输入: [B, input_dim, H, W]
        # 输出: [B, hidden_dim, H, W]

        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        # 第二个 3x3 卷积
        # 输入: [B, hidden_dim, H, W]
        # 输出: [B, 2, H, W]

        self.relu = nn.ReLU(inplace=True)
        # ReLU 激活函数，尺寸不变

    def forward(self, x):
        """
        前向传播。

        参数：
            x: 输入特征图 [B, input_dim, H, W]

        返回：
            光流增量 [B, 2, H, W]
        """
        return self.conv2(self.relu(self.conv1(x)))
        # conv1:
        # [B, input_dim, H, W] -> [B, hidden_dim, H, W]
        # relu:
        # 尺寸不变
        # conv2:
        # [B, hidden_dim, H, W] -> [B, 2, H, W]


class ConvGRU(nn.Module):
    """
    卷积版 GRU。

    作用：
        使用卷积操作更新隐藏状态，同时保留空间结构信息。

    输入：
        h: [B, hidden_dim, H, W]
        x: [B, input_dim, H, W]

    输出：
        h_new: [B, hidden_dim, H, W]
    """

    def __init__(self, hidden_dim=128, input_dim=192+128):
        """
        初始化 ConvGRU。

        参数：
            hidden_dim: 隐藏状态通道数
            input_dim: 输入特征通道数
        """
        super(ConvGRU, self).__init__()

        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        # 更新门 z
        # 输入: [B, hidden_dim + input_dim, H, W]
        # 输出: [B, hidden_dim, H, W]

        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        # 重置门 r
        # 输入: [B, hidden_dim + input_dim, H, W]
        # 输出: [B, hidden_dim, H, W]

        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        # 候选隐藏状态 q
        # 输入: [B, hidden_dim + input_dim, H, W]
        # 输出: [B, hidden_dim, H, W]

    def forward(self, h, x):
        """
        前向传播。

        参数：
            h: 旧隐藏状态 [B, hidden_dim, H, W]
            x: 当前输入特征 [B, input_dim, H, W]

        返回：
            更新后的隐藏状态 [B, hidden_dim, H, W]
        """
        hx = torch.cat([h, x], dim=1)
        # 拼接隐藏状态和输入特征
        # [B, hidden_dim, H, W] + [B, input_dim, H, W]
        # -> [B, hidden_dim + input_dim, H, W]

        z = torch.sigmoid(self.convz(hx))
        # 更新门 z
        # -> [B, hidden_dim, H, W]

        r = torch.sigmoid(self.convr(hx))
        # 重置门 r
        # -> [B, hidden_dim, H, W]

        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))
        # 先计算 r*h:
        # [B, hidden_dim, H, W]
        # 再与 x 拼接:
        # -> [B, hidden_dim + input_dim, H, W]
        # convq 后:
        # -> [B, hidden_dim, H, W]

        h = (1-z) * h + z * q
        # GRU 更新公式
        # 输出: [B, hidden_dim, H, W]

        return h
        # 返回新隐藏状态


class SepConvGRU(nn.Module):
    """
    可分离方向卷积版 GRU。

    作用：
        将隐藏状态更新拆成两个方向进行：
        1. 水平方向卷积 (1x5)
        2. 垂直方向卷积 (5x1)

    输入：
        h: [B, hidden_dim, H, W]
        x: [B, input_dim, H, W]

    输出：
        h_new: [B, hidden_dim, H, W]
    """

    def __init__(self, hidden_dim=128, input_dim=192+128):
        """
        初始化 SepConvGRU。

        参数：
            hidden_dim: 隐藏状态通道数
            input_dim: 输入特征通道数
        """
        super(SepConvGRU, self).__init__()

        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        # 水平方向更新门 z
        # 输入: [B, hidden_dim + input_dim, H, W]
        # 输出: [B, hidden_dim, H, W]

        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        # 水平方向重置门 r

        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        # 水平方向候选状态 q

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        # 垂直方向更新门 z

        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        # 垂直方向重置门 r

        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        # 垂直方向候选状态 q

    def forward(self, h, x):
        """
        前向传播。

        参数：
            h: 旧隐藏状态 [B, hidden_dim, H, W]
            x: 当前输入特征 [B, input_dim, H, W]

        返回：
            更新后的隐藏状态 [B, hidden_dim, H, W]
        """
        # horizontal
        hx = torch.cat([h, x], dim=1)
        # 拼接输入
        # -> [B, hidden_dim + input_dim, H, W]

        z = torch.sigmoid(self.convz1(hx))
        # 水平方向更新门
        # -> [B, hidden_dim, H, W]

        r = torch.sigmoid(self.convr1(hx))
        # 水平方向重置门
        # -> [B, hidden_dim, H, W]

        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        # 水平方向候选状态
        # -> [B, hidden_dim, H, W]

        h = (1-z) * h + z * q
        # 水平方向更新后的隐藏状态
        # [B, hidden_dim, H, W]

        # vertical
        hx = torch.cat([h, x], dim=1)
        # 再次拼接输入
        # -> [B, hidden_dim + input_dim, H, W]

        z = torch.sigmoid(self.convz2(hx))
        # 垂直方向更新门
        # -> [B, hidden_dim, H, W]

        r = torch.sigmoid(self.convr2(hx))
        # 垂直方向重置门
        # -> [B, hidden_dim, H, W]

        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        # 垂直方向候选状态
        # -> [B, hidden_dim, H, W]

        h = (1-z) * h + z * q
        # 垂直方向更新后的隐藏状态
        # [B, hidden_dim, H, W]

        return h
        # 返回最终隐藏状态


class SmallMotionEncoder(nn.Module):
    """
    小型运动编码器。

    作用：
        将当前光流 flow 和相关性特征 corr 编码为运动特征。

    输入：
        flow: [B, 2, H, W]
        corr: [B, cor_planes, H, W]

    输出：
        motion_features: [B, 82, H, W]
    """

    def __init__(self, args):
        """
        初始化小型运动编码器。

        参数：
            args.corr_levels: 相关性层数
            args.corr_radius: 相关性半径
        """
        super(SmallMotionEncoder, self).__init__()

        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        # 相关性输入通道数

        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        # 相关性分支
        # [B, cor_planes, H, W] -> [B, 96, H, W]

        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        # 光流分支第 1 层
        # [B, 2, H, W] -> [B, 64, H, W]

        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        # 光流分支第 2 层
        # [B, 64, H, W] -> [B, 32, H, W]

        self.conv = nn.Conv2d(128, 80, 3, padding=1)
        # 融合层
        # [B, 128, H, W] -> [B, 80, H, W]

    def forward(self, flow, corr):
        """
        前向传播。

        参数：
            flow: 当前光流 [B, 2, H, W]
            corr: 相关性特征 [B, cor_planes, H, W]

        返回：
            运动编码特征 [B, 82, H, W]
        """
        cor = F.relu(self.convc1(corr))
        # [B, cor_planes, H, W] -> [B, 96, H, W]

        flo = F.relu(self.convf1(flow))
        # [B, 2, H, W] -> [B, 64, H, W]

        flo = F.relu(self.convf2(flo))
        # [B, 64, H, W] -> [B, 32, H, W]

        cor_flo = torch.cat([cor, flo], dim=1)
        # [B, 96, H, W] + [B, 32, H, W]
        # -> [B, 128, H, W]

        out = F.relu(self.conv(cor_flo))
        # [B, 128, H, W] -> [B, 80, H, W]

        return torch.cat([out, flow], dim=1)
        # [B, 80, H, W] + [B, 2, H, W]
        # -> [B, 82, H, W]


class BasicMotionEncoder(nn.Module):
    """
    基础运动编码器。

    作用：
        对 flow 和 corr 做更强的特征编码，生成用于更新模块的运动特征。

    输入：
        flow: [B, 2, H, W]
        corr: [B, cor_planes, H, W]

    输出：
        motion_features: [B, 128, H, W]
    """

    def __init__(self, corr_levels, corr_radius):
        """
        初始化基础运动编码器。

        参数：
            corr_levels: 相关性层数
            corr_radius: 相关性半径
        """
        super(BasicMotionEncoder, self).__init__()

        self.corr_levels = corr_levels
        # 保存相关性层数

        self.corr_radius = corr_radius
        # 保存相关性半径

        cor_planes = self.corr_levels * (2*self.corr_radius + 1)**2
        # 相关性输入通道数

        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        # 相关性分支第 1 层
        # [B, cor_planes, H, W] -> [B, 256, H, W]

        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        # 相关性分支第 2 层
        # [B, 256, H, W] -> [B, 192, H, W]

        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        # 光流分支第 1 层
        # [B, 2, H, W] -> [B, 128, H, W]

        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        # 光流分支第 2 层
        # [B, 128, H, W] -> [B, 64, H, W]

        self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)
        # 融合层
        # [B, 256, H, W] -> [B, 126, H, W]

    def forward(self, flow, corr):
        """
        前向传播。

        参数：
            flow: 当前光流 [B, 2, H, W]
            corr: 相关性特征 [B, cor_planes, H, W]

        返回：
            运动编码特征 [B, 128, H, W]
        """
        cor = F.relu(self.convc1(corr))
        # [B, cor_planes, H, W] -> [B, 256, H, W]

        cor = F.relu(self.convc2(cor))
        # [B, 256, H, W] -> [B, 192, H, W]

        flo = F.relu(self.convf1(flow))
        # [B, 2, H, W] -> [B, 128, H, W]

        flo = F.relu(self.convf2(flo))
        # [B, 128, H, W] -> [B, 64, H, W]

        cor_flo = torch.cat([cor, flo], dim=1)
        # [B, 192, H, W] + [B, 64, H, W]
        # -> [B, 256, H, W]

        out = F.relu(self.conv(cor_flo))
        # [B, 256, H, W] -> [B, 126, H, W]

        return torch.cat([out, flow], dim=1)
        # [B, 126, H, W] + [B, 2, H, W]
        # -> [B, 128, H, W]


class SmallUpdateBlock(nn.Module):
    """
    小型更新块。

    作用：
        1. 编码 flow 和 corr 得到运动特征
        2. 将运动特征与上下文 inp 拼接
        3. 使用 ConvGRU 更新隐藏状态 net
        4. 用 FlowHead 预测当前迭代的光流增量 delta_flow

    输入：
        net: [B, hidden_dim, H, W]
        inp: [B, 64, H, W]（常见配置）
        corr: [B, cor_planes, H, W]
        flow: [B, 2, H, W]

    输出：
        net: [B, hidden_dim, H, W]
        mask: None
        delta_flow: [B, 2, H, W]
    """

    def __init__(self, args, hidden_dim=96):
        """
        初始化小型更新块。

        参数：
            args: 配置参数
            hidden_dim: 隐藏状态通道数
        """
        super(SmallUpdateBlock, self).__init__()

        self.encoder = SmallMotionEncoder(args)
        # 小型运动编码器
        # 输出: [B, 82, H, W]

        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82+64)
        # ConvGRU
        # h: [B, hidden_dim, H, W]
        # x: [B, 146, H, W]

        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)
        # 光流预测头
        # [B, hidden_dim, H, W] -> [B, 2, H, W]

    def forward(self, net, inp, corr, flow):
        """
        前向传播。

        参数：
            net: 当前隐藏状态 [B, hidden_dim, H, W]
            inp: 上下文输入 [B, 64, H, W]
            corr: 相关性特征 [B, cor_planes, H, W]
            flow: 当前光流 [B, 2, H, W]

        返回：
            net: 更新后的隐藏状态 [B, hidden_dim, H, W]
            None: 小模型不输出上采样 mask
            delta_flow: 光流增量 [B, 2, H, W]
        """
        motion_features = self.encoder(flow, corr)
        # [B, 2, H, W] + [B, cor_planes, H, W]
        # -> [B, 82, H, W]

        inp = torch.cat([inp, motion_features], dim=1)
        # [B, 64, H, W] + [B, 82, H, W]
        # -> [B, 146, H, W]

        net = self.gru(net, inp)
        # 更新隐藏状态
        # -> [B, hidden_dim, H, W]

        delta_flow = self.flow_head(net)
        # -> [B, 2, H, W]

        return net, None, delta_flow
        # 返回更新后的隐藏状态、空 mask、光流增量


class BasicUpdateBlock256(nn.Module):
    """
    基础更新块 256 版。

    作用：
        1. 用 BasicMotionEncoder 编码 flow 与 corr
        2. 将运动特征与上下文 inp 拼接
        3. 用 SepConvGRU 更新隐藏状态 net
        4. 用 FlowHead 预测光流增量 delta_flow
        5. 用 mask 分支生成上采样掩码

    输入：
        net: [B, hidden_dim, H, W]
        inp: [B, input_dim, H, W]
        corr: [B, cor_planes, H, W]
        flow: [B, 2, H, W]

    输出：
        net: [B, hidden_dim, H, W]
        mask: [B, 64*9, H, W]
        delta_flow: [B, 2, H, W]
    """

    def __init__(self, corr_levels=4, corr_radius=4, hidden_dim=128, input_dim=128):
        """
        初始化基础更新块 256 版。

        参数：
            corr_levels: 相关性层数
            corr_radius: 相关性半径
            hidden_dim: 隐藏状态通道数
            input_dim: 输入上下文通道数
        """
        super(BasicUpdateBlock256, self).__init__()

        self.corr_levels = corr_levels
        # 保存相关性层数

        self.corr_radius = corr_radius
        # 保存相关性半径

        self.encoder = BasicMotionEncoder(self.corr_levels, self.corr_radius)
        # 基础运动编码器
        # 输出: [B, 128, H, W]

        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        # 可分离方向卷积 GRU
        # h: [B, hidden_dim, H, W]
        # x: [B, 128 + hidden_dim, H, W]

        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        # 光流预测头
        # [B, hidden_dim, H, W] -> [B, 2, H, W]

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            # [B, 128, H, W] -> [B, 256, H, W]

            nn.ReLU(inplace=True),
            # 激活，尺寸不变

            nn.Conv2d(256, 64*9, 1, padding=0))
            # [B, 256, H, W] -> [B, 576, H, W]

    def forward(self, net, inp, corr, flow, upsample=True):
        """
        前向传播。

        参数：
            net: 当前隐藏状态 [B, hidden_dim, H, W]
            inp: 上下文输入 [B, hidden_dim, H, W]（常见配置）
            corr: 相关性特征 [B, cor_planes, H, W]
            flow: 当前光流 [B, 2, H, W]
            upsample: 是否上采样，这里参数保留但本函数内部未直接使用

        返回：
            net: 更新后的隐藏状态 [B, hidden_dim, H, W]
            mask: 上采样掩码 [B, 576, H, W]
            delta_flow: 光流增量 [B, 2, H, W]
        """
        motion_features = self.encoder(flow, corr)
        # 编码运动特征
        # -> [B, 128, H, W]

        inp = torch.cat([inp, motion_features], dim=1)
        # 拼接上下文特征与运动特征
        # 若 inp 为 [B, 128, H, W]
        # 则拼接后 -> [B, 256, H, W]

        net = self.gru(net, inp)
        # 更新隐藏状态
        # -> [B, hidden_dim, H, W]

        delta_flow = self.flow_head(net)
        # 预测光流增量
        # -> [B, 2, H, W]

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        # 预测上采样 mask
        # self.mask(net): [B, 128, H, W] -> [B, 576, H, W]
        # 再乘 0.25 缩放梯度，帮助训练稳定

        return net, mask, delta_flow
        # 返回：
        # net: 更新后的隐藏状态
        # mask: 上采样掩码
        # delta_flow: 光流增量
