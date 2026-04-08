import torch
from torch import nn

# RAFT 中的更新模块（通常包含 GRU 风格的迭代更新单元）
from study.SRGAN.model.PIV_esrgan_RAFT.Module.subModule.submodules_RAFT_GRU import BasicUpdateBlock
from study.SRGAN.model.PIV_esrgan_RAFT.Module.subModule.submodules_RAFT_GRU256 import BasicUpdateBlock256
# RAFT 中的特征提取网络
from study.SRGAN.model.PIV_esrgan_RAFT.Module.subModule.submodules_RAFT_extractor import BasicEncoder

# 注意：这里更常见、也更推荐写成：
import torch.nn.functional as F

from study.SRGAN.model.PIV_esrgan_RAFT.Module.subModule.submodules_RAFT_extractor256 import BasicEncoder256
from study.SRGAN.model.PIV_esrgan_RAFT.global_class import global_data

try:
    # 自动混合精度上下文管理器
    # 使用后可以在前向推理中自动选择 float16 / float32，
    # 从而减少显存占用并提升训练速度
    autocast = torch.cuda.amp.autocast
except:
    # 如果 PyTorch 版本过低，不支持 AMP，则定义一个“空壳” autocast
    # 这样后面 with autocast(...): 仍然可以正常运行，只是不会启用混合精度
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """
    对 grid_sample 的封装。
    输入 coords 使用的是“像素坐标系”，而不是 grid_sample 要求的 [-1, 1] 归一化坐标。

    参数:
        img:    [B, C, H, W]，待采样的特征图或图像
        coords: [B, H, W, 2]，每个位置要去 img 中采样的坐标
                最后一维 2 表示 (x, y)
        mode:   插值方式，默认双线性
        mask:   是否返回有效区域掩码

    返回:
        若 mask=False:
            采样结果 [B, C, H, W]
        若 mask=True:
            (采样结果, 有效掩码)
    """
    H, W = img.shape[-2:]

    # 将坐标拆成 x 和 y
    xgrid, ygrid = coords.split([1, 1], dim=-1)

    # 将像素坐标映射到 [-1, 1] 区间
    # grid_sample 要求输入坐标是归一化坐标
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    # 拼成 grid_sample 所需格式 [B, H, W, 2]
    grid = torch.cat([xgrid, ygrid], dim=-1)

    # 在 img 上按照给定网格进行双线性采样
    img = F.grid_sample(img.float(), grid, align_corners=True)

    if mask:
        # 判断采样点是否落在合法区域 [-1,1] 内
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.half()

    return img


def coords_grid(batch, ht, wd):
    """
    生成坐标网格。

    返回形状:
        [B, 2, H, W]

    其中:
        第 0 通道是 x 坐标
        第 1 通道是 y 坐标

    例如:
        coords[:, 0, :, :] -> 每个像素位置的列索引 x
        coords[:, 1, :, :] -> 每个像素位置的行索引 y
    """
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    # coords 默认顺序是 (y, x)，这里通过 coords[::-1] 调整为 (x, y)
    coords = torch.stack(coords[::-1], dim=0).float()
    # 扩展 batch 维度
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    """
    将光流上采样 8 倍。

    RAFT 常在低分辨率特征图上预测 flow，
    最后再把 flow 恢复到更高分辨率。

    注意:
        光流不仅空间尺寸要扩大，数值大小也要同步乘 8，
        因为位移单位也随着分辨率变大。
    """
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


class CorrBlock:
    """
    相关性体（correlation volume）构建模块。

    RAFT 的核心思想之一：
    先计算 fmap1 与 fmap2 所有位置之间的相关性，
    得到一个全局相关性体，然后在每次迭代时根据当前 flow 位置，
    从相关性体中局部采样，作为更新网络的输入。
    """

    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        """
        参数:
            fmap1, fmap2: 两张图像提取出的特征图，形状通常为 [B, C, H, W]
            num_levels:   金字塔层数
            radius:       每层相关性局部搜索半径
        """
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # 计算所有像素对之间的相关性
        # 输出形状大致为 [B, H1, W1, 1, H2, W2]
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape

        # 将前面三个维度压平，方便后面做池化与采样
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        # 第 0 层原始相关性图
        self.corr_pyramid.append(corr)

        # 构建多尺度相关性金字塔
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        """
        根据当前坐标 coords，从多尺度相关性金字塔中采样局部相关特征。

        参数:
            coords: [B, 2, H, W]
                    表示当前第二张图像中每个位置对应的坐标估计

        返回:
            out: [B, C_corr, H, W]
                 多层局部相关特征拼接后的结果
        """
        r = self.radius

        # 调整为 [B, H, W, 2]，方便后面构造采样坐标
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]

            # 在当前层构造一个局部搜索窗口 [-r, r]
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)

            # 生成局部偏移网格，最后一维是 (dy, dx) 对应坐标偏移
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

            # 当前层的中心坐标
            # 由于金字塔每层分辨率减半，所以坐标也要除以 2**i
            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i

            # 偏移网格 reshape 成可广播形式
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)

            # 该层所有局部采样点坐标 = 中心点 + 局部偏移
            coords_lvl = centroid_lvl + delta_lvl

            # 从该层相关性图中采样局部窗口
            corr = bilinear_sampler(corr, coords_lvl)

            # 恢复成 [B, H, W, neighborhood]
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        # 将所有层的局部相关性特征拼接
        out = torch.cat(out_pyramid, dim=-1)

        # 调整成卷积网络常用格式 [B, C, H, W]
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        """
        计算 fmap1 和 fmap2 的全局相关性。

        输入:
            fmap1, fmap2: [B, C, H, W]

        返回:
            corr: [B, H, W, 1, H, W]

        含义:
            对 fmap1 中每一个位置，与 fmap2 中所有位置做点积相似度。
        """
        batch, dim, ht, wd = fmap1.shape

        # 展平空间维度
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        # 相关性 = 特征点积
        # fmap1.transpose(1, 2): [B, H*W, C]
        # fmap2:                [B, C, H*W]
        # 结果:                 [B, H*W, H*W]
        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)

        # 恢复成空间结构
        corr = corr.view(batch, ht, wd, 1, ht, wd)

        # 用 sqrt(dim) 做归一化，防止通道数大时点积值过大
        return corr / torch.sqrt(torch.tensor(dim).float())


def sequence_loss(flow_preds, flow_gt):
    """
    RAFT 的序列损失函数。

    RAFT 会经过多次迭代，不断输出更精细的 flow 预测。
    因此损失不是只看最后一次预测，而是对每次预测都计算损失，
    并给后面的预测更大的权重。

    参数:
        flow_preds: 预测序列，list，每个元素形状通常为 [B, 2, H, W]
        flow_gt:    光流真值，形状 [B, 2, H, W]

    返回:
        flow_loss: 标量损失
        metrics:   一些评估指标
    """
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    for i in range(n_predictions):
        # 越靠后的预测权重越大
        # 最后一轮通常最重要
        i_weight = 0.8 ** (n_predictions - i - 1)

        # 这里使用 L1 误差
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (i_loss).mean()

    # 计算最终预测的 EPE（End-Point Error）
    # 对 flow 两个分量求平方和再开根号，得到每个像素的欧氏距离误差
    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)

    # 常见光流评估指标
    metrics = {
        'epe': epe.mean().item(),                  # 平均 EPE
        '1px': (epe < 1).float().mean().item(),   # 误差小于 1 像素的比例
        '3px': (epe < 3).float().mean().item(),   # 误差小于 3 像素的比例
        '5px': (epe < 5).float().mean().item(),   # 误差小于 5 像素的比例
    }

    return flow_loss, metrics


class RAFT(nn.Module):
    """
    RAFT 主网络。

    基本流程:
    1. 用特征提取器提取两帧图像特征 fmap1, fmap2
    2. 构建全局相关性体 CorrBlock
    3. 用上下文网络 cnet 提取初始隐藏状态和上下文输入
    4. 初始化坐标网格 coords0 和 coords1
    5. 多次迭代更新 coords1，得到 flow = coords1 - coords0
    """

    def __init__(self):
        super(RAFT, self).__init__()

        # 隐状态维度
        self.hidden_dim = 128
        # 上下文特征维度
        self.context_dim = 128

        # 相关性金字塔层数
        self.corr_levels = 4
        # 每层局部搜索半径
        self.corr_radius = 4

        # 特征提取网络：
        # 输入图像，输出用于建立相关性体的特征
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=0.)

        # 上下文编码网络：
        # 通常只对第一帧提取上下文信息
        # 输出会被拆成 hidden state 和 context input
        self.cnet = BasicEncoder(output_dim=self.hidden_dim + self.context_dim,
                                 norm_fn='instance', dropout=0.)

        # 迭代更新模块：
        # 输入当前隐藏状态、上下文、相关性特征、当前 flow
        # 输出新的隐藏状态和 flow 增量
        self.update_block = BasicUpdateBlock(
            hidden_dim=self.hidden_dim,
            corr_levels=self.corr_levels,
            corr_radius=self.corr_radius
        )

    def initialize_flow(self, img):
        """
        初始化光流坐标。

        RAFT 不是直接回归 flow，而是维护两个坐标网格：
            flow = coords1 - coords0

        初始时 coords0 与 coords1 相同，因此初始 flow = 0
        """
        N, C, H, W = img.shape

        # 基准坐标网格
        coords0 = coords_grid(N, H, W).to(img.device)
        # 当前预测坐标网格，初始化为与 coords0 相同
        coords1 = coords_grid(N, H, W).to(img.device)

        return coords0, coords1

    def forward(self, input, flowl0, flow_init=None, upsample=True):
        """
        前向传播。

        参数:
            input:     输入图像对，形状大概率为 [B, 2, H, W]
                       其中第 0 通道是第一帧，第 1 通道是第二帧
            flowl0:    光流真值 [B, 2, H, W]
        =
            flow_init: 可选的初始 flow
            upsample:  是否上采样（当前代码中未实际使用）

        返回:
            flow_predictions: 每次迭代得到的 flow 预测列表
            loss:             sequence_loss 的返回结果
                              即 (flow_loss, metrics)
        """
        # 取出第一帧和第二帧，补成单通道格式 [B, 1, H, W]
        img1 = torch.unsqueeze(input[:, 0, :, :], dim=1)
        img2 = torch.unsqueeze(input[:, 1, :, :], dim=1)

        # 提取两帧图像特征，用于构建相关性体
        with autocast(enabled=global_data.esrgan.AMP):
            fmap1, fmap2 = self.fnet([img1, img2])

        # 构建相关性金字塔
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius, num_levels=self.corr_levels)

        # 对第一帧提取上下文特征
        with autocast(enabled=global_data.esrgan.AMP):
            cnet = self.cnet(img1)

            # 将上下文特征拆分成两部分：
            # net: 初始隐藏状态
            # inp: 上下文输入特征
            net, inp = torch.split(cnet, [self.hidden_dim, self.context_dim], dim=1)

            # 隐状态一般过 tanh，限制范围到 [-1, 1]
            net = torch.tanh(net)
            # 上下文输入做 ReLU 激活
            inp = torch.relu(inp)

        # 初始化 flow 坐标
        coords0, coords1 = self.initialize_flow(img1)

        # 保存每轮迭代的 flow 预测
        flow_predictions = []

        # 迭代更新 flow
        for itr in range(global_data.esrgan.GRU_ITERS):
            # 阻断 coords1 的梯度传播，避免跨迭代的反向传播图过大
            coords1 = coords1.detach()

            # 根据当前 coords1，从相关性体中索引对应局部相关特征
            corr = corr_fn(coords1)

            # 如果提供了初始 flow，则第一轮使用它
            if itr == 0 and flow_init is not None:
                flow = flow_init
            else:
                # 当前 flow = 当前坐标 - 初始坐标
                flow = coords1 - coords0

            # 更新模块输出：
            # net: 更新后的隐藏状态
            # up_mask: 上采样掩码（这里虽然输出了，但后续没用）
            # delta_flow: 当前轮预测的光流增量
            with autocast(enabled=global_data.esrgan.AMP):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # 坐标递推更新
            # F(t+1) = F(t) + Δ(t)
            coords1 = coords1 + delta_flow

            # 更新后的 flow
            flow = coords1 - coords0

            # 保存当前轮预测
            flow_predictions.append(flow)

        # 计算序列损失
        loss = sequence_loss(flow_predictions, flowl0)

        return flow_predictions, loss
class LanczosUpsampling(nn.Module):
    """
    Lanczos4 upsampling module
    """

    def __init__(self, img_shape, new_size, a=4):
        super(LanczosUpsampling, self).__init__()
        torch.pi = torch.acos(torch.zeros(1)).item() * 2
        delta_X = torch.range(0. + 1e-8, 1., 1. / 1024)
        self.B, self.C, self.H, self.W = img_shape
        self.new_size = new_size

        self.lanczos_kernel = torch.stack(((torch.sin((delta_X - 4) * torch.pi) / ((delta_X - 4) * torch.pi)) * (torch.sin((delta_X - 4) / a * torch.pi) / ((delta_X - 4) / a * torch.pi)), \
                                           (torch.sin((delta_X - 3) * torch.pi) / ((delta_X - 3) * torch.pi)) * (torch.sin((delta_X - 3) / a * torch.pi) / ((delta_X - 3) / a * torch.pi)), \
                                           (torch.sin((delta_X - 2) * torch.pi) / ((delta_X - 2) * torch.pi)) * (torch.sin((delta_X - 2) / a * torch.pi) / ((delta_X - 2) / a * torch.pi)), \
                                           (torch.sin((delta_X - 1) * torch.pi) / ((delta_X - 1) * torch.pi)) * (torch.sin((delta_X - 1) / a * torch.pi) / ((delta_X - 1) / a * torch.pi)), \
                                           (torch.sin((delta_X) * torch.pi) / ((delta_X) * torch.pi)) * (torch.sin((delta_X) / a * torch.pi) / ((delta_X) / a * torch.pi)), \
                                           (torch.sin((delta_X + 1) * torch.pi) / ((delta_X + 1) * torch.pi)) * (torch.sin((delta_X + 1) / a * torch.pi) / ((delta_X + 1) / a * torch.pi)), \
                                           (torch.sin((delta_X + 2) * torch.pi) / ((delta_X + 2) * torch.pi)) * (torch.sin((delta_X + 2) / a * torch.pi) / ((delta_X + 2) / a * torch.pi)), \
                                           (torch.sin((delta_X + 3) * torch.pi) / ((delta_X + 3) * torch.pi)) * (torch.sin((delta_X + 3) / a * torch.pi) / ((delta_X + 3) / a * torch.pi)), \
                                           (torch.sin((delta_X + 4) * torch.pi) / ((delta_X + 4) * torch.pi)) * (torch.sin((delta_X + 4) / a * torch.pi) / ((delta_X + 4) / a * torch.pi))) \
                                          ).cuda()

        self.y_init, self.x_init = torch.meshgrid(torch.arange(0, self.H, 1), torch.arange(0, self.W, 1))
        self.y_new, self.x_new = torch.meshgrid(torch.arange(0, self.H, self.H / self.new_size[2]),
                                                torch.arange(0, self.W, self.W / self.new_size[3]))
        self.y_init_up, self.x_init_up = torch.floor(self.y_new.cuda()).long().cuda(), torch.floor(self.x_new.cuda()).long().cuda()

        self.y_sub, self.x_sub = self.y_new.cuda() - self.y_init_up.cuda(), self.x_new.cuda() - self.x_init_up.cuda()

        self.unfold_x = torch.nn.Unfold(kernel_size=(1, 9))
        self.unfold_y = torch.nn.Unfold(kernel_size=(9, 1))

    def forward(self, img, new_size):
        B, C, H, W = img.shape

        y_init, x_init = torch.meshgrid(torch.arange(0, H, 1), torch.arange(0, W, 1))
        y_new, x_new = torch.meshgrid(torch.arange(0, H, H / self.new_size[2]),
                                      torch.arange(0, W, W / self.new_size[3]))
        y_init_up, x_init_up = torch.floor(y_new.cuda()).long().cuda(), torch.floor(x_new.cuda()).long().cuda()

        y_sub, x_sub = y_new.cuda() - y_init_up.cuda(), x_new.cuda() - x_init_up.cuda()
        img_up_rough = img[:, :, y_init_up, x_init_up]

        ### horizontal shift
        # padding
        p1d = (4, 4, 0, 0)
        padded_img_up = F.pad(img_up_rough, p1d, mode='reflect')
        padded_x_sub = F.pad(torch.unsqueeze(torch.unsqueeze(x_sub, dim=0), dim=0), p1d, mode='reflect')

        # unfold patch
        padded_img_unfold = torch.squeeze(self.unfold_x(padded_img_up))
        padded_x_sub_unfold = torch.squeeze(self.unfold_x(padded_x_sub))

        # compute index and select kernel
        center_point = [4]
        center_index = torch.floor(padded_x_sub_unfold / (1.0 / 1024))[center_point, :]
        x_kernel = self.lanczos_kernel[:, center_index.long()].repeat(C, B, 1).permute(1, 0, 2).cuda()
        x_shifted_patch = torch.sum((x_kernel * padded_img_unfold).reshape(B,C,9,-1),dim=2).reshape(new_size)


        ### vertical shift
        # padding
        p2d = (0, 0, 4, 4)
        padded_img_up = F.pad(x_shifted_patch, p2d, mode='reflect')
        padded_y_sub = F.pad(torch.unsqueeze(torch.unsqueeze(y_sub, dim=0), dim=0), p2d, mode='reflect')

        # unfold patch
        padded_img_unfold = torch.squeeze(self.unfold_y(padded_img_up))
        padded_y_sub_unfold = torch.squeeze(self.unfold_y(padded_y_sub))

        # compute index and select kernel
        center_point = [4]
        center_index = torch.floor(padded_y_sub_unfold / (1.0 / 1024))[center_point, :]
        y_kernel = torch.squeeze(self.lanczos_kernel[:, center_index.long()])
        y_kernel = self.lanczos_kernel[:, center_index.long()].repeat(C, B, 1).permute(1, 0, 2).cuda()
        y_shifted_patch = torch.sum((y_kernel * padded_img_unfold).reshape(B,C,9,-1),dim=2).reshape(new_size)

        return y_shifted_patch

class RAFT256(nn.Module):
    """
    RAFT
    """

    def __init__(self, upsample,batch_size):
        super(RAFT256, self).__init__()

        self.hidden_dim = 128
        self.context_dim = 128
        self.corr_levels = 4
        self.corr_radius = 4
        self.flow_size = 32

        self.fnet = BasicEncoder256(output_dim=256, norm_fn='instance', dropout=0.)
        self.cnet = BasicEncoder256(output_dim=self.hidden_dim + self.context_dim, norm_fn='instance', dropout=0.)
        self.update_block = BasicUpdateBlock256(hidden_dim=self.hidden_dim, corr_levels=self.corr_levels,
                                                corr_radius=self.corr_radius)
        if upsample == 'bicubic':
            self.upsample_bicubic = nn.Upsample(scale_factor=2, mode='bicubic')
        elif upsample == 'bicubic8':
            self.upsample_bicubic8 = nn.Upsample(scale_factor=8, mode='bicubic')
        elif upsample == 'lanczos4':
            self.upsample_lanczos2_1 = LanczosUpsampling([batch_size, 2, self.flow_size, self.flow_size],
                                                         [batch_size, 2, self.flow_size * 2, self.flow_size * 2])
        elif upsample == 'lanczos4_8':
            self.upsample_lanczos2_2 = LanczosUpsampling([batch_size, 2, self.flow_size * 2, self.flow_size * 2],
                                                         [batch_size, 2, self.flow_size * 4, self.flow_size * 4])
            self.upsample_lanczos2_3 = LanczosUpsampling([batch_size, 2, self.flow_size * 4, self.flow_size * 4],
                                                         [batch_size, 2, self.flow_size * 8, self.flow_size * 8])
            self.upsample_lanczos8 = LanczosUpsampling([batch_size, 2, self.flow_size, self.flow_size],
                                                       [batch_size, 2, self.flow_size * 8, self.flow_size * 8])

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(4 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, input, flowl0,  flow_init=None):
        img1 = torch.unsqueeze(input[:, 0, :, :], dim=1)
        img2 = torch.unsqueeze(input[:, 1, :, :], dim=1)

        with autocast(enabled=global_data.esrgan.AMP):
            fmap1, fmap2 = self.fnet([img1, img2])

        corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius, num_levels=self.corr_levels)

        with autocast(enabled=global_data.esrgan.AMP):
            cnet = self.cnet(img1)
            net, inp = torch.split(cnet, [self.hidden_dim, self.context_dim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(img1)

        if flow_init is not None:
            flow_init = F.upsample(flow_init, [coords1.size()[2], coords1.size()[3]], mode='bilinear')
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(global_data.esrgan.GRU_ITERS):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=global_data.esrgan.AMP):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            if global_data.esrgan.RAFT_UPSAMPLE == 'convex':
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            elif global_data.esrgan.RAFT_UPSAMPLE == 'bicubic':
                flow_up = self.upsample_bicubic(self.upsample_bicubic(self.upsample_bicubic(coords1 - coords0)))
            elif global_data.esrgan.RAFT_UPSAMPLE == 'bicubic8':
                flow_up = self.upsample_bicubic8(coords1 - coords0)
            elif global_data.esrgan.RAFT_UPSAMPLE == 'lanczos4':
                B_f, C_f, H_f, W_f = coords1.shape
                flow_up = self.upsample_lanczos2_1(coords1 - coords0, new_size=[B_f, C_f, H_f * 2, W_f * 2])
                flow_up = self.upsample_lanczos2_2(flow_up, new_size=[B_f, C_f, H_f * 4, W_f * 4])
                flow_up = self.upsample_lanczos2_3(flow_up, new_size=[B_f, C_f, H_f * 8, W_f * 8])
            elif global_data.esrgan.RAFT_UPSAMPLE == 'lanczos4_8':
                B_f, C_f, H_f, W_f = coords1.shape
                flow_up = self.upsample_lanczos8(coords1 - coords0, new_size=[B_f, C_f, H_f * 8, W_f * 8])
            else:
                raise ValueError('Selected upsample method not supported: ', global_data.esrgan.RAFT_UPSAMPLE)

            flow_predictions.append(flow_up)

        loss = sequence_loss(flow_predictions, flowl0)

        return flow_predictions, loss