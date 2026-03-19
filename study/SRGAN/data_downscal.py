#数据下采样
import torch
import torch.nn.functional as F

def hr_to_lr_maxpool(x_hr, r):
    # x_hr: [N, C, H, W]
    # r: downsampling factor, e.g. 2/4/8
    N, C, H, W = x_hr.shape
    Hc, Wc = (H // r) * r, (W // r) * r   # 裁剪到可整除
    x_hr = x_hr[:, :, :Hc, :Wc]
    x_lr = F.max_pool2d(x_hr, kernel_size=r, stride=r)
    return x_lr