"""
损失函数 start
"""
import torch
from torch import nn
from torchvision.models import vgg19

from study.SRGAN.model.basic_srgan.global_class_srgan import global_data


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
        内容损失（VGG特征空间MSE）：约束生成图在感知特征上接近真值图。
        """

    def __init__(self, vgg):
        """冻结 VGG 特征提取器并初始化 MSE 损失。"""
        super().__init__()
        self.vgg = vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.criterion = nn.MSELoss()

    def forward(self, fake, real):
        """计算 fake 与 real 的 VGG 特征 MSE。"""
        # 给 G 传梯度：fake 不 detach；real 可以 detach
        fake_features = self.vgg(fake)
        real_features = self.vgg(real).detach()
        return self.criterion(fake_features, real_features)
class AdversarialLoss(nn.Module):
    """
    对抗损失：鼓励生成图被判别器判为真实。
    """
    def __init__(self):
        super(AdversarialLoss,self).__init__()

    def forward(self, x):
        loss = torch.sum(-torch.log(x))
        return loss
class PerceptualLoss(nn.Module):
    """
    感知损失 = 内容损失+1e-3 * 对抗损失
    """
    def __init__(self, vgg):
        super(PerceptualLoss,self).__init__()
        self.vgg_loss = ContentLoss(vgg)
        self.adversarial = AdversarialLoss()

    def forward(self, fake, real, x):
        vgg_loss = self.vgg_loss(fake, real)
        adversarial_loss = self.adversarial(x)

        return vgg_loss +global_data.srgan.LAMBDA_PERCEPTION*adversarial_loss,vgg_loss,adversarial_loss
class RegularizationLoss(nn.Module):
    """
    图像平滑正则：惩罚相邻像素突变，抑制高频噪声。
    """

    def __init__(self):
        super(RegularizationLoss,self).__init__()

    def forward(self, x):
        """计算基于局部梯度的平滑正则损失。"""
        a = torch.square(
            x[:, :, :x.shape[2]-1, :x.shape[3]-1] - x[:, :, 1:x.shape[2], :x.shape[3]-1]
        )
        b = torch.square(
            x[:, :, :x.shape[2]-1, :x.shape[3]-1] - x[:, :, :x.shape[2]-1, 1:x.shape[3]]
        )
        loss = torch.sum(torch.pow(a+b, 1.25))
        return loss



class CombinedPixelLoss(nn.Module):
    """
    组合像素损失：
    - 常规模式：L1 + MSE
    - 灰度增强模式：白点加权 + 通道一致性约束
    SAVE_AS_GRAY=True 且 image_pair 时:
      - 用 target 亮度做加权，提升白点召回
      - 加轻量通道一致性约束（防伪彩）
    其余情况:
      - 普通 RGB L1 + MSE
    返回: total, l1_term, mse_term
    """
    def __init__(self, lambda_l1=2e-2, lambda_mse=1e-3, white_alpha=4.0, lambda_cons=1e-2):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_mse = lambda_mse
        self.white_alpha = white_alpha
        self.lambda_cons = lambda_cons
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, pred, target, gray_triplet=False):
        if gray_triplet:
            pred_gray = pred[:, 0:1]
            target_gray = target[:, 0:1]

            # 白点加权
            weight = 1.0 + self.white_alpha * target_gray
            l1_term = (weight * (pred_gray - target_gray).abs()).mean()
            mse_term = self.mse(pred_gray, target_gray)

            # 灰度复制RGB一致性（不要太大）
            cons = (
                self.l1(pred[:, 0:1], pred[:, 1:2]) +
                self.l1(pred[:, 1:2], pred[:, 2:3]) +
                self.l1(pred[:, 0:1], pred[:, 2:3])
            ) / 3.0

            total = self.lambda_l1 * l1_term + self.lambda_mse * mse_term + self.lambda_cons * cons
            return total, l1_term, mse_term

        l1_term = self.l1(pred, target)
        mse_term = self.mse(pred, target)
        total = self.lambda_l1 * l1_term + self.lambda_mse * mse_term
        return total, l1_term, mse_term
"""
损失函数 end
"""
# 实例化loss
# 定义像素损失函数
pixel_loss = CombinedPixelLoss(
    lambda_l1=global_data.srgan.LAMBDA_PIXEL_L1,
    lambda_mse=global_data.srgan.LAMBDA_PIXEL_MSE,
    white_alpha=global_data.srgan.PIXEL_WHITE_ALPHA,
    lambda_cons=global_data.srgan.LAMBDA_GRAY_CONS,
).to(global_data.srgan.device)
# 这里vgg是针对三通道RGB图的
vgg = vgg19(pretrained=True).features[:16].eval()  # 提取 VGG 特征
# vgg模型预测模式
vgg = vgg.to(global_data.srgan.device).eval()

# 感知损失
perceptual_loss = PerceptualLoss(vgg=vgg)
# 判别器的损失函数
loss_d = nn.BCELoss()
# 归一化损失 正则化损失
regularization_loss = RegularizationLoss()