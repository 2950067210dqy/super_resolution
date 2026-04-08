import torch  # 导入 PyTorch 主库
from torch import nn  # 导入神经网络模块基类和常用层接口
import torch.nn.functional as F  # 导入函数式接口，便于直接调用 loss / 激活等函数

from study.SRGAN.model.PIV_esrgan_RAFT.Module.RAFT_Model import RAFT  # 导入 RAFT 光流估计主网络
from study.SRGAN.model.PIV_esrgan_RAFT.Module.loss import (
    descriminator_loss,  # 判别器损失，负责训练 D 区分真/假图像
    image_pair_temporal_loss,  # 图像对时间一致性损失，约束前后帧变化关系
    perceptual_loss,  # 感知损失，内部可选带对抗项
    particle_loss,  # 粒子结构/物理分布损失
    pixel_loss,  # 像素域复合损失，包含 L1/MSE/SSIM/FFT 等项
)
from study.SRGAN.model.PIV_esrgan_RAFT.Module.piv_esrgan_model import Generator, Discriminator  # 导入 ESRGAN 生成器和判别器
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

class PIV_ESRGAN_RAFT(nn.Module):
    """
    PIV_ESRGAN_RAFT 主网络。

    基本流程:
    1.用改进的esrgan 对低分辨率图像对进行超分辨
    2.然后用基本的RAFT进行piv估计
    """

    def __init__(self,inner_chanel):
        super(PIV_ESRGAN_RAFT, self).__init__()  # 调用父类初始化

        self.piv_esrgan_generator = Generator(inner_chanel=inner_chanel)  # 初始化超分生成器，输入 LR 图像对，输出 SR 图像对
        self.piv_esrgan_discriminator = Discriminator(inner_chanel=inner_chanel)  # 初始化判别器，用于区分 SR 图像和真实 HR 图像
        self.piv_RAFT = RAFT()  # 初始化 RAFT，用于根据两帧 SR 图像预测 PIV/光流场

    @staticmethod
    def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
        """
        统一控制某个子模块的参数是否参与梯度计算。

        参数:
            module: 需要切换梯度状态的子模块
            requires_grad: 是否计算梯度
        """
        for param in module.parameters():  # 遍历该子模块的全部可学习参数
            param.requires_grad = requires_grad  # 统一开启或关闭梯度，常用于交替训练 G 和 D

    def _to_raft_frame(self, image: torch.Tensor) -> torch.Tensor:
        """
        将 ESRGAN 输出转换成 RAFT 当前实现所需的单通道帧。

        参数:
            image: [B, C, H, W]

        返回:
            frame: [B, 1, H, W]
        """
        if image.dim() != 4:  # 检查输入是否是标准 4 维图像张量
            raise ValueError(f"Expected image to be 4D [B, C, H, W], got shape={tuple(image.shape)}")  # 如果不是 [B,C,H,W]，直接报错提醒

        if image.size(1) == 1:  # 如果本来就是单通道
            return image  # 则无需转换，直接返回

        # 当前 RAFT 分支按单通道实现；若生成器输出是 3 通道，
        # 这里先在通道维求均值压成单通道，保证接口兼容。
        return image.mean(dim=1, keepdim=True)  # 否则在通道维求均值，将多通道图压成单通道图

    def _compute_generator_frame_terms(self, pred: torch.Tensor, target: torch.Tensor, is_adversarial: bool) -> dict:
        """
        计算单帧生成器损失项。

        参数:
            pred: 生成器输出 [B, C, H, W]
            target: 对应真值 [B, C, H, W]
            is_adversarial: 是否在感知损失中启用对抗项

        返回:
            包含单帧各项损失的字典
        """
        pred_fake = self.piv_esrgan_discriminator(pred)  # 将生成器输出送入判别器，得到“假样本”判别结果
        pred_real = self.piv_esrgan_discriminator(target)  # 将真值图像送入判别器，得到“真样本”判别结果

        perceptual_total, content_loss, adversarial_loss = perceptual_loss(
            pred,  # 当前帧生成图像
            target,  # 当前帧对应的真值图像
            pred_fake,  # 判别器对假图的输出
            pred_real,  # 判别器对真图的输出
            is_adversarial=is_adversarial,  # 是否在感知损失中启用对抗项
        )

        pixel_total, pixel_l1, pixel_mse, pixel_ssim, pixel_fft = pixel_loss(pred, target)  # 计算像素级复合损失及各子项


        return {  # 将单帧所有损失及中间判别结果打包，便于后面 previous/next 两帧统一平均
            "pred_fake": pred_fake,  # 判别器对该帧 SR 图的输出
            "pred_real": pred_real,  # 判别器对该帧 HR 图的输出
            "perceptual_loss": perceptual_total,  # 感知损失总项
            "content_loss": content_loss,  # 感知损失中的内容损失部分
            "adversarial_loss": adversarial_loss,  # 感知损失中的对抗损失部分
            "pixel_total": pixel_total,  # 像素损失总项
            "pixel_l1": pixel_l1,  # 像素损失中的 L1 项
            "pixel_mse": pixel_mse,  # 像素损失中的 MSE 项
            "pixel_ssim": pixel_ssim,  # 像素损失中的 SSIM 项
            "pixel_fft": pixel_fft,  # 像素损失中的频域项

        }

    @staticmethod
    def _mean_loss_term(prev_terms: dict, next_terms: dict, key: str) -> torch.Tensor:
        """
        对前后两帧的同名标量损失做平均。
        """
        return 0.5 * (prev_terms[key] + next_terms[key])  # 对前后两帧的同名损失做平均，保持两帧地位一致

    def _compute_discriminator_loss(self, pred_prev: torch.Tensor, pred_next: torch.Tensor, target_prev: torch.Tensor, target_next: torch.Tensor):
        """
        计算判别器损失。

        参数:
            pred_prev/pred_next: 生成器输出
            target_prev/target_next: 真值

        返回:
            d_loss, fake_loss, real_loss
        """
        pred_fake_prev_d = self.piv_esrgan_discriminator(pred_prev.detach())  # 前一帧假图进入判别器；detach 防止更新 D 时梯度回到 G
        pred_real_prev_d = self.piv_esrgan_discriminator(target_prev)  # 前一帧真图进入判别器
        d_prev, fake_prev, real_prev = descriminator_loss(pred_fake_prev_d, pred_real_prev_d)  # 计算前一帧判别器损失及真假分项

        pred_fake_next_d = self.piv_esrgan_discriminator(pred_next.detach())  # 后一帧假图进入判别器
        pred_real_next_d = self.piv_esrgan_discriminator(target_next)  # 后一帧真图进入判别器
        d_next, fake_next, real_next = descriminator_loss(pred_fake_next_d, pred_real_next_d)  # 计算后一帧判别器损失及真假分项

        d_loss = 0.5 * (d_prev + d_next)  # 两帧判别器总损失取平均
        fake_loss = 0.5 * (fake_prev + fake_next)  # 两帧 fake loss 取平均
        real_loss = 0.5 * (real_prev + real_next)  # 两帧 real loss 取平均

        return d_loss, fake_loss, real_loss  # 返回判别器总损失以及两个监控项

    def forward(
        self,
        input_lr_prev,
        input_lr_next,
        input_gr_prev,
        input_gr_next,
        flowl0,
        flow_init=None,
        upsample=True,
        is_adversarial: bool = False,
    ):
        """
        前向传播。

        参数:
            input_lr_prev,input_lr_next, 输入的低分辨率的图像对
            input_gr_prev,input_gr_next, 输入的低分辨率图像对所对应的真值
            flowl0:    光流真值 [B, 2, H, W]

            flow_init: 可选的初始 flow
            upsample:  是否上采样（当前代码中未实际使用）
            is_adversarial: 是否在生成器损失中启用对抗项

        返回:
            outputs: 一个字典，包含超分结果、RAFT 预测结果以及组合损失
        """
        # 1. 先对低分辨率前后帧做联合超分。
        if hasattr(self.piv_esrgan_generator, "forward_pair"):  # 如果生成器支持双帧联合前向
            pred_prev, pred_next = self.piv_esrgan_generator.forward_pair(input_lr_prev, input_lr_next)  # 直接同时生成前后两帧超分结果
        else:
            pred_prev = self.piv_esrgan_generator(input_lr_prev)  # 否则退化成对前一帧单独超分
            pred_next = self.piv_esrgan_generator(input_lr_next)  # 对后一帧单独超分

        # 2. 计算 ESRGAN 生成器相关损失。
        prev_terms = self._compute_generator_frame_terms(pred_prev, input_gr_prev, is_adversarial=is_adversarial)  # 计算 previous 帧的 G 侧损失项
        next_terms = self._compute_generator_frame_terms(pred_next, input_gr_next, is_adversarial=is_adversarial)  # 计算 next 帧的 G 侧损失项

        perceptual_total = self._mean_loss_term(prev_terms, next_terms, "perceptual_loss")  # 平均后的感知损失
        content_loss = self._mean_loss_term(prev_terms, next_terms, "content_loss")  # 平均后的内容损失
        adversarial_loss = self._mean_loss_term(prev_terms, next_terms, "adversarial_loss")  # 平均后的生成器对抗损失
        pixel_total = self._mean_loss_term(prev_terms, next_terms, "pixel_total")  # 平均后的像素总损失
        pixel_l1 = self._mean_loss_term(prev_terms, next_terms, "pixel_l1")  # 平均后的 L1 损失
        pixel_mse = self._mean_loss_term(prev_terms, next_terms, "pixel_mse")  # 平均后的 MSE 损失
        pixel_ssim = self._mean_loss_term(prev_terms, next_terms, "pixel_ssim")  # 平均后的 SSIM 损失
        pixel_fft = self._mean_loss_term(prev_terms, next_terms, "pixel_fft")  # 平均后的频域损失


        pair_temporal_total, pair_temporal_dict = image_pair_temporal_loss(
            pred_prev,  # SR previous 帧
            pred_next,  # SR next 帧
            input_gr_prev,  # HR previous 真值
            input_gr_next,  # HR next 真值
        )

        sr_loss = perceptual_total + pixel_total + pair_temporal_total  # 将所有 ESRGAN 相关项合成统一 SR 侧损失

        # 3. 将 ESRGAN 输出转换为 RAFT 当前实现所需的单通道输入。
        raft_prev = self._to_raft_frame(pred_prev)  # 将前一帧 SR 图转换成 RAFT 可接受的单通道输入
        raft_next = self._to_raft_frame(pred_next)  # 将后一帧 SR 图转换成 RAFT 可接受的单通道输入

        # 4. RAFT 当前 forward 需要形状 [B, 2, H, W]，
        # 其中第 0 通道是前一帧，第 1 通道是后一帧。
        raft_input = torch.cat([raft_prev, raft_next], dim=1)  # 将两帧单通道图沿通道维拼接成 [B,2,H,W]

        # 5. 调用 RAFT 估计流场，并计算 RAFT 的序列损失。
        flow_predictions, (raft_loss, raft_metrics) = self.piv_RAFT(
            raft_input,  # RAFT 输入图像对
            flowl0,  # 光流真值

            flow_init=flow_init,  # 可选初始光流
            upsample=upsample,  # 可选上采样开关
        )

        # 6. 组合总损失。 不使用这个损失
        # total_loss = sr_loss + raft_loss  # 将 SR 分支和 RAFT 分支的损失合并成总损失，用于更新 G + RAFT

        # 7. 计算判别器损失。
        discriminator_loss, d_fake_loss, d_real_loss = self._compute_discriminator_loss(
            pred_prev,  # previous 生成图
            pred_next,  # next 生成图
            input_gr_prev,  # previous 真值图
            input_gr_next,  # next 真值图
        )

        return pred_prev,pred_next,flow_predictions,{  # 将一次前向中产生的全部关键中间结果和损失打包返回，供日志、可视化和训练使用
            "sr_prev": pred_prev,  # 前一帧超分结果
            "sr_next": pred_next,  # 后一帧超分结果
            "raft_input_prev": raft_prev,  # 送入 RAFT 前的前一帧单通道图
            "raft_input_next": raft_next,  # 送入 RAFT 前的后一帧单通道图
            "flow_predictions": flow_predictions,  # RAFT 每次迭代得到的流场预测序列
            "sr_loss": sr_loss,  # ESRGAN 侧总损失
            "perceptual_loss": perceptual_total,  # 感知损失总项
            "content_loss": content_loss,  # 内容损失
            "adversarial_loss": adversarial_loss,  # 生成器对抗损失
            "pixel_total": pixel_total,  # 像素损失总项
            "pixel_l1": pixel_l1,  # L1 子项
            "pixel_mse": pixel_mse,  # MSE 子项
            "pixel_ssim": pixel_ssim,  # SSIM 子项
            "pixel_fft": pixel_fft,  # FFT 子项

            "pair_temporal_loss": pair_temporal_total,  # 图像对时间一致性总损失
            "pair_delta_loss": pair_temporal_dict["pair_delta_loss"],  # 时间一致性中的差分项
            "pair_gradient_loss": pair_temporal_dict["pair_gradient_loss"],  # 时间一致性中的梯度项
            "raft_loss": raft_loss,  # RAFT 流场序列损失
            "raft_metrics": raft_metrics,  # RAFT 的评估指标字典
            "discriminator_loss": discriminator_loss,  # 判别器总损失
            "d_fake_loss": d_fake_loss,  # 判别器对 fake 的损失
            "d_real_loss": d_real_loss,  # 判别器对 real 的损失
            # "total_loss": total_loss,  # 用于更新 Generator + RAFT 的总损失
        }



    def train_step(
        self,
        input_lr_prev,
        input_lr_next,
        input_gr_prev,
        input_gr_next,
        flowl0,
        generator_optimizer,
        raft_optimizer,
        d_optimizer,
        scaler=None,
        flow_init=None,
        upsample=True,
        is_adversarial: bool = False,
    ) -> dict:
        """
        执行一次完整训练步骤。

        训练顺序:
            1. 前向计算全部损失项
            2. 冻结判别器，更新 Generator + RAFT
            3. 开启判别器，更新 Discriminator
            4. 返回本次训练的主要日志

        参数:
            input_lr_prev/input_lr_next: 低分辨率前后帧
            input_gr_prev/input_gr_next: 高分辨率真值前后帧
            flowl0: 光流真值
            args: RAFT 配置
            generator_optimizer: Generator 的优化器
            raft_optimizer: RAFT 的优化器
            d_optimizer: Discriminator 的优化器
            scaler: 可选 AMP GradScaler
            flow_init: 可选 RAFT 初始流
            upsample: 是否上采样
            is_adversarial: 是否在生成器损失中启用对抗项

        返回:
            包含本次训练主要标量的字典
        """
        pred_prev,pred_next,flow_predictions,outputs = self.forward(
            input_lr_prev=input_lr_prev,  # 前一帧 LR 输入
            input_lr_next=input_lr_next,  # 后一帧 LR 输入
            input_gr_prev=input_gr_prev,  # 前一帧 HR 真值
            input_gr_next=input_gr_next,  # 后一帧 HR 真值
            flowl0=flowl0,  # 光流真值

            flow_init=flow_init,  # 可选 RAFT 初始流
            upsample=upsample,  # 是否上采样
            is_adversarial=is_adversarial,  # 是否在 G 侧启用对抗项
        )

        # 先更新 Generator
        self._set_requires_grad(self.piv_esrgan_discriminator, False)  # 冻结判别器，避免第一阶段更新误伤 D
        generator_optimizer.zero_grad(set_to_none=True)  # 清空 Generator 梯度


        if scaler is not None:  # AMP 模式下更新 G
            scaler.scale(outputs["sr_loss"]).backward()  # 对总损失做缩放后反向传播
            scaler.step(generator_optimizer)  # 更新 Generator 参数
            scaler.step(raft_optimizer)  # 更新 RAFT 参数
        else:
            outputs["sr_loss"].backward()  # 常规反向传播
            generator_optimizer.step()  # 更新 Generator 参数
            raft_optimizer.step()  # 更新 RAFT 参数


        # 继续更新RAFT
        raft_optimizer.zero_grad(set_to_none=True)  # 清空 RAFT 梯度
        if scaler is not None:  # AMP 模式下更新  RAFT
            scaler.scale(outputs["raft_loss"]).backward()  # 对总损失做缩放后反向传播
            scaler.step(raft_optimizer)  # 更新 RAFT 参数
        else:
            outputs["raft_loss"].backward()  # 常规反向传播
            raft_optimizer.step()  # 更新 RAFT 参数




        # 再更新 Discriminator
        self._set_requires_grad(self.piv_esrgan_discriminator, True)  # 打开判别器梯度，准备第二阶段训练 D
        d_optimizer.zero_grad(set_to_none=True)  # 清空 D 梯度

        if scaler is not None:  # AMP 模式下更新 D
            scaler.scale(outputs["discriminator_loss"]).backward()  # 反向传播 D 损失
            scaler.step(d_optimizer)  # 更新 D 参数
            scaler.update()  # 一次 train_step 结束后更新 scaler
        else:
            outputs["discriminator_loss"].backward()  # 常规反向传播 D 损失
            d_optimizer.step()  # 更新 D 参数

        return pred_prev,pred_next,flow_predictions,{  # 返回适合直接写日志或进度条的标量字典
            # "total_loss": float(outputs["total_loss"].detach().item()),  # 总损失
            "sr_loss": float(outputs["sr_loss"].detach().item()),  # ESRGAN 侧总损失
            "perceptual_loss": float(outputs["perceptual_loss"].detach().item()),  # 感知损失
            "content_loss": float(outputs["content_loss"].detach().item()),  # 内容损失
            "adversarial_loss": float(outputs["adversarial_loss"].detach().item()),  # 生成器对抗损失
            "pixel_total": float(outputs["pixel_total"].detach().item()),  # 像素总损失
            "pixel_l1": float(outputs["pixel_l1"].detach().item()),  # L1 子项
            "pixel_mse": float(outputs["pixel_mse"].detach().item()),  # MSE 子项
            "pixel_ssim": float(outputs["pixel_ssim"].detach().item()),  # SSIM 子项
            "pixel_fft": float(outputs["pixel_fft"].detach().item()),  # FFT 子项

            "pair_temporal_loss": float(outputs["pair_temporal_loss"].detach().item()),  # 时间一致性总损失
            "pair_delta_loss": float(outputs["pair_delta_loss"].detach().item()),  # 时间一致性中的差分项
            "pair_gradient_loss": float(outputs["pair_gradient_loss"].detach().item()),  # 时间一致性中的梯度项
            "raft_loss": float(outputs["raft_loss"].detach().item()),  # RAFT 损失
            "discriminator_loss": float(outputs["discriminator_loss"].detach().item()),  # 判别器总损失
            "d_real_loss": float(outputs["d_real_loss"].detach().item()),  # 判别器真样本损失
            "d_fake_loss": float(outputs["d_fake_loss"].detach().item()),  # 判别器假样本损失
            "raft_epe": float(outputs["raft_metrics"]["epe"]),  # RAFT 平均 EPE
            "raft_1px": float(outputs["raft_metrics"]["1px"]),  # EPE < 1 像素的比例
            "raft_3px": float(outputs["raft_metrics"]["3px"]),  # EPE < 3 像素的比例
            "raft_5px": float(outputs["raft_metrics"]["5px"]),  # EPE < 5 像素的比例
        }

