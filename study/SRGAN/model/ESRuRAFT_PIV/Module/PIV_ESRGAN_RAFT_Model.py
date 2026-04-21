import torch  # 导入 PyTorch 主库
from torch import nn  # 导入神经网络模块基类和常用层接口
import torch.nn.functional as F  # 导入函数式接口，便于直接调用 loss / 激活等函数

from study.SRGAN.model.ESRuRAFT_PIV.Module.RAFT_Model import RAFT, RAFT128, RAFT256  # 导入 RAFT 光流估计主网络
from study.SRGAN.model.ESRuRAFT_PIV.Module.loss import (
    descriminator_loss,  # 判别器损失，负责训练 D 区分真/假图像
    flow_warp_consistency_loss,  # GT flow 引导的 SR 前后帧 warp 一致性损失
    perceptual_loss,  # 感知损失，内部可选带对抗项

    pixel_loss,  # 像素域复合损失，包含 L1/MSE/SSIM/FFT 等项
)
from study.SRGAN.model.ESRuRAFT_PIV.Module.piv_esrgan_model import Generator, Discriminator  # 导入 ESRGAN 生成器和判别器
from study.SRGAN.model.ESRuRAFT_PIV.global_class import global_data
from study.SRGAN.util.MTL_METHOD import FAMO

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


def _build_piv_raft(batch_size: int) -> nn.Module:
    """
    根据 global_data.esrgan.RAFT_MODEL_TYPE 创建 piv_RAFT。

    这个函数只替代原来硬编码的 RAFT128 实例化，不改变后续 forward / train_step 接口。
    默认 RAFT_MODEL_TYPE="RAFT128"，因此用户不改配置时，行为与原来完全一致。
    """
    raft_model_type = global_data.esrgan.validate_raft_model_type()
    if raft_model_type == "raft":
        return RAFT()
    if raft_model_type == "raft128":
        return RAFT128(upsample=global_data.esrgan.RAFT_UPSAMPLE, batch_size=batch_size)
    if raft_model_type == "raft256":
        return RAFT256(upsample=global_data.esrgan.RAFT_UPSAMPLE, batch_size=batch_size)
    # validate_raft_model_type 已经拦截非法值；这里保留防御式分支，便于未来扩展时定位问题。
    raise ValueError(f"Unsupported RAFT_MODEL_TYPE: {global_data.esrgan.RAFT_MODEL_TYPE}")


class ESRuRAFT_PIV(nn.Module):
    """
    ESRuRAFT_PIV 主网络。

    基本流程:
    1.用改进的esrgan 对低分辨率图像对进行超分辨
    2.然后用基本的RAFT进行piv估计
    """

    def __init__(self,inner_chanel,batch_size):
        super(ESRuRAFT_PIV, self).__init__()  # 调用父类初始化

        self.piv_esrgan_generator = Generator(inner_chanel=inner_chanel)  # 初始化超分生成器，输入 LR 图像对，输出 SR 图像对
        self.piv_esrgan_discriminator = Discriminator(inner_chanel=inner_chanel)  # 初始化判别器，用于区分 SR 图像和真实 HR 图像
        self.piv_RAFT = _build_piv_raft(batch_size=batch_size)  # 按 RAFT_MODEL_TYPE 选择 RAFT / RAFT128 / RAFT256

        # Generator 侧 FAMO 只在 USE_FAMO=True 时启用。
        # 默认保持 None，训练就会完全走手动权重组合，不改变当前实验行为。
        self.generator_famo_task_names = list(global_data.esrgan.FAMO_GENERATOR_TASK_NAMES)
        self.generator_famo = None
        if bool(global_data.esrgan.USE_FAMO):
            self.generator_famo = FAMO(
                n_tasks=len(self.generator_famo_task_names),
                device=global_data.esrgan.device,
                gamma=global_data.esrgan.FAMO_GAMMA,
                w_lr=global_data.esrgan.FAMO_W_LR,
                task_weights=global_data.esrgan.FAMO_GENERATOR_INIT_WEIGHTS,
                max_norm=global_data.esrgan.FAMO_MAX_NORM,
            )



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


    @staticmethod
    def _to_discriminator_gray_frame(image: torch.Tensor) -> torch.Tensor:
        """
        将送入判别器的图像压成单通道灰度。

        你的颗粒图像本质是灰度图复制成 3 通道，因此判别器不需要同时看 RGB 三个重复通道；
        这里固定只取第 0 个通道，后面再拼成方案 C 的 3 通道时序输入。
        """
        if image.dim() != 4:
            raise ValueError(f"Expected image to be 4D [B, C, H, W], got shape={tuple(image.shape)}")
        return image if image.size(1) == 1 else image[:, 0:1, :, :]

    def _build_temporal_discriminator_input(self, prev: torch.Tensor, next_frame: torch.Tensor) -> torch.Tensor:
        """
        构造方案 C 判别器输入: [prev, next, abs(next - prev)]。

        prev 和 next 都先取单通道灰度，所以最终输入仍是 3 通道：
        第 0 通道看前一帧亮度，第 1 通道看后一帧亮度，第 2 通道看帧间亮度变化。
        """
        prev_gray = self._to_discriminator_gray_frame(prev)
        next_gray = self._to_discriminator_gray_frame(next_frame)
        diff_gray = torch.abs(next_gray - prev_gray)
        return torch.cat([prev_gray, next_gray, diff_gray], dim=1)

    def _to_raft_flow_gt(self, flow: torch.Tensor) -> torch.Tensor:
        """
        将数据集中的流场真值转换成 RAFT 监督所需的 2 通道 uv 形式。

        说明：
            数据集里的 flo 常常是三通道 [u, v, magnitude]，
            但 RAFT 的监督和 EPE 计算只使用前两个通道 [u, v]。
        """
        if flow.dim() != 4:  # 检查输入是否是标准 4 维流场张量
            raise ValueError(f"Expected flow to be 4D [B, C, H, W], got shape={tuple(flow.shape)}")
        if flow.size(1) < 2:  # 至少要有 u、v 两个通道
            raise ValueError(f"Expected flow to have at least 2 channels, got shape={tuple(flow.shape)}")
        # 这里显式只保留前两个通道 u / v，避免把 magnitude 一起传给 RAFT loss。
        return flow[:, :2, :, :]


    def _compute_generator_frame_terms(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        计算单帧生成器的基础图像损失项。

        注意：这里不再调用判别器，也不再把对抗项混入感知损失。
        - perceptual_loss: 只等于 VGG feature L1。
        - pixel_*: 保留 L1 / MSE / SSIM / FFT 四个原始子项，后面按全局权重手动组合。
        """
        vgg_loss = perceptual_loss(pred, target)
        pixel_total, pixel_l1, pixel_mse, pixel_ssim, pixel_fft = pixel_loss(pred, target)

        return {
            "perceptual_loss": vgg_loss,
            "vgg_loss": vgg_loss,
            "pixel_total": pixel_total,
            "pixel_l1": pixel_l1,
            "pixel_mse": pixel_mse,
            "pixel_ssim": pixel_ssim,
            "pixel_fft": pixel_fft,
        }

    @staticmethod
    def _mean_loss_term(prev_terms: dict, next_terms: dict, key: str) -> torch.Tensor:
        """
        对前后两帧的同名标量损失做平均。
        """
        return 0.5 * (prev_terms[key] + next_terms[key])  # 对前后两帧的同名损失做平均，保持两帧地位一致


    def _compute_discriminator_loss(self, pred_prev: torch.Tensor, pred_next: torch.Tensor, target_prev: torch.Tensor, target_next: torch.Tensor):
        """
        计算方案 C 的时序判别器损失。

        判别器看到的 fake/real 不再是单帧图像，而是：
        D_input = cat([prev_gray, next_gray, abs(next_gray - prev_gray)], dim=1)。
        这样 D 会同时约束两帧的颗粒外观和帧间亮度变化，而不是只奖励单帧更像 HR。
        """
        fake_pair = self._build_temporal_discriminator_input(pred_prev.detach(), pred_next.detach())
        real_pair = self._build_temporal_discriminator_input(target_prev, target_next)
        pred_fake_d = self.piv_esrgan_discriminator(fake_pair)
        pred_real_d = self.piv_esrgan_discriminator(real_pair)
        return descriminator_loss(pred_fake_d, pred_real_d)


    def _compute_sr_branch(self, input_lr_prev, input_lr_next, input_gr_prev, input_gr_next, flowl0, is_adversarial: bool):
        """
        单独计算 ESRGAN 生成器分支。

        生成器总损失的语义被拆清楚：
        VGG 感知项 + 内容子项(L1/MSE/SSIM/FFT) + 对抗项 + 图像对一致性项。
        ESRuRAFT_PIV 额外的 EPE 项会在 RAFT 分支得到 raft_epe_tensor 后再加入。
        """
        if hasattr(self.piv_esrgan_generator, "forward_pair"):
            pred_prev, pred_next = self.piv_esrgan_generator.forward_pair(input_lr_prev, input_lr_next)
        else:
            pred_prev = self.piv_esrgan_generator(input_lr_prev)
            pred_next = self.piv_esrgan_generator(input_lr_next)

        prev_terms = self._compute_generator_frame_terms(pred_prev, input_gr_prev)
        next_terms = self._compute_generator_frame_terms(pred_next, input_gr_next)

        vgg_loss = self._mean_loss_term(prev_terms, next_terms, "vgg_loss")
        pixel_l1 = self._mean_loss_term(prev_terms, next_terms, "pixel_l1")
        pixel_mse = self._mean_loss_term(prev_terms, next_terms, "pixel_mse")
        pixel_ssim = self._mean_loss_term(prev_terms, next_terms, "pixel_ssim")
        pixel_fft = self._mean_loss_term(prev_terms, next_terms, "pixel_fft")

        # content_loss 是原来的像素总损失语义，但这里显式由四个子项分别按全局权重相加，
        # 不再把它和 VGG 感知项混成一个含糊的 perceptual/content 复合项。
        content_loss = (
            float(global_data.esrgan.LAMBDA_PIXEL_L1) * pixel_l1 +
            float(global_data.esrgan.LAMBDA_PIXEL_MSE) * pixel_mse +
            float(global_data.esrgan.LAMBDA_SSIM) * pixel_ssim +
            float(global_data.esrgan.LAMBDA_PIXEL_FFT) * pixel_fft
        )

        fake_pair = self._build_temporal_discriminator_input(pred_prev, pred_next)
        real_pair = self._build_temporal_discriminator_input(input_gr_prev, input_gr_next)
        pred_fake = self.piv_esrgan_discriminator(fake_pair)
        pred_real = self.piv_esrgan_discriminator(real_pair)
        adversarial_loss = perceptual_loss.adversarial(pred_fake, pred_real)
        adversarial_weighted_loss = (
            float(global_data.esrgan.LAMBDA_ADVERSARIAL) * adversarial_loss
            if is_adversarial else adversarial_loss.new_zeros(())
        )

        _, flow_warp_dict = flow_warp_consistency_loss(pred_prev, pred_next, flowl0)
        flow_warp_consistency = flow_warp_dict["flow_warp_consistency_loss"]
        flow_warp_weighted_loss = float(global_data.esrgan.LAMBDA_FLOW_WARP_CONSISTENCY) * flow_warp_consistency

        manual_sr_loss = (
            float(global_data.esrgan.LAMBDA_VGG) * vgg_loss +
            content_loss +
            adversarial_weighted_loss +
            flow_warp_weighted_loss
        )

        return pred_prev, pred_next, {
            "sr_loss": manual_sr_loss,
            "manual_sr_loss": manual_sr_loss,
            "perceptual_loss": vgg_loss,
            "vgg_loss": vgg_loss,
            "content_loss": content_loss,
            "adversarial_loss": adversarial_loss,
            "adversarial_weighted_loss": adversarial_weighted_loss,
            "pixel_total": content_loss,
            "pixel_l1": pixel_l1,
            "pixel_mse": pixel_mse,
            "pixel_ssim": pixel_ssim,
            "pixel_fft": pixel_fft,
            "flow_warp_consistency_loss": flow_warp_consistency,
            "flow_warp_consistency_weighted_loss": flow_warp_weighted_loss,
        }

    def _compute_raft_branch(self, pred_prev: torch.Tensor, pred_next: torch.Tensor, flowl0, flow_init=None):
        """
        单独计算 RAFT 分支。

        说明：
            这个分支只负责：
            1. 将超分结果转换为 RAFT 输入
            2. 计算 RAFT 的流场预测与 raft_loss
            3. 额外返回可反向传播的 raft_epe_tensor，供 Generator 侧直接使用
        """
        raft_prev = self._to_raft_frame(pred_prev)  # 将前一帧 SR 图转成 RAFT 单通道输入
        raft_next = self._to_raft_frame(pred_next)  # 将后一帧 SR 图转成 RAFT 单通道输入
        raft_input = torch.cat([raft_prev, raft_next], dim=1)  # 拼成 [B, 2, H, W]
        raft_flow_gt = self._to_raft_flow_gt(flowl0)  # 只保留 uv 两个通道作为监督

        flow_predictions, (raft_loss, raft_metrics) = self.piv_RAFT(
            raft_input,  # RAFT 输入图像对
            raft_flow_gt,  # RAFT 光流真值
            flow_init=flow_init,  # 可选初始光流
        )
        # 这里显式重建一个 tensor 形式的平均端点误差，
        # 避免后续误把 raft_metrics['epe'] 这个 Python 标量拿去给 Generator 反向传播。
        raft_epe_tensor = torch.sum((flow_predictions[-1] - raft_flow_gt) ** 2, dim=1).sqrt().mean()

        return flow_predictions, {
            "raft_input_prev": raft_prev,  # 送入 RAFT 前的前一帧单通道图
            "raft_input_next": raft_next,  # 送入 RAFT 前的后一帧单通道图
            "flow_predictions": flow_predictions,  # RAFT 每次迭代得到的流场预测序列
            "raft_loss": raft_loss,  # RAFT 流场序列损失
            "raft_metrics": raft_metrics,  # RAFT 评估指标
            "raft_epe_tensor": raft_epe_tensor,  # 可反向传播的平均 EPE Tensor
        }


    def _generator_famo_task_losses(self, sr_outputs: dict, raft_outputs: dict) -> list[torch.Tensor]:
        """
        按 FAMO 任务顺序返回 Generator 的非对抗损失项。

        ESRuRAFT_PIV 包含 EPE 约束；GAN 对抗损失不放进 FAMO，继续由动态
        LAMBDA_ADVERSARIAL 单独控制，避免早期对抗项把多任务权重带偏。
        """
        return [
            sr_outputs["vgg_loss"],
            sr_outputs["pixel_l1"],
            sr_outputs["pixel_mse"],
            sr_outputs["pixel_ssim"],
            sr_outputs["pixel_fft"],
            sr_outputs["flow_warp_consistency_loss"],
            raft_outputs["raft_epe_tensor"],
        ]

    def _generator_famo_scalar_logs(self) -> dict:
        """把当前 FAMO 权重展开成日志字段，方便训练日志和 CSV 对齐排查。"""
        if self.generator_famo is None:
            return {}
        with torch.no_grad():
            weights = self.generator_famo.weights.detach().cpu().tolist()
        return {f"generator_famo_{name}_weight": float(weight) for name, weight in zip(self.generator_famo_task_names, weights)}

    def _update_generator_famo(self, sr_outputs: dict, raft_outputs: dict) -> dict:
        """
        Generator optimizer.step() 后，按论文方式用更新后的 loss 调整 FAMO 权重。

        这里传入的是重新前向得到的新 loss；FAMO 内部会比较 step 前后的 log loss 变化，
        更新任务 logits，而不是参与 Generator 参数梯度。
        """
        if self.generator_famo is None:
            return {}
        self.generator_famo.update(self._generator_famo_task_losses(sr_outputs, raft_outputs))
        return self._generator_famo_scalar_logs()

    def _compute_generator_loss(self, sr_outputs: dict, raft_outputs: dict) -> tuple[torch.Tensor, dict]:
        """
        计算 ESRuRAFT_PIV 生成器实际反传的总损失。

        USE_FAMO=False 时保持手动权重：SR 手动加权损失 + RAFT_EPE_WEIGHT * EPE。
        USE_FAMO=True 时只让 FAMO 接管非对抗项；对抗损失仍使用动态 LAMBDA_ADVERSARIAL。
        """
        if self.generator_famo is None:
            epe_weighted_loss = float(global_data.esrgan.RAFT_EPE_WEIGHT) * raft_outputs["raft_epe_tensor"]
            manual_loss = sr_outputs["manual_sr_loss"] + epe_weighted_loss
            return manual_loss, {
                "generator_non_adversarial_loss": manual_loss - sr_outputs["adversarial_weighted_loss"],
                "generator_raft_epe_weighted_loss": epe_weighted_loss,
            }

        famo_loss, _ = self.generator_famo.get_weighted_loss(self._generator_famo_task_losses(sr_outputs, raft_outputs))
        generator_loss = famo_loss + sr_outputs["adversarial_weighted_loss"]
        famo_weights = self.generator_famo.weights.to(device=famo_loss.device, dtype=famo_loss.dtype)
        epe_weighted_loss = famo_weights[-1] * raft_outputs["raft_epe_tensor"]
        logs = {
            "generator_non_adversarial_loss": famo_loss,
            "generator_raft_epe_weighted_loss": epe_weighted_loss,
        }
        logs.update(self._generator_famo_scalar_logs())
        return generator_loss, logs

    def forward(
        self,
        input_lr_prev,
        input_lr_next,
        input_gr_prev,
        input_gr_next,
        flowl0,
        flow_init=None,
        upsample=global_data.esrgan.RAFT_UPSAMPLE,
        is_adversarial: bool = False,
    ):
        """
        前向传播：返回 SR、RAFT 预测和所有主要损失。

        这里复用 train_step 的分支计算逻辑，保证 forward 日志和真实训练时的损失定义一致。
        """
        pred_prev, pred_next, sr_outputs = self._compute_sr_branch(
            input_lr_prev, input_lr_next, input_gr_prev, input_gr_next, flowl0, is_adversarial
        )
        flow_predictions, raft_outputs = self._compute_raft_branch(pred_prev, pred_next, flowl0, flow_init=flow_init)
        g_loss, generator_loss_logs = self._compute_generator_loss(sr_outputs, raft_outputs)
        total_loss = g_loss + raft_outputs["raft_loss"]
        discriminator_loss, d_fake_loss, d_real_loss = self._compute_discriminator_loss(
            pred_prev, pred_next, input_gr_prev, input_gr_next
        )

        return pred_prev, pred_next, flow_predictions, {
            "sr_prev": pred_prev,
            "sr_next": pred_next,
            "raft_input_prev": raft_outputs["raft_input_prev"],
            "raft_input_next": raft_outputs["raft_input_next"],
            "flow_predictions": flow_predictions,
            "sr_loss": sr_outputs["manual_sr_loss"],
            "g_loss": g_loss,
            "perceptual_loss": sr_outputs["perceptual_loss"],
            "content_loss": sr_outputs["content_loss"],
            "adversarial_loss": sr_outputs["adversarial_loss"],
            "adversarial_weighted_loss": sr_outputs["adversarial_weighted_loss"],
            "pixel_total": sr_outputs["pixel_total"],
            "pixel_l1": sr_outputs["pixel_l1"],
            "pixel_mse": sr_outputs["pixel_mse"],
            "pixel_ssim": sr_outputs["pixel_ssim"],
            "pixel_fft": sr_outputs["pixel_fft"],
            "flow_warp_consistency_loss": sr_outputs["flow_warp_consistency_loss"],
            "flow_warp_consistency_weighted_loss": sr_outputs["flow_warp_consistency_weighted_loss"],
            "raft_loss": raft_outputs["raft_loss"],
            "raft_metrics": raft_outputs["raft_metrics"],
            "raft_epe_tensor": raft_outputs["raft_epe_tensor"],
            "discriminator_loss": discriminator_loss,
            "d_fake_loss": d_fake_loss,
            "d_real_loss": d_real_loss,
            "total_loss": total_loss,
            **generator_loss_logs,
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
        upsample=global_data.esrgan.RAFT_UPSAMPLE,
        is_adversarial: bool = False,
    ) -> dict:
        """
        执行一次完整训练步骤。

        训练顺序:
            1. 前向计算全部损失项
            2. 使用单次 RAFT forward，同时完成 Generator 与 RAFT 的双 backward
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
            upsample: 上采样模式
            is_adversarial: 是否在生成器损失中启用对抗项

        返回:
            包含本次训练主要标量的字典
        """
        # 第一阶段：一次前向同时准备 Generator 和 RAFT 的训练目标。
        self._set_requires_grad(self.piv_esrgan_generator, True)  # Generator 需要拿到 sr_loss 和 raft_epe 的梯度
        self._set_requires_grad(self.piv_RAFT, True)  # 同一次 RAFT forward 也要服务于后续 RAFT 自身更新
        self._set_requires_grad(self.piv_esrgan_discriminator, False)  # 训练 G/RAFT 时冻结判别器
        generator_optimizer.zero_grad(set_to_none=True)  # 清空 Generator 梯度
        raft_optimizer.zero_grad(set_to_none=True)  # 清空 RAFT 梯度

        pred_prev, pred_next, sr_outputs = self._compute_sr_branch(
            input_lr_prev=input_lr_prev,
            input_lr_next=input_lr_next,
            input_gr_prev=input_gr_prev,
            input_gr_next=input_gr_next,
            flowl0=flowl0,
            is_adversarial=is_adversarial,
        )
        flow_predictions, raft_outputs = self._compute_raft_branch(
            pred_prev=pred_prev,
            pred_next=pred_next,
            flowl0=flowl0,
            flow_init=flow_init,
        )
        raft_epe_weight = float(global_data.esrgan.RAFT_EPE_WEIGHT)
        generator_g_loss, generator_loss_logs = self._compute_generator_loss(sr_outputs, raft_outputs)
        final_flow_prediction = flow_predictions[-1]

        # 先让 Generator 拿到 sr_loss + 加权 raft_epe 的梯度，
        # 同时保留计算图供 RAFT 再对 raft_loss 单独反向传播。
        if scaler is not None:
            scaled_g_loss = scaler.scale(generator_g_loss)
            scaled_g_loss.backward(retain_graph=True)
        else:
            generator_g_loss.backward(retain_graph=True)

        # 保存 Generator 当前梯度，后面会把第二次 backward 产生的 Generator 梯度清掉，
        # 从而保持“Generator 只吃 sr_loss + 加权 raft_epe，RAFT 只吃 raft_loss”的分离训练语义。
        generator_saved_grads = []
        for param in self.piv_esrgan_generator.parameters():
            if param.grad is None:
                generator_saved_grads.append(None)
            else:
                generator_saved_grads.append(param.grad.detach().clone())

        # Generator 侧的损失经过同一次 RAFT forward 也会在 RAFT 参数上留下梯度，
        # 这里显式清掉，避免后续 raft_optimizer.step() 混入不该有的更新信号。
        for param in self.piv_RAFT.parameters():
            param.grad = None

        # 第二次 backward 只让 RAFT 吃到 raft_loss；
        # Generator 侧即使在图上有梯度路径，最终也会恢复成第一次 backward 保存下来的梯度。
        if scaler is not None:
            scaler.scale(raft_outputs["raft_loss"]).backward()
        else:
            raft_outputs["raft_loss"].backward()

        # 恢复 Generator 梯度，确保 raft_loss 不会反向影响 Generator 的参数更新。
        for param, saved_grad in zip(self.piv_esrgan_generator.parameters(), generator_saved_grads):
            param.grad = saved_grad

        if scaler is not None:
            scaler.step(generator_optimizer)  # Generator 只使用第一次 backward 保留下来的梯度更新
            scaler.step(raft_optimizer)  # RAFT 只使用第二次 backward 后的梯度更新
        else:
            generator_optimizer.step()
            raft_optimizer.step()

        # 如果启用了 FAMO，则在 Generator 参数更新后重新计算一次各任务 loss，
        # 再按论文 update(curr_loss) 调整下一次迭代使用的任务权重。
        if self.generator_famo is not None and bool(global_data.esrgan.FAMO_UPDATE_AFTER_STEP):
            with torch.no_grad():
                pred_prev_famo, pred_next_famo, sr_outputs_famo = self._compute_sr_branch(
                    input_lr_prev=input_lr_prev,
                    input_lr_next=input_lr_next,
                    input_gr_prev=input_gr_prev,
                    input_gr_next=input_gr_next,
                    flowl0=flowl0,
                    is_adversarial=is_adversarial,
                )
                _, raft_outputs_famo = self._compute_raft_branch(
                    pred_prev=pred_prev_famo,
                    pred_next=pred_next_famo,
                    flowl0=flowl0,
                    flow_init=flow_init,
                )
            generator_loss_logs.update(self._update_generator_famo(sr_outputs_famo, raft_outputs_famo))

        # 第三阶段：只更新 Discriminator，严格只使用 discriminator_loss。
        self._set_requires_grad(self.piv_esrgan_generator, False)  # 继续冻结 Generator
        self._set_requires_grad(self.piv_RAFT, False)  # 冻结 RAFT
        self._set_requires_grad(self.piv_esrgan_discriminator, True)  # 只开启判别器梯度
        d_optimizer.zero_grad(set_to_none=True)  # 清空判别器梯度

        if hasattr(self.piv_esrgan_generator, "forward_pair"):
            pred_prev_d, pred_next_d = self.piv_esrgan_generator.forward_pair(input_lr_prev, input_lr_next)
        else:
            pred_prev_d = self.piv_esrgan_generator(input_lr_prev)
            pred_next_d = self.piv_esrgan_generator(input_lr_next)

        discriminator_loss, d_fake_loss, d_real_loss = self._compute_discriminator_loss(
            pred_prev_d,
            pred_next_d,
            input_gr_prev,
            input_gr_next,
        )

        if scaler is not None:
            scaler.scale(discriminator_loss).backward()
            scaler.step(d_optimizer)
            scaler.update()
        else:
            discriminator_loss.backward()
            d_optimizer.step()

        # 恢复三个子模块的梯度开关，避免影响外部后续逻辑。
        self._set_requires_grad(self.piv_esrgan_generator, True)
        self._set_requires_grad(self.piv_RAFT, True)
        self._set_requires_grad(self.piv_esrgan_discriminator, True)

        return pred_prev, pred_next, final_flow_prediction, {
            "sr_loss": float(sr_outputs["sr_loss"].detach().item()),  # ESRGAN 原始 SR 总损失（未叠加 raft_epe）
            "g_loss": float(generator_g_loss.detach().item()),  # Generator 实际回传的总损失
            "perceptual_loss": float(sr_outputs["perceptual_loss"].detach().item()),
            "content_loss": float(sr_outputs["content_loss"].detach().item()),
            "adversarial_loss": float(sr_outputs["adversarial_loss"].detach().item()),
            "pixel_total": float(sr_outputs["pixel_total"].detach().item()),
            "pixel_l1": float(sr_outputs["pixel_l1"].detach().item()),
            "pixel_mse": float(sr_outputs["pixel_mse"].detach().item()),
            "pixel_ssim": float(sr_outputs["pixel_ssim"].detach().item()),
            "pixel_fft": float(sr_outputs["pixel_fft"].detach().item()),
            "flow_warp_consistency_loss": float(sr_outputs["flow_warp_consistency_loss"].detach().item()),
            "flow_warp_consistency_weighted_loss": float(sr_outputs["flow_warp_consistency_weighted_loss"].detach().item()),
            "raft_loss": float(raft_outputs["raft_loss"].detach().item()),
            "discriminator_loss": float(discriminator_loss.detach().item()),
            "d_real_loss": float(d_real_loss.detach().item()),
            "d_fake_loss": float(d_fake_loss.detach().item()),
            "raft_epe": float(raft_outputs["raft_metrics"]["epe"]),
            "generator_raft_epe": float(raft_outputs["raft_epe_tensor"].detach().item()),
            "raft_epe_weight": raft_epe_weight,
            "adversarial_weighted_loss": float(sr_outputs["adversarial_weighted_loss"].detach().item()),
            **{key: float(value.detach().item()) if torch.is_tensor(value) else float(value) for key, value in generator_loss_logs.items()},
            "raft_1px": float(raft_outputs["raft_metrics"]["1px"]),
            "raft_3px": float(raft_outputs["raft_metrics"]["3px"]),
            "raft_5px": float(raft_outputs["raft_metrics"]["5px"]),
        }

