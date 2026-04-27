import torch  # 导入 PyTorch 主库
from torch import nn  # 导入神经网络模块基类和常用层接口
import torch.nn.functional as F  # 导入函数式接口，便于直接调用 loss / 激活等函数
from pathlib import Path  # 用于解析 RAFT256 checkpoint 的相对路径

from loguru import logger
from study.SRGAN.model.ESRuRAFT_PIV_Ground.Module.RAFT_Model import RAFT, RAFT128, RAFT256  # 导入 RAFT 光流估计主网络
from study.SRGAN.model.ESRuRAFT_PIV_Ground.Module.loss import (
    descriminator_loss,  # 判别器损失，负责训练 D 区分真/假图像
    flow_warp_consistency_loss,  # GT flow 引导的 SR 前后帧 warp 一致性损失
    perceptual_loss,  # 感知损失，内部可选带对抗项

    pixel_loss,  # 像素域复合损失，包含 L1/MSE/SSIM/FFT 等项
)
from study.SRGAN.model.ESRuRAFT_PIV_Ground.Module.original_esrgan_model import (
    Discriminator as OriginalESRGANDiscriminator,
    Generator as OriginalESRGANGenerator,
)  # TRAIN_MODE="esrgan_raft" 使用最原始 ESRGAN 结构，不使用 ESRuRAFT_PIV 的改进双帧生成器
from study.SRGAN.model.basic_srgan.Module.model import (
    Discriminator as SRGANDiscriminator,
    Generator as SRGANGenerator,
)  # TRAIN_MODE="srgan_raft" 使用传统 SRGAN 生成器，然后接 RAFT
from study.SRGAN.model.ESRuRAFT_PIV_Ground.global_class import global_data

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


def _resolve_raft256_pretrain_path(path_text: str | Path) -> Path:
    """
    解析 RAFT256 预训练 checkpoint 路径。

    配置中默认写成 "RAFT_CHECKPOINT/ckpt_256.tar"，这是相对 SRGAN 根目录的路径；
    如果用户传入绝对路径，则直接使用绝对路径。这样从 PyCharm、命令行或 notebook
    不同工作目录启动时，都不会因为 cwd 不同而找错 checkpoint。
    """
    path = Path(path_text)
    if path.is_absolute():
        return path
    srgan_root = Path(__file__).resolve().parents[3]
    return srgan_root / path


def _init_raft128_from_raft256_if_enabled(model: nn.Module) -> None:
    """
    可选：用 RAFT256 checkpoint 中 shape 一致的权重初始化 RAFT128。

    这是一个“预训练初始化”而不是无损结构迁移：
    - RAFT256 的 update_block.mask.2 输出通道是 8*8*9=576；
    - RAFT128 的 update_block.mask.2 输出通道是 4*4*9=144；
    - 这类 shape 不一致的层必须跳过，保留 RAFT128 自身初始化。

    因此该功能只在：
    1. global_data.esrgan.RAFT128_INIT_FROM_RAFT256=True；
    2. global_data.esrgan.RAFT_MODEL_TYPE="RAFT128"；
    时生效。迁移后建议继续训练/微调，让 RAFT128 适应 1/4 分辨率的内部特征网格。
    """
    if not bool(global_data.esrgan.RAFT128_INIT_FROM_RAFT256):
        return

    checkpoint_path = _resolve_raft256_pretrain_path(global_data.esrgan.RAFT128_INIT_FROM_RAFT256_CKPT)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"RAFT256 预训练 checkpoint 不存在: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source_state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else None
    if not isinstance(source_state, dict):
        raise ValueError(f"无法从 checkpoint 中读取 RAFT256 state_dict: {checkpoint_path}")

    target_state = model.state_dict()
    matched_state = {}
    skipped = []
    for name, value in source_state.items():
        target_value = target_state.get(name)
        if target_value is not None and hasattr(value, "shape") and target_value.shape == value.shape:
            matched_state[name] = value
        else:
            source_shape = tuple(value.shape) if hasattr(value, "shape") else type(value).__name__
            target_shape = tuple(target_value.shape) if target_value is not None else None
            skipped.append((name, source_shape, target_shape))

    # target_state 本身已经包含 RAFT128 的完整随机初始化权重。
    # 这里只覆盖 shape 完全相同的层；被跳过的层保持 RAFT128 初始化，相当于“重建”不兼容层。
    target_state.update(matched_state)
    model.load_state_dict(target_state)

    logger.info(
        "[ESRuRAFT_PIV_Ground] RAFT128 initialized from RAFT256 checkpoint | "
        f"checkpoint={checkpoint_path}, loaded={len(matched_state)}, skipped={len(skipped)}"
    )
    for name, source_shape, target_shape in skipped[:20]:
        logger.info(
            "[ESRuRAFT_PIV_Ground] skipped RAFT256->RAFT128 weight | "
            f"name={name}, source_shape={source_shape}, target_shape={target_shape}"
        )
    if len(skipped) > 20:
        logger.info(f"[ESRuRAFT_PIV_Ground] skipped weights truncated: {len(skipped) - 20} more")


def _build_piv_raft(batch_size: int) -> nn.Module:
    """
    根据 global_data.esrgan.RAFT_MODEL_TYPE 创建 piv_RAFT。

    这个函数只替代原来硬编码的 RAFT256 实例化，不改变 Ground 分支的 TRAIN_MODE 逻辑。
    默认 RAFT_MODEL_TYPE="RAFT256"，因此用户不改配置时，行为与原来完全一致。
    """
    raft_model_type = global_data.esrgan.validate_raft_model_type()
    if raft_model_type == "raft":
        return RAFT()
    if raft_model_type == "raft128":
        model = RAFT128(upsample=global_data.esrgan.RAFT_UPSAMPLE, batch_size=batch_size)
        _init_raft128_from_raft256_if_enabled(model)
        return model
    if raft_model_type == "raft256":
        return RAFT256(upsample=global_data.esrgan.RAFT_UPSAMPLE, batch_size=batch_size)
    # validate_raft_model_type 已经拦截非法值；这里保留防御式分支，便于未来扩展时定位问题。
    raise ValueError(f"Unsupported RAFT_MODEL_TYPE: {global_data.esrgan.RAFT_MODEL_TYPE}")


def _resolve_generator_pixel_shuffle_scale(sr_scale=None) -> int:
    """
    解析 ESRGAN/SRGAN 生成器每一级 PixelShuffle 的上采样倍率。

    这里不要在 Generator 输出后再插值补尺寸；正确做法是让网络结构本身输出目标尺寸。
    本工程的目录和数据倍率使用 int(SCALE * SCALE)，而 ESRGAN/SRGAN 生成器内部固定有两级
    PixelShuffle，因此：
        - SCALE=2  -> 两级 PixelShuffle(2)，总上采样 x4，对应 lr_data_root_dir/x4；
        - SCALE=4  -> 两级 PixelShuffle(4)，总上采样 x16，对应 lr_data_root_dir/x16。

    sr_scale 由 pipeline 当前循环里的 SCALE 传入；如果外部旧代码没有传，则退回到
    global_data.esrgan.SCALES[0]，保持旧实例化方式仍能工作。
    """
    if sr_scale is None:
        configured_scales = getattr(global_data.esrgan, "SCALES", (2,))
        sr_scale = configured_scales[0] if configured_scales else 2

    scale_value = float(sr_scale)
    scale_int = int(round(scale_value))
    if scale_int < 1 or abs(scale_value - scale_int) > 1e-6:
        raise ValueError(
            "ESRuRAFT_PIV_Ground 的 esrgan_raft/srgan_raft 生成器使用 PixelShuffle，"
            f"每级上采样倍率必须是整数；当前 SCALE={sr_scale}。"
        )
    return scale_int


class ESRuRAFT_PIV(nn.Module):
    """
    ESRuRAFT_PIV_Ground 主网络。

    Ground 版本和 ESRuRAFT_PIV 的训练/评估外壳保持一致，但通过 TRAIN_MODE 切换图像来源和 PIV 估计器：
    1. lr_ground_raft: LR 最近邻对齐到 HR 后送入 RAFT。
    2. hr_ground_raft: HR 真值图像直接送入 RAFT。
    3. bicubic_raft: LR 经 bicubic 上采样后送入 RAFT。
    4. esrgan_raft: 原始 ESRGAN 超分后送入 RAFT。
    5. bicubic_widim: LR 经 bicubic 上采样后进入传统 WIDIM/窗口互相关 PIV。
    6. bicubic_hs: LR 经 bicubic 上采样后进入 Horn-Schunck 光流法。
    7. srgan_raft: 传统 SRGAN 超分后送入 RAFT。
    """

    def __init__(self,inner_chanel,batch_size,sr_scale=None):
        super(ESRuRAFT_PIV, self).__init__()  # 调用父类初始化

        self.train_mode = global_data.esrgan.validate_train_mode()
        self.generator_pixel_shuffle_scale = _resolve_generator_pixel_shuffle_scale(sr_scale)
        self.generator_total_upscale = self.generator_pixel_shuffle_scale * self.generator_pixel_shuffle_scale
        if self.train_mode == "esrgan_raft":
            # esrgan_raft 模式明确要求“超分模块换成最原始的 ESRGAN”。
            # 因此这里使用从 model/esrgan/Module/model.py 复制过来的原始单帧 Generator/Discriminator，
            # 不再使用 ESRuRAFT_PIV 的双帧特征融合生成器。
            # scale 控制每一级 PixelShuffle 的放大倍率；两级之后总倍率就是 scale*scale。
            # 例如当前数据使用 x16 LR 时，pipeline 传入 SCALE=4，这里会构造两级 PixelShuffle(4)，
            # 网络自身直接输出 HR 尺寸，而不是在输出后用插值补尺寸。
            self.piv_esrgan_generator = OriginalESRGANGenerator(
                inner_chanel=inner_chanel,
                scale=self.generator_pixel_shuffle_scale,
            )
            self.piv_esrgan_discriminator = OriginalESRGANDiscriminator(inner_chanel=inner_chanel)
        elif self.train_mode == "srgan_raft":
            # srgan_raft 使用传统 SRGAN 生成器，再把 SR 图像送入 RAFT。
            # 这样可以和 esrgan_raft 的 RRDB/ESRGAN 生成器做结构基线对比。
            # 与 esrgan_raft 保持相同的尺度定义：两级 PixelShuffle，每级倍率来自当前 SCALE。
            self.piv_esrgan_generator = SRGANGenerator(
                inner_chanel=inner_chanel,
                scale=self.generator_pixel_shuffle_scale,
            )
            self.piv_esrgan_discriminator = SRGANDiscriminator(inner_chanel=inner_chanel)
        else:
            # ground / bicubic / WIDIM / HS 是无学习型 SR 生成器的 baseline。
            # 仍然保留同名属性，是为了让 pipeline/evaluate 里已有的统一接口不需要大面积分叉。
            self.piv_esrgan_generator = nn.Identity()
            self.piv_esrgan_discriminator = nn.Identity()
        self.piv_RAFT = _build_piv_raft(batch_size=batch_size)  # 按 RAFT_MODEL_TYPE 选择 RAFT / RAFT128 / RAFT256



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

    def _uses_super_resolution(self) -> bool:
        """
        当前模式是否需要 Generator/Discriminator 参与训练。

        只有 esrgan_raft / srgan_raft 才有可学习的超分模块；其余 baseline 模式不更新 G/D。
        """
        return self.train_mode in {"esrgan_raft", "srgan_raft"}

    def _uses_traditional_piv(self) -> bool:
        """
        当前模式是否使用传统 PIV/光流估计器而不是 RAFT。

        WIDIM/HS 没有可学习参数，也不需要 optimizer.step；训练循环中只计算预测和指标。
        """
        return self.train_mode in {"bicubic_widim", "bicubic_hs"}

    @staticmethod
    def _zero_like_loss(reference: torch.Tensor) -> torch.Tensor:
        """
        构造一个位于同一 device/dtype 的 0 标量。

        ground 模式没有 SR/GAN 损失，但训练日志仍然沿用 ESRuRAFT_PIV 的字段。
        用张量 0 而不是 Python 0，可以保证后续 .detach().item() 和 AMP 分支都能稳定工作。
        """
        return reference.new_zeros(())

    @staticmethod
    def _resize_image_to_target(
        image: torch.Tensor,
        target: torch.Tensor,
        mode: str = "nearest",
    ) -> torch.Tensor:
        """
        将图像 resize 到 target 的空间大小，供非学习型 baseline 使用。

        参数:
            image: 需要放大的 LR 图像，例如 [B, C, 64, 64]。
            target: 尺寸参考图像，例如真实 HR 图像 [B, C, 256, 256]。
            mode:
                - "nearest": 最近邻插值，用于 lr_ground_raft，表示只做最朴素的尺寸对齐；
                - "bicubic": bicubic 双三次插值，用于传统超分 baseline。

        注意:
            这里完全不包含可学习参数，也不会调用 Generator。
            它只是把 LR 图像变成和 HR/flow 监督一致的空间尺寸，让 RAFT 输入保持 256x256。
        """
        if image.shape[-2:] == target.shape[-2:]:
            return image

        mode = str(mode).strip().lower()
        if mode == "nearest":
            # 最近邻插值不需要 align_corners 参数；传入反而会触发 PyTorch 报错。
            return F.interpolate(image, size=target.shape[-2:], mode="nearest")
        if mode == "bicubic":
            # bicubic 是传统图像超分/放大里常用的双三次插值。
            # align_corners=False 是 PyTorch 插值推荐的稳定默认口径，也和大多数 resize 实现更接近。
            return F.interpolate(image, size=target.shape[-2:], mode="bicubic", align_corners=False)
        raise ValueError(f"Unsupported resize mode: {mode}")

    @staticmethod
    def _resize_flow_to_match_image(flow_uv: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        将 HR flow 真值缩放到 RAFT 输入图像的空间大小。

        如果 RAFT 输入尺寸和 flow 真值尺寸不同，预测位移单位也会跟着图像尺寸变化。
        因此不只是 resize flow 场，还必须同步缩放 u/v 数值：
        - u 按 width 比例缩放；
        - v 按 height 比例缩放。
        这样 RAFT 的训练损失才和当前输入分辨率一致。
        """
        if flow_uv.shape[-2:] == image.shape[-2:]:
            return flow_uv
        src_h, src_w = flow_uv.shape[-2:]
        dst_h, dst_w = image.shape[-2:]
        resized = F.interpolate(flow_uv, size=(dst_h, dst_w), mode="bilinear", align_corners=True)
        resized = resized.clone()
        resized[:, 0:1] = resized[:, 0:1] * (dst_w / max(src_w, 1))
        resized[:, 1:2] = resized[:, 1:2] * (dst_h / max(src_h, 1))
        return resized

    @staticmethod
    def _restore_flow_to_target_size(flow_uv: torch.Tensor, target_flow_uv: torch.Tensor) -> torch.Tensor:
        """
        将 RAFT 输出还原到原始 flow 真值大小，并同步恢复 u/v 位移单位。

        evaluate/evaluate_all 现有逻辑都用原始 flow 真值尺寸做统一指标与保存。
        如果某个模式下 RAFT 输入尺寸和 flow 真值不同，这里把预测流场还原回真值尺寸，
        保证外部评估接口不需要额外分叉。
        """
        if flow_uv.shape[-2:] == target_flow_uv.shape[-2:]:
            return flow_uv
        src_h, src_w = flow_uv.shape[-2:]
        dst_h, dst_w = target_flow_uv.shape[-2:]
        restored = F.interpolate(flow_uv, size=(dst_h, dst_w), mode="bilinear", align_corners=True)
        restored = restored.clone()
        restored[:, 0:1] = restored[:, 0:1] * (dst_w / max(src_w, 1))
        restored[:, 1:2] = restored[:, 1:2] * (dst_h / max(src_h, 1))
        return restored

    @staticmethod
    def _standardize_frame_for_traditional_piv(frame: torch.Tensor) -> torch.Tensor:
        """
        给传统 PIV/光流法使用的亮度标准化。

        WIDIM 互相关和 Horn-Schunck 都依赖亮度变化；不同 batch 的颗粒图强度范围可能略有差异。
        这里按每张图自身的均值/方差做标准化，提升互相关和梯度法的数值稳定性。
        """
        mean = frame.mean(dim=(-2, -1), keepdim=True)
        std = frame.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return (frame - mean) / std

    @staticmethod
    def _sample_with_integer_displacement(image: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
        """
        按整数位移采样后一帧图像，用于窗口互相关搜索。

        约定：
            shifted[..., y, x] = image[..., y + dy, x + dx]

        因此如果后一帧相对前一帧向右移动 2 像素，最佳 dx 会接近 +2。
        边界之外用 0 填充，避免 torch.roll 的环绕伪影污染边缘窗口。
        """
        shifted = torch.zeros_like(image)
        _, _, height, width = image.shape

        src_y0 = max(dy, 0)
        src_y1 = min(height + dy, height)
        dst_y0 = max(-dy, 0)
        dst_y1 = dst_y0 + max(src_y1 - src_y0, 0)

        src_x0 = max(dx, 0)
        src_x1 = min(width + dx, width)
        dst_x0 = max(-dx, 0)
        dst_x1 = dst_x0 + max(src_x1 - src_x0, 0)

        if src_y1 > src_y0 and src_x1 > src_x0:
            shifted[:, :, dst_y0:dst_y1, dst_x0:dst_x1] = image[:, :, src_y0:src_y1, src_x0:src_x1]
        return shifted

    def _estimate_widim_flow(self, prev: torch.Tensor, next_frame: torch.Tensor) -> torch.Tensor:
        """
        传统 WIDIM/窗口互相关 PIV baseline。

        说明：
            完整 WIDIM 会包含多级窗口形变和亚像素峰值拟合；这里实现的是工程内可复现、
            无外部依赖的核心互相关版本：在重叠 interrogation window 上做整数位移搜索，
            再把粗网格位移场插值成全分辨率 flow。它用于和 RAFT/SR+RAFT 做传统 PIV 基线对比。
        """
        prev = self._standardize_frame_for_traditional_piv(prev.detach())
        next_frame = self._standardize_frame_for_traditional_piv(next_frame.detach())

        window_size = int(global_data.esrgan.WIDIM_WINDOW_SIZE)
        stride = int(global_data.esrgan.WIDIM_STRIDE)
        search_radius = int(global_data.esrgan.WIDIM_SEARCH_RADIUS)
        padding = window_size // 2

        best_score = None
        best_dx = None
        best_dy = None
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                shifted_next = self._sample_with_integer_displacement(next_frame, dx=dx, dy=dy)
                # 以窗口平均乘积作为互相关分数；输入已标准化，分数越大表示局部匹配越好。
                score = F.avg_pool2d(prev * shifted_next, kernel_size=window_size, stride=stride, padding=padding)
                if best_score is None:
                    best_score = score
                    best_dx = torch.full_like(score, float(dx))
                    best_dy = torch.full_like(score, float(dy))
                    continue
                update_mask = score > best_score
                best_score = torch.where(update_mask, score, best_score)
                best_dx = torch.where(update_mask, torch.full_like(best_dx, float(dx)), best_dx)
                best_dy = torch.where(update_mask, torch.full_like(best_dy, float(dy)), best_dy)

        coarse_flow = torch.cat([best_dx, best_dy], dim=1)
        return F.interpolate(coarse_flow, size=prev.shape[-2:], mode="bilinear", align_corners=True)

    def _estimate_horn_schunck_flow(self, prev: torch.Tensor, next_frame: torch.Tensor) -> torch.Tensor:
        """
        传统 Horn-Schunck 光流 baseline。

        该方法假设亮度守恒并通过全局平滑项约束 flow，适合作为“非学习光流法”的参考。
        输出通道同 RAFT 保持一致：[u, v]。
        """
        prev = self._standardize_frame_for_traditional_piv(prev.detach())
        next_frame = self._standardize_frame_for_traditional_piv(next_frame.detach())
        avg_frame = 0.5 * (prev + next_frame)

        sobel_x = avg_frame.new_tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
        ).view(1, 1, 3, 3) / 8.0
        sobel_y = avg_frame.new_tensor(
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]
        ).view(1, 1, 3, 3) / 8.0
        smooth_kernel = avg_frame.new_tensor(
            [[[1.0 / 12.0, 1.0 / 6.0, 1.0 / 12.0],
              [1.0 / 6.0, 0.0, 1.0 / 6.0],
              [1.0 / 12.0, 1.0 / 6.0, 1.0 / 12.0]]]
        ).view(1, 1, 3, 3)

        ix = F.conv2d(avg_frame, sobel_x, padding=1)
        iy = F.conv2d(avg_frame, sobel_y, padding=1)
        it = next_frame - prev

        alpha = float(global_data.esrgan.HS_ALPHA)
        iterations = int(global_data.esrgan.HS_ITERS)
        u = torch.zeros_like(prev)
        v = torch.zeros_like(prev)
        denom = alpha * alpha + ix * ix + iy * iy + 1e-6

        for _ in range(iterations):
            u_avg = F.conv2d(u, smooth_kernel, padding=1)
            v_avg = F.conv2d(v, smooth_kernel, padding=1)
            residual = ix * u_avg + iy * v_avg + it
            u = u_avg - ix * residual / denom
            v = v_avg - iy * residual / denom

        return torch.cat([u, v], dim=1)

    @staticmethod
    def _flow_metrics_from_prediction(pred_flow_uv: torch.Tensor, target_flow_uv: torch.Tensor) -> dict:
        """
        用最终预测和目标 flow 重新计算 RAFT 常用指标。

        RAFT 内部 sequence_loss 的 metrics 是在训练损失尺寸上算的；这里用最终对齐到真值尺寸的
        预测重新计算一遍指标，保证所有模式对外暴露的是同一个 flow 尺寸口径。
        """
        epe_map = torch.sum((pred_flow_uv - target_flow_uv) ** 2, dim=1).sqrt()
        flat_epe = epe_map.reshape(-1)
        return {
            "epe": flat_epe.mean().item(),
            "1px": (flat_epe < 1).float().mean().item(),
            "3px": (flat_epe < 3).float().mean().item(),
            "5px": (flat_epe < 5).float().mean().item(),
        }

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
        if pred.shape != target.shape:
            # 这里选择直接报清晰错误，而不是偷偷插值修正。
            # esrgan_raft/srgan_raft 的 SR 尺寸应该由 Generator 的 PixelShuffle 结构决定；
            # 如果这里不一致，说明当前 pipeline 传入的 SCALE、LR 数据目录 x{SCALE*SCALE}、
            # 或 Generator 的 inner_chanel/scale 配置没有对上。
            raise ValueError(
                "[ESRuRAFT_PIV_Ground] Generator output shape does not match HR target: "
                f"pred_shape={tuple(pred.shape)}, target_shape={tuple(target.shape)}, "
                f"TRAIN_MODE={self.train_mode}, "
                f"pixel_shuffle_scale={self.generator_pixel_shuffle_scale}, "
                f"total_upscale={self.generator_total_upscale}. "
                "请检查 SCALES 与 LR_DATA_ROOT_DIR/x{int(SCALE*SCALE)} 是否和数据真实倍率一致。"
            )
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


    def _compute_ground_image_terms(self, pred_prev: torch.Tensor, pred_next: torch.Tensor, input_gr_prev: torch.Tensor, input_gr_next: torch.Tensor) -> dict:
        """
        计算 ground 模式的图像诊断项。

        lr_ground_raft/hr_ground_raft/bicubic_raft/bicubic_widim/bicubic_hs 没有 Generator，不应该对图像损失反传；但保留这些数值日志有两个好处：
        1. CSV 字段继续和 ESRuRAFT_PIV 对齐；
        2. 可以直观看到最近邻 LR、bicubic LR 或 HR ground 图像与真实 HR 的图像差异。
        """
        # ground 模式没有 Generator，训练时不需要 VGG feature loss 参与反传。
        # 这里只保留像素/SSIM/FFT 诊断项，避免每个 batch 额外跑 VGG 带来不必要的基线开销。
        _, prev_l1, prev_mse, prev_ssim, prev_fft = pixel_loss(pred_prev, input_gr_prev)
        _, next_l1, next_mse, next_ssim, next_fft = pixel_loss(pred_next, input_gr_next)
        pixel_l1 = 0.5 * (prev_l1 + next_l1)
        pixel_mse = 0.5 * (prev_mse + next_mse)
        pixel_ssim = 0.5 * (prev_ssim + next_ssim)
        pixel_fft = 0.5 * (prev_fft + next_fft)
        content_loss = (
            float(global_data.esrgan.LAMBDA_PIXEL_L1) * pixel_l1 +
            float(global_data.esrgan.LAMBDA_PIXEL_MSE) * pixel_mse +
            float(global_data.esrgan.LAMBDA_SSIM) * pixel_ssim +
            float(global_data.esrgan.LAMBDA_PIXEL_FFT) * pixel_fft
        )
        zero = self._zero_like_loss(content_loss)
        return {
            "sr_loss": content_loss,
            "manual_sr_loss": content_loss,
            "perceptual_loss": zero,
            "vgg_loss": zero,
            "content_loss": content_loss,
            "adversarial_loss": zero,
            "adversarial_weighted_loss": zero,
            "pixel_total": content_loss,
            "pixel_l1": pixel_l1,
            "pixel_mse": pixel_mse,
            "pixel_ssim": pixel_ssim,
            "pixel_fft": pixel_fft,
            "flow_warp_consistency_loss": zero,
            "flow_warp_consistency_weighted_loss": zero,
        }

    def _compute_sr_branch(self, input_lr_prev, input_lr_next, input_gr_prev, input_gr_next, flowl0, is_adversarial: bool):
        """
        计算 Ground 模式下的“图像输出”和“RAFT 输入来源”。

        返回值包含四个核心对象：
        - pred_prev/pred_next: 对外保存和图像指标使用的图像，空间大小始终对齐 HR；
        - raft_prev/raft_next: 真正送进 RAFT 或传统 PIV 的图像，非学习 baseline 会用插值对齐到 HR 尺寸；
        - sr_outputs: 与 ESRuRAFT_PIV 原 CSV 对齐的图像/GAN/一致性损失字典。
        """
        if self.train_mode == "lr_ground_raft":
            # 低分辨率 ground + RAFT：不经过任何可学习超分辨率模块。
            # 由于当前 RAFT/flow 监督按 HR 尺寸组织，例如 256x256，
            # 这里只做最近邻插值把 64x64 LR 图像对齐到 HR 尺寸后送入 RAFT。
            # 这仍然是 LR baseline，而不是 ESRGAN/SR 结果。
            pred_prev = self._resize_image_to_target(input_lr_prev, input_gr_prev, mode="nearest")
            pred_next = self._resize_image_to_target(input_lr_next, input_gr_next, mode="nearest")
            raft_prev = pred_prev
            raft_next = pred_next
            return pred_prev, pred_next, raft_prev, raft_next, self._compute_ground_image_terms(
                pred_prev, pred_next, input_gr_prev, input_gr_next
            )

        if self.train_mode in {"bicubic_raft", "bicubic_widim", "bicubic_hs"}:
            # 传统 bicubic 超分 baseline：
            # 1. 不调用 Generator，不产生任何可学习的 SR 参数；
            # 2. 只用 PyTorch bicubic 双三次插值把 LR previous/next 放大到 HR 尺寸；
            # 3. 放大后的 bicubic 图像既作为 pred_prev/pred_next 参与图像诊断日志，
            #    也作为后续 RAFT/WIDIM/HS 的输入图像。
            # 这样可以单独观察“传统插值超分 + 不同 PIV 估计器”的差异。
            pred_prev = self._resize_image_to_target(input_lr_prev, input_gr_prev, mode="bicubic")
            pred_next = self._resize_image_to_target(input_lr_next, input_gr_next, mode="bicubic")
            raft_prev = pred_prev
            raft_next = pred_next
            return pred_prev, pred_next, raft_prev, raft_next, self._compute_ground_image_terms(
                pred_prev, pred_next, input_gr_prev, input_gr_next
            )

        if self.train_mode == "hr_ground_raft":
            # 高分辨率 ground + RAFT：RAFT 直接看真实 HR 图像，相当于给 RAFT 最理想的图像输入。
            raft_prev = input_gr_prev
            raft_next = input_gr_next
            pred_prev = input_gr_prev
            pred_next = input_gr_next
            return pred_prev, pred_next, raft_prev, raft_next, self._compute_ground_image_terms(
                pred_prev, pred_next, input_gr_prev, input_gr_next
            )

        # esrgan_raft / srgan_raft 模式：使用对应单帧生成器分别超分 previous/next。
        # 这两个生成器都没有 forward_pair，因此这里显式逐帧调用，避免误用 ESRuRAFT_PIV 的双帧融合结构。
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

        return pred_prev, pred_next, pred_prev, pred_next, {
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

    def _compute_raft_branch(self, raft_prev_source: torch.Tensor, raft_next_source: torch.Tensor, flowl0, flow_init=None):
        """
        单独计算 RAFT 分支。

        说明：
            这个分支只负责：
            1. 将当前 TRAIN_MODE 选出的图像转换为 RAFT 输入
            2. 计算 RAFT 的流场预测与 raft_loss
            3. 必要时将预测还原回 flow 真值尺寸，供统一评估和可视化使用
        """
        raft_prev = self._to_raft_frame(raft_prev_source)  # 将前一帧图像转成 RAFT 单通道输入
        raft_next = self._to_raft_frame(raft_next_source)  # 将后一帧图像转成 RAFT 单通道输入
        raft_input = torch.cat([raft_prev, raft_next], dim=1)  # 拼成 [B, 2, H, W]
        flow_gt_full_size = self._to_raft_flow_gt(flowl0)  # 只保留 uv 两个通道作为最终评估监督
        raft_flow_gt = self._resize_flow_to_match_image(flow_gt_full_size, raft_prev)  # RAFT loss 使用和输入图像相同的 flow 尺寸/单位

        flow_predictions_for_loss, (raft_loss, _) = self.piv_RAFT(
            raft_input,  # RAFT 输入图像对
            raft_flow_gt,  # RAFT 光流真值
            flow_init=flow_init,  # 可选初始光流
        )

        flow_predictions = [
            self._restore_flow_to_target_size(pred, flow_gt_full_size)
            for pred in flow_predictions_for_loss
        ]
        # 这里显式重建一个 tensor 形式的平均端点误差，
        # 避免后续误把 raft_metrics['epe'] 这个 Python 标量拿去给 Generator 反向传播。
        raft_epe_tensor = torch.sum((flow_predictions[-1] - flow_gt_full_size) ** 2, dim=1).sqrt().mean()
        raft_metrics = self._flow_metrics_from_prediction(flow_predictions[-1], flow_gt_full_size)

        return flow_predictions, {
            "raft_input_prev": raft_prev,  # 送入 RAFT 前的前一帧单通道图
            "raft_input_next": raft_next,  # 送入 RAFT 前的后一帧单通道图
            "flow_predictions": flow_predictions,  # RAFT 每次迭代得到的流场预测序列
            "raft_loss": raft_loss,  # RAFT 流场序列损失
            "raft_metrics": raft_metrics,  # RAFT 评估指标
            "raft_epe_tensor": raft_epe_tensor,  # 可反向传播的平均 EPE Tensor
        }

    def _compute_traditional_piv_branch(self, piv_prev_source: torch.Tensor, piv_next_source: torch.Tensor, flowl0):
        """
        计算传统 PIV/光流 baseline 分支。

        与 _compute_raft_branch 保持同样的返回字段名，这样 evaluate/train_step 不需要为 WIDIM/HS
        单独维护一套 CSV 和指标字段。这里的 "raft_loss" 字段实际表示传统方法最终 flow 的 EPE，
        仅用于日志兼容；WIDIM/HS 没有可学习参数，不会执行 backward。
        """
        piv_prev = self._to_raft_frame(piv_prev_source)
        piv_next = self._to_raft_frame(piv_next_source)
        flow_gt_full_size = self._to_raft_flow_gt(flowl0)

        with torch.no_grad():
            if self.train_mode == "bicubic_widim":
                flow_prediction = self._estimate_widim_flow(piv_prev, piv_next)
            elif self.train_mode == "bicubic_hs":
                flow_prediction = self._estimate_horn_schunck_flow(piv_prev, piv_next)
            else:
                raise ValueError(f"Unsupported traditional PIV mode: {self.train_mode}")

        flow_prediction = self._restore_flow_to_target_size(flow_prediction, flow_gt_full_size)
        piv_epe_tensor = torch.sum((flow_prediction - flow_gt_full_size) ** 2, dim=1).sqrt().mean()
        piv_metrics = self._flow_metrics_from_prediction(flow_prediction, flow_gt_full_size)
        flow_predictions = [flow_prediction]

        return flow_predictions, {
            "raft_input_prev": piv_prev,  # 为了兼容保存逻辑，字段名沿用 raft_input_prev
            "raft_input_next": piv_next,  # 为了兼容保存逻辑，字段名沿用 raft_input_next
            "flow_predictions": flow_predictions,
            "raft_loss": piv_epe_tensor,  # WIDIM/HS 无序列损失，这里用最终 EPE 作为日志 loss
            "raft_metrics": piv_metrics,
            "raft_epe_tensor": piv_epe_tensor,
            "traditional_piv_method": self.train_mode,
        }


    def _compute_generator_loss(self, sr_outputs: dict, raft_outputs: dict) -> tuple[torch.Tensor, dict]:
        """
        计算 ESRuRAFT_PIV_Ground 生成器实际反传的手动加权总损失。

        自适应多任务权重已移除，恢复为全局权重手动组合：
        VGG/L1/MSE/SSIM/FFT/flow_consistency + 动态对抗损失 + 动态 EPE。
        """
        if not self._uses_super_resolution():
            # ground 模式没有 Generator 参数，不能也不需要对图像诊断损失或 EPE 做 G 侧反传。
            # 这里返回 0，表示“本 batch 没有生成器训练目标”；SR 图像差异仍保存在 sr_outputs 里用于日志。
            zero = self._zero_like_loss(raft_outputs["raft_loss"])
            return zero, {
                "generator_non_adversarial_loss": zero,
                "generator_raft_epe_weighted_loss": zero,
            }

        epe_weighted_loss = float(global_data.esrgan.RAFT_EPE_WEIGHT) * raft_outputs["raft_epe_tensor"]
        manual_loss = sr_outputs["manual_sr_loss"] + epe_weighted_loss
        return manual_loss, {
            "generator_non_adversarial_loss": manual_loss - sr_outputs["adversarial_weighted_loss"],
            "generator_raft_epe_weighted_loss": epe_weighted_loss,
        }

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
        pred_prev, pred_next, raft_prev_source, raft_next_source, sr_outputs = self._compute_sr_branch(
            input_lr_prev, input_lr_next, input_gr_prev, input_gr_next, flowl0, is_adversarial
        )
        if self._uses_traditional_piv():
            # WIDIM/HS 是传统无参数 PIV/光流方法，不走 RAFT，也不参与反向传播。
            flow_predictions, raft_outputs = self._compute_traditional_piv_branch(
                piv_prev_source=raft_prev_source,
                piv_next_source=raft_next_source,
                flowl0=flowl0,
            )
            g_loss = self._zero_like_loss(raft_outputs["raft_loss"])
            generator_loss_logs = {
                "generator_non_adversarial_loss": g_loss,
                "generator_raft_epe_weighted_loss": g_loss,
            }
            total_loss = raft_outputs["raft_loss"]
        else:
            flow_predictions, raft_outputs = self._compute_raft_branch(
                raft_prev_source, raft_next_source, flowl0, flow_init=flow_init
            )
            g_loss, generator_loss_logs = self._compute_generator_loss(sr_outputs, raft_outputs)
            total_loss = g_loss + raft_outputs["raft_loss"]
        if self._uses_super_resolution():
            discriminator_loss, d_fake_loss, d_real_loss = self._compute_discriminator_loss(
                pred_prev, pred_next, input_gr_prev, input_gr_next
            )
        else:
            # ground 模式没有判别器训练；保留 0 值，保证外部日志字段完整。
            discriminator_loss = self._zero_like_loss(total_loss)
            d_fake_loss = self._zero_like_loss(total_loss)
            d_real_loss = self._zero_like_loss(total_loss)

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
        if self._uses_traditional_piv():
            # bicubic_widim / bicubic_hs 是传统无参数 baseline：
            # - LR 先经 bicubic 上采样到 HR；
            # - 再通过 WIDIM 互相关或 Horn-Schunck 光流法得到 PIV；
            # - 不更新 Generator / RAFT / Discriminator，只记录同一套日志字段。
            self._set_requires_grad(self.piv_esrgan_generator, False)
            self._set_requires_grad(self.piv_esrgan_discriminator, False)
            self._set_requires_grad(self.piv_RAFT, False)

            pred_prev, pred_next, piv_prev_source, piv_next_source, sr_outputs = self._compute_sr_branch(
                input_lr_prev=input_lr_prev,
                input_lr_next=input_lr_next,
                input_gr_prev=input_gr_prev,
                input_gr_next=input_gr_next,
                flowl0=flowl0,
                is_adversarial=False,
            )
            flow_predictions, raft_outputs = self._compute_traditional_piv_branch(
                piv_prev_source=piv_prev_source,
                piv_next_source=piv_next_source,
                flowl0=flowl0,
            )

            zero = self._zero_like_loss(raft_outputs["raft_loss"])
            final_flow_prediction = flow_predictions[-1]
            return pred_prev, pred_next, final_flow_prediction, {
                "sr_loss": float(sr_outputs["sr_loss"].detach().item()),
                "g_loss": 0.0,
                "perceptual_loss": float(sr_outputs["perceptual_loss"].detach().item()),
                "content_loss": float(sr_outputs["content_loss"].detach().item()),
                "adversarial_loss": 0.0,
                "pixel_total": float(sr_outputs["pixel_total"].detach().item()),
                "pixel_l1": float(sr_outputs["pixel_l1"].detach().item()),
                "pixel_mse": float(sr_outputs["pixel_mse"].detach().item()),
                "pixel_ssim": float(sr_outputs["pixel_ssim"].detach().item()),
                "pixel_fft": float(sr_outputs["pixel_fft"].detach().item()),
                "flow_warp_consistency_loss": 0.0,
                "flow_warp_consistency_weighted_loss": 0.0,
                "raft_loss": float(raft_outputs["raft_loss"].detach().item()),
                "discriminator_loss": 0.0,
                "d_real_loss": 0.0,
                "d_fake_loss": 0.0,
                "raft_epe": float(raft_outputs["raft_metrics"]["epe"]),
                "generator_raft_epe": 0.0,
                "raft_epe_weight": 0.0,
                "adversarial_weighted_loss": 0.0,
                "generator_non_adversarial_loss": float(zero.detach().item()),
                "generator_raft_epe_weighted_loss": float(zero.detach().item()),
                "raft_1px": float(raft_outputs["raft_metrics"]["1px"]),
                "raft_3px": float(raft_outputs["raft_metrics"]["3px"]),
                "raft_5px": float(raft_outputs["raft_metrics"]["5px"]),
            }

        if not self._uses_super_resolution():
            # lr_ground_raft / hr_ground_raft / bicubic_raft 只训练 RAFT：
            # - 不调用 Generator backward；
            # - 不更新 Discriminator；
            # - 仍然计算图像诊断项，保持 CSV 字段和 ESRuRAFT_PIV 对齐。
            self._set_requires_grad(self.piv_esrgan_generator, False)
            self._set_requires_grad(self.piv_esrgan_discriminator, False)
            self._set_requires_grad(self.piv_RAFT, True)
            raft_optimizer.zero_grad(set_to_none=True)

            pred_prev, pred_next, raft_prev_source, raft_next_source, sr_outputs = self._compute_sr_branch(
                input_lr_prev=input_lr_prev,
                input_lr_next=input_lr_next,
                input_gr_prev=input_gr_prev,
                input_gr_next=input_gr_next,
                flowl0=flowl0,
                is_adversarial=False,
            )
            flow_predictions, raft_outputs = self._compute_raft_branch(
                raft_prev_source=raft_prev_source,
                raft_next_source=raft_next_source,
                flowl0=flowl0,
                flow_init=flow_init,
            )

            if scaler is not None:
                scaler.scale(raft_outputs["raft_loss"]).backward()
                scaler.step(raft_optimizer)
                scaler.update()
            else:
                raft_outputs["raft_loss"].backward()
                raft_optimizer.step()

            self._set_requires_grad(self.piv_RAFT, True)
            zero = self._zero_like_loss(raft_outputs["raft_loss"])
            final_flow_prediction = flow_predictions[-1]
            return pred_prev, pred_next, final_flow_prediction, {
                "sr_loss": float(sr_outputs["sr_loss"].detach().item()),
                "g_loss": 0.0,
                "perceptual_loss": float(sr_outputs["perceptual_loss"].detach().item()),
                "content_loss": float(sr_outputs["content_loss"].detach().item()),
                "adversarial_loss": 0.0,
                "pixel_total": float(sr_outputs["pixel_total"].detach().item()),
                "pixel_l1": float(sr_outputs["pixel_l1"].detach().item()),
                "pixel_mse": float(sr_outputs["pixel_mse"].detach().item()),
                "pixel_ssim": float(sr_outputs["pixel_ssim"].detach().item()),
                "pixel_fft": float(sr_outputs["pixel_fft"].detach().item()),
                "flow_warp_consistency_loss": 0.0,
                "flow_warp_consistency_weighted_loss": 0.0,
                "raft_loss": float(raft_outputs["raft_loss"].detach().item()),
                "discriminator_loss": 0.0,
                "d_real_loss": 0.0,
                "d_fake_loss": 0.0,
                "raft_epe": float(raft_outputs["raft_metrics"]["epe"]),
                "generator_raft_epe": 0.0,
                "raft_epe_weight": 0.0,
                "adversarial_weighted_loss": 0.0,
                "generator_non_adversarial_loss": float(zero.detach().item()),
                "generator_raft_epe_weighted_loss": float(zero.detach().item()),
                "raft_1px": float(raft_outputs["raft_metrics"]["1px"]),
                "raft_3px": float(raft_outputs["raft_metrics"]["3px"]),
                "raft_5px": float(raft_outputs["raft_metrics"]["5px"]),
            }

        # 第一阶段：一次前向同时准备 Generator 和 RAFT 的训练目标。
        self._set_requires_grad(self.piv_esrgan_generator, True)  # Generator 需要拿到 sr_loss 和 raft_epe 的梯度
        self._set_requires_grad(self.piv_RAFT, True)  # 同一次 RAFT forward 也要服务于后续 RAFT 自身更新
        self._set_requires_grad(self.piv_esrgan_discriminator, False)  # 训练 G/RAFT 时冻结判别器
        generator_optimizer.zero_grad(set_to_none=True)  # 清空 Generator 梯度
        raft_optimizer.zero_grad(set_to_none=True)  # 清空 RAFT 梯度

        pred_prev, pred_next, raft_prev_source, raft_next_source, sr_outputs = self._compute_sr_branch(
            input_lr_prev=input_lr_prev,
            input_lr_next=input_lr_next,
            input_gr_prev=input_gr_prev,
            input_gr_next=input_gr_next,
            flowl0=flowl0,
            is_adversarial=is_adversarial,
        )
        flow_predictions, raft_outputs = self._compute_raft_branch(
            raft_prev_source=raft_prev_source,
            raft_next_source=raft_next_source,
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
