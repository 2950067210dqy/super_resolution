from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn.functional as F


EPS = 1e-8  # FAMO 论文实现中的数值稳定项，避免 log(0) 或除 0。


class FAMO:
    """
    Fast Adaptive Multitask Optimization 的轻量实现。

    这里严格保留论文核心形式：
    1. 用 softmax(logits) 得到各任务权重，所有任务权重自动归一化为 1。
    2. 用 loss 相对 min_loss 的 log 改变量更新权重 logits。
    3. task_weights 只用于初始化 logits，之后权重由 FAMO 自己更新。
    """

    def __init__(
        self,
        n_tasks: int,
        device: torch.device | str,
        gamma: float = 1e-5,
        w_lr: float = 0.025,
        task_weights: Sequence[float] | torch.Tensor | None = None,
        max_norm: float = 1.0,
    ) -> None:
        self.n_tasks = int(n_tasks)
        self.device = torch.device(device)
        self.max_norm = float(max_norm)
        self.min_losses = torch.zeros(self.n_tasks, device=self.device)
        self.prev_loss: torch.Tensor | None = None

        if task_weights is None:
            # 不传初始比例时退化为论文示例中的均分初始化。
            init_logits = torch.zeros(self.n_tasks, device=self.device)
        else:
            init_weights = torch.as_tensor(task_weights, dtype=torch.float32, device=self.device)
            if init_weights.numel() != self.n_tasks:
                raise ValueError(
                    f"FAMO task_weights length={init_weights.numel()} does not match n_tasks={self.n_tasks}"
                )
            # 只把用户给的初始比例转成 softmax logits；真正训练时仍由 FAMO 动态调整。
            init_weights = init_weights.clamp_min(EPS)
            init_weights = init_weights / init_weights.sum().clamp_min(EPS)
            init_logits = init_weights.log()

        self.w = init_logits.detach().clone().requires_grad_(True)
        self.w_opt = torch.optim.Adam([self.w], lr=float(w_lr), weight_decay=float(gamma))

    @property
    def weights(self) -> torch.Tensor:
        """返回当前 FAMO 权重，形状为 [n_tasks]，权重和为 1。"""
        return F.softmax(self.w, dim=-1)

    @staticmethod
    def _stack_losses(losses: Iterable[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """把多个标量 loss 统一整理成 [n_tasks] 的 tensor。"""
        if torch.is_tensor(losses):
            return losses.reshape(-1)
        return torch.stack([loss.reshape(()) for loss in losses])

    def set_min_losses(self, losses: Iterable[torch.Tensor] | torch.Tensor) -> None:
        """可选接口：如果外部想显式指定每个任务的历史最小 loss，可以调用它。"""
        self.min_losses = self._stack_losses(losses).detach().to(self.device)

    def get_weighted_loss(self, losses: Iterable[torch.Tensor] | torch.Tensor, **kwargs):
        """
        计算 FAMO 加权后的非对抗生成器损失。

        注意：这里不包含 GAN 对抗损失。对抗损失继续由外部动态权重单独控制，
        避免 FAMO 在训练早期把 GAN 项突然抬高导致生成器失稳。
        """
        del kwargs
        loss_vec = self._stack_losses(losses)
        min_losses = self.min_losses.to(device=loss_vec.device, dtype=loss_vec.dtype)
        z = self.weights.to(device=loss_vec.device, dtype=loss_vec.dtype)

        self.prev_loss = loss_vec.detach()
        D = (loss_vec - min_losses + EPS).clamp_min(EPS)
        c = (z / D).sum().detach().clamp_min(EPS)
        weighted_loss = (D.log() * z / c).sum()
        return weighted_loss, {"weights": z.detach().clone(), "logits": self.w.detach().clone()}

    def update(self, curr_loss: Iterable[torch.Tensor] | torch.Tensor) -> None:
        """
        按论文公式更新任务权重 logits。

        调用时机应在生成器 optimizer.step() 之后，用“更新后的模型”重新计算一遍
        各任务 loss，再交给 update(curr_loss)。
        """
        if self.prev_loss is None:
            return

        curr_loss_vec = self._stack_losses(curr_loss).detach().to(self.device)
        prev_loss = self.prev_loss.to(self.device)
        min_losses = self.min_losses.to(self.device)
        delta = (prev_loss - min_losses + EPS).clamp_min(EPS).log() - (
            curr_loss_vec - min_losses + EPS
        ).clamp_min(EPS).log()

        with torch.enable_grad():
            grad = torch.autograd.grad(
                F.softmax(self.w, dim=-1),
                self.w,
                grad_outputs=delta.detach(),
            )[0]

        self.w_opt.zero_grad()
        self.w.grad = grad
        self.w_opt.step()
