"""
多任务学习优化 FAMO
"""
from abc import abstractmethod
from typing import Union, List, Tuple

import torch
import torch.nn.functional as F

class WeightMethod:
    def __init__(self, n_tasks: int, device: torch.device, max_norm=1.0):
        super().__init__()
        self.n_tasks = n_tasks
        self.device = device
        self.max_norm = max_norm

    @abstractmethod
    def get_weighted_loss(
            self,
            losses: torch.Tensor,
            shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
            task_specific_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ],
            last_shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
            representation: Union[torch.nn.parameter.Parameter, torch.Tensor],
            **kwargs,
    ):
        pass

    def backward(
            self,
            losses: torch.Tensor,
            shared_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
            task_specific_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
            last_shared_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
            representation: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
            **kwargs,
    ) -> Tuple[Union[torch.Tensor, None], Union[dict, None]]:
        """

        Parameters
        ----------
        losses :
        shared_parameters :
        task_specific_parameters :
        last_shared_parameters : parameters of last shared layer/block
        representation : shared representation
        kwargs :

        Returns
        -------
        Loss, extra outputs
        """
        loss, extra_outputs = self.get_weighted_loss(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            last_shared_parameters=last_shared_parameters,
            representation=representation,
            **kwargs,
        )

        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)

        loss.backward()
        return loss, extra_outputs

    def __call__(
            self,
            losses: torch.Tensor,
            shared_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
            task_specific_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
            **kwargs,
    ):
        return self.backward(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            **kwargs,
        )

    def parameters(self) -> List[torch.Tensor]:
        """return learnable parameters"""
        return []


class FAMO(WeightMethod):
    """
    FAMO: Fast Adaptive Multitask Optimization。

    这个实现维护一组可学习的任务权重 logits `self.w`，通过 softmax 得到每个任务的权重。
    训练时先用 `get_weighted_loss(losses)` 得到当前 batch 的自适应加权总损失；
    参数更新后，再用 `update(curr_loss)` 根据任务损失下降幅度更新任务权重。
    """

    def __init__(
            self,
            n_tasks: int,
            device: torch.device,
            gamma: float = 1e-5,
            w_lr: float = 0.025,
            task_weights: Union[List[float], torch.Tensor] = None,
            max_norm: float = 1.0,
    ):
        super().__init__(n_tasks, device=device)
        self.min_losses = torch.zeros(n_tasks).to(device)
        self.w = torch.tensor([0.0] * n_tasks, device=device, requires_grad=True)
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma)
        self.max_norm = max_norm
        self.eps = 1e-8

    def set_min_losses(self, losses):
        # min_losses 是 FAMO 中的参考最小损失/utopia point。
        # 这里只保存数值，不应该持有计算图，避免跨 batch 保留 autograd graph。
        self.min_losses = losses.detach().to(self.device)

    def get_weighted_loss(self, losses, store_prev: bool = True, **kwargs):
        # prev_loss 只用于后续 update 比较“优化前后任务损失下降幅度”，
        # 因此保存 detached 版本即可，避免额外占用显存。
        # store_prev=False 用于“参数更新后重算当前 weighted loss”，避免覆盖优化前的 prev_loss。
        if store_prev:
            self.prev_loss = losses.detach()
        z = F.softmax(self.w, -1)
        # D 必须为正才能取 log；clamp_min 能防止数值误差或偶发负值导致 NaN。
        D = (losses - self.min_losses).clamp_min(self.eps)
        c = (z / D).sum().detach()
        loss = (D.log() * z / c).sum()
        return loss, {"weights": z, "logits": self.w.detach().clone()}

    def update(self, curr_loss):
        # update 使用优化前后的任务损失变化来更新任务权重。
        # curr_loss 同样只需要数值，不需要梯度图。
        curr_loss = curr_loss.detach().to(self.device)
        prev_D = (self.prev_loss - self.min_losses).clamp_min(self.eps)
        curr_D = (curr_loss - self.min_losses).clamp_min(self.eps)
        delta = prev_D.log() - curr_D.log()
        with torch.enable_grad():
            d = torch.autograd.grad(F.softmax(self.w, -1),
                                    self.w,
                                    grad_outputs=delta.detach())[0]
        self.w_opt.zero_grad()
        self.w.grad = d
        self.w_opt.step()
