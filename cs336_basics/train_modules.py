import torch
from torch import nn
from jaxtyping import Int, Float
from typing import Iterable, Tuple, Callable, IO, BinaryIO
import os
import math
import numpy as np
import numpy.typing as npt

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()

    def forward(
            self,
            pred_logits: Float[torch.Tensor, '... vocab_size'],
            target_idxs: Int[torch.Tensor, '...'],
    ) -> torch.Tensor:
        # shift
        pred_logits_max = torch.max(pred_logits, dim = -1, keepdim = True)[0]
        pred_logits = pred_logits - pred_logits_max

        # sum
        log_exp_sum_logits = torch.log(torch.exp(pred_logits).sum(dim = -1))
        target_logits = torch.gather(pred_logits, -1, target_idxs.unsqueeze(-1)).squeeze(-1)

        return (log_exp_sum_logits - target_logits).mean()
    
class CustomAdamW(torch.optim.Optimizer):
    def __init__(
            self,
            params: Iterable[nn.parameter.Parameter],
            lr: Float = 1e-3,
            betas: Tuple[Float, Float] = (0.9, 0.99),
            weight_decay: Float = 1e-2,
            eps: Float = 1e-8,
    ):
        # validate the hyperparams
        if lr <= 0:
            raise ValueError(f'Invalid alpha: {lr}')
        if not (0 < betas[0] < 1):
            raise ValueError(f'Invalid beta1: {betas[0]}')
        if not (0 < betas[1] < 1):
            raise ValueError(f'Invalid beta2: {betas[1]}')
        if not 0 <= weight_decay:
            raise ValueError(f'Invalid weight_decay: {weight_decay}')
        if not 0 < eps:
            raise ValueError(f'Invalid epsilon: {eps}')
        defaults = {
            'alpha': lr, 'beta': betas,
            'weight_decay': weight_decay, 'epsilon': eps,
        }
        super().__init__(params, defaults)

    def step(
            self,
            closure: Callable | None = None
    ):
        loss = closure() if closure else None
        for group in self.param_groups:
            alpha = group['alpha']
            beta = group['beta']
            weight_decay = group['weight_decay']
            epsilon = group['epsilon']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                
                state = self.state[p]
                
                t = state.get(
                    't',
                    1
                )

                m = state.get(
                    'm',
                    torch.zeros_like(g)
                )
                m = beta[0] * m + (1 - beta[0]) * g

                v = state.get(
                    'v',
                    torch.zeros_like(g)
                )
                v = beta[1] * v + (1 - beta[1]) * (g ** 2)

                alpha_t = alpha * (math.sqrt(1 - beta[1] ** t) / (1 - beta[0] ** t))
                p.data -= alpha_t * m / (torch.sqrt(v) + epsilon)
                p.data *= (1 - alpha * weight_decay)

                state['m'] = m
                state['v'] = v
                state['t'] = t+1
                
        
        return loss
    
def get_lr(
        t,
        lr_max: Float,
        lr_min: Float,
        T_w: int,
        T_c: int,
):
    if t < T_w:
        return t * lr_max / T_w
    elif T_w <= t <= T_c:
        return lr_min + (lr_max - lr_min) / 2 * (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi))
    elif t > T_c:
        return lr_min
    else:
        raise ValueError(f'Invalid t: {t}')

def gradient_clipping(
        params: Iterable[nn.Parameter],
        max_norm: Float,
        epsilon: Float = 1e-6,
):
    norm_sum = 0
    valid_gs = [p.grad for p in params if p.grad is not None]
    for g in valid_gs:
        norm_sum += (g.data ** 2).sum()
    norm_sum = torch.sqrt(norm_sum)

    if norm_sum > max_norm:
        discount = max_norm / (norm_sum + epsilon)
        for g in valid_gs:
            g.data *= discount

def get_batch(
        dataset: npt.NDArray, 
        batch_size: int, 
        context_length: int, 
        device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    data_len = len(dataset)
    max_start = data_len - context_length
    start_idxs = np.random.randint(0, max_start, size = batch_size) # [0, data_len - context_length)
    input_seqs = torch.from_numpy(np.stack([dataset[idx: idx+context_length] for idx in start_idxs])).long().to(device)
    target_seqs = torch.from_numpy(np.stack([dataset[idx+1: idx+1+context_length] for idx in start_idxs])).long().to(device)
    return input_seqs, target_seqs

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    check_point = torch.load(src)
    model.load_state_dict(check_point['model_state_dict'])
    optimizer.load_state_dict(check_point['optimizer_state_dict'])
    iteration = check_point['iteration']
    return iteration