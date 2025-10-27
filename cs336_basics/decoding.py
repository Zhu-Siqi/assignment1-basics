import torch
import torch.nn.functional as F
from jaxtyping import Int, Float

def low_temperature_sampling_step(
        model: torch.nn.Module,
        input_idxs: Int[torch.Tensor, '... seq_len'],
        temperature: Float = 0.1,
) -> Int[torch.Tensor, '... seq_len_new']:
    model.eval()
    with torch.no_grad():
        logits = model(input_idxs)[..., -1, :]
    probs = F.softmax(logits / temperature, dim = -1)

    if probs.dim() == 1:
        probs = probs.unsqueeze(0)

    new_idxs = torch.multinomial(probs, num_samples=1)

    if input_idxs.dim() == 1:
        new_idxs = new_idxs.squeeze(0)

    return torch.concat([input_idxs, new_idxs], dim=-1)