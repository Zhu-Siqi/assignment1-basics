import torch
from torch import tensor
import torch.nn as nn
from einops import einsum, rearrange
from jaxtyping import Float
import math

class CustomLinear(nn.Module):
# custom implementation of linear layer
    def __init__(
            self,
            in_features: int,
            out_features: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
        ):
        '''
        Initialize the custom implementation of linear layer
        '''
        super(CustomLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), 
                                                device = device, dtype = dtype))
        self.reset_parameters()
        
    def reset_parameters(self):
        '''
        Initialize the parameters with truncated normalization
        '''
        sd = math.sqrt(2 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, 
                              mean = 0,
                              std = sd,
                              a = -3 * sd,
                              b = 3 * sd)
    
    def forward(self,
                x: torch.Tensor
        ) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Input tensor has wrong last dimension {x.shape[-1]}, expected {self.in_features}")
        return einsum(x, self.weight, '... d_in, d_out d_in -> ... d_out')
    

class CustomEmbedding(nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
        ):
        super(CustomEmbedding, self).__init__()
        self.num_embedding = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(
                (num_embeddings, embedding_dim),
                device = device, dtype = dtype,
            )
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight,
                              mean = 0,
                              std = 1,
                              a = -3,
                              b = 3,
                              )
        
    def forward(self,
                x: torch.Tensor
            ) -> torch.Tensor:
        return self.weight[x]
    

class CustomRMSNorm(nn.Module):
    def __init__(
            self,
            d_in: int,
            epsilon : float = 1e-5,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,              
    ):
        super(CustomRMSNorm, self).__init__()
        self.d_in = d_in
        self.weight = nn.Parameter(
            torch.empty(
                d_in,
                device=device, dtype=dtype
            )
        )
        self.epsilon = epsilon
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, 1)

    def forward(
            self,
            x: float
    ) -> Float[torch.Tensor, '... d_in']:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        x_rms = torch.sqrt((x ** 2).mean(-1).unsqueeze(-1) + self.epsilon)
        x = x / x_rms
        x = einsum(x.to(in_dtype), self.weight, '... d_in, d_in -> ... d_in')
        return x
    
class CustomSwiGLU(nn.Module):
    def __init__(
      self,
      d_model: int,
      d_ff: int,
      device: torch.device | None = None,
      dtype: torch.dtype | None = None,      
    ):
        super(CustomSwiGLU, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear1 = CustomLinear(d_model, d_ff,
                                    device=device, dtype=dtype)
        self.linear2 = CustomLinear(d_ff, d_model,
                                    device=device, dtype=dtype)
        self.linear3 = CustomLinear(d_model, d_ff,
                                    device=device, dtype=dtype)
        
    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        weight = self.linear1(x)
        weight = weight * torch.sigmoid(weight)
        return self.linear2(weight * self.linear3(x))
    
class CustomRoPE(nn.Module):
    def __init__(
            self,
            theta:float,
            d_k:int,
            max_seq_len: int,
            device: torch.device | None = None,
    ):
        super(CustomRoPE, self).__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        angles = 1 / (theta ** (torch.arange(0, d_k, 2, device=device) / d_k))
        self.register_buffer('angles', angles)

    def _rearrange_x(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        x = rearrange(x, '... (k r) -> ... k r', r = 2)
        x = torch.cat((-x[..., 1].unsqueeze(-1), x[..., 0].unsqueeze(-1)), dim = -1)
        x = rearrange(x, '... k r -> ... (k r)')
        return x
    
    def forward(
            self, 
            x: torch.Tensor, 
            token_positions: torch.Tensor
    ) -> torch.Tensor:
        seq_angles = einsum(token_positions, self.angles, '... k, d -> ... k d')

        ang_cos = seq_angles.cos().repeat_interleave(2, dim = -1)
        ang_sin = seq_angles.sin().repeat_interleave(2, dim = -1)

        return x * ang_cos + self._rearrange_x(x) * ang_sin