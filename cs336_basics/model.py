import torch
from torch import tensor
import torch.nn as nn
from einops import einsum, rearrange
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