import torch
from torch import tensor
import torch.nn as nn
from einops import einsum, rearrange
from jaxtyping import Float, Bool, Int
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
            token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if token_positions is None:
            token_positions = torch.arange(0, x.shape[-2], device=x.device)
            if x.dim() == 3:
                token_positions = token_positions.unsqueeze(0)
            elif x.dim() == 4:
                token_positions = token_positions.unsqueeze(0).unsqueeze(0)
        seq_angles = einsum(token_positions, self.angles, '... k, d -> ... k d')

        ang_cos = seq_angles.cos().repeat_interleave(2, dim = -1)
        ang_sin = seq_angles.sin().repeat_interleave(2, dim = -1)

        return x * ang_cos + self._rearrange_x(x) * ang_sin

class CustomSoftmax(nn.Module):
    def __init__(self):
        super(CustomSoftmax, self).__init__()
        pass
    
    def forward(
            self,
            x: torch.Tensor,
            dim: int,
    ) -> torch.Tensor:
        # shift
        x = x - x.max(dim=dim, keepdim=True)[0] # max() returns max values and idxs
        x = torch.exp(x)
        x = x / x.sum(dim=dim, keepdim=True)
        return x
    
class CustomScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(CustomScaledDotProductAttention, self).__init__()
        self.softmax = CustomSoftmax()

    def forward(
            self,
            query: Float[torch.Tensor, '... query d_k'],
            key: Float[torch.Tensor, '... key d_k'],
            value: Float[torch.Tensor, '... key d_v'],
            mask: Bool[torch.Tensor, '... query key']
    ) -> Float[torch.Tensor, '... query d_v']:
        d_k = query.shape[-1]
        attention = einsum(query, key, '... query d_k, ... key d_k -> ... query key') / math.sqrt(d_k)
        attention = attention.masked_fill(~mask, -torch.inf)
        attention = self.softmax(attention, -1)
        return einsum(attention, value, '... query key, ... key d_v -> ... query d_v')

class CustomMultiheadSelfAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            use_rope: Bool = False,
            theta: Float | None = None,
            max_seq_len: int | None = None,
    ):
        super(CustomMultiheadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = CustomLinear(d_model, d_model)
        self.k_proj = CustomLinear(d_model, d_model)
        self.v_proj = CustomLinear(d_model, d_model)
        self.o_proj = CustomLinear(d_model, d_model)

        self.use_rope = use_rope if theta and max_seq_len else False
        if use_rope:
            self.rope_layer = CustomRoPE(
                theta=theta, d_k=self.d_k, 
                max_seq_len=max_seq_len
            )
        
        self.attention_layer = CustomScaledDotProductAttention()

    def _generate_mask(
            self,
            seq_len: int,
            device: torch.device,
    ):
        mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool()
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
            self,
            x: torch.Tensor,
            token_positions: Int[torch.Tensor, " ... sequence_length"] | None = None,
    ) -> torch.Tensor:
        q = self.q_proj(x)
        q = rearrange(q, '... seq_len (head d_k) -> ... head seq_len d_k', 
                      head = self.num_heads, d_k = self.d_k)
        k = self.k_proj(x)
        k = rearrange(k, '... seq_len (head d_k) -> ... head seq_len d_k', 
                      head = self.num_heads, d_k = self.d_k)
        v = self.v_proj(x)
        v = rearrange(v, '... seq_len (head d_k) -> ... head seq_len d_k', 
                      head = self.num_heads, d_k = self.d_k)
        
        if self.use_rope:
            q = self.rope_layer(q, token_positions)
            k = self.rope_layer(k, token_positions)
        
        mask = self._generate_mask(
            q.shape[-2], q.device
        )

        attention_out = self.attention_layer(
            q,k,v,mask
        )
        attention_out = rearrange(attention_out, '... head seq_len d_k -> ... seq_len (head d_k)')
        attention_out = self.o_proj(attention_out)
        return attention_out
    

class CustomTransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            theta: float,
            max_seq_len: int,
    ):
        super(CustomTransformerBlock, self).__init__()
        self.norm1 = CustomRMSNorm(d_model)
        self.MHA_layer = CustomMultiheadSelfAttention(
            d_model, num_heads, 
            use_rope=True, theta=theta, max_seq_len=max_seq_len
        )
        self.norm2 = CustomRMSNorm(d_model)
        self.FFN = CustomSwiGLU(d_model, d_ff)

    def forward(
            self,
            x: torch.Tensor,
            token_positions: Int[torch.Tensor, " ... sequence_length"] | None = None,
    ) -> torch.Tensor:
        x = x + self.MHA_layer(self.norm1(x), token_positions)
        x = x + self.FFN(self.norm2(x))
        return x
    

class CustomTransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size:int,
            context_length:int,
            num_layers:int,
            d_model: int,
            num_heads: int,
            d_ff: int,
            theta: float,
    ):
        super(CustomTransformerLM, self).__init__()
        self.embedding = CustomEmbedding(
            vocab_size, d_model
        )
        self.blocks = nn.ModuleList(
            [
                CustomTransformerBlock(
                    d_model, num_heads, d_ff, theta, context_length
                ) for _ in range(num_layers)
            ]
        )
        self.last_norm = CustomRMSNorm(d_model)
        self.last_linear = CustomLinear(d_model, vocab_size)
        # self.last_sftm = CustomSoftmax()

    def forward(
            self,
            idxs: Int[torch.Tensor, ' batch_size seq_len'],
            token_positions: Int[torch.Tensor, " ... sequence_length"] | None = None,
    ) -> Float[torch.Tensor, ' batch_size seq_len vocab_size']:
        x = self.embedding(idxs)
        for block in self.blocks:
            x = block(x, token_positions)
        # x = self.last_sftm(self.last_linear(self.last_norm(x)), dim = -1)
        x = self.last_linear(self.last_norm(x))
        return x