import torch
import torch.nn as nn


class Linear(nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T

class Embedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(torch.nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        
        # Square each element, then mean over d_model (last dim)
        x_sq = x ** 2  # or torch.square(x)
        rms = (x_sq.mean(dim=-1, keepdim=True) + self.eps) ** 0.5
        return (x / rms) * self.weight


class SWIGLU(torch.nn.Module):

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_ff, d_model)
        self.w3 = Linear(d_ff, d_model)
        self.w2 = Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor):
        z = self.w1(x)
        gate = z * torch.sigmoid(z)
        value = self.w3(x)
        return self.w2(gate * value)


class RotaryPositionalEmbedding(nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        freqs = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        denominators = theta ** (freqs / d_k)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = positions[:, None] / denominators[None, :]
        self.register_buffer("cos_cache", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cache", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x_pairs = x.unflatten(-1, (-1, 2))
        x1 = x_pairs[..., 0]
        x2 = x_pairs[..., 1]
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos
        return torch.stack((x1_rot, x2_rot), dim=-1).flatten(-2)

