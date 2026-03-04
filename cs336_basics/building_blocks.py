import token
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
        self.w1 = Linear(d_model, d_ff)
        self.w3 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)

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

def softmax(x: torch.Tensor, i: int):

    max_vals, _ = x.max(dim=i, keepdim=True)
    subtract = x - max_vals
    expo = torch.exp(subtract)

    denominator = expo.sum(dim=i, keepdim=True)
    
    return expo/denominator

def causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """Boolean mask for causal attention: position i can attend to positions j <= i.
    Returns shape (seq_len, seq_len), True where attention is allowed."""
    return torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    )


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
    scale = k.shape[-1] ** 0.5
    scores = q @ k.transpose(-2, -1) / scale

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = softmax(scores, -1)
    return weights @ v

class MultiHeadAttention(torch.nn.Module):

    def __init__(self, d_model, num_heads, max_seq_len=None, theta=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

        if max_seq_len is not None and theta is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, token_positions=None):
        seq_len = x.size(-2)
        mask = causal_mask(seq_len, device=x.device)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        # (..., seq_len, num_heads * d_k/d_v) -> (..., num_heads, seq_len, d_k/d_v)
        Q = Q.unflatten(-1, (self.num_heads, self.d_k)).transpose(-3, -2)
        K = K.unflatten(-1, (self.num_heads, self.d_k)).transpose(-3, -2)
        V = V.unflatten(-1, (self.num_heads, self.d_v)).transpose(-3, -2)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        attn_out = scaled_dot_product_attention(Q, K, V, mask)
        # (..., num_heads, seq_len, d_v) -> (..., seq_len, num_heads * d_v)
        attn_out = attn_out.transpose(-3, -2).flatten(-2)
        return self.output_proj(attn_out)


class TransformerBlock(torch.nn.Module):

    def __init__(self, d_model, num_heads, d_ff, max_seq_len=None, theta=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, max_seq_len, theta)
        self.ffn = SWIGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        return x

class TransformerLM(torch.nn.Module):

    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)
        x = self.lm_head(x)
        return x



    




