import math
import os
from collections.abc import Callable
from typing import IO, BinaryIO, Optional

import numpy as np
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

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross entropy loss: ℓ_i = -log(softmax(o_i)[x_{i+1}]).

    Uses max subtraction for numerical stability and log-sum-exp to cancel log/exp.
    Batch dimensions come first; returns the average loss over the batch.

    Args:
        logits: Predicted logits, shape (..., vocab_size).
        targets: Target class indices, shape (...).

    Returns:
        Scalar tensor: average cross entropy over the batch.
    """
    # Subtract max over last dim for numerical stability
    max_vals = logits.max(dim=-1, keepdim=True).values
    logits_stable = logits - max_vals
    # -log(softmax(o)[t]) = log(sum(exp(o))) - o[t] = log_sum_exp(o) - o[t]
    log_sum_exp = max_vals.squeeze(-1) + torch.log(torch.exp(logits_stable).sum(dim=-1))
    logits_at_targets = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    loss_per_example = log_sum_exp - logits_at_targets
    return loss_per_example.mean()


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

class SGD(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data -= lr * p.grad.data
        return loss


class AdamW(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                state["t"] += 1
                t = state["t"]
                m, v = state["m"], state["v"]
                grad = p.grad.data

                p.data -= lr * weight_decay * p.data

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                p.data -= lr * m_hat / (v_hat.sqrt() + eps)

        return loss


def gradient_clipping(parameters: list[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    total_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale)


def lr_cosine_schedule(t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int) -> float:
    if t < T_w:
        return (t / T_w) * alpha_max
    elif t <= T_c:
        return alpha_min + 0.5 * (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi)) * (alpha_max - alpha_min)
    else:
        return alpha_min


def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str):
    starts = np.random.randint(0, len(x) - context_length, size=batch_size)
    inputs = torch.stack([torch.from_numpy(x[i:i + context_length].astype(np.int64)) for i in starts])
    targets = torch.stack([torch.from_numpy(x[i + 1:i + 1 + context_length].astype(np.int64)) for i in starts])
    return inputs.to(device), targets.to(device)


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


def generate(
    model: nn.Module,
    prompt: torch.Tensor,
    max_tokens: int,
    temperature: float = 1.0,
    top_p: float = 0.0,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Auto-regressively generate tokens from a language model.

    Args:
        model: A TransformerLM that maps (batch, seq_len) -> (batch, seq_len, vocab_size) logits.
        prompt: Int tensor of shape (seq_len,) with the initial token ids.
        max_tokens: Maximum number of new tokens to generate.
        temperature: Softmax temperature. Lower -> sharper distribution.
        top_p: Nucleus sampling threshold in (0, 1]. 0 disables nucleus sampling.
        eos_token_id: If not None, stop when this token is sampled.

    Returns:
        Int tensor of generated token ids (excluding the prompt).
    """
    model.eval()
    generated: list[int] = []
    context = prompt.unsqueeze(0) if prompt.dim() == 1 else prompt  # (1, seq_len)

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(context)                      # (1, seq_len, vocab_size)
            next_logits = logits[0, -1, :]               # (vocab_size,)

            if temperature != 1.0:
                next_logits = next_logits / temperature

            probs = softmax(next_logits.unsqueeze(0), -1).squeeze(0)  # (vocab_size,)

            if 0.0 < top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=0)
                # Find the smallest set whose cumulative probability >= top_p
                cutoff_mask = cumulative - sorted_probs >= top_p
                sorted_probs[cutoff_mask] = 0.0
                sorted_probs /= sorted_probs.sum()
                # Sample from the truncated distribution
                idx_in_sorted = torch.multinomial(sorted_probs, num_samples=1)
                next_token = sorted_indices[idx_in_sorted].item()
            else:
                next_token = torch.multinomial(probs, num_samples=1).item()

            generated.append(next_token)

            if eos_token_id is not None and next_token == eos_token_id:
                break

            next_tensor = torch.tensor([[next_token]], device=context.device, dtype=context.dtype)
            context = torch.cat([context, next_tensor], dim=1)

    return torch.tensor(generated, dtype=prompt.dtype, device=prompt.device)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]


