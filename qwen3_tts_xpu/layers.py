import torch
import torch.nn.functional as F
from torch import nn

from .config import TransformerConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def rope_table(
    length: int, dim: int, theta: float, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim)
    )
    pos = torch.arange(length, device=device, dtype=torch.float32)
    freqs = pos[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def _fuse_linears(*linears: nn.Linear) -> nn.Linear:
    w = torch.cat([lin.weight for lin in linears])
    fused = nn.Linear(w.shape[1], w.shape[0], bias=False, device=w.device, dtype=w.dtype)
    fused.weight = nn.Parameter(w, requires_grad=False)
    return fused


class Attention(nn.Module):
    """Qwen3-style GQA attention with per-head q/k RMSNorm, writing into an external KV cache.

    Batch size is fixed to 1.
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        q_dim = cfg.num_heads * cfg.head_dim
        kv_dim = cfg.num_kv_heads * cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, q_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.split = (q_dim, kv_dim, kv_dim)
        self.qkv_proj: nn.Linear | None = None

    def fuse(self) -> None:
        self.qkv_proj = _fuse_linears(self.q_proj, self.k_proj, self.v_proj)
        del self.q_proj, self.k_proj, self.v_proj

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot: int | torch.Tensor,
        kv_len: int,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Attend `x` (shape [1, L, D]) over the cache after writing its keys/values.

        `slot` is the first cache index to write (int) or the positions to write (LongTensor of
        shape [L]). Attention covers `cache[:kv_len]`; it is causal when L > 1 (then the block
        must start at 0 and fill `kv_len`), otherwise `mask` (bool, [kv_len]) selects valid keys.
        """
        _, length, _ = x.shape
        h, kvh, hd = self.num_heads, self.num_kv_heads, self.head_dim
        if self.qkv_proj is not None:
            q, k, v = self.qkv_proj(x).split(self.split, dim=-1)
        else:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = self.q_norm(q.view(1, length, h, hd)).transpose(1, 2)
        k = self.k_norm(k.view(1, length, kvh, hd)).transpose(1, 2)
        v = v.view(1, length, kvh, hd).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if isinstance(slot, int):
            k_cache[:, :, slot : slot + length] = k
            v_cache[:, :, slot : slot + length] = v
        else:
            k_cache.index_copy_(2, slot, k)
            v_cache.index_copy_(2, slot, v)
        keys = k_cache[:, :, :kv_len]
        values = v_cache[:, :, :kv_len]

        if length == 1:
            # Single query: fold the GQA group into the query axis instead of repeating K/V.
            qg = q.reshape(1, kvh, h // kvh, hd)
            attn_mask = None if mask is None else mask.view(1, 1, 1, kv_len)
            out = F.scaled_dot_product_attention(qg, keys, values, attn_mask=attn_mask)
            out = out.reshape(1, 1, h * hd)
        else:
            n_rep = h // kvh
            out = F.scaled_dot_product_attention(
                q, repeat_kv(keys, n_rep), repeat_kv(values, n_rep), is_causal=True
            )
            out = out.transpose(1, 2).reshape(1, length, h * hd)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.gate_up_proj: nn.Linear | None = None

    def fuse(self) -> None:
        self.gate_up_proj = _fuse_linears(self.gate_proj, self.up_proj)
        del self.gate_proj, self.up_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_up_proj is not None:
            gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        else:
            gate, up = self.gate_proj(x), self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)


class DecoderLayer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg.hidden_size, cfg.intermediate_size)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, k_cache, v_cache, slot, kv_len, mask):
        x = x + self.self_attn(
            self.input_layernorm(x), cos, sin, k_cache, v_cache, slot, kv_len, mask
        )
        return x + self.mlp(self.post_attention_layernorm(x))


class Decoder(nn.Module):
    """Decoder-only transformer body (`layers` + final `norm`) without embeddings or heads."""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def fuse(self) -> None:
        for layer in self.layers:
            layer.self_attn.fuse()
            layer.mlp.fuse()

    def new_cache(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        c = self.cfg
        return torch.zeros(
            c.num_layers, 2, 1, c.num_kv_heads, length, c.head_dim, device=device, dtype=dtype
        )

    def forward(self, x, cos, sin, cache, slot, kv_len, mask=None):
        cos, sin = cos[None, None], sin[None, None]
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, cache[i, 0], cache[i, 1], slot, kv_len, mask)
        return self.norm(x)
