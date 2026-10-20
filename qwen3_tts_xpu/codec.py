"""Qwen3-TTS 12Hz speech tokenizer: Mimi encoder (audio -> codes) and decoder (codes -> audio)."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import CodecConfig, CodecDecoderConfig, MimiConfig
from .layers import RMSNorm, apply_rope, repeat_kv, rope_table


def window_causal_mask(length: int, window: int | None, device: torch.device) -> torch.Tensor:
    q = torch.arange(length, device=device)[:, None]
    k = torch.arange(length, device=device)[None, :]
    mask = k <= q
    if window is not None:
        mask &= k > q - window
    return mask


class SeqAttention(nn.Module):
    """Full-sequence multi-head attention with RoPE, no KV cache."""

    def __init__(self, hidden: int, heads: int, kv_heads: int, head_dim: int):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)

    def forward(self, x, cos, sin, mask):
        b, length, _ = x.shape
        q = self.q_proj(x).view(b, length, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, length, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, length, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        n_rep = self.heads // self.kv_heads
        out = F.scaled_dot_product_attention(
            q, repeat_kv(k, n_rep), repeat_kv(v, n_rep), attn_mask=mask
        )
        return self.o_proj(out.transpose(1, 2).reshape(b, length, -1))


class LayerScale(nn.Module):
    def __init__(self, dim: int, init: float):
        super().__init__()
        self.scale = nn.Parameter(torch.full((dim,), init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x


# ---------------------------------------------------------------------------------------------
# Decoder


class CausalConv(nn.Module):
    def __init__(self, cin, cout, kernel, dilation=1, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, kernel, stride=stride, dilation=dilation, groups=groups)
        self.stride = stride
        self.kernel = (kernel - 1) * dilation + 1
        self.padding = self.kernel - stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        n_frames = (length - self.kernel + self.padding) / self.stride + 1
        ideal = (math.ceil(n_frames) - 1) * self.stride + (self.kernel - self.padding)
        x = F.pad(x, (self.padding, ideal - length))
        return self.conv(x)


class CausalTransConv(nn.Module):
    def __init__(self, cin, cout, kernel, stride=1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(cin, cout, kernel, stride=stride)
        self.right_pad = kernel - stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.right_pad > 0:
            x = x[..., : x.shape[-1] - self.right_pad]
        return x


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = CausalConv(dim, dim, 7, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dwconv(x).permute(0, 2, 1)
        h = self.pwconv2(F.gelu(self.pwconv1(self.norm(h))))
        return x + (self.gamma * h).permute(0, 2, 1)


class SnakeBeta(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(self.alpha[None, :, None])
        beta = torch.exp(self.beta[None, :, None])
        return x + (1.0 / (beta + 1e-9)) * torch.pow(torch.sin(x * alpha), 2)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int):
        super().__init__()
        self.act1 = SnakeBeta(dim)
        self.conv1 = CausalConv(dim, dim, 7, dilation=dilation)
        self.act2 = SnakeBeta(dim)
        self.conv2 = CausalConv(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.act2(self.conv1(self.act1(x))))


class DecoderBlock(nn.Module):
    def __init__(self, cfg: CodecDecoderConfig, idx: int):
        super().__init__()
        cin = cfg.decoder_dim // 2**idx
        cout = cfg.decoder_dim // 2 ** (idx + 1)
        rate = cfg.upsample_rates[idx]
        self.block = nn.ModuleList(
            [
                SnakeBeta(cin),
                CausalTransConv(cin, cout, 2 * rate, rate),
                *(ResidualUnit(cout, d) for d in (1, 3, 9)),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.block:
            x = b(x)
        return x


class DecoderTransformerLayer(nn.Module):
    def __init__(self, cfg: CodecDecoderConfig):
        super().__init__()
        d = cfg.hidden_size
        self.self_attn = SeqAttention(d, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim)
        self.mlp = _GatedMLP(d, cfg.intermediate_size)
        self.input_layernorm = RMSNorm(d, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(d, cfg.rms_norm_eps)
        self.self_attn_layer_scale = LayerScale(d, cfg.layer_scale_initial_scale)
        self.mlp_layer_scale = LayerScale(d, cfg.layer_scale_initial_scale)

    def forward(self, x, cos, sin, mask):
        x = x + self.self_attn_layer_scale(self.self_attn(self.input_layernorm(x), cos, sin, mask))
        return x + self.mlp_layer_scale(self.mlp(self.post_attention_layernorm(x)))


class _GatedMLP(nn.Module):
    def __init__(self, d: int, inter: int):
        super().__init__()
        self.gate_proj = nn.Linear(d, inter, bias=False)
        self.up_proj = nn.Linear(d, inter, bias=False)
        self.down_proj = nn.Linear(inter, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderTransformer(nn.Module):
    def __init__(self, cfg: CodecDecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.latent_dim, cfg.hidden_size)
        self.layers = nn.ModuleList([DecoderTransformerLayer(cfg) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.output_proj = nn.Linear(cfg.hidden_size, cfg.latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        length = x.shape[1]
        cos, sin = rope_table(length, self.cfg.head_dim, self.cfg.rope_theta, x.device, x.dtype)
        mask = window_causal_mask(length, self.cfg.sliding_window, x.device)
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        return self.output_proj(self.norm(x))


class Codebook(nn.Module):
    def __init__(self, size: int, dim: int):
        super().__init__()
        self.cluster_usage = nn.Parameter(torch.ones(size))
        self.embedding_sum = nn.Parameter(torch.zeros(size, dim))

    def embedding(self) -> torch.Tensor:
        return self.embedding_sum / self.cluster_usage.clamp(min=1e-5)[:, None]


class _VQ(nn.Module):
    def __init__(self, size: int, dim: int):
        super().__init__()
        self._codebook = Codebook(size, dim)


class _VQLayers(nn.Module):
    def __init__(self, n: int, size: int, dim: int):
        super().__init__()
        self.layers = nn.ModuleList([_VQ(size, dim) for _ in range(n)])


class RVQDecoder(nn.Module):
    def __init__(self, n_q: int, size: int, dim: int, io_dim: int):
        super().__init__()
        self.input_proj = nn.Conv1d(io_dim, dim, 1, bias=False)
        self.output_proj = nn.Conv1d(dim, io_dim, 1, bias=False)
        self.vq = _VQLayers(n_q, size, dim)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes [B, K, T] -> [B, io_dim, T]"""
        q = sum(
            F.embedding(codes[:, i], layer._codebook.embedding()).transpose(1, 2)
            for i, layer in enumerate(self.vq.layers[: codes.shape[1]])
        )
        return self.output_proj(q)


class SplitRVQDecoder(nn.Module):
    def __init__(self, cfg: CodecDecoderConfig):
        super().__init__()
        dim = cfg.codebook_dim // 2
        self.rvq_first = RVQDecoder(1, cfg.codebook_size, dim, cfg.codebook_dim)
        self.rvq_rest = RVQDecoder(cfg.num_quantizers - 1, cfg.codebook_size, dim, cfg.codebook_dim)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        q = self.rvq_first.decode(codes[:, :1])
        if codes.shape[1] > 1:
            q = q + self.rvq_rest.decode(codes[:, 1:])
        return q


class CodecDecoder(nn.Module):
    def __init__(self, cfg: CodecDecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.total_upsample = math.prod(cfg.upsample_rates + cfg.upsampling_ratios)
        self.pre_transformer = DecoderTransformer(cfg)
        self.quantizer = SplitRVQDecoder(cfg)
        self.pre_conv = CausalConv(cfg.codebook_dim, cfg.latent_dim, 3)
        self.upsample = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        CausalTransConv(cfg.latent_dim, cfg.latent_dim, f, f),
                        ConvNeXtBlock(cfg.latent_dim),
                    ]
                )
                for f in cfg.upsampling_ratios
            ]
        )
        out_dim = cfg.decoder_dim // 2 ** len(cfg.upsample_rates)
        self.decoder = nn.ModuleList(
            [
                CausalConv(cfg.latent_dim, cfg.decoder_dim, 7),
                *(DecoderBlock(cfg, i) for i in range(len(cfg.upsample_rates))),
                SnakeBeta(out_dim),
                CausalConv(out_dim, 1, 7),
            ]
        )

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """codes [B, K, T] -> wav [B, 1, T * total_upsample]"""
        h = self.quantizer.decode(codes)
        h = self.pre_conv(h).transpose(1, 2)
        h = self.pre_transformer(h).permute(0, 2, 1)
        for blocks in self.upsample:
            for block in blocks:
                h = block(h)
        for block in self.decoder:
            h = block(h)
        return h.clamp(min=-1, max=1)

    def chunked(
        self, codes: torch.Tensor, chunk: int = 300, left_context: int = 25
    ) -> torch.Tensor:
        wavs = []
        start = 0
        while start < codes.shape[-1]:
            end = min(start + chunk, codes.shape[-1])
            ctx = left_context if start - left_context > 0 else start
            wav = self(codes[..., start - ctx : end])
            wavs.append(wav[..., ctx * self.total_upsample :])
            start = end
        return torch.cat(wavs, dim=-1)


# ---------------------------------------------------------------------------------------------
# Encoder (Mimi encoder half, as in transformers' MimiModel)


class MimiConv(nn.Module):
    def __init__(self, cin, cout, kernel, stride=1, dilation=1, bias=True, pad_mode="constant"):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, kernel, stride, dilation=dilation, bias=bias)
        self.stride = stride
        self.kernel = (kernel - 1) * dilation + 1
        self.padding_total = self.kernel - stride
        self.pad_mode = pad_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        n_frames = math.ceil((length - self.kernel + self.padding_total) / self.stride + 1) - 1
        extra = n_frames * self.stride + self.kernel - self.padding_total - length
        x = F.pad(x, (self.padding_total, extra), mode=self.pad_mode)
        return self.conv(x)


class MimiResnetBlock(nn.Module):
    def __init__(self, cfg: MimiConfig, dim: int, dilations: tuple[int, int]):
        super().__init__()
        hidden = dim // cfg.compress
        kernels = (cfg.residual_kernel_size, 1)
        block: list[nn.Module] = []
        for i, (k, d) in enumerate(zip(kernels, dilations, strict=True)):
            cin = dim if i == 0 else hidden
            cout = dim if i == len(kernels) - 1 else hidden
            block += [nn.ELU(), MimiConv(cin, cout, k, dilation=d, pad_mode=cfg.pad_mode)]
        self.block = nn.ModuleList(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.block:
            h = layer(h)
        return x + h


class MimiSEANetEncoder(nn.Module):
    def __init__(self, cfg: MimiConfig):
        super().__init__()
        pm = cfg.pad_mode
        layers: list[nn.Module] = [MimiConv(1, cfg.num_filters, cfg.kernel_size, pad_mode=pm)]
        scale = 1
        for ratio in reversed(cfg.upsampling_ratios):
            dim = scale * cfg.num_filters
            layers += [
                MimiResnetBlock(cfg, dim, (cfg.dilation_growth_rate**j, 1))
                for j in range(cfg.num_residual_layers)
            ]
            layers += [nn.ELU(), MimiConv(dim, dim * 2, ratio * 2, stride=ratio, pad_mode=pm)]
            scale *= 2
        layers += [
            nn.ELU(),
            MimiConv(scale * cfg.num_filters, cfg.hidden_size, cfg.last_kernel_size, pad_mode=pm),
        ]
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class _MimiMLP(nn.Module):
    def __init__(self, d: int, inter: int):
        super().__init__()
        self.fc1 = nn.Linear(d, inter, bias=False)
        self.fc2 = nn.Linear(inter, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class MimiTransformerLayer(nn.Module):
    def __init__(self, cfg: MimiConfig):
        super().__init__()
        d = cfg.hidden_size
        self.self_attn = SeqAttention(d, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim)
        self.mlp = _MimiMLP(d, cfg.intermediate_size)
        self.input_layernorm = nn.LayerNorm(d, eps=cfg.norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(d, eps=cfg.norm_eps)
        self.self_attn_layer_scale = LayerScale(d, 0.01)
        self.mlp_layer_scale = LayerScale(d, 0.01)

    def forward(self, x, cos, sin, mask):
        x = x + self.self_attn_layer_scale(self.self_attn(self.input_layernorm(x), cos, sin, mask))
        return x + self.mlp_layer_scale(self.mlp(self.post_attention_layernorm(x)))


class MimiTransformer(nn.Module):
    def __init__(self, cfg: MimiConfig):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([MimiTransformerLayer(cfg) for _ in range(cfg.num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        cos, sin = rope_table(length, self.cfg.head_dim, self.cfg.rope_theta, x.device, x.dtype)
        # transformers' Mimi uses a plain causal mask here (its sliding_window is not applied).
        mask = window_causal_mask(length, None, x.device)
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        return x


class MimiCodebook(nn.Module):
    def __init__(self, size: int, dim: int):
        super().__init__()
        self.register_buffer("initialized", torch.ones(1))
        self.register_buffer("cluster_usage", torch.ones(size))
        self.register_buffer("embed_sum", torch.zeros(size, dim))

    def embed(self) -> torch.Tensor:
        return self.embed_sum / self.cluster_usage.clamp(min=1e-5)[:, None]


class _MimiVQ(nn.Module):
    def __init__(self, size: int, dim: int):
        super().__init__()
        self.codebook = MimiCodebook(size, dim)


class MimiRVQ(nn.Module):
    def __init__(self, cfg: MimiConfig, n_q: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [_MimiVQ(cfg.codebook_size, cfg.codebook_dim) for _ in range(n_q)]
        )
        self.input_proj = nn.Conv1d(cfg.hidden_size, cfg.vq_hidden_dim, 1, bias=False)
        self.output_proj = nn.Conv1d(cfg.vq_hidden_dim, cfg.hidden_size, 1, bias=False)

    def encode(self, x: torch.Tensor, n_q: int) -> torch.Tensor:
        """x [B, hidden, T] -> codes [n_q, B, T]"""
        residual = self.input_proj(x)
        codes = []
        for layer in self.layers[:n_q]:
            embed = layer.codebook.embed()
            flat = residual.permute(0, 2, 1)
            shape = flat.shape
            flat = flat.reshape(-1, shape[-1])
            idx = torch.cdist(flat[None].float(), embed[None].float(), p=2)[0].argmin(dim=-1)
            idx = idx.view(*shape[:-1])
            residual = residual - F.embedding(idx, embed).permute(0, 2, 1)
            codes.append(idx)
        return torch.stack(codes)


class MimiSplitRVQ(nn.Module):
    def __init__(self, cfg: MimiConfig):
        super().__init__()
        self.n_semantic = cfg.num_semantic_quantizers
        self.semantic_residual_vector_quantizer = MimiRVQ(cfg, cfg.num_semantic_quantizers)
        self.acoustic_residual_vector_quantizer = MimiRVQ(
            cfg, cfg.num_quantizers - cfg.num_semantic_quantizers
        )

    def encode(self, x: torch.Tensor, n_q: int) -> torch.Tensor:
        codes = self.semantic_residual_vector_quantizer.encode(x, self.n_semantic)
        if n_q > self.n_semantic:
            acoustic = self.acoustic_residual_vector_quantizer.encode(x, n_q - self.n_semantic)
            codes = torch.cat([codes, acoustic], dim=0)
        return codes


class CodecEncoder(nn.Module):
    def __init__(self, cfg: MimiConfig):
        super().__init__()
        self.encoder = MimiSEANetEncoder(cfg)
        self.encoder_transformer = MimiTransformer(cfg)
        hop = math.prod(cfg.upsampling_ratios)
        encodec_frame_rate = math.ceil(cfg.sampling_rate / hop)
        self.downsample = MimiConv(
            cfg.hidden_size,
            cfg.hidden_size,
            2 * int(encodec_frame_rate / cfg.frame_rate),
            stride=2,
            bias=False,
            pad_mode="replicate",
        )
        self.quantizer = MimiSplitRVQ(cfg)

    def forward(self, wav: torch.Tensor, n_q: int) -> torch.Tensor:
        """wav [B, 1, N] -> codes [B, n_q, T]"""
        h = self.encoder(wav)
        h = self.encoder_transformer(h.transpose(1, 2)).transpose(1, 2)
        h = self.downsample(h)
        return self.quantizer.encode(h, n_q).transpose(0, 1)


class SpeechTokenizer(nn.Module):
    def __init__(self, cfg: CodecConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = CodecEncoder(cfg.encoder)
        self.decoder = CodecDecoder(cfg.decoder)

    def _dtype(self) -> torch.dtype:
        return self.decoder.pre_conv.conv.weight.dtype

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """Mono waveform [N] at `cfg.sample_rate` -> codes [T, Q]."""
        n = wav.shape[-1]
        x = wav.to(self._dtype()).view(1, 1, n)
        codes = self.encoder(x, self.cfg.encoder_valid_num_quantizers)[0]
        return codes[:, : -(-n // self.cfg.downsample_rate)].transpose(0, 1)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes [T, Q] -> float32 waveform [T * upsample_rate]"""
        wav = self.decoder.chunked(codes.clamp(min=0).transpose(0, 1)[None])
        return wav[0, 0, : codes.shape[0] * self.cfg.upsample_rate].float()
