import json
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class TransformerConfig:
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    rope_theta: float
    rms_norm_eps: float


@dataclass(frozen=True)
class TalkerConfig(TransformerConfig):
    text_vocab_size: int
    text_hidden_size: int
    num_code_groups: int
    codec_eos_id: int
    codec_think_id: int
    codec_nothink_id: int
    codec_think_bos_id: int
    codec_think_eos_id: int
    codec_pad_id: int
    codec_bos_id: int
    spk_id: dict[str, int]
    spk_is_dialect: dict[str, str | bool]
    codec_language_id: dict[str, int]


@dataclass(frozen=True)
class SpeakerEncoderConfig:
    enc_dim: int
    mel_dim: int = 128
    enc_channels: tuple[int, ...] = (512, 512, 512, 512, 1536)
    enc_kernel_sizes: tuple[int, ...] = (5, 3, 3, 3, 1)
    enc_dilations: tuple[int, ...] = (1, 2, 3, 4, 1)
    enc_attention_channels: int = 128
    enc_res2net_scale: int = 8
    enc_se_channels: int = 128
    sample_rate: int = 24000


@dataclass(frozen=True)
class ModelConfig:
    talker: TalkerConfig
    code_predictor: TransformerConfig
    speaker_encoder: SpeakerEncoderConfig | None
    model_type: str
    model_size: str
    tts_pad_token_id: int
    tts_bos_token_id: int
    tts_eos_token_id: int


@dataclass(frozen=True)
class CodecDecoderConfig:
    codebook_size: int
    codebook_dim: int
    hidden_size: int
    latent_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    num_layers: int
    rope_theta: float
    rms_norm_eps: float
    sliding_window: int
    layer_scale_initial_scale: float
    num_quantizers: int
    upsample_rates: tuple[int, ...]
    upsampling_ratios: tuple[int, ...]
    decoder_dim: int


@dataclass(frozen=True)
class MimiConfig:
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rope_theta: float
    norm_eps: float
    num_filters: int
    kernel_size: int
    last_kernel_size: int
    residual_kernel_size: int
    dilation_growth_rate: int
    compress: int
    num_residual_layers: int
    upsampling_ratios: tuple[int, ...]
    pad_mode: str
    codebook_size: int
    codebook_dim: int
    num_quantizers: int
    num_semantic_quantizers: int
    vq_hidden_dim: int
    sampling_rate: int
    frame_rate: float


@dataclass(frozen=True)
class CodecConfig:
    decoder: CodecDecoderConfig
    encoder: MimiConfig
    encoder_valid_num_quantizers: int
    sample_rate: int
    upsample_rate: int
    downsample_rate: int


@dataclass(frozen=True)
class GenerationParams:
    do_sample: bool = True
    top_k: int = 50
    top_p: float = 1.0
    temperature: float = 0.9
    repetition_penalty: float = 1.05
    subtalker_dosample: bool = True
    subtalker_top_k: int = 50
    subtalker_top_p: float = 1.0
    subtalker_temperature: float = 0.9
    max_new_tokens: int = 2048


def _transformer(c: dict) -> dict:
    return dict(
        hidden_size=c["hidden_size"],
        intermediate_size=c["intermediate_size"],
        num_layers=c["num_hidden_layers"],
        num_heads=c["num_attention_heads"],
        num_kv_heads=c["num_key_value_heads"],
        head_dim=c.get("head_dim", c["hidden_size"] // c["num_attention_heads"]),
        vocab_size=c["vocab_size"],
        rope_theta=c["rope_theta"],
        rms_norm_eps=c["rms_norm_eps"],
    )


def _check_plain_attention(name: str, c: dict) -> None:
    # Only full causal attention with plain RoPE is implemented. The talker's interleaved M-RoPE
    # reduces to plain RoPE because all three position streams are identical for text/codec input.
    if c.get("use_sliding_window") or c.get("sliding_window") is not None:
        raise NotImplementedError(f"{name}: sliding window attention")
    if c.get("hidden_act", "silu") != "silu" or c.get("attention_bias"):
        raise NotImplementedError(f"{name}: unexpected activation or attention bias")
    rope = c.get("rope_scaling") or {}
    if rope.get("rope_type", "default") != "default":
        raise NotImplementedError(f"{name}: rope type {rope['rope_type']}")


def load_model_config(path: Path) -> ModelConfig:
    c = json.loads((path / "config.json").read_text())
    t = c["talker_config"]
    cp = t["code_predictor_config"]
    _check_plain_attention("talker", t)
    _check_plain_attention("code_predictor", cp)
    if c.get("tokenizer_type") != "qwen3_tts_tokenizer_12hz":
        raise NotImplementedError(f"tokenizer type {c.get('tokenizer_type')}")
    talker = TalkerConfig(
        **_transformer(t),
        text_vocab_size=t["text_vocab_size"],
        text_hidden_size=t["text_hidden_size"],
        num_code_groups=t["num_code_groups"],
        codec_eos_id=t["codec_eos_token_id"],
        codec_think_id=t["codec_think_id"],
        codec_nothink_id=t["codec_nothink_id"],
        codec_think_bos_id=t["codec_think_bos_id"],
        codec_think_eos_id=t["codec_think_eos_id"],
        codec_pad_id=t["codec_pad_id"],
        codec_bos_id=t["codec_bos_id"],
        spk_id={k.lower(): v for k, v in (t.get("spk_id") or {}).items()},
        spk_is_dialect={k.lower(): v for k, v in (t.get("spk_is_dialect") or {}).items()},
        codec_language_id={k.lower(): v for k, v in (t.get("codec_language_id") or {}).items()},
    )
    speaker = None
    if c["tts_model_type"] == "base":
        se = c.get("speaker_encoder_config") or {}
        speaker = SpeakerEncoderConfig(
            **{k: tuple(v) if isinstance(v, list) else v for k, v in se.items()}
        )
    return ModelConfig(
        talker=talker,
        code_predictor=TransformerConfig(**_transformer(cp)),
        speaker_encoder=speaker,
        model_type=c["tts_model_type"],
        model_size=c["tts_model_size"],
        tts_pad_token_id=c["tts_pad_token_id"],
        tts_bos_token_id=c["tts_bos_token_id"],
        tts_eos_token_id=c["tts_eos_token_id"],
    )


def load_codec_config(path: Path) -> CodecConfig:
    c = json.loads((path / "config.json").read_text())
    d = c["decoder_config"]
    e = c["encoder_config"]
    decoder = CodecDecoderConfig(
        codebook_size=d["codebook_size"],
        codebook_dim=d["codebook_dim"],
        hidden_size=d["hidden_size"],
        latent_dim=d["latent_dim"],
        num_heads=d["num_attention_heads"],
        num_kv_heads=d["num_key_value_heads"],
        head_dim=d.get("head_dim", d["hidden_size"] // d["num_attention_heads"]),
        intermediate_size=d["intermediate_size"],
        num_layers=d["num_hidden_layers"],
        rope_theta=d["rope_theta"],
        rms_norm_eps=d["rms_norm_eps"],
        sliding_window=d["sliding_window"],
        layer_scale_initial_scale=d["layer_scale_initial_scale"],
        num_quantizers=d["num_quantizers"],
        upsample_rates=tuple(d["upsample_rates"]),
        upsampling_ratios=tuple(d["upsampling_ratios"]),
        decoder_dim=d["decoder_dim"],
    )
    hop = 1
    for r in e["upsampling_ratios"]:
        hop *= r
    encoder = MimiConfig(
        hidden_size=e["hidden_size"],
        intermediate_size=e["intermediate_size"],
        num_layers=e["num_hidden_layers"],
        num_heads=e["num_attention_heads"],
        num_kv_heads=e["num_key_value_heads"],
        head_dim=e.get("head_dim") or e["hidden_size"] // e["num_attention_heads"],
        rope_theta=e["rope_theta"],
        norm_eps=e["norm_eps"],
        num_filters=e["num_filters"],
        kernel_size=e["kernel_size"],
        last_kernel_size=e["last_kernel_size"],
        residual_kernel_size=e["residual_kernel_size"],
        dilation_growth_rate=e["dilation_growth_rate"],
        compress=e["compress"],
        num_residual_layers=e["num_residual_layers"],
        upsampling_ratios=tuple(e["upsampling_ratios"]),
        pad_mode=e["pad_mode"],
        codebook_size=e["codebook_size"],
        codebook_dim=e.get("codebook_dim") or e["hidden_size"],
        num_quantizers=e["num_quantizers"],
        num_semantic_quantizers=e["num_semantic_quantizers"],
        vq_hidden_dim=e["vector_quantization_hidden_dimension"],
        sampling_rate=e["sampling_rate"],
        frame_rate=e.get("_frame_rate") or e["sampling_rate"] / (hop * 2),
    )
    if not e.get("use_causal_conv", True) or e.get("use_conv_shortcut"):
        raise NotImplementedError("Mimi encoder: non-causal conv or conv shortcut")
    return CodecConfig(
        decoder=decoder,
        encoder=encoder,
        encoder_valid_num_quantizers=c["encoder_valid_num_quantizers"],
        sample_rate=c["output_sample_rate"],
        upsample_rate=c["decode_upsample_rate"],
        downsample_rate=c["encode_downsample_rate"],
    )


def load_generation_defaults(path: Path) -> GenerationParams:
    f = path / "generation_config.json"
    if not f.exists():
        return GenerationParams()
    c = json.loads(f.read_text())
    names = {f.name for f in fields(GenerationParams)}
    return GenerationParams(**{k: v for k, v in c.items() if k in names})
