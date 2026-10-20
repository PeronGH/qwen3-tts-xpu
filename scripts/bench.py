"""Benchmark generation and decode speed. Run from the repository root:

    python -m scripts.bench Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --frames 250

`--random-weights` measures speed without the checkpoints; it only needs the config and
tokenizer files, e.g.:

    hf download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --exclude "*.safetensors" --local-dir cfg
    python -m scripts.bench cfg --random-weights
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from qwen3_tts_xpu import Qwen3TTS, VoiceClonePrompt
from qwen3_tts_xpu.codec import SpeechTokenizer
from qwen3_tts_xpu.config import (
    load_codec_config,
    load_generation_defaults,
    load_model_config,
)
from qwen3_tts_xpu.engine import resolve_model_dir
from qwen3_tts_xpu.talker import Qwen3TTSModel
from qwen3_tts_xpu.text import TextTokenizer


def _randomize(module: torch.nn.Module, device: str, dtype: torch.dtype) -> torch.nn.Module:
    module = module.to_empty(device=device)
    with torch.no_grad():
        for name, t in [*module.named_parameters(), *module.named_buffers()]:
            if name.endswith(("norm.weight", "layernorm.weight", "cluster_usage", "initialized")):
                t.fill_(1.0)
            elif name.endswith(("alpha", "beta", "bias")):
                t.zero_()
            else:
                t.normal_(0.0, 0.02)
    return module.to(dtype).eval()


def random_tts(path: Path, device: str, dtype: torch.dtype, **kw) -> Qwen3TTS:
    cfg = load_model_config(path)
    codec_cfg = load_codec_config(path / "speech_tokenizer")
    with torch.device("meta"):
        model = Qwen3TTSModel(cfg)
        codec = SpeechTokenizer(codec_cfg)
    model = _randomize(model, device, dtype)
    model.fuse()
    codec = _randomize(codec, device, dtype)
    return Qwen3TTS(model, codec, TextTokenizer(path), load_generation_defaults(path), **kw)


def build_prompt(tts: Qwen3TTS, text: str):
    kind = tts.cfg.model_type
    if kind == "custom_voice":
        return tts.custom_voice_prompt(text, tts.supported_speakers[0], "auto")
    if kind == "voice_design":
        return tts.voice_design_prompt(text, "A calm, warm voice.", "auto")
    sr = 24000
    t = np.arange(3 * sr) / sr
    ref = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    clone = tts.create_voice_clone_prompt((ref, sr), ref_text="This is a reference.")
    assert isinstance(clone, VoiceClonePrompt)
    return tts.voice_clone_prompt(text, clone, "auto")


def sync(device: str) -> None:
    if device.startswith("xpu"):
        torch.xpu.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--random-weights", action="store_true")
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--frames", type=int, default=250, help="frames to generate (EOS ignored)")
    ap.add_argument("--no-graphs", action="store_true")
    ap.add_argument("--text", default="The quick brown fox jumps over the lazy dog. " * 3)
    args = ap.parse_args()

    kw = dict(max_seq_len=2048, use_graphs=not args.no_graphs)
    t0 = time.perf_counter()
    if args.random_weights:
        tts = random_tts(Path(args.model), args.device, torch.bfloat16, **kw)
    else:
        tts = Qwen3TTS.from_pretrained(resolve_model_dir(args.model), device=args.device, **kw)
    sync(args.device)
    print(f"load: {time.perf_counter() - t0:.1f}s  ({tts.cfg.model_type} {tts.cfg.model_size})")

    with torch.inference_mode():
        prompt = build_prompt(tts, args.text)
        gen = tts._params({"max_new_tokens": args.frames})
        print(f"prompt: {prompt.embeds.shape[1]} tokens")
        for label in ("warmup (captures graphs)", "timed"):
            sync(args.device)
            t0 = time.perf_counter()
            frames = list(
                tts.generator.generate(prompt.embeds, prompt.trailing, gen, ignore_eos=True)
            )
            sync(args.device)
            dt = time.perf_counter() - t0
            n = sum(f.shape[0] for f in frames)
            audio_s = n / 12.5
            print(
                f"{label}: {n} frames in {dt:.2f}s = {n / dt:.1f} frames/s, "
                f"{dt / n * 1000:.1f} ms/frame, RTF {dt / audio_s:.3f}"
            )
        codes = torch.cat(frames)
        for label in ("decode warmup", "decode"):
            sync(args.device)
            t0 = time.perf_counter()
            wav = tts.codec.decode(codes)
            sync(args.device)
            dt = time.perf_counter() - t0
            print(
                f"{label}: {wav.shape[0] / tts.sample_rate:.1f}s audio in {dt:.2f}s "
                f"(RTF {dt / (wav.shape[0] / tts.sample_rate):.3f})"
            )


if __name__ == "__main__":
    main()
