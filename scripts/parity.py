"""Compare this implementation against upstream Qwen3-TTS on the same device and weights.

Needs a checkout of https://github.com/QwenLM/Qwen3-TTS and an environment with an XPU build of
PyTorch plus upstream's pinned dependencies (keep it separate from the main environment):

    pip install transformers==4.57.3 accelerate==1.12.0 librosa einops

Then, from this repository's root (Base models also need --ref-audio and --ref-text):

    python -m scripts.parity --upstream PATH/TO/Qwen3-TTS MODEL

Checks: text token ids, prefill embeddings, talker prefill logits, greedy codes, codec decode
and, for Base models, reference codes and speaker embedding. bf16 kernels differ slightly
between the two (fused projections, attention paths), so greedy codes may diverge late.
"""

import argparse
import contextlib
import sys
import types

import numpy as np
import torch

TEXT = "Hello world. This is a quick test of the speech synthesis system."


def stub_25hz_deps() -> None:
    # Upstream imports the 25Hz tokenizer eagerly; none of these are used by the 12Hz models.
    for name in ("sox", "onnxruntime", "torchaudio", "torchaudio.compliance"):
        sys.modules.setdefault(name, types.ModuleType(name))
    kaldi = types.ModuleType("torchaudio.compliance.kaldi")
    sys.modules["torchaudio.compliance.kaldi"] = kaldi
    sys.modules["torchaudio.compliance"].kaldi = kaldi


def report(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    a, b = a.float().cpu(), b.float().cpu()
    if a.shape != b.shape:
        print(f"[{name}] SHAPE MISMATCH {tuple(a.shape)} vs {tuple(b.shape)}")
        return
    diff = (a - b).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    print(f"[{name}] max|diff|={diff:.4g} cos={cos:.6f} scale={b.abs().max().item():.4g}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--upstream", required=True, help="path to the Qwen3-TTS repo")
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--ref-audio")
    ap.add_argument("--ref-text")
    args = ap.parse_args()

    stub_25hz_deps()
    sys.path.insert(0, args.upstream)
    from qwen_tts import Qwen3TTSModel as Upstream

    from qwen3_tts_xpu import Qwen3TTS

    dev = args.device
    up = Upstream.from_pretrained(
        args.model, device_map=dev, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    ours = Qwen3TTS.from_pretrained(args.model, device=dev)
    kind = ours.cfg.model_type

    # 1. tokenizer
    for s in [TEXT, "其实我真的有发现,我是一个特别善于观察别人情绪的人。"]:
        a = ours.tokenizer.encode(ours.tokenizer.assistant_text(s))
        b = up._tokenize_texts([up._build_assistant_text(s)])[0][0].tolist()
        print(f"[tokens] equal={a == b} ({len(a)} vs {len(b)})")

    # 2. inputs to the upstream talker.generate, captured by patching it
    captured = {}

    class Stop(Exception):
        pass

    def capture(**kw):
        captured.update(kw)
        raise Stop

    speaker = ours.supported_speakers[0] if kind == "custom_voice" else None
    instruct = "Speak slowly in a warm, deep voice." if kind == "voice_design" else None
    ids = up._tokenize_texts([up._build_assistant_text(TEXT)])
    upstream_kwargs = dict(input_ids=ids, languages=["auto"])
    with torch.inference_mode():
        if kind == "custom_voice":
            upstream_kwargs.update(speakers=[speaker], instruct_ids=[None], non_streaming_mode=True)
            prompt = ours.custom_voice_prompt(TEXT, speaker)
        elif kind == "voice_design":
            ins = up._tokenize_texts([up._build_instruct_text(instruct)])[0]
            upstream_kwargs.update(instruct_ids=[ins], non_streaming_mode=True)
            prompt = ours.voice_design_prompt(TEXT, instruct)
        else:
            if not args.ref_audio:
                ap.error("--ref-audio/--ref-text are required for Base models")
            items = up.create_voice_clone_prompt(args.ref_audio, args.ref_text)
            clone = ours.create_voice_clone_prompt(args.ref_audio, args.ref_text)
            a, b = clone.ref_code.cpu(), items[0].ref_code.cpu()
            match = (a == b).float().mean().item() if a.shape == b.shape else 0.0
            print(f"[ref codes] {tuple(a.shape)} vs {tuple(b.shape)} match={match:.4f}")
            report("speaker embedding", clone.spk_embedding, items[0].ref_spk_embedding)
            clone.ref_code = items[0].ref_code  # isolate the talker comparison from encoder diffs
            ref_ids = [up._tokenize_texts([up._build_ref_text(args.ref_text)])[0]]
            upstream_kwargs.update(
                ref_ids=ref_ids,
                voice_clone_prompt=up._prompt_items_to_voice_clone_prompt(items),
                non_streaming_mode=False,
            )
            prompt = ours.voice_clone_prompt(TEXT, clone)

        talker_generate = up.model.talker.generate
        up.model.talker.generate = capture
        with contextlib.suppress(Stop):
            up.model.generate(**upstream_kwargs)
        up.model.talker.generate = talker_generate
        report("prefill embeds", prompt.embeds, captured["inputs_embeds"])
        n = prompt.trailing.shape[0] - 1
        report("trailing text", prompt.trailing[:n], captured["trailing_text_hidden"][0, :n])

        # 3. talker prefill logits on identical inputs
        emb = captured["inputs_embeds"]
        out = up.model.talker(
            inputs_embeds=emb,
            attention_mask=torch.ones(emb.shape[:2], device=emb.device, dtype=torch.long),
            trailing_text_hidden=captured["trailing_text_hidden"],
            tts_pad_embed=captured["tts_pad_embed"],
            use_cache=False,
        )
        g = ours.generator
        length = emb.shape[1]
        h = ours.model.talker.model(emb, g.cos[:length], g.sin[:length], g.cache, 0, length)
        report("prefill logits", ours.model.talker.codec_head(h[:, -1]), out.logits[:, -1])

        # 4. greedy codes
        greedy = dict(do_sample=False, subtalker_dosample=False, max_new_tokens=args.frames)
        up_codes = up.model.generate(**upstream_kwargs, **greedy)[0][0].cpu()
        our_codes = torch.cat(list(ours.generate_codes(prompt, **greedy))).cpu()
        same = 0
        while same < min(len(up_codes), len(our_codes)) and torch.equal(
            up_codes[same], our_codes[same]
        ):
            same += 1
        print(
            f"[greedy codes] upstream {len(up_codes)} frames, ours {len(our_codes)}, "
            f"identical prefix {same}"
        )

        # 5. codec decode of the same codes
        up_wav = up.model.speech_tokenizer.decode([{"audio_codes": up_codes}])[0][0]
        our_wav = ours.codec.decode(up_codes.to(dev)).cpu().numpy()
        report("decoded wav", torch.from_numpy(our_wav), torch.from_numpy(np.asarray(up_wav)))


if __name__ == "__main__":
    main()
