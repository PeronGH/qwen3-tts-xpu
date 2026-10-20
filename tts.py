"""Synthesize speech to a wav file.

python tts.py -m Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --speaker ryan "Hello there." -o out.wav
python tts.py -m Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --instruct "A calm, low voice." "Hi."
python tts.py -m Qwen/Qwen3-TTS-12Hz-0.6B-Base --ref-audio ref.wav --ref-text "..." "Hi."
"""

import argparse
import time

import numpy as np
import soundfile as sf
import torch

from qwen3_tts_xpu import Qwen3TTS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("text")
    ap.add_argument("-m", "--model", required=True, help="HF repo id or local directory")
    ap.add_argument("-o", "--out", default="out.wav")
    ap.add_argument("--language", default="auto")
    ap.add_argument("--speaker", help="CustomVoice speaker name")
    ap.add_argument("--instruct", help="style/voice instruction (CustomVoice 1.7B, VoiceDesign)")
    ap.add_argument("--ref-audio", help="reference audio for Base models")
    ap.add_argument("--ref-text", help="transcript of --ref-audio (omit for x-vector-only)")
    ap.add_argument("--stream", action="store_true", help="decode audio in chunks while generating")
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--max-new-tokens", type=int)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--no-graphs", action="store_true")
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    tts = Qwen3TTS.from_pretrained(args.model, device=args.device, use_graphs=not args.no_graphs)

    kind = tts.cfg.model_type
    if kind == "custom_voice":
        if not args.speaker:
            ap.error(f"--speaker is required; one of {tts.supported_speakers}")
        prompt = tts.custom_voice_prompt(args.text, args.speaker, args.language, args.instruct)
    elif kind == "voice_design":
        prompt = tts.voice_design_prompt(args.text, args.instruct or "", args.language)
    else:
        if not args.ref_audio:
            ap.error("--ref-audio is required for Base models")
        clone = tts.create_voice_clone_prompt(
            args.ref_audio, args.ref_text, x_vector_only_mode=not args.ref_text
        )
        prompt = tts.voice_clone_prompt(args.text, clone, args.language)

    gen = {}
    if args.max_new_tokens:
        gen["max_new_tokens"] = args.max_new_tokens
    if args.greedy:
        gen.update(do_sample=False, subtalker_dosample=False)

    t0 = time.perf_counter()
    if args.stream:
        chunks = []
        for chunk in tts.stream(prompt, **gen):
            if not chunks:
                print(f"first audio after {time.perf_counter() - t0:.2f}s")
            chunks.append(chunk)
        wav = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    else:
        wav = tts.synthesize(prompt, **gen)
    dt = time.perf_counter() - t0
    secs = wav.shape[0] / tts.sample_rate
    print(f"{secs:.2f}s audio in {dt:.2f}s (RTF {dt / max(secs, 1e-9):.3f}) -> {args.out}")
    sf.write(args.out, wav, tts.sample_rate)


if __name__ == "__main__":
    main()
