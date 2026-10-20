# qwen3-tts-xpu

A standalone, plain-PyTorch inference engine for the [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 12Hz models, tuned for Intel GPUs (XPU). It reproduces upstream's forward pass but replaces the transformers `generate` loop with a static-KV-cache decoder in which each audio frame (talker step + the 15 code-predictor steps) is one replayed XPU graph.

Supports all released 12Hz checkpoints (0.6B/1.7B CustomVoice, 1.7B VoiceDesign, 0.6B/1.7B Base voice cloning), batch size 1.

Other PyTorch devices (`--device cuda`, `mps`, `cpu`) are best effort: they run the same code without graph replay and are untested.

## Install

Install an XPU build of PyTorch first (e.g. `pip install torch --index-url https://download.pytorch.org/whl/xpu`), then:

```
pip install -r requirements.txt
```

On Arch Linux the same dependencies are available as system packages:

```
pacman -S python-pytorch-opt-xpu python-safetensors python-tokenizers python-soundfile python-soxr python-huggingface-hub
```

## Usage

```
python tts.py -m Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --speaker ryan "Hello there." -o out.wav
python tts.py -m Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --instruct "A calm, low voice." "Hi."
```

`-m` takes a Hugging Face repo id (downloaded on first use) or a local directory; `--stream` decodes audio in chunks while generating.

### Voice cloning

Voice cloning uses the Base models. Give a reference recording and its transcript; without `--ref-text` only the speaker embedding is used, which is easier but less faithful.

```
python tts.py -m Qwen/Qwen3-TTS-12Hz-0.6B-Base --ref-audio ref.wav --ref-text "Transcript." "Hi."
```

To reuse a voice without re-encoding the reference, save it once (text is optional here) and load it later:

```
python tts.py -m Qwen/Qwen3-TTS-12Hz-0.6B-Base --ref-audio ref.wav --ref-text "Transcript." --save-voice voice.pt
python tts.py -m Qwen/Qwen3-TTS-12Hz-0.6B-Base --voice voice.pt "Hi."
```

Voice files use the same format as the official web demo's "Save / Load Voice", so they work in both directions. A voice only fits models of the same size (0.6B or 1.7B).

### Python

```python
from qwen3_tts_xpu import Qwen3TTS

tts = Qwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
prompt = tts.custom_voice_prompt("Hello there.", speaker="ryan")
wav = tts.synthesize(prompt)  # float32 numpy array at tts.sample_rate
for chunk in tts.stream(prompt):  # or stream audio chunks
    ...

# Voice cloning (Base models)
from qwen3_tts_xpu import VoiceClonePrompt

base = Qwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base")
voice = base.create_voice_clone_prompt("ref.wav", ref_text="Transcript.")
voice.save("voice.pt")
wav = base.synthesize(base.voice_clone_prompt("Hi.", VoiceClonePrompt.load("voice.pt")))
```

## Tools

Run from the repository root:

- `python -m scripts.bench MODEL [--random-weights]`: generation and decode speed; `--random-weights` only needs the config and tokenizer files.
- `python -m scripts.parity --upstream PATH/TO/Qwen3-TTS MODEL`: compares tokens, prompt embeddings, logits, greedy codes and decoded audio against upstream (see the script's docstring for its dependencies).
