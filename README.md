# qwen3-tts-xpu

A standalone, plain-PyTorch inference engine for the [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 12Hz models, tuned for Intel GPUs (XPU). It reproduces upstream's forward pass but replaces the transformers `generate` loop with a static-KV-cache decoder in which each audio frame (talker step + the 15 code-predictor steps) is one replayed XPU graph.

Supports all released 12Hz checkpoints (0.6B/1.7B CustomVoice, 1.7B VoiceDesign, 0.6B/1.7B Base voice cloning), batch size 1.

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
python tts.py -m Qwen/Qwen3-TTS-12Hz-0.6B-Base --ref-audio ref.wav --ref-text "Transcript." "Hi."
```

`-m` takes a Hugging Face repo id (downloaded on first use) or a local directory; `--stream` decodes audio in chunks while generating.

Base models can save a cloned voice with `--save-voice voice.pt` and reuse it with `--voice voice.pt`. The file format is the same as the official web demo's "Save / Load Voice", so files work in both directions; a voice only fits models of the same size (0.6B or 1.7B).

```python
from qwen3_tts_xpu import Qwen3TTS

tts = Qwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
prompt = tts.custom_voice_prompt("Hello there.", speaker="ryan")
wav = tts.synthesize(prompt)  # float32 numpy array at tts.sample_rate
for chunk in tts.stream(prompt):  # or stream audio chunks
    ...
```

## Tools

Run from the repository root:

- `python -m scripts.bench MODEL [--random-weights]`: generation and decode speed; `--random-weights` only needs the config and tokenizer files.
- `python -m scripts.parity --upstream PATH/TO/Qwen3-TTS MODEL`: compares tokens, prompt embeddings, logits, greedy codes and decoded audio against upstream (see the script's docstring for its dependencies).
