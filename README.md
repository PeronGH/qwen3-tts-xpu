# qwen3-tts-xpu

A standalone, plain-PyTorch inference engine for the [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 12Hz models, tuned for Intel GPUs (XPU). It reproduces upstream's forward pass but replaces the transformers `generate` loop with a static-KV-cache decoder in which each audio frame (talker step + the 15 code-predictor steps) is one replayed XPU graph.

Supports all released 12Hz checkpoints (0.6B/1.7B CustomVoice, 1.7B VoiceDesign, 0.6B/1.7B Base voice cloning), batch size 1.

## Install

Arch Linux packages:

```
pacman -S python-pytorch-opt-xpu python-safetensors python-tokenizers python-soundfile python-soxr python-huggingface-hub
```

or `pip install -r requirements.txt` with an XPU build of PyTorch.

## Usage

```
python tts.py -m Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --speaker ryan "Hello there." -o out.wav
python tts.py -m Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --instruct "A calm, low voice." "Hi."
python tts.py -m Qwen/Qwen3-TTS-12Hz-0.6B-Base --ref-audio ref.wav --ref-text "Transcript." "Hi."
```

`-m` takes a Hugging Face repo id (downloaded on first use) or a local directory; `--stream` decodes audio in chunks while generating.

```python
from qwen3_tts_xpu import Qwen3TTS

tts = Qwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
prompt = tts.custom_voice_prompt("Hello there.", speaker="ryan")
wav = tts.synthesize(prompt)  # float32 numpy array at tts.sample_rate
for chunk in tts.stream(prompt):  # or stream audio chunks
    ...
```

## Tools

- `scripts/bench.py MODEL [--random-weights]`: generation and decode speed; `--random-weights` only needs the config/tokenizer files.
- `scripts/parity.py`: compares tokens, prompt embeddings, logits, greedy codes and decoded audio against upstream (see its docstring for the venv setup).
