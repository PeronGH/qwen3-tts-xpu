import base64
import io
import urllib.request
from urllib.parse import urlparse

import numpy as np
import soundfile as sf
import soxr

AudioInput = str | tuple[np.ndarray, int]


def _read_bytes(data: bytes) -> tuple[np.ndarray, int]:
    with io.BytesIO(data) as f:
        audio, sr = sf.read(f, dtype="float32", always_2d=False)
    return audio, sr


def load_audio(x: AudioInput) -> tuple[np.ndarray, int]:
    """Load a path, http(s) URL, base64 string or (waveform, sr) as mono float32."""
    if isinstance(x, tuple):
        audio, sr = np.asarray(x[0], dtype=np.float32), int(x[1])
    elif urlparse(x).scheme in ("http", "https"):
        with urllib.request.urlopen(x) as resp:
            audio, sr = _read_bytes(resp.read())
    elif x.startswith("data:audio") or ("/" not in x and "\\" not in x and len(x) > 256):
        audio, sr = _read_bytes(
            base64.b64decode(x.split(",", 1)[1] if x.startswith("data:") else x)
        )
    else:
        audio, sr = sf.read(x, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    return audio.astype(np.float32), int(sr)


def resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return audio
    return soxr.resample(audio, sr, target_sr, quality="HQ").astype(np.float32)
