"""Stream safetensors checkpoints into device tensors without mmap or page cache.

The file is read with O_DIRECT through one reusable page-aligned host buffer, so host memory
stays at the buffer size and nothing is left in the page cache (which matters on WSL, where
cached pages are slow to return to the host). Filesystems without O_DIRECT fall back to
buffered reads whose pages are dropped right after use.
"""

import json
import mmap
import os
import struct
from pathlib import Path

import torch
from torch import nn

_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}
_ALIGN = 4096


def _fadvise(fd: int, offset: int, length: int, advice: str) -> None:
    if hasattr(os, "posix_fadvise"):  # no-op where unavailable (e.g. macOS)
        os.posix_fadvise(fd, offset, length, getattr(os, advice))


class _Reader:
    def __init__(self, path: Path):
        self.path = path
        try:
            self.fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECT", 0))
            self.direct = True
        except OSError:
            self._open_buffered()

    def _open_buffered(self) -> None:
        self.fd = os.open(self.path, os.O_RDONLY)
        self.direct = False
        # No readahead: it would cache pages beyond each chunk that DONTNEED never sees.
        _fadvise(self.fd, 0, 0, "POSIX_FADV_RANDOM")

    def read(self, buf: mmap.mmap, pos: int) -> int:
        """Fill `buf` from the page-aligned file offset `pos`; returns bytes read."""
        try:
            n = os.preadv(self.fd, [buf], pos)
        except OSError:
            if not self.direct:
                raise
            os.close(self.fd)  # O_DIRECT accepted at open but refused on read
            self._open_buffered()
            n = os.preadv(self.fd, [buf], pos)
        if not self.direct:
            _fadvise(self.fd, pos, n, "POSIX_FADV_DONTNEED")
        return n

    def close(self) -> None:
        if not self.direct:
            _fadvise(self.fd, 0, 0, "POSIX_FADV_DONTNEED")
        os.close(self.fd)


def _header(path: Path) -> tuple[int, dict]:
    with open(path, "rb", buffering=0) as f:
        _fadvise(f.fileno(), 0, 0, "POSIX_FADV_RANDOM")
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        _fadvise(f.fileno(), 0, 0, "POSIX_FADV_DONTNEED")
    header.pop("__metadata__", None)
    return 8 + n, header


@torch.no_grad()
def load_safetensors(module: nn.Module, files: list[Path], buffer_mb: int = 128) -> None:
    """Copy all tensors of `files` into the matching (already allocated) state of `module`.

    Strict: every state entry must be present with the same shape. Dtype conversion, if the
    file dtype differs from the parameter's, happens on the parameter's device.
    """
    targets = module.state_dict()
    seen: set[str] = set()
    buf = mmap.mmap(-1, buffer_mb << 20)
    host = torch.frombuffer(buf, dtype=torch.uint8)
    try:
        for path in files:
            data_start, header = _header(path)
            entries = []
            for name, spec in header.items():
                if name not in targets:
                    raise KeyError(f"{path.name}: unexpected tensor {name}")
                dst = targets[name]
                if tuple(spec["shape"]) != tuple(dst.shape):
                    raise ValueError(f"{name}: shape {spec['shape']} != {tuple(dst.shape)}")
                src_dtype = _DTYPES[spec["dtype"]]
                staging = dst if src_dtype == dst.dtype else torch.empty_like(dst, dtype=src_dtype)
                start, end = (data_start + o for o in spec["data_offsets"])
                entries.append((start, end, name, staging))
                seen.add(name)
            entries.sort(key=lambda e: e[0])
            _stream(path, entries, buf, host)
            for _, _, name, staging in entries:
                if staging is not targets[name]:
                    targets[name].copy_(staging)
    finally:
        del host
        buf.close()
    missing = set(targets) - seen
    if missing:
        raise KeyError(f"missing tensors: {sorted(missing)[:5]} ({len(missing)} total)")


def _stream(path: Path, entries: list, buf: mmap.mmap, host: torch.Tensor) -> None:
    if not entries:
        return
    reader = _Reader(path)
    try:
        pos = entries[0][0] // _ALIGN * _ALIGN
        first = 0
        while first < len(entries):
            got = reader.read(buf, pos)
            if got <= 0:
                raise EOFError(f"{path}: truncated at byte {pos}")
            window_end = pos + got
            i = first
            while i < len(entries) and entries[i][0] < window_end:
                start, end, _, staging = entries[i]
                lo, hi = max(start, pos), min(end, window_end)
                if hi > lo:
                    dst = staging.view(-1).view(torch.uint8)
                    dst[lo - start : hi - start].copy_(host[lo - pos : hi - pos])
                if end <= window_end and i == first:
                    first += 1
                i += 1
            if first < len(entries) and got < len(buf):
                raise EOFError(f"{path}: truncated at byte {window_end}")
            pos = window_end // _ALIGN * _ALIGN
    finally:
        reader.close()
