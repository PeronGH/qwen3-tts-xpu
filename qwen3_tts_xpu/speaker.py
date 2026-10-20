"""ECAPA-TDNN speaker encoder (x-vector) used by the Base models for voice cloning."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .config import SpeakerEncoderConfig


def _conv(cin: int, cout: int, kernel: int, dilation: int = 1) -> nn.Conv1d:
    return nn.Conv1d(cin, cout, kernel, dilation=dilation, padding="same", padding_mode="reflect")


class TimeDelayNetBlock(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int, dilation: int):
        super().__init__()
        self.conv = _conv(cin, cout, kernel, dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv(x))


class Res2NetBlock(nn.Module):
    def __init__(self, cin: int, cout: int, scale: int, kernel: int, dilation: int):
        super().__init__()
        self.scale = scale
        self.blocks = nn.ModuleList(
            [
                TimeDelayNetBlock(cin // scale, cout // scale, kernel, dilation)
                for _ in range(scale - 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        out = None
        for i, part in enumerate(torch.chunk(x, self.scale, dim=1)):
            if i == 0:
                out = part
            elif i == 1:
                out = self.blocks[i - 1](part)
            else:
                out = self.blocks[i - 1](part + out)
            outputs.append(out)
        return torch.cat(outputs, dim=1)


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, cin: int, se_channels: int, cout: int):
        super().__init__()
        self.conv1 = _conv(cin, se_channels, 1)
        self.conv2 = _conv(se_channels, cout, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=2, keepdim=True)
        s = torch.sigmoid(self.conv2(F.relu(self.conv1(s))))
        return x * s


class SqueezeExcitationRes2NetBlock(nn.Module):
    def __init__(self, cin, cout, res2net_scale, se_channels, kernel, dilation):
        super().__init__()
        self.tdnn1 = TimeDelayNetBlock(cin, cout, 1, 1)
        self.res2net_block = Res2NetBlock(cout, cout, res2net_scale, kernel, dilation)
        self.tdnn2 = TimeDelayNetBlock(cout, cout, 1, 1)
        self.se_block = SqueezeExcitationBlock(cout, se_channels, cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.se_block(self.tdnn2(self.res2net_block(self.tdnn1(x))))
        return h + x


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, channels: int, attention_channels: int):
        super().__init__()
        self.eps = 1e-12
        self.tdnn = TimeDelayNetBlock(channels * 3, attention_channels, 1, 1)
        self.conv = _conv(attention_channels, channels, 1)

    def _stats(self, x: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = (m * x).sum(2)
        std = torch.sqrt((m * (x - mean.unsqueeze(2)).pow(2)).sum(2).clamp(self.eps))
        return mean, std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Single unpadded utterance: the upstream length mask is all ones, but its dtype-rounded
        # 1/L weights are kept for numerical parity.
        length = x.shape[-1]
        mask = torch.ones(x.shape[0], 1, length, dtype=x.dtype, device=x.device)
        total = mask.sum(dim=2, keepdim=True)
        mean, std = self._stats(x, mask / total)
        mean = mean.unsqueeze(2).repeat(1, 1, length)
        std = std.unsqueeze(2).repeat(1, 1, length)
        attn = self.conv(torch.tanh(self.tdnn(torch.cat([x, mean, std], dim=1))))
        attn = F.softmax(attn, dim=2)
        mean, std = self._stats(x, attn)
        return torch.cat((mean, std), dim=1).unsqueeze(2)


class SpeakerEncoder(nn.Module):
    def __init__(self, cfg: SpeakerEncoderConfig):
        super().__init__()
        ch, ks, dil = cfg.enc_channels, cfg.enc_kernel_sizes, cfg.enc_dilations
        self.cfg = cfg
        self.blocks = nn.ModuleList([TimeDelayNetBlock(cfg.mel_dim, ch[0], ks[0], dil[0])])
        for i in range(1, len(ch) - 1):
            self.blocks.append(
                SqueezeExcitationRes2NetBlock(
                    ch[i - 1], ch[i], cfg.enc_res2net_scale, cfg.enc_se_channels, ks[i], dil[i]
                )
            )
        self.mfa = TimeDelayNetBlock(ch[-1], ch[-1], ks[-1], dil[-1])
        self.asp = AttentiveStatisticsPooling(ch[-1], cfg.enc_attention_channels)
        self.fc = _conv(ch[-1] * 2, cfg.enc_dim, 1)

    def forward(self, mels: torch.Tensor) -> torch.Tensor:
        """mels: [B, T, mel_dim] -> [B, enc_dim]"""
        x = mels.transpose(1, 2)
        outs = []
        for layer in self.blocks:
            x = layer(x)
            outs.append(x)
        x = self.mfa(torch.cat(outs[1:], dim=1))
        x = self.fc(self.asp(x))
        return x.squeeze(-1)


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    f_sp = 200.0 / 3
    mels = f / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(
        f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep, mels
    )


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    f_sp = 200.0 / 3
    freqs = f_sp * m
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), freqs)


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Slaney-scale, Slaney-normalized filterbank; equals `librosa.filters.mel` defaults."""
    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)
    mel_f = _mel_to_hz(
        np.linspace(_hz_to_mel(np.array(fmin)), _hz_to_mel(np.array(fmax)), n_mels + 2)
    )
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    weights = np.zeros((n_mels, 1 + n_fft // 2), dtype=np.float32)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, None]
    return weights


def mel_spectrogram(wav: np.ndarray, sr: int = 24000) -> torch.Tensor:
    """Log-mel features of a mono float32 waveform -> [1, T, 128] (computed on CPU)."""
    n_fft, hop, win, n_mels = 1024, 256, 1024, 128
    y = torch.from_numpy(wav).float().unsqueeze(0)
    basis = torch.from_numpy(mel_filterbank(sr, n_fft, n_mels, 0, 12000)).float()
    pad = (n_fft - hop) // 2
    y = F.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop,
        win_length=win,
        window=torch.hann_window(win),
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
    mel = torch.log(torch.clamp(basis @ spec, min=1e-5))
    return mel.transpose(1, 2)
