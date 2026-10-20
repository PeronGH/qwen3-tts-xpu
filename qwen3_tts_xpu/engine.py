from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .audio import AudioInput, load_audio, resample
from .codec import SpeechTokenizer
from .config import (
    GenerationParams,
    ModelConfig,
    load_codec_config,
    load_generation_defaults,
    load_model_config,
)
from .generator import Generator
from .speaker import mel_spectrogram
from .talker import Qwen3TTSModel
from .text import TextTokenizer
from .weights import load_safetensors


@dataclass
class VoiceClonePrompt:
    ref_code: torch.Tensor | None  # [T, Q]; None in x-vector-only mode
    spk_embedding: torch.Tensor  # [D]
    ref_text: str | None = None

    def save(self, path: str | Path) -> None:
        """Write the voice file format of upstream's web demo ("Save / Load Voice")."""
        item = {
            "ref_code": None if self.ref_code is None else self.ref_code.cpu(),
            "ref_spk_embedding": self.spk_embedding.cpu(),
            "x_vector_only_mode": self.ref_code is None,
            "icl_mode": self.ref_code is not None,
            "ref_text": self.ref_text,
        }
        torch.save({"items": [item]}, path)

    @classmethod
    def load(cls, path: str | Path, index: int = 0) -> VoiceClonePrompt:
        """Read a voice file saved by `save` or by upstream's web demo."""
        item = torch.load(path, map_location="cpu", weights_only=True)["items"][index]
        icl = bool(item.get("icl_mode", not item.get("x_vector_only_mode", False)))
        ref_code = item.get("ref_code") if icl else None
        if ref_code is not None and not item.get("ref_text"):
            raise ValueError(f"{path}: ICL voice file without ref_text")
        return cls(
            ref_code=None if ref_code is None else torch.as_tensor(ref_code, dtype=torch.long),
            spk_embedding=torch.as_tensor(item["ref_spk_embedding"]),
            ref_text=item.get("ref_text"),
        )


@dataclass
class Prompt:
    embeds: torch.Tensor  # [1, L, D] talker prefill input
    trailing: torch.Tensor  # [T, D] per-step text embeddings, last row is tts_pad
    ref_code: torch.Tensor | None = None  # prepended for decoding (ICL voice clone)


def resolve_model_dir(name_or_path: str | Path) -> Path:
    path = Path(name_or_path)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(name_or_path)))


def _load_weights(module_factory, files: list[Path], device, dtype) -> nn.Module:
    with torch.device("meta"):
        module = module_factory().to(dtype)
    module = module.to_empty(device=device)
    load_safetensors(module, files)
    return module.eval()


class Qwen3TTS:
    def __init__(
        self,
        model: Qwen3TTSModel,
        codec: SpeechTokenizer,
        tokenizer: TextTokenizer,
        gen_defaults: GenerationParams,
        max_seq_len: int = 4096,
        use_graphs: bool = True,
    ):
        self.model = model
        self.cfg: ModelConfig = model.cfg
        self.codec = codec
        self.tokenizer = tokenizer
        self.gen_defaults = gen_defaults
        w = model.talker.codec_head.weight
        self.device, self.dtype = w.device, w.dtype
        with torch.inference_mode():
            self.generator = Generator(model.talker, max_seq_len, use_graphs)

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str | Path,
        device: str = "xpu",
        dtype: torch.dtype = torch.bfloat16,
        codec_dtype: torch.dtype | None = None,
        max_seq_len: int = 4096,
        use_graphs: bool = True,
    ) -> Qwen3TTS:
        path = resolve_model_dir(name_or_path)
        cfg = load_model_config(path)
        model = _load_weights(
            lambda: Qwen3TTSModel(cfg), sorted(path.glob("*.safetensors")), device, dtype
        )
        model.fuse()
        codec_path = path / "speech_tokenizer"
        codec_cfg = load_codec_config(codec_path)
        codec = _load_weights(
            lambda: SpeechTokenizer(codec_cfg),
            sorted(codec_path.glob("*.safetensors")),
            device,
            codec_dtype or dtype,
        )
        return cls(
            model,
            codec,
            TextTokenizer(path),
            load_generation_defaults(path),
            max_seq_len=max_seq_len,
            use_graphs=use_graphs,
        )

    @property
    def sample_rate(self) -> int:
        return self.codec.cfg.sample_rate

    @property
    def supported_speakers(self) -> list[str]:
        return sorted(self.cfg.talker.spk_id)

    @property
    def supported_languages(self) -> list[str]:
        langs = [k for k in self.cfg.talker.codec_language_id if "dialect" not in k]
        return ["auto", *sorted(langs)]

    # -----------------------------------------------------------------------------------------
    # Prompt construction (mirrors upstream Qwen3TTSForConditionalGeneration.generate, batch 1)

    def _ids(self, text: str) -> torch.Tensor:
        return torch.tensor([self.tokenizer.encode(text)], dtype=torch.long, device=self.device)

    def _check_language(self, language: str) -> str:
        language = language.lower()
        if language not in self.supported_languages:
            raise ValueError(f"unsupported language {language!r}; use {self.supported_languages}")
        return language

    def _build(
        self,
        text: str,
        language: str,
        speaker: str | None = None,
        instruct: str | None = None,
        clone: VoiceClonePrompt | None = None,
        non_streaming_mode: bool = True,
    ) -> Prompt:
        cfg = self.cfg.talker
        talker = self.model.talker
        language = self._check_language(language)
        input_id = self._ids(self.tokenizer.assistant_text(text))

        def codec(ids: list[int]) -> torch.Tensor:
            return talker.embed_codec(torch.tensor([ids], dtype=torch.long, device=self.device))

        speaker_embed = None
        if clone is not None:
            speaker_embed = clone.spk_embedding.to(self.device, self.dtype)
        elif speaker:
            speaker = speaker.lower()
            if speaker not in cfg.spk_id:
                raise ValueError(f"unsupported speaker {speaker!r}; use {self.supported_speakers}")
            speaker_embed = talker.embed_codec(
                torch.tensor(cfg.spk_id[speaker], dtype=torch.long, device=self.device)
            )

        language_id = None if language == "auto" else cfg.codec_language_id[language]
        if language in ("chinese", "auto") and speaker:
            dialect = cfg.spk_is_dialect.get(speaker, False)
            if dialect is not False:
                language_id = cfg.codec_language_id[dialect]

        tts_bos, tts_eos, tts_pad = talker.embed_text(
            torch.tensor(
                [[self.cfg.tts_bos_token_id, self.cfg.tts_eos_token_id, self.cfg.tts_pad_token_id]],
                dtype=torch.long,
                device=self.device,
            )
        ).chunk(3, dim=1)

        if language_id is None:
            prefill = [cfg.codec_nothink_id, cfg.codec_think_bos_id, cfg.codec_think_eos_id]
        else:
            prefill = [
                cfg.codec_think_id,
                cfg.codec_think_bos_id,
                language_id,
                cfg.codec_think_eos_id,
            ]
        codec_parts = [codec(prefill)]
        if speaker_embed is not None:
            codec_parts.append(speaker_embed.view(1, 1, -1))
        codec_parts.append(codec([cfg.codec_pad_id, cfg.codec_bos_id]))
        codec_embed = torch.cat(codec_parts, dim=1)

        role = talker.embed_text(input_id[:, :3])
        prefix = (
            torch.cat([tts_pad.expand(-1, codec_embed.shape[1] - 2, -1), tts_bos], dim=1)
            + codec_embed[:, :-1]
        )
        embeds = torch.cat([role, prefix], dim=1)

        if clone is not None and clone.ref_code is not None:
            ref_id = self._ids(self.tokenizer.ref_text(clone.ref_text or ""))
            icl, trailing = self._icl_prompt(
                input_id[:, 3:-5],
                ref_id[:, 3:-2],
                clone.ref_code.to(self.device),
                tts_pad,
                tts_eos,
                non_streaming_mode,
            )
            embeds = torch.cat([embeds, icl], dim=1)
        else:
            first = talker.embed_text(input_id[:, 3:4]) + codec_embed[:, -1:]
            embeds = torch.cat([embeds, first], dim=1)
            if non_streaming_mode:
                body = input_id[:, 3:-5]
                embeds = torch.cat(
                    [
                        embeds[:, :-1],
                        torch.cat([talker.embed_text(body), tts_eos], dim=1)
                        + codec([cfg.codec_pad_id] * (body.shape[1] + 1)),
                        tts_pad + codec([cfg.codec_bos_id]),
                    ],
                    dim=1,
                )
                trailing = tts_pad
            else:
                trailing = torch.cat([talker.embed_text(input_id[:, 4:-5]), tts_eos], dim=1)

        if instruct:
            instruct_id = self._ids(self.tokenizer.instruct_text(instruct))
            embeds = torch.cat([talker.embed_text(instruct_id), embeds], dim=1)

        trailing = torch.cat([trailing, tts_pad], dim=1)[0]
        return Prompt(embeds, trailing, clone.ref_code if clone is not None else None)

    def _icl_prompt(self, text_id, ref_id, ref_code, tts_pad, tts_eos, non_streaming_mode):
        cfg = self.cfg.talker
        talker = self.model.talker
        text_embed = talker.embed_text(torch.cat([ref_id, text_id], dim=-1))
        text_embed = torch.cat([text_embed, tts_eos], dim=1)
        bos = torch.tensor([[cfg.codec_bos_id]], dtype=torch.long, device=self.device)
        codec_embed = torch.cat(
            [talker.embed_codec(bos), talker.embed_frames(ref_code)[None]], dim=1
        )
        text_lens, codec_lens = text_embed.shape[1], codec_embed.shape[1]
        if non_streaming_mode:
            pad = torch.full((1, text_lens), cfg.codec_pad_id, dtype=torch.long, device=self.device)
            icl = text_embed + talker.embed_codec(pad)
            return torch.cat([icl, codec_embed + tts_pad], dim=1), tts_pad
        if text_lens > codec_lens:
            return text_embed[:, :codec_lens] + codec_embed, text_embed[:, codec_lens:]
        text_embed = torch.cat([text_embed] + [tts_pad] * (codec_lens - text_lens), dim=1)
        return text_embed + codec_embed, tts_pad

    @torch.inference_mode()
    def custom_voice_prompt(
        self,
        text: str,
        speaker: str,
        language: str = "auto",
        instruct: str | None = None,
        non_streaming_mode: bool = True,
    ) -> Prompt:
        if self.cfg.model_type != "custom_voice":
            raise ValueError(f"{self.cfg.model_type} model does not support custom voice")
        if self.cfg.model_size == "0b6":
            instruct = None  # the 0.6B CustomVoice model ignores instructions upstream
        return self._build(
            text,
            language,
            speaker=speaker,
            instruct=instruct,
            non_streaming_mode=non_streaming_mode,
        )

    @torch.inference_mode()
    def voice_design_prompt(
        self, text: str, instruct: str, language: str = "auto", non_streaming_mode: bool = True
    ) -> Prompt:
        if self.cfg.model_type != "voice_design":
            raise ValueError(f"{self.cfg.model_type} model does not support voice design")
        return self._build(text, language, instruct=instruct, non_streaming_mode=non_streaming_mode)

    @torch.inference_mode()
    def voice_clone_prompt(
        self,
        text: str,
        clone: VoiceClonePrompt,
        language: str = "auto",
        non_streaming_mode: bool = False,
    ) -> Prompt:
        if self.cfg.model_type != "base":
            raise ValueError(f"{self.cfg.model_type} model does not support voice cloning")
        if clone.spk_embedding.numel() != self.cfg.talker.hidden_size:
            raise ValueError(
                f"speaker embedding has {clone.spk_embedding.numel()} values but this model "
                f"expects {self.cfg.talker.hidden_size}; the voice was made with another model size"
            )
        return self._build(text, language, clone=clone, non_streaming_mode=non_streaming_mode)

    @torch.inference_mode()
    def create_voice_clone_prompt(
        self, ref_audio: AudioInput, ref_text: str | None = None, x_vector_only_mode: bool = False
    ) -> VoiceClonePrompt:
        if self.model.speaker_encoder is None:
            raise ValueError(f"{self.cfg.model_type} model does not support voice cloning")
        if not x_vector_only_mode and not ref_text:
            raise ValueError("ref_text is required unless x_vector_only_mode=True")
        wav, sr = load_audio(ref_audio)
        ref_code = None
        if not x_vector_only_mode:
            wav_codec = resample(wav, sr, self.codec.cfg.sample_rate)
            ref_code = self.codec.encode(torch.from_numpy(wav_codec).to(self.device))
        se = self.model.speaker_encoder
        wav_spk = resample(wav, sr, se.cfg.sample_rate)
        mels = mel_spectrogram(wav_spk, se.cfg.sample_rate).to(self.device, self.dtype)
        return VoiceClonePrompt(ref_code, se(mels)[0], ref_text)

    # -----------------------------------------------------------------------------------------
    # Synthesis

    def _params(self, overrides: dict) -> GenerationParams:
        return replace(self.gen_defaults, **overrides)

    @torch.inference_mode()
    def generate_codes(self, prompt: Prompt, **gen) -> Iterator[torch.Tensor]:
        """Yield generated frames ([n, Q] codes) as they become available."""
        yield from self.generator.generate(prompt.embeds, prompt.trailing, self._params(gen))

    @torch.inference_mode()
    def synthesize(self, prompt: Prompt, **gen) -> np.ndarray:
        frames = list(self.generate_codes(prompt, **gen))
        if not frames:
            return np.zeros(0, dtype=np.float32)
        codes = torch.cat(frames)
        if prompt.ref_code is None:
            return self.codec.decode(codes).cpu().numpy()
        ref = prompt.ref_code.to(codes.device)
        wav = self.codec.decode(torch.cat([ref, codes])).cpu().numpy()
        total = ref.shape[0] + codes.shape[0]
        return wav[int(ref.shape[0] / total * wav.shape[0]) :]

    @torch.inference_mode()
    def stream(
        self, prompt: Prompt, chunk_frames: int = 12, left_context: int = 25, **gen
    ) -> Iterator[np.ndarray]:
        """Yield audio chunks of about `chunk_frames` frames (12 frames = 0.96 s) as they decode."""
        ref = prompt.ref_code
        codes = ref.to(self.device) if ref is not None else None
        emitted = codes.shape[0] if codes is not None else 0
        rate = self.codec.cfg.upsample_rate
        pending = 0
        for frames in self.generate_codes(prompt, **gen):
            codes = frames if codes is None else torch.cat([codes, frames])
            pending += frames.shape[0]
            if pending < chunk_frames:
                continue
            yield self._decode_tail(codes, emitted, left_context, rate)
            emitted, pending = codes.shape[0], 0
        if pending:
            yield self._decode_tail(codes, emitted, left_context, rate)

    def _decode_tail(self, codes, start, left_context, rate) -> np.ndarray:
        ctx = min(left_context, start)
        wav = self.codec.decode(codes[start - ctx :])
        return wav[ctx * rate :].cpu().numpy()

    # Convenience wrappers with upstream-like names.

    def generate_custom_voice(
        self, text, speaker, language="auto", instruct=None, non_streaming_mode=True, **gen
    ) -> tuple[np.ndarray, int]:
        prompt = self.custom_voice_prompt(text, speaker, language, instruct, non_streaming_mode)
        return self.synthesize(prompt, **gen), self.sample_rate

    def generate_voice_design(
        self, text, instruct, language="auto", non_streaming_mode=True, **gen
    ) -> tuple[np.ndarray, int]:
        prompt = self.voice_design_prompt(text, instruct, language, non_streaming_mode)
        return self.synthesize(prompt, **gen), self.sample_rate

    def generate_voice_clone(
        self,
        text,
        language="auto",
        ref_audio=None,
        ref_text=None,
        x_vector_only_mode=False,
        voice_clone_prompt=None,
        non_streaming_mode=False,
        **gen,
    ) -> tuple[np.ndarray, int]:
        clone = voice_clone_prompt or self.create_voice_clone_prompt(
            ref_audio, ref_text, x_vector_only_mode
        )
        prompt = self.voice_clone_prompt(text, clone, language, non_streaming_mode)
        return self.synthesize(prompt, **gen), self.sample_rate
