import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig, TalkerConfig, TransformerConfig
from .layers import Decoder
from .speaker import SpeakerEncoder


class TalkerModel(Decoder):
    def __init__(self, cfg: TalkerConfig):
        super().__init__(cfg)
        self.codec_embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.text_embedding = nn.Embedding(cfg.text_vocab_size, cfg.text_hidden_size)


class TextProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear_fc1 = nn.Linear(in_dim, in_dim, bias=True)
        self.linear_fc2 = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(F.silu(self.linear_fc1(x)))


class CodePredictorModel(Decoder):
    def __init__(self, cfg: TransformerConfig, embedding_dim: int, num_code_groups: int):
        super().__init__(cfg)
        self.codec_embedding = nn.ModuleList(
            [nn.Embedding(cfg.vocab_size, embedding_dim) for _ in range(num_code_groups - 1)]
        )


class CodePredictor(nn.Module):
    """Predicts codebooks 1..G-1 of a frame from the talker hidden state and codebook 0."""

    def __init__(self, cfg: TransformerConfig, talker_cfg: TalkerConfig):
        super().__init__()
        groups = talker_cfg.num_code_groups
        self.model = CodePredictorModel(cfg, talker_cfg.hidden_size, groups)
        self.lm_head = nn.ModuleList(
            [nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False) for _ in range(groups - 1)]
        )
        if cfg.hidden_size != talker_cfg.hidden_size:
            self.small_to_mtp_projection = nn.Linear(
                talker_cfg.hidden_size, cfg.hidden_size, bias=True
            )
        else:
            self.small_to_mtp_projection = nn.Identity()


class Talker(nn.Module):
    def __init__(self, cfg: TalkerConfig, cp_cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.model = TalkerModel(cfg)
        self.text_projection = TextProjection(cfg.text_hidden_size, cfg.hidden_size)
        self.codec_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.code_predictor = CodePredictor(cp_cfg, cfg)

    def embed_text(self, ids: torch.Tensor) -> torch.Tensor:
        return self.text_projection(self.model.text_embedding(ids))

    def embed_codec(self, ids: torch.Tensor) -> torch.Tensor:
        return self.model.codec_embedding(ids)

    def embed_frames(self, codes: torch.Tensor) -> torch.Tensor:
        """Sum of all codebook embeddings for frames `codes` of shape [T, G] -> [T, D]."""
        tables = [self.model.codec_embedding, *self.code_predictor.model.codec_embedding]
        return torch.stack([t(codes[:, i]) for i, t in enumerate(tables)], dim=1).sum(1)


class Qwen3TTSModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.talker = Talker(cfg.talker, cfg.code_predictor)
        self.speaker_encoder = (
            SpeakerEncoder(cfg.speaker_encoder) if cfg.speaker_encoder is not None else None
        )

    def fuse(self) -> None:
        self.talker.model.fuse()
        self.talker.code_predictor.model.fuse()
