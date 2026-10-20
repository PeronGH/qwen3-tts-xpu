import json
from pathlib import Path

from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers

# Qwen2 pre-tokenizer. Upstream passes `fix_mistral_regex=True`, which swaps in a different regex
# only when loading from a local directory; the hub-id path (the documented usage) keeps this one.
QWEN2_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+|\s+(?!\S)|\s+"
)


class TextTokenizer:
    """Qwen2 byte-level BPE tokenizer built from `vocab.json`/`merges.txt` (no transformers)."""

    def __init__(self, path: Path):
        tok = Tokenizer(
            models.BPE.from_file(
                str(path / "vocab.json"),
                str(path / "merges.txt"),
                unk_token=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
                byte_fallback=False,
            )
        )
        tok.normalizer = normalizers.NFC()
        tok.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(QWEN2_REGEX), behavior="isolated", invert=False),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
        tok.decoder = decoders.ByteLevel()
        added = json.loads((path / "tokenizer_config.json").read_text())["added_tokens_decoder"]
        for idx, spec in sorted(added.items(), key=lambda kv: int(kv[0])):
            token = AddedToken(
                spec["content"],
                single_word=spec["single_word"],
                lstrip=spec["lstrip"],
                rstrip=spec["rstrip"],
                normalized=spec["normalized"],
                special=spec["special"],
            )
            if spec["special"]:
                tok.add_special_tokens([token])
            else:
                tok.add_tokens([token])
            if tok.token_to_id(spec["content"]) != int(idx):
                raise ValueError(f"added token {spec['content']!r} did not get id {idx}")
        self.tok = tok

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    @staticmethod
    def assistant_text(text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    @staticmethod
    def ref_text(text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n"

    @staticmethod
    def instruct_text(text: str) -> str:
        return f"<|im_start|>user\n{text}<|im_end|>\n"
