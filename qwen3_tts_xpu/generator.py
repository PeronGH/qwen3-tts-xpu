"""Static-buffer decode loop for the talker + code predictor, replayed through XPU graphs."""

from collections.abc import Iterator

import torch

from .config import GenerationParams
from .layers import rope_table
from .sampling import SamplerParams, apply_repetition_penalty, sample
from .talker import Talker


def _bucket(n: int, cap: int) -> int:
    return min(cap, max(256, 1 << (n - 1).bit_length()))


class Generator:
    """Owns the KV caches and all per-step state so one frame is a single fixed-shape graph.

    A frame step runs the code predictor (G-1 sampled codebooks) for the previously sampled
    codebook-0 token, feeds the summed frame embedding to the talker and samples the next
    codebook-0 token. Graphs are captured lazily per (KV bucket, sampling params).
    """

    def __init__(self, talker: Talker, max_seq_len: int, use_graphs: bool):
        cfg = talker.cfg
        cp_cfg = talker.code_predictor.model.cfg
        w = talker.codec_head.weight
        dev, dt = w.device, w.dtype
        groups = cfg.num_code_groups
        self.talker = talker
        self.max_seq_len = max_seq_len
        self.use_graphs = use_graphs and dev.type == "xpu"
        self.groups = groups
        self.eos = cfg.codec_eos_id
        self.cache = talker.model.new_cache(max_seq_len, dev, dt)
        self.cp_cache = talker.code_predictor.model.new_cache(groups, dev, dt)
        self.cos, self.sin = rope_table(max_seq_len, cfg.head_dim, cfg.rope_theta, dev, dt)
        self.cp_cos, self.cp_sin = rope_table(groups, cp_cfg.head_dim, cp_cfg.rope_theta, dev, dt)
        self.arange = torch.arange(max_seq_len, device=dev)

        self.tok = torch.zeros(1, dtype=torch.long, device=dev)
        self.past_hidden = torch.zeros(1, 1, cfg.hidden_size, dtype=dt, device=dev)
        self.pos = torch.zeros(1, dtype=torch.long, device=dev)
        self.step = torch.zeros(1, dtype=torch.long, device=dev)
        self.seen = torch.zeros(cfg.vocab_size, dtype=torch.float32, device=dev)
        self.bias = torch.zeros(cfg.vocab_size, dtype=torch.float32, device=dev)
        self.suppress = torch.zeros(cfg.vocab_size, dtype=torch.float32, device=dev)
        self.suppress[cfg.vocab_size - 1024 :] = float("-inf")
        self.suppress[self.eos] = 0.0
        self.trailing = torch.zeros(max_seq_len, cfg.hidden_size, dtype=dt, device=dev)
        self.trailing_len = torch.zeros(1, dtype=torch.long, device=dev)
        self.noise = torch.ones(
            groups, max(cfg.vocab_size, cp_cfg.vocab_size), dtype=torch.float32, device=dev
        )
        self.codes = torch.zeros(max_seq_len, groups, dtype=torch.long, device=dev)
        self.graphs: dict[tuple, torch.xpu.XPUGraph] = {}
        self.pool = None

    def _state(self) -> tuple[torch.Tensor, ...]:
        return (self.tok, self.past_hidden, self.pos, self.step, self.seen)

    def _sample_talker(self, h: torch.Tensor, tp: SamplerParams, rep: float) -> torch.Tensor:
        logits = self.talker.codec_head(h[:, -1]).float()
        logits = apply_repetition_penalty(logits, self.seen, rep) + self.bias
        width = tp.width(logits.shape[-1])
        return sample(logits, tp, self.noise[self.groups - 1 :, :width])

    def _step(self, kv_len: int, tp: SamplerParams, cp: SamplerParams, rep: float) -> None:
        talker = self.talker
        cpm = talker.code_predictor
        e0 = talker.model.codec_embedding(self.tok.view(1, 1))
        x = cpm.small_to_mtp_projection(torch.cat([self.past_hidden, e0], dim=1))
        h = cpm.model(x, self.cp_cos[:2], self.cp_sin[:2], self.cp_cache, 0, 2)
        codes, embs = [self.tok], [e0]
        for j in range(self.groups - 1):
            logits = cpm.lm_head[j](h[:, -1]).float()
            c = sample(logits, cp, self.noise[j : j + 1, : cp.width(logits.shape[-1])])
            emb = cpm.model.codec_embedding[j](c.view(1, 1))
            codes.append(c)
            embs.append(emb)
            if j < self.groups - 2:
                p = j + 2
                h = cpm.model(
                    cpm.small_to_mtp_projection(emb),
                    self.cp_cos[p : p + 1],
                    self.cp_sin[p : p + 1],
                    self.cp_cache,
                    p,
                    p + 1,
                )

        x = torch.cat(embs, dim=1).sum(1, keepdim=True)
        idx = torch.minimum(self.step, self.trailing_len)
        x = x + self.trailing.index_select(0, idx).view(1, 1, -1)
        mask = self.arange[:kv_len] <= self.pos
        cos = self.cos.index_select(0, self.pos)
        sin = self.sin.index_select(0, self.pos)
        h = talker.model(x, cos, sin, self.cache, self.pos, kv_len, mask)
        nxt = self._sample_talker(h, tp, rep)

        self.codes.index_copy_(0, self.step, torch.cat(codes).view(1, -1))
        self.seen.index_fill_(0, nxt, 1.0)
        self.tok.copy_(nxt)
        self.past_hidden.copy_(h)
        self.pos.add_(1)
        self.step.add_(1)

    def _run_step(self, kv_len: int, tp: SamplerParams, cp: SamplerParams, rep: float) -> None:
        if not self.use_graphs:
            self._step(kv_len, tp, cp, rep)
            return
        key = (kv_len, tp, cp, rep)
        graph = self.graphs.get(key)
        if graph is None:
            saved = [t.clone() for t in self._state()]

            def restore() -> None:
                for dst, src in zip(self._state(), saved, strict=True):
                    dst.copy_(src)

            side = torch.xpu.Stream()
            side.wait_stream(torch.xpu.current_stream())
            with torch.xpu.stream(side):
                for _ in range(2):
                    restore()
                    self._step(kv_len, tp, cp, rep)
            torch.xpu.current_stream().wait_stream(side)
            restore()
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph, pool=self.pool):
                self._step(kv_len, tp, cp, rep)
            if self.pool is None:
                self.pool = graph.pool()
            restore()
            self.graphs[key] = graph
        graph.replay()

    def generate(
        self,
        embeds: torch.Tensor,
        trailing: torch.Tensor,
        gen: GenerationParams,
        check_every: int = 4,
        ignore_eos: bool = False,
    ) -> Iterator[torch.Tensor]:
        """Prefill `embeds` [1, L, D] and yield newly finished frames ([n, G] codes) until EOS.

        `trailing` [T, D] holds the per-step text embeddings; its last row (the tts_pad
        embedding) is reused once the steps run past it.
        """
        length = embeds.shape[1]
        if length >= self.max_seq_len:
            raise ValueError(f"prompt of {length} tokens exceeds max_seq_len={self.max_seq_len}")
        if trailing.shape[0] > self.max_seq_len:
            raise ValueError("trailing text longer than max_seq_len")
        tp = SamplerParams(gen.do_sample, gen.top_k, gen.top_p, gen.temperature)
        cp = SamplerParams(
            gen.subtalker_dosample,
            gen.subtalker_top_k,
            gen.subtalker_top_p,
            gen.subtalker_temperature,
        )
        rep = gen.repetition_penalty

        self.trailing[: trailing.shape[0]] = trailing
        self.trailing_len.fill_(trailing.shape[0] - 1)
        self.seen.zero_()
        self.bias.copy_(self.suppress)
        self.bias[self.eos] = float("-inf")
        self.noise.exponential_()
        h = self.talker.model(embeds, self.cos[:length], self.sin[:length], self.cache, 0, length)
        nxt = self._sample_talker(h, tp, rep)
        self.tok.copy_(nxt)
        self.seen.index_fill_(0, nxt, 1.0)
        self.past_hidden.copy_(h[:, -1:])
        self.pos.fill_(length)
        self.step.zero_()

        max_frames = min(gen.max_new_tokens, self.max_seq_len - length)
        done = 0
        while done < max_frames:
            start = done
            for _ in range(min(check_every, max_frames - done)):
                self.noise.exponential_()
                self._run_step(_bucket(length + done + 1, self.max_seq_len), tp, cp, rep)
                if done == 0:
                    # transformers' min_new_tokens=2: EOS is banned for the first two samples.
                    self.bias[self.eos] = 0.0
                done += 1
            end, stop = done, False
            if not ignore_eos:
                # Row i of `codes` holds the codebook-0 token fed at step i; `tok` is the next one.
                firsts = torch.cat([self.codes[start:done, 0], self.tok]).tolist()
                if self.eos in firsts:
                    end, stop = start + firsts.index(self.eos), True
            if end > start:
                yield self.codes[start:end].clone()
            if stop:
                return
