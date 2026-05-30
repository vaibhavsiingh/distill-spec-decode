"""Autoregressive and speculative decoding loops (batch size 1).

Every model is wrapped in a KV backend that owns its cache and tracks how many
committed tokens are cached (`n`). The loops feed `seq[n:]` (plus any draft
tokens) and, after verification, `rollback()` each backend to the accepted
prefix so rejected draft tokens never leak into later steps.

Backends:
  none    - no cache; re-encode the whole sequence every step
  dynamic - HF DynamicCache, rollback by cropping
  static  - preallocated StaticCache + CUDA graphs (torch.compile
            reduce-overhead); rollback is O(1): move the write pointer back,
            stale slots beyond it are hidden by the causal mask and
            overwritten on the next write
"""

from dataclasses import dataclass, field

import torch
from transformers import DynamicCache, StaticCache


@dataclass
class GenStats:
    new_tokens: int = 0
    target_calls: int = 0
    draft_calls: int = 0
    drafted: int = 0
    accepted: int = 0
    accept_lens: list = field(default_factory=list)

    @property
    def acceptance_rate(self):
        return self.accepted / max(self.drafted, 1)

    @property
    def tokens_per_target_call(self):
        return self.new_tokens / max(self.target_calls, 1)


class NoCacheKV:
    def __init__(self, model):
        self.model, self.n = model, 0

    def reset(self):
        pass

    def forward(self, seq, extra=None):
        """Logits for positions n..end of seq (+extra tokens); caches them."""
        x = seq if extra is None else torch.cat([seq, extra], dim=1)
        logits = self.model(input_ids=x, use_cache=False).logits[0, self.n:]
        self.n = x.shape[1]
        return logits

    def rollback(self, n):
        self.n = min(self.n, n)


class DynamicKV(NoCacheKV):
    def reset(self):
        self.cache, self.n = DynamicCache(), 0

    def forward(self, seq, extra=None):
        x = seq[:, self.n:] if extra is None else torch.cat([seq[:, self.n:], extra], dim=1)
        logits = self.model(input_ids=x, past_key_values=self.cache, use_cache=True).logits[0]
        self.n += x.shape[1]
        return logits

    def rollback(self, n):
        if self.n > n:
            self.cache.crop(n - self.n)  # negative = drop that many trailing tokens
            self.n = n


class StaticKV(NoCacheKV):
    """StaticCache with CUDA-graph decode steps.

    Prefill (variable length) runs eagerly; short decode/verify steps (q <=
    max_graph_len tokens) run through a compiled forward, one graph per q.
    """

    def __init__(self, model, max_len=4096, max_graph_len=16, compile=True):
        super().__init__(model)
        self.max_graph_len = max_graph_len
        self.cache = StaticCache(config=model.config, max_cache_len=max_len)
        self.slots = torch.arange(max_len, device=model.device)
        self.step = model.forward
        if compile:
            # draft and target share Qwen2ForCausalLM.forward, so their graphs count toward one limit
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
            self.step = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True, dynamic=False)

    def reset(self):
        self.cache.reset()
        self.n = 0

    @torch.no_grad()
    def warmup(self, qs, prefill=32):
        dev = self.model.device
        for q in qs:
            for _ in range(3):  # cudagraph trees record on the 2nd/3rd call
                self.reset()
                self.forward(torch.zeros(1, prefill + q, dtype=torch.long, device=dev)[:, :prefill])
                self.forward(torch.zeros(1, prefill + q, dtype=torch.long, device=dev))
        self.reset()

    def forward(self, seq, extra=None):
        x = seq[:, self.n:] if extra is None else torch.cat([seq[:, self.n:], extra], dim=1)
        q = x.shape[1]
        pos = torch.arange(self.n, self.n + q, device=x.device)
        # explicit 4D mask over the whole static cache: slot j visible to query i iff j <= n + i.
        # This is what hides stale slots after a rollback, and it bypasses HF's
        # data-dependent mask construction, which breaks the CUDA graph.
        mask = (self.slots[None, :] <= pos[:, None])[None, None]
        fn = self.step if q <= self.max_graph_len else self.model.forward
        out = fn(input_ids=x, attention_mask=mask, position_ids=pos[None], cache_position=pos,
                 past_key_values=self.cache, use_cache=True)
        self.n += q
        return out.logits[0].clone()  # graph output buffers are reused on the next replay

    def rollback(self, n):
        self.n = min(self.n, n)


def make_kv(model, kind, **kw):
    return {"none": NoCacheKV, "dynamic": DynamicKV, "static": StaticKV}[kind](model, **kw)


def _probs(logits, temperature):
    if temperature == 0:
        return None
    return torch.softmax(logits.float() / temperature, dim=-1)


def _pick(logits, temperature):
    if temperature == 0:
        return int(logits.argmax(-1))
    return int(torch.multinomial(_probs(logits, temperature), 1))


def _as_kv(m, cache):
    if isinstance(m, NoCacheKV):
        return m
    return make_kv(m, "dynamic" if cache else "none")


def _done(seq, prompt_len, max_new_tokens, eos_ids):
    n = seq.shape[1] - prompt_len
    return n >= max_new_tokens or (n > 0 and int(seq[0, -1]) in eos_ids)


@torch.no_grad()
def autoregressive_generate(model, input_ids, max_new_tokens, temperature=0.0,
                            eos_token_id=None, use_cache=True):
    """`model` is an nn.Module (use_cache picks none/dynamic) or a KV backend."""
    eos_ids = set(eos_token_id or [])
    kv = _as_kv(model, use_cache)
    kv.reset()
    seq = input_ids
    prompt_len = seq.shape[1]
    stats = GenStats()
    while not _done(seq, prompt_len, max_new_tokens, eos_ids):
        logits = kv.forward(seq)[-1]
        stats.target_calls += 1
        tok = _pick(logits, temperature)
        seq = torch.cat([seq, seq.new_tensor([[tok]])], dim=1)
    stats.new_tokens = seq.shape[1] - prompt_len
    return seq, stats


@torch.no_grad()
def speculative_generate(target, draft, input_ids, max_new_tokens, gamma=5,
                         temperature=0.0, eos_token_id=None, use_cache=True):
    """Speculative decoding (Leviathan et al. 2023; Chen et al. 2023).

    Greedy (temperature=0) output is identical to target-only greedy decoding
    (up to floating-point noise); with sampling, rejection sampling preserves
    the target distribution. `target`/`draft` are nn.Modules or KV backends.
    """
    eos_ids = set(eos_token_id or [])
    t_kv, d_kv = _as_kv(target, use_cache), _as_kv(draft, use_cache)
    t_kv.reset()
    d_kv.reset()
    vocab = d_kv.model.config.vocab_size
    seq = input_ids
    prompt_len = seq.shape[1]
    stats = GenStats()

    while not _done(seq, prompt_len, max_new_tokens, eos_ids):
        base = seq.shape[1]
        k = min(gamma, max_new_tokens - (base - prompt_len))

        # 1) draft k tokens autoregressively
        drafts, q_dists = [], []
        cur = seq
        for _ in range(k):
            logits = d_kv.forward(cur)[-1]
            stats.draft_calls += 1
            tok = _pick(logits, temperature)
            drafts.append(tok)
            q_dists.append(_probs(logits, temperature))
            cur = torch.cat([cur, cur.new_tensor([[tok]])], dim=1)
            if tok in eos_ids:
                break
        k = len(drafts)

        # 2) verify all drafts with one target forward
        # target may have a padded vocab (Qwen2.5-7B: 152064 vs 151936); extra rows are unused
        t_logits = t_kv.forward(seq, cur[:, base:])[-(k + 1):, :vocab]
        stats.target_calls += 1

        # 3) accept / reject
        n = 0
        if temperature == 0:
            t_argmax = t_logits.argmax(-1).tolist()
            while n < k and drafts[n] == t_argmax[n]:
                n += 1
            nxt = t_argmax[n]
        else:
            p_dists = _probs(t_logits, temperature)
            nxt = None
            for i in range(k):
                p, q, d = p_dists[i], q_dists[i], drafts[i]
                if torch.rand(()) < (p[d] / q[d]).clamp(max=1.0):
                    n += 1
                    continue
                resid = (p - q).clamp(min=0)
                nxt = int(torch.multinomial(resid / resid.sum(), 1))
                break
            if nxt is None:
                nxt = int(torch.multinomial(p_dists[k], 1))

        stats.drafted += k
        stats.accepted += n
        stats.accept_lens.append(n)

        accepted = drafts[:n]
        if not (accepted and accepted[-1] in eos_ids):
            accepted = accepted + [nxt]
        seq = torch.cat([seq, seq.new_tensor([accepted])], dim=1)

        # 4) roll caches back to the committed prefix (exclude the newest token)
        t_kv.rollback(base + n)
        d_kv.rollback(base + n)

    # trim overshoot past max_new_tokens / eos
    out = seq[0, prompt_len:].tolist()[:max_new_tokens]
    for i, t in enumerate(out):
        if t in eos_ids:
            out = out[:i + 1]
            break
    seq = torch.cat([input_ids, input_ids.new_tensor([out])], dim=1)
    stats.new_tokens = len(out)
    return seq, stats
