"""Greedy speculative decoding must reproduce target-only greedy decoding exactly.

Uses fp32 small models so bf16 kernel nondeterminism can't cause spurious diffs.
  python -m pytest tests -q
"""

import os
import sys

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from specdec.decoding import autoregressive_generate, make_kv, speculative_generate  # noqa: E402

TARGET = os.environ.get("TEST_TARGET", "Qwen/Qwen2.5-1.5B-Instruct")
DRAFT = os.environ.get("TEST_DRAFT", "Qwen/Qwen2.5-0.5B-Instruct")
PROMPTS = ["Explain how a hash map works.", "Write a haiku about GPUs.", "What is 17 * 23? Show your work."]


@pytest.fixture(scope="module")
def models():
    tok = AutoTokenizer.from_pretrained(TARGET)
    kw = dict(dtype=torch.float32, attn_implementation="eager")
    t = AutoModelForCausalLM.from_pretrained(TARGET, **kw).cuda().eval()
    d = AutoModelForCausalLM.from_pretrained(DRAFT, **kw).cuda().eval()
    eos = [tok.convert_tokens_to_ids("<|im_end|>")]
    return tok, t, d, eos


def _ids(tok, p):
    text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()


@pytest.mark.parametrize("gamma", [1, 4, 8])
def test_greedy_equivalence(models, gamma):
    tok, t, d, eos = models
    for p in PROMPTS:
        ids = _ids(tok, p)
        ref, _ = autoregressive_generate(t, ids, 96, eos_token_id=eos)
        out, st = speculative_generate(t, d, ids, 96, gamma=gamma, eos_token_id=eos)
        assert out.tolist() == ref.tolist(), tok.decode(out[0, ids.shape[1]:])
        assert st.target_calls < st.new_tokens  # actually got speedup in calls


def test_cache_matches_nocache(models):
    tok, t, d, eos = models
    ids = _ids(tok, PROMPTS[0])
    a, _ = speculative_generate(t, d, ids, 48, gamma=4, eos_token_id=eos, use_cache=True)
    b, _ = speculative_generate(t, d, ids, 48, gamma=4, eos_token_id=eos, use_cache=False)
    assert a.tolist() == b.tolist()


def test_self_draft_accepts_everything(models):
    # draft == target => every greedy draft token is accepted
    tok, t, _, eos = models
    ids = _ids(tok, PROMPTS[1])
    _, st = speculative_generate(t, t, ids, 48, gamma=4, eos_token_id=eos)
    assert st.acceptance_rate == 1.0


def test_sampling_runs(models):
    tok, t, d, eos = models
    torch.manual_seed(0)
    ids = _ids(tok, PROMPTS[2])
    out, st = speculative_generate(t, d, ids, 64, gamma=4, temperature=1.0, eos_token_id=eos)
    assert 0 < st.new_tokens <= 64 and 0 <= st.acceptance_rate <= 1


def test_static_cache_matches_dynamic(models):
    # static backend without compile: same math, different cache/rollback mechanism
    tok, t, d, eos = models
    ids = _ids(tok, PROMPTS[0])
    a, _ = speculative_generate(t, d, ids, 64, gamma=4, eos_token_id=eos)
    tk, dk = make_kv(t, "static", max_len=1024, compile=False), make_kv(d, "static", max_len=1024, compile=False)
    b, _ = speculative_generate(tk, dk, ids, 64, gamma=4, eos_token_id=eos)
    c, _ = autoregressive_generate(tk, ids, 64, eos_token_id=eos)
    assert a.tolist() == b.tolist() == c.tolist()


@pytest.mark.parametrize("kind", ["none", "dynamic", "static"])
def test_backend_reuse_across_prompts(models, kind):
    # benchmark reuses one backend per model across prompts; reset() must fully clear state
    tok, t, d, eos = models
    kw = {"max_len": 1024, "compile": False} if kind == "static" else {}
    tk, dk = make_kv(t, kind, **kw), make_kv(d, kind, **kw)
    for p in PROMPTS[:2]:
        ids = _ids(tok, p)
        ref, _ = autoregressive_generate(t, ids, 32, eos_token_id=eos)
        autoregressive_generate(tk, ids, 32, eos_token_id=eos)
        out, _ = speculative_generate(tk, dk, ids, 32, gamma=4, eos_token_id=eos)
        assert out.tolist() == ref.tolist()
