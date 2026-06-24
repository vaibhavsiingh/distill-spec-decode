"""Benchmark decoding methods: wall-clock speed, acceptance rate, greedy correctness.

Example:
  python scripts/benchmark.py --draft_ft checkpoints/draft-distill/epoch2 \
      --datasets mtbench ultrachat gsm8k humaneval --n 50 --out results/main.json
"""

import argparse
import json
import os
import sys
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from specdec.decoding import autoregressive_generate, make_kv, speculative_generate  # noqa: E402


def load_prompts(name, n):
    if name == "mtbench":
        ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        ps = [r["prompt"][0] for r in ds]
    elif name == "ultrachat":
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
        ps = [r["messages"][0]["content"] for r in ds.select(range(n))]
    elif name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        ps = [r["question"] + "\nLet's think step by step." for r in ds.select(range(n))]
    elif name == "humaneval":
        ds = load_dataset("openai/openai_humaneval", split="test")
        ps = ["Complete the following Python function:\n```python\n" + r["prompt"] + "```"
              for r in ds.select(range(n))]
    else:
        raise ValueError(name)
    return ps[:n]


def load(path):
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
                                                attn_implementation="sdpa").cuda().eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--draft_base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--draft_ft", default=None)
    ap.add_argument("--datasets", nargs="+", default=["mtbench", "ultrachat", "gsm8k", "humaneval"])
    ap.add_argument("--methods", nargs="+",
                    default=["ar_nocache", "ar_dynamic", "ar_static", "hf_generate", "hf_assisted",
                             "sd_base_static", "sd_ft_nocache", "sd_ft_dynamic", "sd_ft_static"],
                    help="ar_{kv}, sd_{base|ft}_{kv}, hf_generate, hf_assisted; kv in none|dynamic|static")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--n_nocache", type=int, default=20, help="fewer prompts for the slow no-cache runs")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--gamma", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", default="results/benchmark.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.target)
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    target = load(args.target)
    drafts = {"base": load(args.draft_base)}
    if args.draft_ft:
        drafts["ft"] = load(args.draft_ft)
    else:
        args.methods = [m for m in args.methods if not m.startswith("sd_ft")]

    hf_assistant = None
    if "hf_assisted" in args.methods:
        # HF requires identical vocab sizes; Qwen2.5-0.5B is 151936 vs 7B's padded 152064.
        # Pad with zero rows (never chosen) so HF's assisted generation accepts the pair.
        hf_assistant = load(args.draft_ft or args.draft_base)
        n_old = hf_assistant.config.vocab_size
        hf_assistant.resize_token_embeddings(target.config.vocab_size, mean_resizing=False)
        with torch.no_grad():
            hf_assistant.get_input_embeddings().weight[n_old:] = 0
            hf_assistant.get_output_embeddings().weight[n_old:] = 0

    # one KV backend per (model, kind), reused across prompts
    kvs = {}

    def kv(name, kind):
        if (name, kind) not in kvs:
            model = target if name == "target" else drafts[name]
            kvs[name, kind] = make_kv(model, {"nocache": "none"}.get(kind, kind))
            if kind == "static":
                # compile every graph shape up front so no compilation lands in timed runs
                kvs[name, kind].warmup(range(1, args.gamma + 2))
        return kvs[name, kind]

    def run(method, ids):
        g = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature, eos_token_id=eos)
        if method.startswith("ar_"):
            return autoregressive_generate(kv("target", method[3:]), ids, **g)
        if method.startswith("sd_"):
            _, which, kind = method.split("_")
            return speculative_generate(kv("target", kind), kv(which, kind), ids, gamma=args.gamma, **g)
        if method in ("hf_generate", "hf_assisted"):
            # override Qwen's generation_config (repetition_penalty=1.05, top_k/top_p) for a fair comparison
            kw = dict(max_new_tokens=args.max_new_tokens, eos_token_id=eos, pad_token_id=tok.pad_token_id,
                      do_sample=args.temperature > 0, repetition_penalty=1.0, top_k=None, top_p=None)
            if args.temperature > 0:
                kw.update(temperature=args.temperature)
            if method == "hf_assisted":
                kw["assistant_model"] = hf_assistant
            out = target.generate(ids, attention_mask=torch.ones_like(ids), **kw)
            return out, None
        raise ValueError(method)

    results = {"config": vars(args), "rows": []}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ds_name in args.datasets:
        prompts = load_prompts(ds_name, args.n)
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                         add_generation_prompt=True) for p in prompts]
        enc = [tok(t, return_tensors="pt", add_special_tokens=False).input_ids.cuda() for t in texts]
        # warmup
        for m in args.methods:
            run(m, enc[0][:, :32])
        ref, ref_method = {}, next((m for m in args.methods if m.startswith("ar_")), None)
        for m in args.methods:
            n = args.n_nocache if m.endswith("nocache") else args.n
            torch.manual_seed(args.seed)
            tot_tok = tot_time = drafted = accepted = t_calls = 0
            match = 0
            for i, ids in enumerate(enc[:n]):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out, st = run(m, ids)
                torch.cuda.synchronize()
                tot_time += time.perf_counter() - t0
                gen = out[0, ids.shape[1]:].tolist()
                tot_tok += len(gen)
                if m == ref_method:
                    ref[i] = gen
                elif i in ref:
                    match += gen == ref[i]
                if st is not None:
                    drafted += st.drafted
                    accepted += st.accepted
                    t_calls += st.target_calls
            row = {"dataset": ds_name, "method": m, "n": n, "tokens": tot_tok, "time_s": tot_time,
                   "tok_per_s": tot_tok / tot_time}
            if m.startswith("sd_"):
                row["acceptance_rate"] = accepted / max(drafted, 1)
                row["tokens_per_target_call"] = tot_tok / max(t_calls, 1)
            if m != ref_method and ref:
                row["greedy_match_vs_ar"] = match / min(n, len(ref))
            results["rows"].append(row)
            print(json.dumps(row), flush=True)
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)

    # speedups relative to target-only decoding
    for ds_name in args.datasets:
        rows = {r["method"]: r for r in results["rows"] if r["dataset"] == ds_name}
        for r in rows.values():
            for base in ("ar_nocache", "ar_dynamic", "ar_static"):
                if base in rows:
                    r[f"speedup_vs_{base}"] = r["tok_per_s"] / rows[base]["tok_per_s"]
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
