"""Generate distillation data: target-model responses to UltraChat prompts (vLLM)."""

import argparse
import json
import random

from datasets import load_dataset
from vllm import LLM, SamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--num_prompts", type=int, default=60000)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--out", default="data/distill_train.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    idx = list(range(len(ds)))
    random.Random(args.seed).shuffle(idx)
    prompts = []
    for i in idx[:args.num_prompts]:
        msg = ds[i]["messages"][0]
        if msg["role"] == "user" and msg["content"].strip():
            prompts.append(msg["content"])

    llm = LLM(args.target, tensor_parallel_size=args.tp, dtype="bfloat16",
              max_model_len=4096, gpu_memory_utilization=0.85, seed=args.seed)
    params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    convs = [[{"role": "user", "content": p}] for p in prompts]
    outs = llm.chat(convs, params)

    n = 0
    with open(args.out, "w") as f:
        for p, o in zip(prompts, outs):
            c = o.outputs[0]
            if c.finish_reason != "stop":  # drop truncated responses
                continue
            f.write(json.dumps({"messages": [{"role": "user", "content": p},
                                             {"role": "assistant", "content": c.text}]}) + "\n")
            n += 1
    print(f"wrote {n}/{len(prompts)} examples to {args.out}")


if __name__ == "__main__":
    main()
