"""Fine-tune the draft model on target-generated data.

loss = alpha * CE(draft, target tokens) + (1 - alpha) * KL(p_target || q_draft)
computed on assistant tokens only. The target is kept frozen on the same GPU
and supplies full-vocabulary soft labels (DistillSpec-style forward KL).

Launch: torchrun --nproc_per_node N scripts/train_draft.py ...
"""

import argparse
import json
import math
import os
import random
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer


def encode(tok, messages, max_len):
    """Token ids + mask that is 1 on assistant-response tokens."""
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(messages, tokenize=False)
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    f_ids = tok(full, add_special_tokens=False).input_ids[:max_len]
    mask = [0] * len(p_ids) + [1] * (len(f_ids) - len(p_ids))
    return f_ids, mask[:len(f_ids)]


def batches(examples, tokens_per_batch, seed):
    """Length-bucketed batches with a padded-token budget."""
    rng = random.Random(seed)
    order = sorted(range(len(examples)), key=lambda i: len(examples[i][0]) + rng.random())
    out, cur, cur_max = [], [], 0
    for i in order:
        L = len(examples[i][0])
        if cur and max(cur_max, L) * (len(cur) + 1) > tokens_per_batch:
            out.append(cur)
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, L)
    if cur:
        out.append(cur)
    rng.shuffle(out)
    return out


def collate(examples, idxs, pad_id, device):
    L = max(len(examples[i][0]) for i in idxs)
    ids = torch.full((len(idxs), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(idxs), L), dtype=torch.long)
    lm = torch.zeros((len(idxs), L), dtype=torch.bool)
    for r, i in enumerate(idxs):
        x, m = examples[i]
        ids[r, :len(x)] = torch.tensor(x)
        att[r, :len(x)] = 1
        lm[r, :len(m)] = torch.tensor(m, dtype=torch.bool)
    return ids.to(device), att.to(device), lm.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--target", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--data", default="data/distill_train.jsonl")
    ap.add_argument("--out", default="checkpoints/draft-distill")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--tokens_per_batch", type=int, default=16384)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--alpha", type=float, default=0.5, help="CE weight; 1.0 disables KD")
    ap.add_argument("--kd_temp", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--grad_ckpt", action="store_true", help="only needed for much larger drafts")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.draft)
    with open(args.data) as f:
        rows = [json.loads(l) for l in f]
    examples = [encode(tok, r["messages"], args.max_len) for r in rows]
    examples = [e for e in examples if sum(e[1]) > 0]
    if rank == 0:
        n_tok = sum(sum(m) for _, m in examples)
        print(f"{len(examples)} examples, {n_tok / 1e6:.1f}M response tokens")

    draft = AutoModelForCausalLM.from_pretrained(args.draft, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(device)
    if args.grad_ckpt:
        draft.gradient_checkpointing_enable()
    draft.config.use_cache = False
    use_kd = args.alpha < 1.0
    if use_kd:
        target = AutoModelForCausalLM.from_pretrained(args.target, dtype=torch.bfloat16,
                                                      attn_implementation="sdpa").to(device).eval()
        target.requires_grad_(False)
    model = DDP(draft, device_ids=[device.index]) if ddp else draft
    V = draft.config.vocab_size

    opt = torch.optim.AdamW(draft.parameters(), lr=args.lr, weight_decay=0.0,
                            betas=(0.9, 0.95), fused=True)
    per_epoch = [batches(examples, args.tokens_per_batch, args.seed + e)[rank::world]
                 for e in range(args.epochs)]
    n_min = min(len(b) for b in per_epoch)
    total_steps = args.epochs * (n_min // args.grad_accum)
    warmup = int(args.warmup * total_steps)

    def lr_at(s):
        if s < warmup:
            return args.lr * (s + 1) / warmup
        return args.lr * 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(total_steps - warmup, 1)))

    step, t0 = 0, time.time()
    log = {"ce": 0.0, "kl": 0.0, "acc": 0.0, "n": 0}
    model.train()
    for epoch in range(args.epochs):
        bl = per_epoch[epoch][:n_min - n_min % args.grad_accum]
        for bi, idxs in enumerate(bl):
            ids, att, lm = collate(examples, idxs, tok.pad_token_id, device)
            # position t predicts token t+1
            sel = lm[:, 1:]
            labels = ids[:, 1:][sel]
            logits = model(input_ids=ids, attention_mask=att).logits[:, :-1][sel].float()
            ce = F.cross_entropy(logits, labels)
            loss = args.alpha * ce
            if use_kd:
                with torch.no_grad():
                    t_logits = target(input_ids=ids, attention_mask=att).logits[:, :-1][sel][:, :V].float()
                T = args.kd_temp
                kl = F.kl_div(F.log_softmax(logits / T, -1), F.log_softmax(t_logits / T, -1),
                              log_target=True, reduction="batchmean") * T * T
                loss = loss + (1 - args.alpha) * kl
                log["kl"] += kl.item()
                # how often draft argmax == target argmax: a proxy for greedy acceptance
                log["acc"] += (logits.argmax(-1) == t_logits.argmax(-1)).float().mean().item()
            log["ce"] += ce.item()
            log["n"] += 1
            (loss / args.grad_accum).backward()

            if (bi + 1) % args.grad_accum == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if rank == 0 and step % args.log_every == 0:
                    n = log["n"]
                    print(f"ep {epoch} step {step}/{total_steps} lr {lr_at(step):.2e} "
                          f"ce {log['ce'] / n:.4f} kl {log['kl'] / n:.4f} "
                          f"top1-match {log['acc'] / n:.4f} {time.time() - t0:.0f}s", flush=True)
                    log = {k: 0.0 for k in log}

        if rank == 0:
            path = f"{args.out}/epoch{epoch + 1}"
            draft.save_pretrained(path)
            tok.save_pretrained(path)
            print(f"saved {path}", flush=True)

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
