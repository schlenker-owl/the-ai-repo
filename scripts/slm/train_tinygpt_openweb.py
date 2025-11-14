# scripts/slm/train_tinygpt_openweb.py
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import time
from collections import deque
from typing import Deque, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from airoad.transformers.tinygpt_modern import ByteTokenizer, TinyGPTModern


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


def amp_autocast(device_str: str):
    dtype = torch.float16 if device_str == "mps" else torch.bfloat16
    return torch.autocast(device_type=device_str, dtype=dtype)


# data helpers (unchanged) ...
# from datasets import load_dataset if False else None  # placate linters


def _iter_hf(name: str) -> Iterable[str]:
    from datasets import load_dataset

    key = name.lower()
    if key in {"fineweb", "fw"}:
        ds = load_dataset("HuggingFaceFW/fineweb", name="default", split="train", streaming=True)
        col = "text"
    elif key in {"fineweb-edu", "fw-edu"}:
        ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
        col = "text"
    elif key in {"c4"}:
        ds = load_dataset("c4", "en", split="train", streaming=True)
        col = "text"
    elif key in {"openwebtext", "owt2"}:
        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        col = "text"
    elif key in {"redpajama", "rj"}:
        ds = load_dataset(
            "togethercomputer/RedPajama-Data-V2", "sample", split="train", streaming=True
        )
        col = "text"
    else:
        raise ValueError(f"Unknown dataset {name}")
    for ex in ds:
        t = ex.get(col)
        if t:
            yield t


def _iter_file(path: str) -> Iterable[str]:
    with io.open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s:
                yield s


def ascii_ratio(s: str) -> float:
    if not s:
        return 0.0
    a = sum(1 for ch in s if ord(ch) < 128)
    return a / max(1, len(s))


def stream_docs(
    src: Iterable[str], min_chars=200, max_chars=5000, min_ascii=0.5, dedup_window=100_000
) -> Iterable[str]:
    seen: Deque[int] = deque(maxlen=dedup_window)
    for raw in src:
        s = raw.strip()
        if len(s) < min_chars or len(s) > max_chars:
            continue
        if ascii_ratio(s) < min_ascii:
            continue
        h = int(hashlib.blake2b(s.encode("utf-8", "ignore"), digest_size=8).hexdigest(), 16)
        if h in seen:
            continue
        seen.append(h)
        yield s


def pack_stream(
    tok: ByteTokenizer, it: Iterable[str], seq_len: int, batch_size: int, device
) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    sep = [10]
    toks = []
    need = batch_size * (seq_len + 1)
    for doc in it:
        toks.extend(tok.encode(doc).tolist())
        toks.extend(sep)
        while len(toks) >= need:
            arr = toks[:need]
            toks = toks[need:]
            buf = torch.tensor(arr, dtype=torch.long).view(batch_size, seq_len + 1)
            x = buf[:, :-1].contiguous().to(device)
            y = buf[:, 1:].contiguous().to(device)
            yield x, y
    if 0 < len(toks) < need:
        toks.extend([0] * (need - len(toks)))
        buf = torch.tensor(toks[:need], dtype=torch.long).view(batch_size, seq_len + 1)
        x = buf[:, :-1].contiguous().to(device)
        y = buf[:, 1:].contiguous().to(device)
        yield x, y


@torch.no_grad()
def eval_ppl(
    model: nn.Module, it: Iterable[Tuple[torch.Tensor, torch.Tensor]], steps: int, device_str: str
) -> float:
    model.eval()
    loss_sum = 0.0
    n_tok = 0
    for i, (x, y) in enumerate(it):
        if i >= steps:
            break
        with amp_autocast(device_str):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))
        n = y.numel()
        loss_sum += loss.item() * n
        n_tok += n
    model.train()
    return math.exp(loss_sum / max(1, n_tok)) if n_tok else float("inf")


def cosine_lr(step, total, base_lr, min_lr=1e-5, warmup=0):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


def human(n: float) -> str:
    if n >= 1e12:
        return f"{n/1e12:.2f}T"
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.2f}K"
    return f"{n:.0f}"


def bytes_approx(params: int, dtype_bytes: int = 2) -> str:
    b = params * dtype_bytes
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if b < 1024 or unit == "TB":
            return f"{b:.1f}{unit}"
        b /= 1024


def fmt_hms(seconds: float) -> str:
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    s = s % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        type=str,
        default="fineweb",
        help="fineweb | fineweb-edu | c4 | openwebtext | redpajama",
    )
    ap.add_argument("--text-file", type=str, default=None)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--ffn-mult", type=float, default=8 / 3)
    ap.add_argument(
        "--ffn-type",
        type=str,
        default="swiglu",
        choices=["swiglu", "gelu_mlp", "geglu", "reglu", "conv_glu", "moe"],
    )
    ap.add_argument("--conv-kernel", type=int, default=5)
    ap.add_argument("--conv-dilation", type=int, default=1)
    ap.add_argument("--moe-experts", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument(
        "--eval-steps", type=int, default=8, help="number of eval batches per eval call"
    )
    ap.add_argument(
        "--skip-eval", action="store_true", help="disable periodic eval to maximize throughput"
    )
    ap.add_argument(
        "--log-every", type=int, default=20, help="training log frequency (optimizer steps)"
    )
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--ckpt-dir", type=str, default="ckpts_tinygpt")
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--max-chars", type=int, default=5000)
    ap.add_argument("--min-ascii", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--target-tokens",
        type=float,
        default=0.0,
        help="Override compute-optimal token target (0 = auto: 20×params)",
    )
    ap.add_argument("--metrics-csv", type=str, default="tinygpt_metrics.csv")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device, dev_str = pick_device()

    # data
    if args.text_file:
        src = _iter_file(args.text_file)
        data_label = f"file:{os.path.basename(args.text_file)}"
    else:
        src = _iter_hf(args.dataset)
        data_label = f"hf:{args.dataset}"
    stream = stream_docs(
        src, min_chars=args.min_chars, max_chars=args.max_chars, min_ascii=args.min_ascii
    )
    tok = ByteTokenizer()
    train_iter = pack_stream(tok, stream, seq_len=args.seq, batch_size=args.batch, device=device)

    def make_eval_iter():
        if args.text_file:
            src2 = _iter_file(args.text_file)
        else:
            src2 = _iter_hf(args.dataset)
        s2 = stream_docs(
            src2, min_chars=args.min_chars, max_chars=args.max_chars, min_ascii=args.min_ascii
        )
        return pack_stream(tok, s2, seq_len=args.seq, batch_size=args.batch, device=device)

    # model
    model = TinyGPTModern(
        vocab_size=tok.vocab_size,
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        n_kv_heads=args.kv_heads,
        ffn_mult=args.ffn_mult,
        dropout=args.dropout,
        ffn_type=args.ffn_type,
        conv_kernel=args.conv_kernel,
        conv_dilation=args.conv_dilation,
        moe_experts=args.moe_experts,
    ).to(device)

    # optim
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # telemetry header
    n_trainable = count_params(model, trainable_only=True)
    n_total = count_params(model, trainable_only=False)
    compute_optimal_tokens = args.target_tokens if args.target_tokens > 0 else 20.0 * n_total
    tokens_per_update = args.batch * args.seq * args.grad_accum
    planned_tokens = tokens_per_update * args.steps
    weights_mem_fp16 = bytes_approx(n_total, 2)
    opt_states_mem = bytes_approx(n_total * 12)

    header = {
        "device": dev_str,
        "dataset": data_label,
        "seed": args.seed,
        "d_model": args.d_model,
        "layers": args.layers,
        "heads": args.heads,
        "kv_heads": args.kv_heads,
        "ffn_type": args.ffn_type,
        "ffn_mult": args.ffn_mult,
        "conv_kernel": args.conv_kernel,
        "conv_dilation": args.conv_dilation,
        "moe_experts": args.moe_experts,
        "seq": args.seq,
        "batch": args.batch,
        "grad_accum": args.grad_accum,
        "trainable_params": n_trainable,
        "total_params": n_total,
        "params_human": human(n_total),
        "weights_mem_fp16": weights_mem_fp16,
        "opt_states_mem_est": opt_states_mem,
        "tokens_per_update": tokens_per_update,
        "planned_steps": args.steps,
        "planned_tokens": planned_tokens,
        "compute_optimal_tokens": compute_optimal_tokens,
    }
    with open(os.path.join(args.ckpt_dir, "experiment_manifest.json"), "w") as f:
        json.dump(header, f, indent=2)

    print("\n===== TinyGPT Training =====")
    print(f" Device:            {dev_str}")
    print(f" Data:              {data_label}")
    print(
        f" Model:             d_model={args.d_model} | L={args.layers} | heads={args.heads}/{args.kv_heads} (Q/KV)"
    )
    print(
        f" FFN:               type={args.ffn_type} | mult={args.ffn_mult} "
        f"{'(k='+str(args.conv_kernel)+',d='+str(args.conv_dilation)+')' if args.ffn_type=='conv_glu' else ''}"
        f"{'(experts='+str(args.moe_experts)+')' if args.ffn_type=='moe' else ''}"
    )
    print(f" Params:            {human(n_total)} (trainable {human(n_trainable)})")
    print(f" Weights (fp16):    ~{weights_mem_fp16} | Optimizer states est: ~{opt_states_mem}")
    print(
        f" Batch/Seq/Accum:   {args.batch} / {args.seq} / {args.grad_accum} → tokens/update={tokens_per_update:,}"
    )
    print(f" Planned:           steps={args.steps:,} → tokens={planned_tokens:,}")
    print(
        f" Target tokens:     {human(compute_optimal_tokens)} (≈20× params)  | Planned/Target = {planned_tokens/compute_optimal_tokens:.2%}\n"
    )

    # metrics CSV (elapsed_sec)
    csv_path = os.path.join(args.ckpt_dir, args.metrics_csv)
    csv_new = not os.path.exists(csv_path)
    if csv_new:
        with open(csv_path, "w", newline="") as cf:
            csv.writer(cf).writerow(
                [
                    "step",
                    "lr",
                    "loss_ma",
                    "ppl",
                    "tok_per_sec",
                    "tokens_cum",
                    "planned_vs_target",
                    "elapsed_sec",
                ]
            )

    # train
    model.train()
    accum = 0
    running = 0.0
    loss_window = []
    tokens_cum = 0
    t0 = time.time()
    t_start = time.time()

    def write_csv_row(step, lr, loss_ma, ppl_str, tok_per_sec, tokens_cum, planned_vs_target):
        elapsed_sec = time.time() - t_start
        with open(csv_path, "a", newline="") as cf:
            csv.writer(cf).writerow(
                [
                    step,
                    lr,
                    f"{loss_ma:.6f}",
                    ppl_str,
                    tok_per_sec,
                    tokens_cum,
                    f"{planned_vs_target:.6f}",
                    f"{elapsed_sec:.2f}",
                ]
            )

    for step in range(1, args.steps + 1):
        x, y = next(train_iter)
        with amp_autocast(dev_str):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))

        (loss / args.grad_accum).backward()
        running += loss.item()
        loss_window.append(loss.item())
        if len(loss_window) > 50:
            loss_window.pop(0)
        accum += 1

        if accum == args.grad_accum:
            lr = cosine_lr(step, args.steps, args.lr, min_lr=args.lr * 0.1, warmup=args.warmup)
            for pg in opt.param_groups:
                pg["lr"] = lr
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            accum = 0

            if step % args.log_every == 0 or step == 1:
                dt = time.time() - t0
                tok_per_sec = (
                    0 if step == 1 else int(tokens_per_update * args.log_every / max(1e-6, dt))
                )
                tokens_cum += tokens_per_update * (1 if step == 1 else args.log_every)
                loss_ma = sum(loss_window) / max(1, len(loss_window))
                planned_vs_target = tokens_cum / compute_optimal_tokens
                elapsed_hms = fmt_hms(time.time() - t_start)
                print(
                    f"step {step:5d} | loss(ma) {loss_ma:.3f} | lr {lr:.2e} | ~{tok_per_sec:,} tok/s | tokens {tokens_cum:,} ({planned_vs_target:.2%} of target) | elapsed {elapsed_hms}"
                )
                write_csv_row(step, lr, loss_ma, "", tok_per_sec, tokens_cum, planned_vs_target)
                t0 = time.time()
                running = 0.0

        if not args.skip_eval and (step % args.eval_every == 0 or step == args.steps):
            ppl = eval_ppl(model, make_eval_iter(), steps=args.eval_steps, device_str=dev_str)
            loss_ma = sum(loss_window) / max(1, len(loss_window))
            elapsed_hms = fmt_hms(time.time() - t_start)
            print(f"[eval] step {step} | perplexity {ppl:.2f} | elapsed {elapsed_hms}")
            write_csv_row(
                step,
                lr,
                loss_ma,
                f"{ppl:.4f}",
                "",
                tokens_cum,
                (tokens_cum / compute_optimal_tokens),
            )

        if step % args.save_every == 0 or step == args.steps:
            path = os.path.join(args.ckpt_dir, f"tinygpt_step{step}.pt")
            torch.save({"model": model.state_dict(), "cfg": vars(args), "step": step}, path)
            print(f"[ckpt] saved {path}")

    total_elapsed = time.time() - t_start
    print(f"\n=== TRAINING DONE — elapsed {fmt_hms(total_elapsed)} ({total_elapsed:.2f}s) ===")

    model.eval()
    prompt = "Transformers attend to what matters.\n"
    ids = ByteTokenizer.encode(prompt).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=200, temperature=0.9, top_k=50)
    print("\n=== SAMPLE ===\n", ByteTokenizer.decode(out[0].cpu()), "\n==============")


if __name__ == "__main__":
    main()
