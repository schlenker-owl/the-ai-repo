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


# --- device selection (MPS-first) ---
def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


def amp_autocast(device_str: str):
    # float16 on mps is the practical sweet spot; bf16 elsewhere
    dtype = torch.float16 if device_str == "mps" else torch.bfloat16
    return torch.autocast(device_type=device_str, dtype=dtype)


# --- tiny data pipeline (unchanged) ---
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
    """
    Packs documents into fixed-length batches for next-token LM training.
    For each batch, we need B*(T+1) tokens so that x has shape [B,T] and y is x shifted by 1.
    """
    sep = [10]  # newline
    toks: list[int] = []
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

    # Optional tail pad (kept identical to your working version)
    if 0 < len(toks) < need:
        pad_len = need - len(toks)
        toks.extend([0] * pad_len)
        buf = torch.tensor(toks[:need], dtype=torch.long).view(batch_size, seq_len + 1)
        x = buf[:, :-1].contiguous().to(device)
        y = buf[:, 1:].contiguous().to(device)
        yield x, y


# --- eval perplexity (unchanged math) ---
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


# --- tiny helpers for telemetry (no training behavior changes) ---
def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


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
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--ckpt-dir", type=str, default="ckpts_tinygpt")
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--max-chars", type=int, default=5000)
    ap.add_argument("--min-ascii", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    # Telemetry-only extras
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

    # data (unchanged)
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

    def make_eval_iter(n_batches=8):
        if args.text_file:
            src2 = _iter_file(args.text_file)
        else:
            src2 = _iter_hf(args.dataset)
        s2 = stream_docs(
            src2, min_chars=args.min_chars, max_chars=args.max_chars, min_ascii=args.min_ascii
        )
        return pack_stream(tok, s2, seq_len=args.seq, batch_size=args.batch, device=device)

    # model (unchanged)
    model = TinyGPTModern(
        vocab_size=tok.vocab_size,
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        n_kv_heads=args.kv_heads,
        ffn_mult=args.ffn_mult,
        dropout=args.dropout,
    ).to(device)

    # optim (unchanged)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ---------- Telemetry header ----------
    n_trainable = count_params(model, trainable_only=True)
    n_total = count_params(model, trainable_only=False)
    compute_optimal_tokens = args.target_tokens if args.target_tokens > 0 else 20.0 * n_total
    tokens_per_update = args.batch * args.seq * args.grad_accum
    planned_tokens = tokens_per_update * args.steps
    weights_mem_fp16 = bytes_approx(n_total, 2)
    opt_states_mem = bytes_approx(n_total * 12)  # AdamW m+v (fp32) + grads (fp16) ≈ 12 bytes/param

    header = {
        "device": dev_str,
        "dataset": data_label,
        "seed": args.seed,
        "d_model": args.d_model,
        "layers": args.layers,
        "heads": args.heads,
        "kv_heads": args.kv_heads,
        "ffn_mult": args.ffn_mult,
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
        f" Model:             d_model={args.d_model} | L={args.layers} | heads={args.heads}/{args.kv_heads} (Q/KV) | ffn_mult={args.ffn_mult}"
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

    # Metrics CSV
    csv_path = os.path.join(args.ckpt_dir, args.metrics_csv)
    csv_new = not os.path.exists(csv_path)
    if csv_new:
        with open(csv_path, "w", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow(
                ["step", "lr", "loss_ma", "ppl", "tok_per_sec", "tokens_cum", "planned_vs_target"]
            )

    # ---------- Training loop (unchanged math) ----------
    model.train()
    accum = 0
    running = 0.0
    loss_window = []  # for moving average display only
    tokens_cum = 0
    t0 = time.time()

    for step in range(1, args.steps + 1):
        x, y = next(train_iter)
        with amp_autocast(dev_str):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, tok.vocab_size), y.view(-1))

        (loss / args.grad_accum).backward()
        running += loss.item()
        loss_window.append(loss.item())
        if len(loss_window) > 50:  # MA over last 50 minibatches
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

            # Throughput + telemetry every 20 optimizer steps
            if step % 20 == 0 or step == 1:
                dt = time.time() - t0
                tok_per_sec = 0 if step == 1 else int(tokens_per_update * 20 / max(1e-6, dt))
                tokens_cum += tokens_per_update * (1 if step == 1 else 20)
                loss_ma = sum(loss_window) / max(1, len(loss_window))
                planned_vs_target = tokens_cum / compute_optimal_tokens
                print(
                    f"step {step:5d} | loss(ma) {loss_ma:.3f} | lr {lr:.2e} | ~{tok_per_sec:,} tok/s | tokens {tokens_cum:,} ({planned_vs_target:.2%} of target)"
                )
                # append to CSV
                with open(csv_path, "a", newline="") as cf:
                    writer = csv.writer(cf)
                    writer.writerow(
                        [
                            step,
                            lr,
                            f"{loss_ma:.6f}",
                            "",
                            tok_per_sec,
                            tokens_cum,
                            f"{planned_vs_target:.6f}",
                        ]
                    )
                t0 = time.time()
                running = 0.0  # keep your original running reset semantics

        # eval (unchanged math)
        if step % args.eval_every == 0 or step == args.steps:
            ppl = eval_ppl(model, make_eval_iter(), steps=8, device_str=dev_str)
            loss_ma = sum(loss_window) / max(1, len(loss_window))
            print(f"[eval] step {step} | perplexity {ppl:.2f}")
            # append eval row to CSV
            with open(csv_path, "a", newline="") as cf:
                writer = csv.writer(cf)
                writer.writerow(
                    [
                        step,
                        lr,
                        f"{loss_ma:.6f}",
                        f"{ppl:.4f}",
                        "",
                        tokens_cum,
                        f"{(tokens_cum/compute_optimal_tokens):.6f}",
                    ]
                )

        # checkpoint (unchanged)
        if step % args.save_every == 0 or step == args.steps:
            path = os.path.join(args.ckpt_dir, f"tinygpt_step{step}.pt")
            torch.save({"model": model.state_dict(), "cfg": vars(args), "step": step}, path)
            print(f"[ckpt] saved {path}")

    # quick sample (unchanged)
    model.eval()
    prompt = "Transformers attend to what matters.\n"
    ids = ByteTokenizer.encode(prompt).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=200, temperature=0.9, top_k=50)
    text = ByteTokenizer.decode(out[0].cpu())
    print("\n=== SAMPLE ===\n", text, "\n==============")


if __name__ == "__main__":
    main()
