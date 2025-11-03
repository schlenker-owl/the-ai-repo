import typer
from typing import List, Tuple
from airoad.sft.lora_sft import build_alpaca_like_examples, format_example
from airoad.sft.infer import load_base, load_with_lora, merge_and_save_lora, generate, GenConfig
from airoad.sft.eval_sft import evaluate_pairs

app = typer.Typer(add_completion=False)

@app.command()
def main(
    base_model: str = "sshleifer/tiny-gpt2",
    lora_dir: str = "outputs/sft-lora",
    use_merged: bool = False,
    merged_out: str = "outputs/sft-lora-merged",
    max_new_tokens: int = 64,
):
    cfg = GenConfig(max_new_tokens=max_new_tokens, temperature=0.0, do_sample=False)

    # toy refs
    exs = build_alpaca_like_examples()
    prompts: List[str] = []
    refs: List[str] = []
    for ex in exs:
        p, t = format_example(ex)
        prompts.append(p)
        refs.append(t)

    # before
    base, tok = load_base(base_model)
    before = generate(base, tok, prompts, cfg)

    # after
    if use_merged:
        merge_and_save_lora(base_model, lora_dir, merged_out)
        merged, tokm = load_base(merged_out)
        after = generate(merged, tokm, prompts, cfg)
    else:
        lora_model, tokL = load_with_lora(base_model, lora_dir)
        after = generate(lora_model, tokL, prompts, cfg)

    # evaluate
    pairs_before: List[Tuple[str, str]] = list(zip(before, refs))
    pairs_after: List[Tuple[str, str]]  = list(zip(after, refs))
    m_before = evaluate_pairs(pairs_before)
    m_after  = evaluate_pairs(pairs_after)

    print("=== SFT toy metrics ===")
    print("Before:", m_before)
    print("After: ", m_after)

if __name__ == "__main__":
    app()
