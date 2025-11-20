import typer

from airoad.sft.infer import (
    GenConfig,
    generate,
    load_base,
    load_with_lora,
    merge_and_save_lora,
)
from airoad.sft.lora_sft import build_alpaca_like_examples, format_example

app = typer.Typer(add_completion=False)

_PROMPTS = [format_example(ex)[0] for ex in build_alpaca_like_examples()][:5]  # first 5


@app.command()
def main(
    base_model: str = "sshleifer/tiny-gpt2",
    lora_dir: str = "outputs/sft-lora",
    merged_out: str = "outputs/sft-lora-merged",
    use_merged: bool = False,
    max_new_tokens: int = 64,
):
    cfg = GenConfig(max_new_tokens=max_new_tokens, temperature=0.0, do_sample=False)

    if use_merged:
        # merge if not merged yet; then load from merged_out as a base model
        print(f"[merge] Merging adapters from {lora_dir} -> {merged_out}")
        merge_and_save_lora(base_model, lora_dir, merged_out)
        base, tok = load_base(merged_out)
        after_texts = generate(base, tok, _PROMPTS, cfg)
        before, tok0 = load_base(base_model)
        before_texts = generate(before, tok0, _PROMPTS, cfg)
    else:
        # LoRA at inference
        base, tok = load_base(base_model)
        before_texts = generate(base, tok, _PROMPTS, cfg)
        lora_model, tokL = load_with_lora(base_model, lora_dir)
        after_texts = generate(lora_model, tokL, _PROMPTS, cfg)

    for i, prompt in enumerate(_PROMPTS):
        print("\n" + "=" * 80)
        print(f"PROMPT:\n{prompt.strip()}\n")
        print("BEFORE:")
        print(before_texts[i])
        print("\nAFTER:")
        print(after_texts[i])


if __name__ == "__main__":
    app()
