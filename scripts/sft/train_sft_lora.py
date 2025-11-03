import typer
from airoad.sft.lora_sft import SFTConfig, run_lora_sft

app = typer.Typer(add_completion=False)

@app.command()
def main(
    base_model: str = "sshleifer/tiny-gpt2",
    output_dir: str = "outputs/sft-lora",
    max_steps: int = 50,
    learning_rate: float = 5e-4,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    max_seq_len: int = 256,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 1,
    use_trl: bool = True,
):
    cfg = SFTConfig(
        base_model=base_model,
        output_dir=output_dir,
        max_steps=max_steps,
        learning_rate=learning_rate,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_trl=use_trl,
    )
    run_lora_sft(cfg)

if __name__ == "__main__":
    app()
