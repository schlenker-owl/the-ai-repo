SLM_CFG = configs/slm/qwen05b_lora.yaml
SLM_OUT = outputs/qwen05b_lora
BASE    = Qwen/Qwen2.5-0.5B-Instruct

slm-lora-train:
	python scripts/slm/qwen05b_lora_train.py --config $(SLM_CFG)

slm-lora-merge:
	python scripts/slm/qwen05b_lora_merge.py --base "$(BASE)" --adapters $(SLM_OUT) --out checkpoints/qwen05b_merged

slm-lora-infer:
	python scripts/slm/qwen05b_lora_infer.py --base "$(BASE)" --adapters $(SLM_OUT) --prompt "List three benefits of on-device AI."

slm-lora-chat:
	python scripts/slm/qwen05b_lora_infer.py --base checkpoints/qwen05b_merged --adapters "" --prompt "How do I reset my passphrase?"

train-fast:
	uv run python scripts/slm/qwen05b_lora_train.py --config configs/slm/qwen05b_lora_fast.yaml

train-reg:
	uv run python scripts/slm/qwen05b_lora_train.py --config configs/slm/qwen05b_lora_reg.yaml

train-long:
	uv run python scripts/slm/qwen05b_lora_train.py --config configs/slm/qwen05b_lora_long.yaml

ai-build:
	docker compose build ai-server

ai-up:
	docker compose up -d ai-server

ai-logs:
	docker compose logs -f ai-server

ai-down:
	docker compose down
