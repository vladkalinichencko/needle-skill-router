#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

GREEN='\033[0;32m'; DIM='\033[0;2m'; NC='\033[0m'
say() { printf "${GREEN}%s${NC}\n" "$*"; }
note() { printf "${DIM}%s${NC}\n" "$*"; }

if [ ! -x .venv/bin/python ]; then
  echo "сначала подними окружение: ./scripts/run.sh и пункт 1, либо дай ему создать venv"
  exit 1
fi

say "проверяю бэкенд и устройства"
.venv/bin/python - <<'PY'
import jax
print("jax", jax.__version__, "backend", jax.default_backend())
for device in jax.devices():
    print("  устройство:", device.device_kind)
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  note "память карты: $(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader | head -1)"
fi

say "делаю три строки из обучающего набора"
head -3 data/processed/train.jsonl > data/processed/smoke.jsonl
wc -l data/processed/smoke.jsonl

say "один шаг обучения: батч 1, длина 8192"
note "это проверка трубы, а не обучение: важно увидеть backend gpu и loss больше нуля"
NEEDLE_TELEMETRY=0 .venv/bin/needle finetune data/processed/smoke.jsonl \
  --epochs 1 --lora-rank 16 --batch-size 1 --max-len 8192 \
  --val-split 0 --seed 42 --out checkpoints/smoke_lora.safetensors

say "готово"
note "если здесь всё прошло, а полный прогон падает — дело в батче, а не в коде"
