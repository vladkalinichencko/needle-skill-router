#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

GREEN='\033[0;32m'; DIM='\033[0;2m'; NC='\033[0m'
say() { printf "${GREEN}%s${NC}\n" "$*"; }
note() { printf "${DIM}%s${NC}\n" "$*"; }

GPU=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then GPU=1; fi

bootstrap() {
  if [ ! -d .venv ]; then
    say "создаю окружение"
    if command -v uv >/dev/null 2>&1; then
      uv venv --python 3.12 .venv
      if [ "$GPU" = "1" ]; then
        uv pip install --python .venv/bin/python "cactus-needle[train,gpu]" pyyaml matplotlib
      else
        uv pip install --python .venv/bin/python "cactus-needle[train]" pyyaml matplotlib
      fi
    else
      python3 -m venv .venv
      .venv/bin/pip install -q --upgrade pip
      if [ "$GPU" = "1" ]; then
        .venv/bin/pip install -q "cactus-needle[train,gpu]" pyyaml matplotlib
      else
        .venv/bin/pip install -q "cactus-needle[train]" pyyaml matplotlib
      fi
    fi
  fi
  .venv/bin/python - <<'PY'
import jax
print("jax", jax.__version__, "backend", jax.default_backend())
PY
}

train() {
  bootstrap
  say "обучаю LoRA (ранг 16, 10 эпох, контекст 8192)"
  NEEDLE_TELEMETRY=0 .venv/bin/needle finetune data/processed/train.jsonl \
    --epochs 10 --lora-rank 16 --lora-alpha 32 --batch-size 16 \
    --max-len 8192 --val-split 0 --seed 42 \
    --out checkpoints/router_lora.safetensors
}

build() {
  mkdir -p models
  say "собираю архив на 8 слоёв"
  NEEDLE_TELEMETRY=0 .venv/bin/needle build checkpoints/needle3.safetensors \
    --lora checkpoints/router_lora.safetensors --layers 8 --out models/router.cact
  ls -lh models/router.cact
}

evaluate() {
  mkdir -p reports
  say "оцениваю на отложенных диалогах"
  PYTHONPATH=src NEEDLE_TELEMETRY=0 .venv/bin/python -m router.cli eval \
    ${1:+--weights "$1"} --out reports/eval.json
}

collect() {
  bootstrap
  say "пересобираю данные из локальных сессий"
  PYTHONPATH=src .venv/bin/python -m router.cli collect
  PYTHONPATH=src .venv/bin/python -m router.cli dataset
  PYTHONPATH=src .venv/bin/python -m router.cli eda
}

menu() {
  echo
  echo "1) обучение на GPU  (train 10 эпох)"
  echo "2) собрать архив из готового адаптера"
  echo "3) оценка базовой модели"
  echo "4) оценка обученного архива"
  echo "5) пересобрать данные из сессий"
  echo "6) графики и статистика"
  echo "7) всё подряд: обучение, сборка, оценка"
  echo "0) выход"
  echo
}

if [ "${1:-}" != "" ]; then
  case "$1" in
    train) train ;;
    build) build ;;
    eval) evaluate "${2:-}" ;;
    collect) collect ;;
    *) echo "неизвестная команда: $1"; exit 1 ;;
  esac
  exit 0
fi

while true; do
  menu
  read -r -p "выбор: " choice
  case "$choice" in
    1) train ;;
    2) build ;;
    3) evaluate "" ;;
    4) evaluate "models/router.cact" ;;
    5) collect ;;
    6) PYTHONPATH=src .venv/bin/python -m router.cli eda ;;
    7) train && build && evaluate "models/router.cact" ;;
    0) exit 0 ;;
    *) note "нет такого пункта" ;;
  esac
done
