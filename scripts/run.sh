#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

GREEN='\033[0;32m'; DIM='\033[0;2m'; YELL='\033[0;33m'; NC='\033[0m'
say() { printf "${GREEN}%s${NC}\n" "$*"; }
note() { printf "${DIM}%s${NC}\n" "$*"; }
warn() { printf "${YELL}%s${NC}\n" "$*"; }

STEP=0
step() { STEP=$((STEP + 1)); printf "\n${GREEN}[%d]${NC} %s\n" "$STEP" "$*"; }

MAXLEN="${MAXLEN:-8192}"
EPOCHS="${EPOCHS:-10}"
RANK="${RANK:-16}"

gpu_present() { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; }

gpu_memory_mib() {
  nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

pick_batch() {
  local mib="${BATCH:-0}"
  if [ "$mib" != "0" ]; then BATCH="$mib"; return; fi
  if ! gpu_present; then BATCH=8; note "карты нет, батч $BATCH"; return; fi
  mib="$(gpu_memory_mib)"
  if [ -z "$mib" ]; then BATCH=8; note "объём памяти не определился, батч $BATCH"; return; fi
  if [ "$mib" -ge 70000 ]; then
    BATCH=16
  elif [ "$mib" -ge 30000 ]; then
    BATCH=8
  else
    BATCH=4
  fi
  note "карта ${mib} МиБ → батч $BATCH (длина $MAXLEN, эпох $EPOCHS)"
}

environment_ready() {
  [ -x .venv/bin/python ] && .venv/bin/python -c "import needle" >/dev/null 2>&1
}

bootstrap() {
  step "проверяю окружение"
  if environment_ready; then
    note "окружение уже собрано, установку пропускаю"
  else
    step "создаю виртуальное окружение в .venv"
    note "python 3.12, это занимает считанные секунды"
    if command -v uv >/dev/null 2>&1; then
      uv venv --python 3.12 .venv
    else
      python3 -m venv .venv
      .venv/bin/pip install -q --upgrade pip
    fi

    step "ставлю зависимости"
    note "качается JAX и jaxlib — около 500 МБ, плюс needle, yaml, matplotlib"
    note "на быстром канале 1–3 минуты, на медленном до десяти; прогресс печатает сам установщик"
    local extra="[train]"
    if gpu_present; then extra="[train,gpu]"; note "карта найдена, ставлю CUDA-сборку JAX"; fi
    if command -v uv >/dev/null 2>&1; then
      uv pip install --python .venv/bin/python "cactus-needle${extra}" pyyaml matplotlib
    else
      .venv/bin/pip install "cactus-needle${extra}" pyyaml matplotlib
    fi
  fi

  step "проверяю бэкенд"
  .venv/bin/python - <<'PY'
import jax
print("jax", jax.__version__, "backend", jax.default_backend())
print("устройства:", [d.device_kind for d in jax.devices()])
PY
  if gpu_present; then
    note "если backend не gpu — значит CUDA-сборка не встала, обучение пойдёт на CPU и будет медленным"
  fi
}

train() {
  bootstrap
  pick_batch
  step "обучаю LoRA: ранг $RANK, $EPOCHS эпох, контекст $MAXLEN, батч $BATCH"
  note "валидация по эпохам на data/processed/val.jsonl, чекпоинт после каждой эпохи, ранний стоп по patience"
  local attempt batch="$BATCH"
  for attempt in 1 2 3; do
    if NEEDLE_TELEMETRY=0 .venv/bin/python scripts/finetune_plus.py data/processed/train.jsonl \
        --val-path data/processed/val.jsonl \
        --epochs "$EPOCHS" --lora-rank "$RANK" --lora-alpha 32 --batch-size "$batch" \
        --max-len "$MAXLEN" --seed 42 \
        --out checkpoints/router_lora.safetensors; then
      say "адаптер готов: checkpoints/router_lora.safetensors"
      return 0
    fi
    batch=$((batch / 2))
    [ "$batch" -lt 1 ] && break
    warn "прогон упал, пробую батч $batch (обычно причина — нехватка памяти под активации)"
  done
  warn "обучение не прошло; пришли лог, подберём длину и батч"
  return 1
}

build() {
  mkdir -p models
  step "собираю архив на 8 слоёв"
  NEEDLE_TELEMETRY=0 .venv/bin/needle build checkpoints/needle3.safetensors \
    --lora checkpoints/router_lora.safetensors --layers 8 --out models/router.cact
  ls -lh models/router.cact
}

evaluate() {
  mkdir -p reports
  step "оцениваю на отложенных диалогах"
  PYTHONPATH=src NEEDLE_TELEMETRY=0 .venv/bin/python -m router.cli eval \
    ${1:+--weights "$1"} --out reports/eval.json
}

collect() {
  bootstrap
  step "пересобираю данные из локальных сессий"
  PYTHONPATH=src .venv/bin/python -m router.cli collect
  PYTHONPATH=src .venv/bin/python -m router.cli dataset
  PYTHONPATH=src .venv/bin/python -m router.cli eda
}

menu() {
  echo
  echo "1) обучение на GPU  (rank $RANK, $EPOCHS эпох, контекст $MAXLEN)"
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
