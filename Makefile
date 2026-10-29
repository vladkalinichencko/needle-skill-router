PY := .venv/bin/python
export PYTHONPATH := src
export NEEDLE_TELEMETRY := 0

venv:
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) "cactus-needle[train]" pyyaml matplotlib

skills:
	$(PY) -m router.cli skills

collect:
	$(PY) -m router.cli collect

dataset: skills
	$(PY) -m router.cli dataset

eda:
	$(PY) -m router.cli eda

all: skills collect dataset eda

.PHONY: venv skills collect dataset eda all
