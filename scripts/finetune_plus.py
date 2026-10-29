#!/usr/bin/env python3
"""Обучение LoRA с чекпоинтом после каждой эпохи и валидацией на отдельном файле."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from router.vendor.finetune import finetune_local


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl_path")
    parser.add_argument("--val-path", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--max-len", type=int, default=8192)
    parser.add_argument("--val-split", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-epoch-saves", action="store_true")
    ns = parser.parse_args()

    args = SimpleNamespace(
        jsonl_path=ns.jsonl_path,
        val_path=ns.val_path,
        epochs=ns.epochs,
        batch_size=ns.batch_size,
        lr=ns.lr,
        lora_rank=ns.lora_rank,
        lora_alpha=ns.lora_alpha,
        max_len=ns.max_len,
        val_split=ns.val_split,
        seed=ns.seed,
        patience=ns.patience,
        checkpoint=ns.checkpoint,
        checkpoint_dir=ns.checkpoint_dir,
        out=ns.out,
        save_every_epoch=not ns.no_epoch_saves,
        generate=0,
        model=None,
        workers=8,
        score=True,
    )
    finetune_local(args)


if __name__ == "__main__":
    main()
