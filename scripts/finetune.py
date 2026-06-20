#!/usr/bin/env python3
"""scripts/finetune.py

Standalone fine-tuning script for the Chell project.

Loads the training config, builds a ChellTrainer, and runs training.

Usage
-----
    python scripts/finetune.py
    python scripts/finetune.py --config configs/train.yaml --model-config configs/model.yaml
    python scripts/finetune.py --subset 50   # smoke-test on first 50 cases
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_yaml(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        print("ERROR: pyyaml not installed. Run: pip install pyyaml", file=sys.stderr)
        sys.exit(1)
    p = Path(path)
    if not p.exists():
        print(f"ERROR: config file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a Chell model using ChellTrainer."
    )
    parser.add_argument(
        "--config",
        default="configs/train.yaml",
        metavar="PATH",
        help="Path to training config YAML (default: configs/train.yaml).",
    )
    parser.add_argument(
        "--model-config",
        default="configs/model.yaml",
        metavar="PATH",
        help="Path to model config YAML (default: configs/model.yaml).",
    )
    parser.add_argument(
        "--cases-dir",
        default="data/curated/",
        metavar="DIR",
        help="Directory with curated JSON case files (default: data/curated/).",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        metavar="N",
        help="Limit training to the first N cases (useful for smoke tests).",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA adapters (full fine-tuning).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load configs
    # ------------------------------------------------------------------
    print(f"Loading training config from {args.config} ...")
    train_config = _load_yaml(args.config)

    print(f"Loading model config from {args.model_config} ...")
    model_config = _load_yaml(args.model_config)

    backend_type = model_config.get("backend", "hf")
    if backend_type == "api":
        print(
            "ERROR: Fine-tuning requires a local HF backend, not an API backend.\n"
            "Set 'backend: hf' in your model config.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Imports
    # ------------------------------------------------------------------
    try:
        from chell.models.registry import build_backend
        from chell.data.dataset import ChellDataset
        from chell.training.trainer import ChellTrainer
    except ImportError as exc:
        print(f"Import error: {exc}", file=sys.stderr)
        print("Ensure all dependencies are installed: pip install -e .", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print(f"Loading training dataset from {args.cases_dir} ...")
    dataset = ChellDataset(cases_dir=args.cases_dir, split="train")

    if args.subset is not None:
        try:
            from torch.utils.data import Subset
            n = min(args.subset, len(dataset))
            dataset = Subset(dataset, list(range(n)))
            print(f"Using subset of {n} cases.")
        except ImportError:
            print("WARNING: torch not available — cannot create Subset; using full dataset.")
    else:
        print(f"Loaded {len(dataset)} training cases.")

    # ------------------------------------------------------------------
    # Build backend
    # ------------------------------------------------------------------
    print("Building backend ...")
    try:
        backend = build_backend(model_config)
    except Exception as exc:
        print(f"Failed to build backend: {exc}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    use_lora = not args.no_lora
    lora_cfg = train_config.get("lora", {})
    use_lora = use_lora and lora_cfg.get("enabled", True)

    trainer = ChellTrainer(
        backend=backend,
        dataset=dataset,
        config=train_config,
        use_lora=use_lora,
    )

    print("\nStarting training...")
    try:
        metrics = trainer.train()
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print("\n=== Training Complete ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    output_dir = train_config.get("output_dir", "results/checkpoints")
    print(f"\nCheckpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
