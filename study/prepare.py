"""
Load 200 items from ARC-Challenge, BoolQ, and SQuAD.
Apply all 8 prompt templates per item - write prompts.jsonl.

Usage:
    python study/prepare.py --n 200 --seed 42 --out-dir study/output
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

# Allow importing templates.py from same directory
sys.path.insert(0, str(Path(__file__).parent))
from templates import all_templates


def load_arc_challenge(n: int, seed: int) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset("ai2_arc", "ARC-Challenge", split="test")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))
    items = []
    for idx in indices:
        row = ds[idx]
        labels = row["choices"]["label"]
        texts  = row["choices"]["text"]
        options = [f"{lbl}) {txt}" for lbl, txt in zip(labels, texts)]
        items.append({
            "id":          f"arc_{idx}",
            "dataset":     "arc_challenge",
            "question":    row["question"],
            "options":     options,
            "gold_answer": row["answerKey"],
            "task_type":   "mcq",
        })
    return items

def load_boolq(n: int, seed: int) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset("google/boolq", split="validation")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))
    items = []
    for idx in indices:
        row = ds[idx]
        items.append({
            "id":          f"boolq_{idx}",
            "dataset":     "boolq",
            "question":    row["question"],
            "passage":     row["passage"][:800],
            "gold_answer": "yes" if row["answer"] else "no",
            "task_type":   "boolean",
        })
    return items


def load_squad(n: int, seed: int) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad", split="validation")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))
    items = []
    for idx in indices:
        row = ds[idx]
        items.append({
            "id":               f"squad_{idx}",
            "dataset":          "squad",
            "question":         row["question"],
            "context":          row["context"][:800],
            "gold_answer":      row["answers"]["text"][0],
            "all_gold_answers": row["answers"]["text"],
            "task_type":        "open_ended",
        })
    return items

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare prompts for the study.")
    parser.add_argument("--n",       type=int, default=200,            help="Items per dataset")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--out-dir", default="study/output")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaders = [
        ("arc_challenge", load_arc_challenge),
        ("boolq",         load_boolq),
        ("squad",         load_squad),
    ]

    all_prompts: List[Dict[str, Any]] = []
    for dataset_name, loader in loaders:
        items = loader(args.n, args.seed)
        print(f"  Loaded {len(items)} items from {dataset_name}")
        for item in items:
            for tmpl in all_templates(item, dataset_name):
                all_prompts.append({**item, **tmpl})

    out_file = out_dir / "prompts.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for row in all_prompts:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    expected = args.n * 3 * 8
    print(f"Wrote {len(all_prompts)} prompts to {out_file}  (expected {expected})")


if __name__ == "__main__":
    main()



