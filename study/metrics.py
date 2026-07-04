"""
Compute all sensitivity metrics from judged.jsonl.

Per instance (item × model):
  sens_heuristic  — std of heuristic correctness scores across 8 templates
  sens_judge      — std of judge_correct across 8 templates
  eas             — |sens_heuristic - sens_judge|  (Evaluation-Attributable Sensitivity)
  signed_eas      — sens_heuristic - sens_judge  (positive = heuristic overestimates)
  stab_sem        — avg pairwise cosine similarity of 8 responses (semantic stability)
  trizone         — "artifact" | "genuine" | "stable"
  effect_{factor}_{eval} — main effect of each structural factor on each eval method

Outputs:
  metrics_instance.csv  — one row per (item, model)
  metrics_dataset.csv   — aggregated by (dataset, model)

Usage:
    python study/metrics.py \
        --in-files study/output/llama_judged.jsonl study/output/smollm_judged.jsonl
"""

import argparse
import json
import re
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# I/O 

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# Heuristic evaluation  

def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_letter(text: str, options: List[str]) -> str:
    upper = text.upper().strip()
    valid = {opt.split(")")[0].strip() for opt in options}
    for m in re.findall(r"\b([A-Z])\b", upper):
        if m in valid:
            return m
    matches = re.findall(r"\b([A-Z])\b", upper)
    return matches[0] if matches else upper[:1]


def _extract_yes_no(text: str) -> str:
    norm = _normalize(text)
    for word in norm.split():
        if word == "yes":
            return "yes"
        if word == "no":
            return "no"
    if "yes" in norm:
        return "yes"
    if "no" in norm:
        return "no"
    return norm.split()[0] if norm else ""


def _token_f1(pred: str, gold: str) -> float:
    pred_tok = _normalize(pred).split()
    gold_tok = _normalize(gold).split()
    if not pred_tok or not gold_tok:
        return 0.0
    common = set(pred_tok) & set(gold_tok)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tok)
    recall    = len(common) / len(gold_tok)
    return 2 * precision * recall / (precision + recall)


def heuristic_score(row: Dict[str, Any]) -> float:
    task     = row["task_type"]
    gold     = row["gold_answer"]
    response = row.get("raw_response", "") or ""

    if task == "mcq":
        pred = _extract_letter(response, row.get("options", []))
        return float(_normalize(pred) == _normalize(gold))
    elif task == "boolean":
        pred = _extract_yes_no(response)
        return float(_normalize(pred) == _normalize(gold))
    else:
        # open_ended: use max token F1 over all gold answers
        all_golds = row.get("all_gold_answers", [gold])
        return max(_token_f1(response, g) for g in all_golds)


#Semantic Stability 

def sem_stability(responses: List[str], model) -> float:
    """Average pairwise cosine similarity (embeddings already L2-normalised)."""
    if len(responses) < 2:
        return 1.0
    vecs = model.encode(responses, convert_to_numpy=True, normalize_embeddings=True)
    sims = [float(np.dot(vecs[i], vecs[j]))
            for i, j in combinations(range(len(vecs)), 2)]
    return float(np.mean(sims))


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, ci: float = 0.95) -> tuple:
    rng  = np.random.default_rng(42)
    boot = np.array([rng.choice(values, len(values), replace=True).mean()
                     for _ in range(n_boot)])
    lo = float(np.percentile(boot, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot, (1 + ci) / 2 * 100))
    return lo, hi


# Four-zone classification
#
# Classifies each instance by whether heuristic (SensH) and judge (SensJ)
# independently detect sensitivity, using the 75th-percentile of each as
# the "high sensitivity" threshold.
#
# Zone        | SensH high? | SensJ high? | Meaning
# ------------|-------------|-------------|-----------------------------------
# stable      |     no      |     no      | Both methods agree: no sensitivity
# artifact    |    yes      |     no      | Heuristic overcounts; judge doesn't see it
# underdetected|    no      |    yes      | Judge detects sensitivity heuristic misses
# genuine     |    yes      |    yes      | Both methods agree: real instability

def classify_fourzone(sens_heuristic: float, sens_judge: float,
                      sh_thresh: float, sj_thresh: float) -> str:
    h = sens_heuristic > sh_thresh
    j = sens_judge     > sj_thresh
    if h and not j:
        return "artifact"
    if not h and j:
        return "underdetected"
    if h and j:
        return "genuine"
    return "stable"


# Main 

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute sensitivity metrics.")
    parser.add_argument("--in-files", nargs="+",
                        default=["study/output/judged.jsonl"],
                        help="One or more judged.jsonl files (one per model).")
    parser.add_argument("--out-instance", default="study/output/metrics_instance.csv")
    parser.add_argument("--out-dataset",  default="study/output/metrics_dataset.csv")
    parser.add_argument("--no-sem", action="store_true",
                        help="Skip semantic similarity computation (faster).")
    args = parser.parse_args()

    # Load and combine all input files
    all_rows: List[Dict[str, Any]] = []
    for path_str in args.in_files:
        chunk = read_jsonl(Path(path_str))
        all_rows.extend(chunk)
        print(f"  Loaded {len(chunk)} rows from {path_str}")

    df = pd.DataFrame(all_rows)
    df["heuristic_score"] = df.apply(heuristic_score, axis=1)
    df["judge_correct"]   = df["judge_correct"].astype(float)

    # Load sentence-transformers once
    sem_model: Optional[Any] = None
    if not args.no_sem:
        try:
            from sentence_transformers import SentenceTransformer
            print("Loading sentence-transformers (all-MiniLM-L6-v2)...")
            sem_model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            print("sentence-transformers not installed — skipping StabSem. "
                  "Install with: pip install sentence-transformers")

    # Compute per-instance metrics
    group_cols = ["id", "dataset", "model_name"]
    instance_records: List[Dict[str, Any]] = []

    for (item_id, dataset, model_name), grp in df.groupby(group_cols):
        n = len(grp)

        sens_heuristic = float(grp["heuristic_score"].std(ddof=0)) if n > 1 else 0.0
        sens_judge     = float(grp["judge_correct"].std(ddof=0))    if n > 1 else 0.0
        eas            = abs(sens_heuristic - sens_judge)
        mean_heuristic = float(grp["heuristic_score"].mean())
        mean_judge     = float(grp["judge_correct"].mean())

        stab_sem: Optional[float] = None
        if sem_model is not None:
            responses = grp["raw_response"].fillna("").tolist()
            stab_sem  = sem_stability(responses, sem_model)

        # Factor main effects: mean score when factor=1 minus when factor=0
        def factor_effect(score_col: str, factor: str) -> float:
            g1 = grp.loc[grp[factor] == 1, score_col].mean()
            g0 = grp.loc[grp[factor] == 0, score_col].mean()
            if pd.isna(g1) or pd.isna(g0):
                return 0.0
            return float(g1 - g0)

        instance_records.append({
            "id":          item_id,
            "dataset":     dataset,
            "model_name":  model_name,
            "task_type":   grp["task_type"].iloc[0],
            "gold_answer": grp["gold_answer"].iloc[0],
            "n_templates": n,
            # Core metrics
            "sens_heuristic": sens_heuristic,
            "sens_judge":     sens_judge,
            "eas":            eas,
            "signed_eas":     sens_heuristic - sens_judge,
            "mean_heuristic": mean_heuristic,
            "mean_judge":     mean_judge,
            "stab_sem":       stab_sem,
            # Structural factor effects on heuristic
            "effect_role_heuristic":   factor_effect("heuristic_score", "role"),
            "effect_fmt_heuristic":    factor_effect("heuristic_score", "fmt"),
            "effect_prefix_heuristic": factor_effect("heuristic_score", "prefix"),
            # Structural factor effects on judge
            "effect_role_judge":       factor_effect("judge_correct", "role"),
            "effect_fmt_judge":        factor_effect("judge_correct", "fmt"),
            "effect_prefix_judge":     factor_effect("judge_correct", "prefix"),
        })

    inst_df = pd.DataFrame(instance_records)

    # Four-zone thresholds: 75th percentile of SensH and SensJ independently.
    # Using the 75th percentile means ~25% of items are "high sensitivity" by
    # each method, creating a balanced four-quadrant space.  The old approach
    # (median EAS as threshold) collapsed to zero because 60%+ of items have
    # EAS = 0, making "eas < 0" mathematically unreachable.
    sh_thresh = float(inst_df["sens_heuristic"].quantile(0.75))
    sj_thresh = float(inst_df["sens_judge"].quantile(0.75))
    print(f"\nFour-zone thresholds — SensH 75th pct: {sh_thresh:.4f}  "
          f"SensJ 75th pct: {sj_thresh:.4f}")

    inst_df["trizone"] = inst_df.apply(
        lambda r: classify_fourzone(
            r["sens_heuristic"], r["sens_judge"], sh_thresh, sj_thresh
        ),
        axis=1,
    )

    # Save instance-level CSV
    inst_df.to_csv(args.out_instance, index=False)
    print(f"Saved instance metrics → {args.out_instance}")

    # Quick summary
    print("\nFOUR-ZONE COUNTS BY (DATASET, MODEL)")
    summary = (inst_df.groupby(["dataset", "model_name", "trizone"])
               .size().unstack(fill_value=0))
    print(summary.to_string())

    print("\nMEAN EAS BY DATASET")
    print(inst_df.groupby("dataset")[["sens_heuristic", "sens_judge", "eas"]].mean().round(4).to_string())

    # Dataset-level aggregation
    agg = (inst_df
           .groupby(["dataset", "model_name", "task_type"])
           .agg(
               sens_heuristic_mean=("sens_heuristic", "mean"),
               sens_heuristic_std=("sens_heuristic",  "std"),
               sens_judge_mean=("sens_judge", "mean"),
               sens_judge_std=("sens_judge",  "std"),
               eas_mean=("eas", "mean"),
               eas_std=("eas",  "std"),
               mean_heuristic=("mean_heuristic", "mean"),
               mean_judge=("mean_judge",     "mean"),
               stab_sem_mean=("stab_sem", "mean"),
               n_artifact=("trizone",      lambda x: (x == "artifact").sum()),
               n_genuine=("trizone",        lambda x: (x == "genuine").sum()),
               n_underdetected=("trizone",  lambda x: (x == "underdetected").sum()),
               n_stable=("trizone",         lambda x: (x == "stable").sum()),
               n_items=("id", "count"),
           )
           .reset_index())

    # Bootstrap 95% CI for EAS, SensHeuristic, SensJudge
    ci_rows: List[Dict[str, Any]] = []
    for (ds, mn, tt), grp in inst_df.groupby(["dataset", "model_name", "task_type"]):
        row: Dict[str, Any] = {"dataset": ds, "model_name": mn, "task_type": tt}
        for col, key in [("eas", "eas"), ("sens_heuristic", "sh"), ("sens_judge", "sj")]:
            lo, hi = bootstrap_ci(grp[col].values)
            row[f"{key}_ci_lo"] = lo
            row[f"{key}_ci_hi"] = hi
        ci_rows.append(row)
    agg = agg.merge(pd.DataFrame(ci_rows), on=["dataset", "model_name", "task_type"])

    agg.to_csv(args.out_dataset, index=False)
    print(f"\nSaved dataset metrics → {args.out_dataset}")

    # Structural ablation summary
    print("\nSTRUCTURAL ABLATION: FACTOR MAIN EFFECTS")
    print(f"{'Model':<35} {'Factor':<8} {'ΔHeuristic':>12} {'ΔJudge':>10} {'Δ(H-J)':>10}")
    for model in sorted(inst_df["model_name"].unique()):
        mdf = inst_df[inst_df["model_name"] == model]
        for factor in ("role", "fmt", "prefix"):
            eh = mdf[f"effect_{factor}_heuristic"].mean()
            ej = mdf[f"effect_{factor}_judge"].mean()
            print(f"  {model:<33} {factor:<8} {eh:>+12.4f} {ej:>+10.4f} {eh-ej:>+10.4f}")


if __name__ == "__main__":
    main()
