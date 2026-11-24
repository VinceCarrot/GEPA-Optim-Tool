"""Seed trial run: score the seed prompt on a small balanced sample before a full run."""

from __future__ import annotations

import asyncio
import json
import os
import time

from . import evaluator as ev
from . import inference as infer
from . import llm as llm_mod
from . import objective as obj
from . import protect
from .dataset import INPUT_COL, TARGET_COL, balanced_sample, load_splits
from .spec import RunConfig
from .store import EventLog


async def run_trial(cfg: RunConfig, out_dir: str, n: int = 50) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    events = EventLog(out_dir)
    llm_mod.init_models(cfg.models)
    blocks = protect.parse(cfg.seed_prompt)

    train_df, _ = load_splits(cfg.data)
    sample = balanced_sample(train_df, n, seed=cfg.random_seed)
    events.emit("trial_start", n=len(sample))
    t0 = time.time()

    def prog(_pi, done, total, n_err):
        events.emit("eval_progress", candidate_id="seed", done=done, total=total, n_errors=n_err)

    preds = await infer.run_one_async(cfg, cfg.seed_prompt, sample, prog)
    elapsed = time.time() - t0
    bundle = ev.evaluate(cfg, preds, sample)
    summary = ev.summarize(bundle)

    n_err = bundle["n_errors"]
    parse_fail_rate = n_err / max(1, len(sample))
    rows = []
    merged = preds.merge(sample[[cfg.data.id_col, INPUT_COL, TARGET_COL]], on=cfg.data.id_col, how="left")
    for rec in merged.to_dict(orient="records"):
        gold = int(rec.get(TARGET_COL, 0) or 0)
        pred_label = rec.get("classification")
        pred = 1 if pred_label == cfg.task.positive_label else 0
        rows.append({
            "id": str(rec.get(cfg.data.id_col)),
            "input": str(rec.get(INPUT_COL, ""))[:4000],
            "gold": cfg.task.positive_label if gold else cfg.task.negative_label,
            "predicted": pred_label,
            "risk_score": rec.get("risk_score"),
            "reason": rec.get("classification_reason"),
            "outcome": "error" if pred_label == "ERROR" else
                       ("TP" if gold and pred else "FN" if gold else "FP" if pred else "TN"),
        })

    # Rough budget estimate: wall-clock per row at this concurrency, scaled to the full run.
    b = cfg.budget
    n_scoring = min(b.scoring_set_size, len(train_df))
    task_calls = n_scoring * (1 + b.max_rounds * b.proposals_per_round)
    reflection_calls = b.max_rounds * (cfg.reflection.n_false_positives + cfg.reflection.n_false_negatives + 2)
    per_row = elapsed / max(1, len(sample))
    eff_conc = min(b.max_workers, len(sample))
    est_seconds = task_calls * per_row * eff_conc / max(1, b.max_workers) if eff_conc else 0

    warnings = []
    if parse_fail_rate > 0.05:
        warnings.append(f"{parse_fail_rate:.0%} of rows failed or did not parse; check the prompt and the output labels")
    if not bundle["feasible"] and cfg.objective.constraints:
        gaps = [f"{obj.describe_constraint(c)} (now {bundle['metrics'][c.metric]:.2f})"
                for c in cfg.objective.constraints if obj.constraint_gap(c, bundle["metrics"]) > 0]
        warnings.append("Seed misses: " + "; ".join(gaps) + ". The run will start in the constraint phase.")
        if any(obj.constraint_gap(c, bundle["metrics"]) > 0.2 for c in cfg.objective.constraints):
            warnings.append("Seed is far from a constraint (>0.20); expect a long constraint phase. "
                            "Consider more rounds or threshold scoring.")
    warnings.append("Trial sample is balanced and small; its numbers are not the run baseline.")

    result = {
        "n": len(sample), "n_errors": n_err, "parse_fail_rate": parse_fail_rate,
        "blocked": parse_fail_rate > 0.05,
        "metrics": summary, "feasible": bundle["feasible"],
        "objective_text": obj.describe_objective(cfg.objective),
        "protected_sections": [{"name": b.name, "chars": len(b.body)} for b in blocks],
        "elapsed_seconds": elapsed,
        "estimate": {"task_calls": task_calls, "reflection_calls": reflection_calls,
                     "approx_hours": est_seconds / 3600.0},
        "warnings": warnings, "rows": rows,
    }
    with open(os.path.join(out_dir, "trial_result.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    events.emit("trial_end", status="completed")
    return result


def run_trial_sync(cfg: RunConfig, out_dir: str, n: int = 50) -> int:
    events = EventLog(out_dir) if os.path.isdir(out_dir) else None
    try:
        asyncio.run(run_trial(cfg, out_dir, n))
        return 0
    except Exception as e:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "trial_result.json"), "w") as f:
            json.dump({"error": str(e)}, f)
        (events or EventLog(out_dir)).emit("trial_end", status="failed", error=str(e)[:2000])
        return 1