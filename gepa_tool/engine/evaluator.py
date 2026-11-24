"""Turn raw predictions into the metrics bundle the optimizer and UI use."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import objective as obj
from .dataset import INPUT_COL, TARGET_COL
from .inference import ERROR_LABEL
from .spec import RunConfig

PRED_COL = "pred_flag"


def merge_predictions(cfg: RunConfig, preds: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Drop errored rows, map label -> 0/1, and join gold input + target."""
    id_col = cfg.data.id_col
    preds = preds[preds["classification"] != ERROR_LABEL].copy()
    preds[PRED_COL] = np.where(preds["classification"] == cfg.task.positive_label, 1, 0)
    return pd.merge(preds, gold[[id_col, INPUT_COL, TARGET_COL]], how="left", on=id_col)


def evaluate(cfg: RunConfig, preds: pd.DataFrame, gold: pd.DataFrame,
             save_path: Optional[str] = None) -> Dict:
    """Metrics bundle.

    label_metrics:     metrics on the model's discrete labels
    threshold:         calibrated operating point on risk_score (always computed, informational in label mode)
    metrics:           the operating point used for accept/reject (label or threshold per scoring_mode)
    feasible/violation/optimization_score: from the ObjectiveSpec
    error_rate / inference_failed: share of rows that still errored after the retry passes, and
                       whether it exceeds budget.max_error_rate (the candidate is then not scored)
    """
    spec = cfg.objective
    n_errors = int((preds["classification"] == ERROR_LABEL).sum()) if len(preds) else 0
    err_rate = n_errors / len(gold) if len(gold) else 0.0
    failed = err_rate > cfg.budget.max_error_rate
    merged = merge_predictions(cfg, preds, gold)
    if save_path:
        merged.to_parquet(save_path, index=False)

    if merged.empty:
        empty = obj.metrics_from_counts(0, 0, 0, 0, spec.beta)
        return {
            "label_metrics": empty, "metrics": empty,
            "threshold": {"threshold": 0.5, "metrics": empty, "feasible": False},
            "feasible": False, "violation": obj.violation(spec, empty),
            "optimization_score": 0.0, "n_errors": n_errors, "merged": merged,
            "error_rate": err_rate, "inference_failed": True,
        }

    y_true = merged[TARGET_COL].to_numpy()
    label_m = obj.metrics_from_arrays(y_true, merged[PRED_COL].to_numpy(), spec.beta)
    thr = obj.calibrate_threshold(spec, y_true, merged["risk_score"].to_numpy())
    op = label_m if spec.scoring_mode == "label" else thr["metrics"]
    return {
        "label_metrics": label_m,
        "threshold": thr,
        "metrics": op,
        "feasible": obj.is_feasible(spec, op),
        "violation": obj.violation(spec, op),
        "optimization_score": obj.optimization_score(spec, op),
        "n_errors": n_errors,
        "error_rate": err_rate,
        "inference_failed": failed,
        "merged": merged,
    }


def summarize(bundle: Dict) -> Dict:
    """JSON-safe subset of a bundle for events / results."""
    m = bundle["metrics"]
    keys = ["precision", "recall", "f1", "fbeta", "accuracy", "specificity", "npv", "fpr", "fnr",
            "predicted_positive_rate", "positive_rate", "n", "n_positive", "n_predicted_positive",
            "tp", "fp", "fn", "tn"]
    out = {k: m[k] for k in keys if k in m}
    out.update({
        "optimization_score": bundle["optimization_score"],
        "feasible": bundle["feasible"],
        "violation": bundle["violation"],
        "n_errors": bundle["n_errors"],
        "error_rate": bundle.get("error_rate", 0.0),
        "inference_failed": bundle.get("inference_failed", False),
        "threshold": bundle["threshold"]["threshold"],
        "label_precision": bundle["label_metrics"]["precision"],
        "label_recall": bundle["label_metrics"]["recall"],
    })
    return out


def per_instance_correct(cfg: RunConfig, merged: pd.DataFrame) -> Dict[str, int]:
    correct = (merged[PRED_COL] == merged[TARGET_COL]).astype(int)
    return dict(zip(merged[cfg.data.id_col].astype(str), correct))


def _records(cfg: RunConfig, df: pd.DataFrame) -> List[dict]:
    cols = [cfg.data.id_col, INPUT_COL, "classification", "risk_score", "classification_reason", TARGET_COL]
    cols = [c for c in cols if c in df.columns]
    return df[cols].to_dict(orient="records")


def sample_errors(cfg: RunConfig, merged: pd.DataFrame, n_fp: int, n_fn: int, seed: int) -> Dict[str, List[dict]]:
    """Sample FPs/FNs with full text for per-error analysis (same sampling as legacy)."""
    if merged.empty:
        return {"false_positives": [], "false_negatives": []}
    y_true, y_pred = merged[TARGET_COL], merged[PRED_COL]
    fp = merged[(y_true == 0) & (y_pred == 1)]
    fn = merged[(y_true == 1) & (y_pred == 0)]

    def _s(df, n):
        if not len(df) or n <= 0:
            return []
        return _records(cfg, df.sample(n=min(n, len(df)), random_state=seed))

    return {"false_positives": _s(fp, n_fp), "false_negatives": _s(fn, n_fn)}