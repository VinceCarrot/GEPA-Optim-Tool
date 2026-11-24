"""Evaluation metrics for the GEPA optimization loop.

Turns raw model predictions (issue_classification + risk_score per row) into
the signals the optimizer needs: label-based precision/recall, a
confidence-calibrated operating point that satisfies a recall floor, and
sampled false positive / false negative / uncertain examples used to build
the reflective dataset for prompt mutation.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve

import config

# issue_classification is enum-constrained in CaseNoteOutput (inference_run.py), so this is
# the single canonical positive value regardless of how a mutated prompt phrases the task.
AMEX_ERROR_LABELS = ["Amex Error"]


def map_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Map the free-text issue_classification field to a binary llm_pred_flag."""
    pred_df = pred_df.copy()
    pred_df["llm_pred_flag"] = np.where(
        pred_df["issue_classification"].isin(AMEX_ERROR_LABELS), 1, 0
    )
    return pred_df


def merge_predictions(pred_df: pd.DataFrame, raw_df: pd.DataFrame,
                       id_col: str = config.ID_COL, text_col: str = config.TEXT_COL) -> pd.DataFrame:
    """Drop errored rows, map predictions to binary flags, and merge in gold target + text."""
    n_errors = int((pred_df["issue_classification"] == "ERROR").sum())
    if n_errors:
        print(f"Dropping {n_errors} errored prediction rows.")
    pred_df = pred_df[pred_df["issue_classification"] != "ERROR"]
    pred_df = map_predictions(pred_df)
    merged = pd.merge(
        pred_df, raw_df[[id_col, text_col, config.TARGET_COL]], how="left", on=id_col
    )
    return merged


def compute_metrics(merged_df: pd.DataFrame) -> Dict[str, float]:
    """Label-based precision/recall/F1 using the discrete issue_classification mapping."""
    y_true = merged_df[config.TARGET_COL]
    y_pred = merged_df["llm_pred_flag"]
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n": int(len(merged_df)),
        "n_positive": int(y_true.sum()),
        "n_predicted_positive": int(y_pred.sum()),
    }


def calibrate_threshold(merged_df: pd.DataFrame, recall_target: float = config.RECALL_TARGET) -> Dict[str, float]:
    """Find the risk_score threshold giving the best precision while meeting the recall floor.

    Falls back to the highest achievable recall (and its precision) if no threshold meets
    the floor, so callers can distinguish "met the floor" from "closest available".
    """
    y_true = merged_df[config.TARGET_COL].to_numpy()
    scores = merged_df["risk_score"].to_numpy()
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)

    # precision_recall_curve returns len(thresholds) == len(precisions) - 1
    precisions, recalls = precisions[:-1], recalls[:-1]

    meets_floor = recalls >= recall_target
    if meets_floor.any():
        idx = np.argmax(np.where(meets_floor, precisions, -1))
        return {
            "threshold": float(thresholds[idx]),
            "precision_at_recall": float(precisions[idx]),
            "achieved_recall": float(recalls[idx]),
            "met_recall_floor": True,
        }

    # No threshold meets the floor: report the closest operating point (highest achievable recall).
    idx = int(np.argmax(recalls))
    return {
        "threshold": float(thresholds[idx]) if len(thresholds) else 0.5,
        "precision_at_recall": float(precisions[idx]),
        "achieved_recall": float(recalls[idx]),
        "met_recall_floor": False,
    }


def get_error_examples(
    merged_df: pd.DataFrame,
    text_col: str = config.TEXT_COL,
    n_fp: int = config.N_FALSE_POSITIVES_FOR_REFLECTION,
    n_fn: int = config.N_FALSE_NEGATIVES_FOR_REFLECTION,
    n_uncertain: int = config.N_UNCERTAIN_FOR_REFLECTION,
    uncertain_band: float = 0.15,
    seed: int = config.RANDOM_SEED,
) -> Dict[str, List[dict]]:
    """Sample false positives, false negatives, and low-confidence-margin predictions for reflection."""
    y_true = merged_df[config.TARGET_COL]
    y_pred = merged_df["llm_pred_flag"]

    fp = merged_df[(y_true == 0) & (y_pred == 1)]
    fn = merged_df[(y_true == 1) & (y_pred == 0)]
    uncertain = merged_df[(merged_df["risk_score"] - 0.5).abs() <= uncertain_band]

    def _sample(df: pd.DataFrame, n: int) -> List[dict]:
        sample = df.sample(n=min(n, len(df)), random_state=seed) if len(df) else df
        cols = [config.ID_COL, text_col, "issue_classification", "risk_score",
                "issue_classification_reason", config.TARGET_COL]
        cols = [c for c in cols if c in sample.columns]
        return sample[cols].to_dict(orient="records")

    return {
        "false_positives": _sample(fp, n_fp),
        "false_negatives": _sample(fn, n_fn),
        "uncertain": _sample(uncertain, n_uncertain),
    }


def get_analysis_errors(
    merged_df: pd.DataFrame,
    text_col: str = config.TEXT_COL,
    n_fp: int = config.N_FALSE_POSITIVES_FOR_ANALYSIS,
    n_fn: int = config.N_FALSE_NEGATIVES_FOR_ANALYSIS,
    seed: int = config.RANDOM_SEED,
) -> Dict[str, List[dict]]:
    """Sample false positives and false negatives with full (untruncated) text for per-error analysis."""
    y_true = merged_df[config.TARGET_COL]
    y_pred = merged_df["llm_pred_flag"]

    fp = merged_df[(y_true == 0) & (y_pred == 1)]
    fn = merged_df[(y_true == 1) & (y_pred == 0)]

    def _sample(df: pd.DataFrame, n: int) -> List[dict]:
        sample = df.sample(n=min(n, len(df)), random_state=seed) if len(df) else df
        cols = [config.ID_COL, text_col, "issue_classification", "risk_score",
                "issue_classification_reason", config.TARGET_COL]
        cols = [c for c in cols if c in sample.columns]
        return sample[cols].to_dict(orient="records")

    return {
        "false_positives": _sample(fp, n_fp),
        "false_negatives": _sample(fn, n_fn),
    }


def evaluate(
    pred_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    recall_target: Optional[float] = None,
    text_col: str = config.TEXT_COL,
    save_path: Optional[str] = None,
) -> Dict:
    """Main entry point: merge predictions with gold labels and compute the full metrics bundle."""
    recall_target = config.RECALL_TARGET if recall_target is None else recall_target
    merged = merge_predictions(pred_df, raw_df, text_col=text_col)

    if save_path:
        # Persisted pre-metrics so a run can be debugged from raw predictions alone.
        merged.to_parquet(save_path, index=False)

    if merged.empty:
        # Every row in this batch errored out (e.g. sustained API failures); score as
        # a total failure rather than crashing the optimization loop.
        print("Warning: all predictions errored/unmatched for this evaluation; scoring as 0.")
        return {
            "label_metrics": {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                               "n": 0, "n_positive": 0, "n_predicted_positive": 0},
            "threshold_metrics": {"threshold": 0.5, "precision_at_recall": 0.0,
                                   "achieved_recall": 0.0, "met_recall_floor": False},
            "error_examples": {"false_positives": [], "false_negatives": [], "uncertain": []},
            "optimization_score": 0.0,
        }

    label_metrics = compute_metrics(merged)
    threshold_metrics = calibrate_threshold(merged, recall_target=recall_target)
    error_examples = get_error_examples(merged, text_col=text_col)

    # Below-floor candidates get gradient signal but can never outscore above-floor ones
    BELOW_FLOOR_WEIGHT = 0.1

    if config.OPTIMIZATION_METRIC == "label_precision":
        if label_metrics["recall"] >= recall_target:
            opt_score = label_metrics["precision"]
        else:
            opt_score = BELOW_FLOOR_WEIGHT * label_metrics["precision"] * (label_metrics["recall"] / recall_target) ** 2
    else:
        if threshold_metrics["met_recall_floor"]:
            opt_score = threshold_metrics["precision_at_recall"]
        else:
            achieved = threshold_metrics["achieved_recall"]
            opt_score = BELOW_FLOOR_WEIGHT * threshold_metrics["precision_at_recall"] * (achieved / recall_target) ** 2

    return {
        "label_metrics": label_metrics,
        "threshold_metrics": threshold_metrics,
        "error_examples": error_examples,
        "optimization_score": opt_score,
    }


if __name__ == "__main__":
    import dataset as ds

    raw_df = ds.load_raw_dataset()
    pred_df = pd.read_parquet(config.GEPA_DIR + "/sample_op.parquet")
    result = evaluate(pred_df, raw_df)
    print(f"Label-based Precision: {result['label_metrics']['precision']*100:.2f}")
    print(f"Label-based Recall: {result['label_metrics']['recall']*100:.2f}")
    print(f"Precision at recall>={config.RECALL_TARGET}: {result['threshold_metrics']['precision_at_recall']*100:.2f} "
          f"(met floor: {result['threshold_metrics']['met_recall_floor']})")