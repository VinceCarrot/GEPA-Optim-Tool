"""Metrics, feasibility and accept/reject logic driven by an ObjectiveSpec.

Generalizes the legacy GEPA rules (maximize precision subject to recall >= target)
to "maximize one confusion-matrix metric subject to N hard >=/<= constraints".
With a single recall constraint and strategy="legacy" every number and decision
reproduces GEPA/optimizer.py + GEPA/evaluator.py exactly.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .spec import METRIC_LABELS, AcceptanceSpec, Constraint, ObjectiveSpec

BELOW_FLOOR_WEIGHT = 0.1  # legacy: infeasible candidates get signal but never outscore feasible ones


# ---------------------------------------------------------------------------
# Confusion-matrix metrics
# ---------------------------------------------------------------------------

def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int, beta: Optional[float] = None) -> Dict[str, float]:
    n = tp + fp + fn + tn
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    out = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": _safe_div(tp + tn, n),
        "specificity": _safe_div(tn, tn + fp),
        "npv": _safe_div(tn, tn + fn),
        "fpr": _safe_div(fp, fp + tn),
        "fnr": _safe_div(fn, fn + tp),
        "predicted_positive_rate": _safe_div(tp + fp, n),
        "positive_rate": _safe_div(tp + fn, n),
        "n": int(n),
        "n_positive": int(tp + fn),
        "n_predicted_positive": int(tp + fp),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
    if beta:
        b2 = beta * beta
        out["fbeta"] = _safe_div((1 + b2) * precision * recall, b2 * precision + recall)
    else:
        out["fbeta"] = f1
    return out


def metrics_from_arrays(y_true: np.ndarray, y_pred: np.ndarray, beta: Optional[float] = None) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return metrics_from_counts(tp, fp, fn, tn, beta)


# ---------------------------------------------------------------------------
# Constraints / feasibility
# ---------------------------------------------------------------------------

def constraint_gap(c: Constraint, m: Dict[str, float]) -> float:
    """How far a metric bundle is from satisfying one constraint (0 when satisfied)."""
    v = m[c.metric]
    return max(0.0, c.value - v) if c.op == ">=" else max(0.0, v - c.value)


def violation(spec: ObjectiveSpec, m: Dict[str, float]) -> float:
    return float(sum(constraint_gap(c, m) for c in spec.constraints))


def is_feasible(spec: ObjectiveSpec, m: Dict[str, float]) -> bool:
    for c in spec.constraints:
        v = m[c.metric]
        if c.op == ">=" and not v >= c.value:
            return False
        if c.op == "<=" and not v <= c.value:
            return False
    return True


def _penalty_ratio(c: Constraint, m: Dict[str, float]) -> float:
    v = m[c.metric]
    if c.op == ">=":
        return min(1.0, _safe_div(v, c.value)) if c.value else 1.0
    # <=: v above the ceiling shrinks the ratio
    return min(1.0, _safe_div(c.value, v)) if v else 1.0


def optimization_score(spec: ObjectiveSpec, m: Dict[str, float]) -> float:
    """Scalar used for ranking. Feasible: the objective. Infeasible: heavily discounted objective."""
    obj = m[spec.maximize]
    if is_feasible(spec, m):
        return float(obj)
    penalty = 1.0
    for c in spec.constraints:
        if constraint_gap(c, m) > 0:
            penalty *= _penalty_ratio(c, m) ** 2
    return float(BELOW_FLOOR_WEIGHT * obj * penalty)


# ---------------------------------------------------------------------------
# Threshold calibration (scoring_mode = "threshold")
# ---------------------------------------------------------------------------

def calibrate_threshold(spec: ObjectiveSpec, y_true: np.ndarray, scores: np.ndarray) -> Dict:
    """Pick the score threshold that maximizes the objective among feasible operating points.

    Falls back to the least-violating point when none is feasible. Thresholds are the distinct
    scores (predict positive when score >= threshold), matching sklearn's precision_recall_curve.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    if len(y_true) == 0:
        m = metrics_from_counts(0, 0, 0, 0, spec.beta)
        return {"threshold": 0.5, "metrics": m, "feasible": False}

    order = np.argsort(-scores, kind="mergesort")
    s_sorted = scores[order]
    y_sorted = y_true[order]
    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1 - y_sorted)
    total_pos = int(y_true.sum())
    total_neg = int(len(y_true) - total_pos)
    # last index of each distinct score value -> predicting positive for all scores >= that value
    distinct_last = np.where(np.diff(s_sorted) != 0)[0]
    idxs = np.r_[distinct_last, len(s_sorted) - 1]

    best: Optional[Tuple] = None
    fallback: Optional[Tuple] = None
    for i in idxs:
        tp, fp = int(tp_cum[i]), int(fp_cum[i])
        m = metrics_from_counts(tp, fp, total_pos - tp, total_neg - fp, spec.beta)
        thr = float(s_sorted[i])
        if is_feasible(spec, m):
            key = (m[spec.maximize], -thr)
            if best is None or key > best[0]:
                best = (key, thr, m)
        else:
            key = (-violation(spec, m), m[spec.maximize])
            if fallback is None or key > fallback[0]:
                fallback = (key, thr, m)
    if best is not None:
        return {"threshold": best[1], "metrics": best[2], "feasible": True}
    return {"threshold": fallback[1], "metrics": fallback[2], "feasible": False}


# ---------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------

REASONS = {
    "feasible_improved": "Meets all constraints and the objective did not drop by more than epsilon",
    "feasible_regressed": "Still meets constraints but the objective dropped by more than epsilon",
    "left_feasible_region": "Parent met all constraints; this child breaks at least one",
    "entered_feasible_region": "Parent missed a constraint; this child meets all of them",
    "violation_reduced": "Still misses a constraint, but is closer to meeting it",
    "violation_reduced_objective_drop": "Closer to the constraint, but the objective dropped by more than delta",
    "same_violation_objective_up": "Same constraint gap and objective not lower",
    "same_violation_objective_down": "Same constraint gap and lower objective",
    "violation_increased": "Further away from meeting the constraints",
    "inference_errors": "Too many rows failed with API errors (after retries); not scored",
}


def decide(spec: ObjectiveSpec, acc: AcceptanceSpec, parent_m: Dict, child_m: Dict) -> Tuple[bool, str]:
    """Return (accepted, reason_code) for a child versus its parent (operating-point metrics)."""
    if acc.strategy == "legacy":
        return _decide_legacy(spec, acc, parent_m, child_m)

    obj_p, obj_c = parent_m[spec.maximize], child_m[spec.maximize]
    p_feas, c_feas = is_feasible(spec, parent_m), is_feasible(spec, child_m)
    if p_feas:
        if not c_feas:
            return False, "left_feasible_region"
        if obj_c >= obj_p - acc.epsilon:
            return True, "feasible_improved"
        return False, "feasible_regressed"
    if c_feas:
        return True, "entered_feasible_region"
    v_p, v_c = violation(spec, parent_m), violation(spec, child_m)
    if v_c < v_p - 1e-9:
        if obj_c >= obj_p - acc.infeasible_delta:
            return True, "violation_reduced"
        return False, "violation_reduced_objective_drop"
    if abs(v_c - v_p) <= 1e-9:
        if obj_c >= obj_p:
            return True, "same_violation_objective_up"
        return False, "same_violation_objective_down"
    return False, "violation_increased"


def _decide_legacy(spec: ObjectiveSpec, acc: AcceptanceSpec, parent_m: Dict, child_m: Dict) -> Tuple[bool, str]:
    """Exact port of GEPA/optimizer.py feasibility-first rule (no delta).

    With a single `recall >= target` constraint, violation = target - recall, so
    "violation shrank" is exactly the legacy "child_r > parent_r" branch.
    """
    obj_p, obj_c = parent_m[spec.maximize], child_m[spec.maximize]
    p_feas, c_feas = is_feasible(spec, parent_m), is_feasible(spec, child_m)
    if p_feas:
        if c_feas and obj_c >= obj_p - acc.epsilon:
            return True, "feasible_improved"
        return False, ("feasible_regressed" if c_feas else "left_feasible_region")
    if c_feas:
        return True, "entered_feasible_region"
    v_p, v_c = violation(spec, parent_m), violation(spec, child_m)
    if v_c < v_p:
        return True, "violation_reduced"
    if abs(v_c - v_p) < 1e-9:
        if obj_c >= obj_p:
            return True, "same_violation_objective_up"
        return False, "same_violation_objective_down"
    return False, "violation_increased"


# ---------------------------------------------------------------------------
# Human-readable rendering (used by the UI and the NL->spec confirmation)
# ---------------------------------------------------------------------------

def describe_constraint(c: Constraint) -> str:
    sym = "≥" if c.op == ">=" else "≤"
    return f"{METRIC_LABELS[c.metric]} {sym} {c.value:.2f}"


def describe_objective(spec: ObjectiveSpec) -> str:
    target = METRIC_LABELS[spec.maximize]
    if spec.maximize == "fbeta":
        target = f"F-beta (beta={spec.beta})"
    cons = "; ".join(describe_constraint(c) for c in spec.constraints) or "no hard constraints"
    mode = "label decisions" if spec.scoring_mode == "label" else "best score threshold"
    return f"Maximize {target} subject to: {cons} (scored on {mode})"


def violated_sides(spec: ObjectiveSpec, m: Dict[str, float]) -> List[str]:
    """Which error type ('fn' or 'fp') must shrink to fix currently violated constraints."""
    fn_side = {"recall", "fnr", "npv"}
    fp_side = {"precision", "specificity", "fpr"}
    sides = []
    for c in spec.constraints:
        if constraint_gap(c, m) <= 0:
            continue
        metric, op = c.metric, c.op
        if metric in fn_side:
            sides.append("fn")
        elif metric in fp_side:
            sides.append("fp")
        elif metric == "predicted_positive_rate":
            sides.append("fp" if op == "<=" else "fn")
    return sides


def is_finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False