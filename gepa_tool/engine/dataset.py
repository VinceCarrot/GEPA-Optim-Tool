"""Dataset loading, input rendering and train/test splitting for one run.

The raw data is reduced to three canonical columns used by the rest of the engine:
  ID_COL     -> original id column (name preserved)
  INPUT_COL  -> the text the task model sees, rendered from data.input_template
  TARGET_COL -> 1 if label_col == positive_value else 0

Auto splits reproduce GEPA/dataset.py exactly (same stratified permutation), so a run
configured like the legacy one scores on identical rows.
"""

from __future__ import annotations

import os
import re
import string
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .spec import DataSpec

INPUT_COL = "__input__"
TARGET_COL = "target"


def read_table(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext in (".csv", ".txt"):
        return pd.read_csv(path)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported data file type: {ext} (use parquet, csv or xlsx)")


def _profanity_pattern(wordlist: Optional[str]):
    """Same matches as the legacy r"\\b(w1|w2|...)\\b" (IGNORECASE), but alternatives are grouped by
    their first character so the engine only tries words that can match at a position. Relative
    order inside a group is preserved, so leftmost-first alternation picks the same word."""
    if not wordlist or not os.path.exists(wordlist):
        return None
    words = [w.strip() for w in Path(wordlist).read_text().splitlines() if w.strip()]
    if not words:
        return None
    groups: dict = {}
    for w in words:
        groups.setdefault(w[0].lower(), []).append(w)
    parts = []
    for first, ws in groups.items():
        rests = "|".join(re.escape(w[1:]) for w in ws)
        parts.append(re.escape(first) + "(?:" + rests + ")")
    return re.compile(r"\b(" + "|".join(parts) + r")\b", re.IGNORECASE)


def template_fields(template: str):
    return [f for _, f, _, _ in string.Formatter().parse(template) if f]


def render_inputs(df: pd.DataFrame, spec: DataSpec) -> pd.Series:
    pattern = _profanity_pattern(spec.profanity_wordlist)

    def _clean(v):
        s = "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
        return pattern.sub("**", s) if pattern is not None else s

    cols = {c: df[c].map(_clean) for c in spec.input_columns}
    if spec.input_template.strip() == "{" + spec.input_columns[0] + "}" and len(spec.input_columns) == 1:
        return cols[spec.input_columns[0]]
    rows = [dict(zip(cols.keys(), vals)) for vals in zip(*cols.values())]
    return pd.Series([spec.input_template.format(**r) for r in rows], index=df.index)


def prepare(df: pd.DataFrame, spec: DataSpec) -> pd.DataFrame:
    """Map a raw table onto (id, __input__, target). Rows with a missing id or empty input are dropped."""
    missing = [c for c in [spec.id_col, spec.label_col, *spec.input_columns] if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")
    df = df.dropna(subset=[spec.id_col, *spec.input_columns], how="any").copy()
    df[INPUT_COL] = render_inputs(df, spec)
    df[TARGET_COL] = np.where(df[spec.label_col].astype(str) == str(spec.positive_value), 1, 0)
    return df[[spec.id_col, INPUT_COL, TARGET_COL]]


def stratified_split(df: pd.DataFrame, train_frac: float, dev_frac: float, seed: int):
    """Identical algorithm to GEPA/dataset._stratified_split (dev part is returned but unused)."""
    rng = np.random.RandomState(seed)
    train_parts, dev_parts, test_parts = [], [], []
    for _, group in df.groupby(TARGET_COL):
        idx = rng.permutation(group.index.values)
        n = len(idx)
        n_train = int(round(n * train_frac))
        n_dev = int(round(n * dev_frac))
        train_parts.append(group.loc[idx[:n_train]])
        dev_parts.append(group.loc[idx[n_train:n_train + n_dev]])
        test_parts.append(group.loc[idx[n_train + n_dev:]])

    def _shuffle(parts):
        out = pd.concat(parts, ignore_index=True)
        return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return _shuffle(train_parts), _shuffle(dev_parts), _shuffle(test_parts)


def load_splits(spec: DataSpec, out_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) in canonical form; optionally persist them under out_dir."""
    if spec.split.mode == "provided":
        train_df = prepare(read_table(spec.split.train_path), spec).reset_index(drop=True)
        test_df = prepare(read_table(spec.split.test_path), spec).reset_index(drop=True)
    else:
        full = prepare(read_table(spec.path), spec)
        train_df, _dev, test_df = stratified_split(full, spec.split.train_frac, spec.split.dev_frac, spec.split.seed)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        train_df.to_parquet(os.path.join(out_dir, "train.parquet"), index=False)
        test_df.to_parquet(os.path.join(out_dir, "test.parquet"), index=False)
    return train_df, test_df


def sample_rows(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Same semantics as GEPA/dataset.sample_minibatch."""
    if n >= len(df):
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def balanced_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Up to n rows, half positive / half negative where possible (used by the seed trial run)."""
    pos = df[df[TARGET_COL] == 1]
    neg = df[df[TARGET_COL] == 0]
    n_pos = min(len(pos), n // 2)
    n_neg = min(len(neg), n - n_pos)
    n_pos = min(len(pos), n - n_neg)
    out = pd.concat([pos.sample(n=n_pos, random_state=seed), neg.sample(n=n_neg, random_state=seed)])
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def profile(df_raw: pd.DataFrame, spec: DataSpec) -> dict:
    """Data-quality summary shown in the setup wizard."""
    issues = []
    n_raw = len(df_raw)
    dup_ids = int(df_raw[spec.id_col].duplicated().sum()) if spec.id_col in df_raw else 0
    if dup_ids:
        issues.append(f"{dup_ids} duplicate ids")
    prepared = prepare(df_raw, spec)
    n = len(prepared)
    if n < n_raw:
        issues.append(f"{n_raw - n} rows dropped for missing id/input")
    n_pos = int(prepared[TARGET_COL].sum())
    if n_pos == 0:
        issues.append(f"no rows where {spec.label_col} == {spec.positive_value!r}")
    if n_pos == n:
        issues.append("every row is positive")
    lengths = prepared[INPUT_COL].str.len()
    long_rows = int((lengths > 20000).sum())
    if long_rows:
        issues.append(f"{long_rows} inputs longer than 20k characters")
    if n < 200:
        issues.append("fewer than 200 usable rows; scores will be noisy")
    return {
        "n_raw": n_raw,
        "n_usable": n,
        "n_positive": n_pos,
        "positive_rate": (n_pos / n) if n else 0.0,
        "duplicate_ids": dup_ids,
        "input_chars": {
            "min": int(lengths.min()) if n else 0,
            "median": int(lengths.median()) if n else 0,
            "p95": int(lengths.quantile(0.95)) if n else 0,
            "max": int(lengths.max()) if n else 0,
        },
        "label_values": df_raw[spec.label_col].astype(str).value_counts().head(20).to_dict(),
        "issues": issues,
        "example_input": prepared[INPUT_COL].iloc[0] if n else "",
    }


# ---------------------------------------------------------------------------
# Pre-flight data check (setup wizard "Check data"; also run by config validation)
# ---------------------------------------------------------------------------

def _ids(df: pd.DataFrame, id_col: str) -> pd.Series:
    return df[id_col].astype(str).str.strip()


def _file_summary(prepared: pd.DataFrame, n_raw: int, id_col: str) -> dict:
    n = len(prepared)
    n_pos = int(prepared[TARGET_COL].sum()) if n else 0
    return {"n_raw": n_raw, "n_usable": n, "n_positive": n_pos, "positive_rate": n_pos / n if n else 0.0,
            "duplicate_ids": int(_ids(prepared, id_col).duplicated().sum()) if n else 0}


def _examples(values, k: int = 5) -> str:
    vals = sorted(map(str, values))
    more = f" … (+{len(vals) - k} more)" if len(vals) > k else ""
    return ", ".join(vals[:k]) + more


def check_data(spec: DataSpec, scoring_set_size: Optional[int] = None) -> dict:
    """Blocking errors and warnings for the configured data, before a trial or run.

    auto split:     the dataset must have unique ids and usable rows of both classes.
    provided split: train and test must each be readable, have every configured column, have unique ids,
                    contain both classes, and share no ids (leakage). If a full dataset is also given, every
                    train/test id must exist in it with the same label and the same rendered input; full rows
                    that are in neither split are reported as a warning.
    """
    errors: List[str] = []
    warnings: List[str] = []
    files: Dict[str, dict] = {}
    raws: Dict[str, pd.DataFrame] = {}
    cols = [spec.id_col, spec.label_col, *spec.input_columns]

    def load(name: str, path: Optional[str]) -> Optional[pd.DataFrame]:
        if not path:
            return None
        try:
            raw = read_table(path)
        except Exception as e:  # unreadable / missing file
            errors.append(f"{name}: cannot read {os.path.basename(path)} ({e})")
            return None
        missing = [c for c in cols if c not in raw.columns]
        if missing:
            errors.append(f"{name}: missing column(s) {missing}")
            return None
        raws[name] = raw
        prepared = prepare(raw, spec)
        files[name] = _file_summary(prepared, len(raw), spec.id_col)
        if len(prepared) < len(raw):
            warnings.append(f"{name}: {len(raw) - len(prepared)} rows dropped (missing id or input)")
        return prepared

    def need_both_classes(name: str) -> None:
        s = files[name]
        if s["n_usable"] == 0:
            errors.append(f"{name}: no usable rows")
        elif s["n_positive"] == 0:
            errors.append(f"{name}: no positive rows ({spec.label_col} == {spec.positive_value!r})")
        elif s["n_positive"] == s["n_usable"]:
            errors.append(f"{name}: every row is positive (no negatives)")

    def need_unique_ids(name: str) -> None:
        if files[name]["duplicate_ids"]:
            errors.append(f"{name}: {files[name]['duplicate_ids']} duplicate ids (ids must be unique; "
                          "predictions are joined to gold labels by id)")

    if spec.split.mode == "auto":
        full = load("full", spec.path)
        if full is not None:
            need_both_classes("full")
            need_unique_ids("full")
            if not errors:
                tr, _dev, te = stratified_split(full, spec.split.train_frac, spec.split.dev_frac, spec.split.seed)
                files["train"] = _file_summary(tr, len(tr), spec.id_col)
                files["test"] = _file_summary(te, len(te), spec.id_col)
    else:
        full = load("full", spec.path)
        train = load("train", spec.split.train_path)
        test = load("test", spec.split.test_path)
        if not spec.split.train_path:
            errors.append("train file is not set")
        if not spec.split.test_path:
            errors.append("test file is not set")
        for name, p in (("train", train), ("test", test)):
            if p is not None:
                need_both_classes(name)
                need_unique_ids(name)
        if train is not None and test is not None:
            overlap = set(_ids(train, spec.id_col)) & set(_ids(test, spec.id_col))
            if overlap:
                errors.append(f"train and test share {len(overlap)} ids (test leakage): {_examples(overlap)}")
            tr_rate, te_rate = files["train"]["positive_rate"], files["test"]["positive_rate"]
            if tr_rate and te_rate and abs(tr_rate - te_rate) > max(0.05, 0.5 * min(tr_rate, te_rate)):
                warnings.append(f"positive rate differs between train ({tr_rate:.1%}) and test ({te_rate:.1%}); "
                                "test metrics may not reflect what the optimizer saw")
            tr_labels = set(raws["train"][spec.label_col].astype(str))
            te_labels = set(raws["test"][spec.label_col].astype(str))
            if te_labels - tr_labels:
                warnings.append(f"label values only in test: {_examples(te_labels - tr_labels)}")
        if full is not None:
            if files["full"]["duplicate_ids"]:
                warnings.append(f"full: {files['full']['duplicate_ids']} duplicate ids; the consistency check "
                                "uses the first row of each id")
            f = full.assign(_id=_ids(full, spec.id_col)).drop_duplicates("_id").set_index("_id")
            used: set = set()
            for name, p in (("train", train), ("test", test)):
                if p is None:
                    continue
                q = p.assign(_id=_ids(p, spec.id_col)).drop_duplicates("_id").set_index("_id")
                used |= set(q.index)
                absent = q.index.difference(f.index)
                if len(absent):
                    errors.append(f"{name}: {len(absent)} ids are not in the full dataset: {_examples(absent)}")
                common = q.index.intersection(f.index)
                lab = common[(q.loc[common, TARGET_COL] != f.loc[common, TARGET_COL]).to_numpy()]
                if len(lab):
                    errors.append(f"{name}: {len(lab)} ids have a different label than in the full dataset: "
                                  f"{_examples(lab)}")
                txt = common[(q.loc[common, INPUT_COL] != f.loc[common, INPUT_COL]).to_numpy()]
                if len(txt):
                    errors.append(f"{name}: {len(txt)} ids have different input text than in the full dataset: "
                                  f"{_examples(txt)}")
            unused = set(f.index) - used
            if unused and train is not None and test is not None:
                warnings.append(f"{len(unused)} rows of the full dataset ({len(unused) / max(1, len(f)):.1%}) "
                                "are in neither train nor test; they are not used")

    tr_sum = files.get("train")
    if scoring_set_size and tr_sum and tr_sum["n_usable"] < scoring_set_size:
        warnings.append(f"train has {tr_sum['n_usable']} usable rows, fewer than the scoring set size "
                        f"({scoring_set_size}); every train row will be scored")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "files": files}
