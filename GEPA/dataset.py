"""Dataset loading and train/dev/test splitting for the GEPA optimization loop.

Builds the binary target from the raw complaint data and caches stratified
train/dev/test splits under data/splits/ so every optimizer run, baseline
check and final locked evaluation operate on the exact same partitions.
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config


def _profanity_pattern():
    toxic_path = Path(config.GEPA_DIR) / "toxic_words.txt"
    if not toxic_path.exists():
        toxic_path = Path(config.PROJECT_ROOT) / "toxic_words.txt"
    if not toxic_path.exists():
        return re.compile(r"$.^")
    words = [
        re.escape(w.strip()) for w in toxic_path.read_text().splitlines() if w.strip()
    ]
    return re.compile(r"\b(" + "|".join(words) + r")\b", re.IGNORECASE)


_PROFANITY_PATTERN = _profanity_pattern()


def remove_profanity(text: str) -> str:
    if not text:
        return ""
    return _PROFANITY_PATTERN.sub("**", text)


def load_raw_dataset() -> pd.DataFrame:
    """Load the raw parquet data and derive the binary `target` column."""
    df = pd.read_parquet(config.RAW_DATA_PATH)
    df[config.TEXT_COL] = df[config.TEXT_COL].apply(remove_profanity)
    df[config.TARGET_COL] = np.where(
        df[config.LABEL_SOURCE_COL] == config.POSITIVE_LABEL_VALUE, 1, 0
    )
    return df[[config.ID_COL, config.TEXT_COL, config.TARGET_COL]].dropna(
        subset=[config.ID_COL, config.TEXT_COL]
    )


def _stratified_split(df: pd.DataFrame, seed: int):
    rng = np.random.RandomState(seed)
    train_parts, dev_parts, test_parts = [], [], []
    for _, group in df.groupby(config.TARGET_COL):
        idx = rng.permutation(group.index.values)
        n = len(idx)
        n_train = int(round(n * config.TRAIN_FRAC))
        n_dev = int(round(n * config.DEV_FRAC))
        train_idx, dev_idx, test_idx = (
            idx[:n_train],
            idx[n_train:n_train + n_dev],
            idx[n_train + n_dev:],
        )
        train_parts.append(group.loc[train_idx])
        dev_parts.append(group.loc[dev_idx])
        test_parts.append(group.loc[test_idx])

    def _shuffle(parts):
        out = pd.concat(parts, ignore_index=True)
        return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return _shuffle(train_parts), _shuffle(dev_parts), _shuffle(test_parts)


def make_splits(seed: int = config.RANDOM_SEED) -> None:
    """Build and persist train/dev/test splits from the raw dataset."""
    df = load_raw_dataset()
    train_df, dev_df, test_df = _stratified_split(df, seed)

    os.makedirs(config.SPLITS_DIR, exist_ok=True)
    train_df.to_parquet(os.path.join(config.SPLITS_DIR, "train.parquet"), index=False)
    dev_df.to_parquet(os.path.join(config.SPLITS_DIR, "dev.parquet"), index=False)
    test_df.to_parquet(os.path.join(config.SPLITS_DIR, "test.parquet"), index=False)

    print(
        f"Splits written to {config.SPLITS_DIR} "
        f"(train={len(train_df)}, dev={len(dev_df)}, test={len(test_df)})"
    )


def load_splits():
    """Regenerate and load fresh train/dev/test splits on every call."""
    make_splits()
    paths = {
        split: os.path.join(config.SPLITS_DIR, f"{split}.parquet")
        for split in ("train", "dev", "test")
    }
    return tuple(pd.read_parquet(paths[split]) for split in ("train", "dev", "test"))


def sample_minibatch(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample a minibatch of n rows without replacement (or all rows if n >= len(df))."""
    if n >= len(df):
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    make_splits()