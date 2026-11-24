"""Run configuration schema for the GEPA tool.

`RunConfig` is the single contract between the UI/server and the engine: the server
writes it as YAML, the engine loads and validates it, and a copy is stored in every
run directory. Everything that used to be a module-level constant in GEPA/config.py
lives here so one run never depends on global state.
"""

from __future__ import annotations

from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# Metrics that can be computed from a binary confusion matrix. Used both as the
# objective and in constraints; the NL->spec translator is restricted to this list.
METRICS = (
    "precision", "recall", "f1", "fbeta", "accuracy", "specificity", "npv",
    "fpr", "fnr", "predicted_positive_rate",
)
MetricName = Literal[
    "precision", "recall", "f1", "fbeta", "accuracy", "specificity", "npv",
    "fpr", "fnr", "predicted_positive_rate",
]

METRIC_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "fbeta": "F-beta",
    "accuracy": "Accuracy",
    "specificity": "Specificity (true negative rate)",
    "npv": "Negative predictive value",
    "fpr": "False positive rate",
    "fnr": "False negative (miss) rate",
    "predicted_positive_rate": "Share of cases predicted positive",
}


class SplitSpec(BaseModel):
    mode: Literal["auto", "provided"] = "auto"
    train_frac: float = 0.75
    dev_frac: float = 0.15          # kept only so auto splits reproduce the legacy partitions; dev is never scored
    test_frac: float = 0.10
    seed: int = 42
    train_path: Optional[str] = None  # mode == "provided"
    test_path: Optional[str] = None

    @model_validator(mode="after")
    def _check(self):
        if self.mode == "auto":
            total = self.train_frac + self.dev_frac + self.test_frac
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"train/dev/test fractions must sum to 1 (got {total:.3f})")
        else:
            if not self.train_path or not self.test_path:
                raise ValueError("split.mode=provided needs train_path and test_path")
        return self


class DataSpec(BaseModel):
    path: Optional[str] = None                 # raw dataset (mode=auto); parquet or csv
    id_col: str
    label_col: str
    positive_value: str                        # value in label_col that means the positive class
    input_columns: List[str]
    input_template: str                        # e.g. "{note}" or "Note: {note}\nProduct: {product}"
    profanity_wordlist: Optional[str] = None   # optional newline-separated word list masked in inputs
    split: SplitSpec = Field(default_factory=SplitSpec)

    @model_validator(mode="after")
    def _check(self):
        if self.split.mode == "auto" and not self.path:
            raise ValueError("data.path is required when split.mode=auto")
        for col in self.input_columns:
            if "{" + col + "}" not in self.input_template:
                raise ValueError(f"input_template does not reference input column {{{col}}}")
        return self


class TaskSpec(BaseModel):
    positive_label: str = "Positive"           # display/output name of the positive class
    negative_label: str = "Negative"
    description: str = ""                       # what is being classified and why
    positive_definition: str = ""
    negative_definition: str = ""
    fp_cost: str = ""                           # business cost of a false positive (free text)
    fn_cost: str = ""                           # business cost of a false negative (free text)
    input_placeholder: str = "INPUT"            # seed prompt must contain {INPUT_PLACEHOLDER}

    @field_validator("input_placeholder")
    @classmethod
    def _identifier(cls, v):
        if not v.isidentifier():
            raise ValueError("input_placeholder must be a valid identifier (letters, digits, underscore)")
        return v

    @model_validator(mode="after")
    def _distinct(self):
        if self.positive_label.strip() == self.negative_label.strip():
            raise ValueError("positive_label and negative_label must differ")
        return self


class Constraint(BaseModel):
    metric: MetricName
    op: Literal[">=", "<="]
    value: float

    @field_validator("value")
    @classmethod
    def _range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("constraint value must be between 0 and 1")
        return v


class ObjectiveSpec(BaseModel):
    maximize: MetricName = "precision"
    beta: Optional[float] = None               # only for fbeta
    scoring_mode: Literal["label", "threshold"] = "label"
    constraints: List[Constraint] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self):
        if self.maximize == "fbeta" and not self.beta:
            raise ValueError("beta is required when maximizing fbeta")
        if self.maximize in ("fpr", "fnr", "predicted_positive_rate"):
            raise ValueError(f"{self.maximize} is a lower-is-better metric and cannot be maximized; use it as a <= constraint")
        seen = {}
        for c in self.constraints:
            if c.metric in seen:
                raise ValueError(f"metric {c.metric} appears in more than one constraint")
            seen[c.metric] = c
        return self


class AcceptanceSpec(BaseModel):
    strategy: Literal["feasibility_first", "legacy"] = "feasibility_first"
    epsilon: float = 0.01                      # feasible parent: child objective may drop by at most this
    infeasible_delta: float = 0.05             # infeasible parent: objective may drop by at most this while violation shrinks
    feasibility_patience: Optional[int] = None  # rounds without any feasible candidate before an alert; None = 30% of rounds


class ModelSpec(BaseModel):
    model_config_path: str                     # safechain config_gpt.yml
    env_file: Optional[str] = None
    task_model_id: str
    reflection_model_id: str
    error_analysis_model_id: str
    task_temperature: float = 1.0
    reflection_temperature: float = 1.0


class BudgetSpec(BaseModel):
    max_rounds: int = 20
    scoring_set_size: int = 1800
    proposals_per_round: int = 3
    max_workers: int = 200
    request_delay: float = 0.5
    row_timeout_seconds: int = 90
    reflection_timeout_seconds: int = 180
    parallel_children: bool = True            # evaluate a round's children concurrently (shared worker pool)
    final_test_eval: bool = True
    # Inference-error handling: when more than max_error_rate of a prompt's rows failed (after the
    # per-call retries), re-run only the failed rows, up to error_retry_passes times. A child that is
    # still above max_error_rate is marked "error" (excluded like a rejected child).
    max_error_rate: float = 0.01
    error_retry_passes: int = 3

    @field_validator("max_error_rate")
    @classmethod
    def _rate(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("must be between 0 and 1")
        return v

    @field_validator("error_retry_passes")
    @classmethod
    def _nonneg(cls, v):
        if v < 0:
            raise ValueError("must be >= 0")
        return v

    @field_validator("max_rounds", "scoring_set_size", "proposals_per_round", "max_workers")
    @classmethod
    def _positive(cls, v):
        if v < 1:
            raise ValueError("must be >= 1")
        return v


class ReflectionSpec(BaseModel):
    n_false_positives: int = 50
    n_false_negatives: int = 50
    phase_a_focus_ratio: float = 0.7           # share of analysed errors from the violating side while infeasible
    mutation_history_max: int = 10
    analysis_concurrency: int = 50
    analysis_request_delay: float = 0.5


class RunConfig(BaseModel):
    name: str = "gepa_run"
    seed_prompt: str                           # full seed prompt text (may contain PROTECTED blocks)
    data: DataSpec
    task: TaskSpec
    objective: ObjectiveSpec = Field(default_factory=ObjectiveSpec)
    acceptance: AcceptanceSpec = Field(default_factory=AcceptanceSpec)
    models: ModelSpec
    budget: BudgetSpec = Field(default_factory=BudgetSpec)
    reflection: ReflectionSpec = Field(default_factory=ReflectionSpec)
    random_seed: int = 42

    @model_validator(mode="after")
    def _seed_has_placeholder(self):
        ph = "{" + self.task.input_placeholder + "}"
        if ph not in self.seed_prompt:
            raise ValueError(f"seed_prompt must contain the input placeholder {ph}")
        return self

    @property
    def patience(self) -> int:
        if self.acceptance.feasibility_patience is not None:
            return self.acceptance.feasibility_patience
        return max(1, round(0.3 * self.budget.max_rounds))


def load_config(path: str) -> RunConfig:
    with open(path, encoding="utf-8") as f:
        return RunConfig.model_validate(yaml.safe_load(f))


def dump_config(cfg: RunConfig, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.model_dump(mode="json"), f, sort_keys=False, allow_unicode=True, width=120)