"""Natural language -> ObjectiveSpec translation (GPT-5 via SafeChain).

The LLM only fills a whitelisted structure; the result is validated with Pydantic and
rendered back to the user for confirmation. Ambiguity -> clarification questions;
out-of-scope requests -> explicit "unsupported" items. Nothing is silently dropped.
"""

from __future__ import annotations

import asyncio
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from . import llm as llm_mod
from . import objective as obj
from .spec import METRIC_LABELS, Constraint, MetricName, ModelSpec, ObjectiveSpec


class NLConstraint(BaseModel):
    metric: MetricName
    op: Literal[">=", "<="]
    value: float = Field(..., description="Threshold as a fraction between 0 and 1 (85% -> 0.85).")
    source_quote: str = Field(..., description="The exact words in the user's text this constraint comes from.")


class NLObjective(BaseModel):
    maximize: Optional[MetricName] = Field(None, description="Metric to maximize; null if the user did not say.")
    maximize_source_quote: str = ""
    beta: Optional[float] = None
    constraints: List[NLConstraint] = Field(default_factory=list)
    clarifications: List[str] = Field(default_factory=list, description=(
        "Questions to ask the user when wording is ambiguous (e.g. '准确率' could be accuracy or precision). "
        "Never guess on ambiguous wording."))
    unsupported: List[str] = Field(default_factory=list, description=(
        "Requirements that cannot be expressed as 'maximize one metric subject to >=/<= thresholds' "
        "(weighted combinations, per-segment rules, soft targets, conditional rules), quoted verbatim."))


SYSTEM = """\
You translate a business owner's description of what they want from a BINARY classifier into a
strict specification. The positive class is "{positive}" and the negative class is "{negative}".

Allowed metrics (use exactly these names):
- precision: of the cases predicted {positive}, share that truly are
- recall: of the true {positive} cases, share that are caught (1 - miss rate)
- f1, fbeta (needs beta), accuracy
- specificity: of true {negative} cases, share predicted {negative}
- npv: of the cases predicted {negative}, share that truly are
- fpr: false positive rate = 1 - specificity
- fnr: miss rate = 1 - recall
- predicted_positive_rate: share of all cases predicted {positive}

Rules:
1. Output ONE metric to maximize and zero or more HARD constraints of the form metric >= value or metric <= value.
2. Values are fractions in [0, 1]: "85%" / "85" / "八成五" -> 0.85. "不超过两成" -> 0.20.
3. Prefer the natural metric: "漏掉不超过15%" / "miss at most 15%" -> recall >= 0.85 (not fnr). "误报率" means fpr.
4. Lower-is-better metrics (fpr, fnr, predicted_positive_rate) can only be <= constraints, never maximized.
5. Ambiguous words ("准确率", "accuracy" used loosely, "be accurate") -> add a clarification question; do not guess.
   If the metric to maximize is not stated, leave maximize null and ask.
6. Soft wishes ("ideally", "尽量", "希望") are NOT hard constraints: put them in unsupported (soft constraints are not supported).
7. Weighted combinations, per-segment rules, conditional rules, cost-weighted objectives -> unsupported.
8. Every constraint and the objective must quote the user's words that justify it.
"""


async def translate_async(text: str, models: ModelSpec, nl_model_id: str,
                          positive: str, negative: str) -> dict:
    llm_mod.init_models(models)
    structured = await llm_mod.build_structured(nl_model_id, 1.0, NLObjective)
    payload = SYSTEM.format(positive=positive, negative=negative) + "\n\nUser description:\n" + text
    raw: NLObjective = await asyncio.wait_for(structured.ainvoke(payload), timeout=120)
    return finalize(raw)


def finalize(raw: NLObjective) -> dict:
    """Validate the LLM draft and build what the UI renders for confirmation."""
    problems: List[str] = []
    clar = list(raw.clarifications)
    if raw.maximize is None:
        clar.append("Which metric should be maximized?")
    spec = None
    if raw.maximize is not None:
        try:
            spec = ObjectiveSpec(
                maximize=raw.maximize,
                beta=raw.beta,
                constraints=[Constraint(metric=c.metric, op=c.op, value=c.value) for c in raw.constraints],
            )
        except ValidationError as e:
            problems.extend(err["msg"] for err in e.errors())
    readable = []
    if spec is not None:
        readable.append({"kind": "objective", "text": f"Maximize {METRIC_LABELS[spec.maximize]}",
                         "source": raw.maximize_source_quote})
        for c, src in zip(spec.constraints, raw.constraints):
            readable.append({"kind": "constraint", "text": obj.describe_constraint(c), "source": src.source_quote})
    return {
        "spec": spec.model_dump() if spec else None,
        "readable": readable,
        "clarifications": clar,
        "unsupported": raw.unsupported,
        "problems": problems,
        "ok": spec is not None and not clar and not raw.unsupported and not problems,
    }