"""Three-stage reflective mutation, generalized from GEPA/reflection.py.

  Stage 0 (per-error analysis): each sampled FP/FN analysed individually with full text.
  Stage 1 (clustering): analyses grouped into N failure clusters.
  Stage 2 (targeted mutation): one revised prompt per cluster.

Task wording, label names, objective/constraints and the current phase (A: satisfy
constraints / B: optimize objective) are injected from RunConfig. Protected sections are
masked as [[PROTECTED:name]] tokens for Stage 2 and validated / restored afterwards.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Dict, List, Optional

from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from . import llm as llm_mod
from . import objective as obj
from . import protect
from .dataset import INPUT_COL, TARGET_COL
from .spec import METRIC_LABELS, RunConfig

logger = logging.getLogger("gepa_tool.reflection")
StageFn = Callable[[str, str], None]  # (stage, status)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ErrorAnalysis(BaseModel):
    responsible_prompt_aspect: str = Field(..., description="The specific prompt instruction/section/gap that caused this error.")
    why_wrong: str = Field(..., description="Concise explanation of why the classifier got this example wrong.")


class ErrorCluster(BaseModel):
    name: str = Field(..., description="Short theme/name for this failure cluster.")
    responsible_prompt_sections: str = Field(..., description="Prompt section(s) or missing guidance responsible.")
    fp_vs_fn: str = Field(..., description="Rough count of false positives vs false negatives in this cluster.")
    root_cause: str = Field(..., description="Root-cause summary for this cluster.")
    constraint_risk: str = Field(..., description="Whether fixing this risks breaking a hard constraint.")


class ClusteringOutput(BaseModel):
    diagnosis: str = Field(..., description="Overall natural-language diagnosis across all clusters.")
    clusters: List[ErrorCluster] = Field(..., description="Distinct failure clusters, one per intended mutation.")


class PromptProposal(BaseModel):
    target_cluster: str = Field(..., description="The error cluster name this proposal targets.")
    rationale: str = Field(..., description="What this revision changes and why, based on the targeted cluster.")
    new_prompt: str = Field(..., description="The full revised system prompt text.")


class MutationOutput(BaseModel):
    proposals: List[PromptProposal] = Field(..., description="Proposed revised prompts, one per cluster.")


class ReflectionResult(BaseModel):
    diagnosis: str
    clusters: List[ErrorCluster]
    proposals: List[PromptProposal]
    rejected: List[dict] = Field(default_factory=list)  # proposals dropped for protected-section violations
    n_analyses: int = 0


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

def _task_block(cfg: RunConfig) -> str:
    t = cfg.task
    lines = [
        f"The classifier reads an input and must output exactly \"{t.positive_label}\" (positive) "
        f"or \"{t.negative_label}\" (negative).",
    ]
    if t.description:
        lines.append(f"Task description: {t.description}")
    if t.positive_definition:
        lines.append(f"Definition of \"{t.positive_label}\": {t.positive_definition}")
    if t.negative_definition:
        lines.append(f"Definition of \"{t.negative_label}\": {t.negative_definition}")
    if t.fp_cost:
        lines.append(f"Cost of a false positive: {t.fp_cost}")
    if t.fn_cost:
        lines.append(f"Cost of a false negative: {t.fn_cost}")
    return "\n".join(lines)


def _objective_block(cfg: RunConfig) -> str:
    spec = cfg.objective
    cons = [obj.describe_constraint(c) for c in spec.constraints]
    out = [f"Objective: maximize {METRIC_LABELS[spec.maximize]}."]
    if cons:
        out.append("HARD constraints (a candidate that breaks any of them is nearly worthless): " + "; ".join(cons) + ".")
    return "\n".join(out)


def _phase_block(cfg: RunConfig, phase: str, metrics: Dict) -> str:
    spec = cfg.objective
    if phase == "A" and spec.constraints:
        missing = [obj.describe_constraint(c) for c in spec.constraints if obj.constraint_gap(c, metrics) > 0]
        return (
            "CURRENT PRIORITY (constraint phase): the prompt does not yet satisfy "
            + "; ".join(missing)
            + ". The first priority is to close this gap. Do not sacrifice "
            + METRIC_LABELS[spec.maximize]
            + " more than necessary, and never make the classifier predict one class for almost everything."
        )
    if spec.constraints:
        return (
            "CURRENT PRIORITY (optimization phase): the constraints are met. Improve "
            + METRIC_LABELS[spec.maximize]
            + " while keeping every constraint satisfied."
        )
    return f"CURRENT PRIORITY: improve {METRIC_LABELS[spec.maximize]}."


def _metrics_block(cfg: RunConfig, m: Dict, score: float) -> str:
    return (
        f"Precision: {m['precision']:.3f}  Recall: {m['recall']:.3f}  F1: {m['f1']:.3f}  "
        f"Specificity: {m['specificity']:.3f}\n"
        f"Optimization score: {score:.3f}\n"
        f"Sample size: {m['n']} total, {m['n_positive']} actual positives, "
        f"{m['n_predicted_positive']} predicted positives"
    )


ERROR_ANALYSIS_PROMPT = """\
You are analyzing a single misclassification made by an LLM-based binary classifier.
{task_block}

You are given the EXACT system prompt the classifier used, the FULL input text it saw,
the output it produced, and the correct gold label. Your job is to explain precisely why
the classifier got this example wrong and which specific instruction(s) in the system
prompt led it astray.
{protected_note}
=== SYSTEM PROMPT USED ===
{prompt_text}

=== FULL INPUT TEXT ===
{input_text}

=== CLASSIFIER OUTPUT ===
Predicted: {predicted}
risk_score: {risk_score}
Model's stated reasoning: {model_reason}

=== CORRECT GOLD LABEL ===
{gold_label}  ({error_type})

Explain, in 2-4 sentences:
1. Why the classifier reached the wrong answer on THIS example.
2. Which specific prompt instruction, definition, or missing guidance caused it.
Be concrete and reference the actual prompt wording where possible.
"""

CLUSTERING_PROMPT = """\
You are grouping individual error analyses of an LLM-based binary classifier into
distinct failure clusters.
{task_block}

{objective_block}
{phase_block}

Below are per-error analyses. Each was produced by examining one misclassified
example with the full input text and the classifier's system prompt. Each notes
the responsible prompt aspect and why the classifier was wrong.
{protected_note}
=== CURRENT SYSTEM PROMPT ===
{prompt_text}

=== METRICS ===
{metrics_block}

=== PER-ERROR ANALYSES ===
{error_analyses}

Group these errors into exactly {n_clusters} distinct clusters, each capturing a
different root-cause failure mode. Make the clusters as non-overlapping as
possible. For each cluster provide:
- a short name/theme
- the responsible prompt section(s) or missing guidance
- how many false positives vs false negatives fall in it
- a root-cause summary
- whether fixing it risks breaking a hard constraint
"""

MUTATION_PROMPT = """\
You are optimizing the system prompt of an LLM-based binary classifier.
{task_block}
The output schema is fixed and enforced by the calling code (the classification is
constrained to exactly "{positive_label}" or "{negative_label}", plus a risk_score
field) -- do not invent a different output format or field names.

{objective_block}
{phase_block}

=== CURRENT SYSTEM PROMPT ===
{prompt_text}

=== OVERALL DIAGNOSIS ===
{diagnosis}

=== ERROR CLUSTERS (one mutation should target each) ===
{clusters}

=== MUTATION HISTORY (do not re-try strategies that already failed) ===
{mutation_history}

Produce exactly {n_proposals} distinct, complete, revised system prompts -- one
targeting each error cluster above. For each proposal:
- Focus the change on the responsible prompt section(s) for that cluster.
- Prefer surgical edits: keep sections that are working verbatim and rewrite
  only what is needed to fix the targeted cluster. Rewrite the whole prompt only
  if the failure is structural.
- Do NOT break a hard constraint to chase the objective.
- Each proposal must be self-contained and usable as-is.
{protected_rule}
CRITICAL REQUIREMENT: Every proposed prompt MUST contain the exact placeholder
{{{placeholder}}} (including the curly braces), normally on the last line. This is the
template variable where the actual input text is injected at runtime. If you
omit it, the model will never see the input and cannot classify anything.
"""

PROTECTED_NOTE = """
NOTE: parts of the system prompt are wrapped in <!-- PROTECTED:name --> ... <!-- /PROTECTED -->
markers. Those sections are locked and cannot be changed. Do not attribute errors to them;
attribute errors to the editable text or to guidance that is missing from it.
"""

PROTECTED_RULE = """
LOCKED SECTIONS: the prompt contains tokens of the form [[PROTECTED:name]]. Each stands for a
locked section that will be re-inserted verbatim. Every proposal MUST contain each of these
tokens exactly once, unchanged and in the same order: {tokens}.
Do not edit, paraphrase, duplicate, remove or reorder them, and do not add new ones.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_analyses(analyses: List[dict]) -> str:
    if not analyses:
        return "(none)"
    return "\n".join(
        f"- id={a['id']} [{a['error_type']}] risk_score={a['risk_score']}\n"
        f"  responsible_prompt_aspect: {a['responsible_prompt_aspect']}\n"
        f"  why_wrong: {a['why_wrong']}"
        for a in analyses
    )


def _format_clusters(clusters: List[ErrorCluster]) -> str:
    return "\n".join(
        f"Cluster {i}: {c.name}\n"
        f"  responsible_prompt_sections: {c.responsible_prompt_sections}\n"
        f"  fp_vs_fn: {c.fp_vs_fn}\n"
        f"  root_cause: {c.root_cause}\n"
        f"  constraint_risk: {c.constraint_risk}"
        for i, c in enumerate(clusters, 1)
    )


def _format_history(history: Optional[List[dict]], max_n: int) -> str:
    if not history:
        return "(no prior mutations)"
    entries = history if max_n <= 0 else history[-max_n:]
    return "\n".join(
        f"- round {h['round']}: \"{h['rationale']}\" -> "
        f"precision {h['parent_precision']:.2f}->{h['child_precision']:.2f}, "
        f"recall {h['parent_recall']:.2f}->{h['child_recall']:.2f} "
        f"(opt {h['parent_score']:.3f}->{h['child_score']:.3f}, "
        f"{'ACCEPTED' if h['accepted'] else 'REJECTED'})"
        for h in entries
    )


def _retrying(timeout: float):
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30), reraise=True)
    async def _invoke(structured_llm, payload: str):
        return await asyncio.wait_for(structured_llm.ainvoke(payload), timeout=timeout)
    return _invoke


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

async def analyze_errors(cfg: RunConfig, prompt_text: str, errors: Dict[str, List[dict]]) -> List[dict]:
    r = cfg.reflection
    invoke = _retrying(cfg.budget.reflection_timeout_seconds)
    sem = asyncio.Semaphore(r.analysis_concurrency)
    structured = await llm_mod.build_structured(cfg.models.error_analysis_model_id,
                                                cfg.models.reflection_temperature, ErrorAnalysis)
    fallback = None
    if cfg.models.task_model_id != cfg.models.error_analysis_model_id:
        fallback = await llm_mod.build_structured(cfg.models.task_model_id,
                                                  cfg.models.reflection_temperature, ErrorAnalysis)
    has_protected = bool(protect.parse(prompt_text))
    pos, neg = cfg.task.positive_label, cfg.task.negative_label

    async def _one(ex: dict, error_type: str, delay: float):
        await asyncio.sleep(delay)
        payload = ERROR_ANALYSIS_PROMPT.format(
            task_block=_task_block(cfg),
            protected_note=PROTECTED_NOTE if has_protected else "",
            prompt_text=prompt_text,
            input_text=str(ex.get(INPUT_COL, "")),
            predicted=ex.get("classification"),
            risk_score=ex.get("risk_score"),
            model_reason=str(ex.get("classification_reason", "")),
            gold_label=pos if ex.get(TARGET_COL) == 1 else neg,
            error_type=error_type,
        )
        async with sem:
            try:
                res = await invoke(structured, payload)
            except asyncio.CancelledError:
                raise
            except Exception:
                if fallback is None:
                    raise
                res = await invoke(fallback, payload)
        return {
            "id": ex.get(cfg.data.id_col),
            "error_type": error_type,
            "risk_score": ex.get("risk_score"),
            "responsible_prompt_aspect": res.responsible_prompt_aspect,
            "why_wrong": res.why_wrong,
        }

    tasks = []
    for i, ex in enumerate(errors.get("false_positives", [])):
        tasks.append(_one(ex, "false_positive", i * r.analysis_request_delay))
    for i, ex in enumerate(errors.get("false_negatives", [])):
        tasks.append(_one(ex, "false_negative", i * r.analysis_request_delay))
    if not tasks:
        return []
    return list(await asyncio.gather(*tasks))


async def cluster_errors(cfg: RunConfig, prompt_text: str, bundle: Dict, analyses: List[dict],
                         phase: str, n_clusters: int) -> ClusteringOutput:
    m = bundle["metrics"]
    payload = CLUSTERING_PROMPT.format(
        task_block=_task_block(cfg),
        objective_block=_objective_block(cfg),
        phase_block=_phase_block(cfg, phase, m),
        protected_note=PROTECTED_NOTE if protect.parse(prompt_text) else "",
        prompt_text=prompt_text,
        metrics_block=_metrics_block(cfg, m, bundle["optimization_score"]),
        error_analyses=_format_analyses(analyses),
        n_clusters=n_clusters,
    )
    structured = await llm_mod.build_structured(cfg.models.reflection_model_id,
                                                cfg.models.reflection_temperature, ClusteringOutput)
    return await _retrying(cfg.budget.reflection_timeout_seconds)(structured, payload)


def _ensure_placeholder(cfg: RunConfig, text: str) -> str:
    ph = "{" + cfg.task.input_placeholder + "}"
    if ph in text:
        return text
    return text.rstrip() + f"\n\n---\n\nNow analyze the following input:\n\n{ph}"


async def mutate(cfg: RunConfig, prompt_text: str, bundle: Dict, clustering: ClusteringOutput,
                 history: Optional[List[dict]], phase: str, n_proposals: int):
    """Returns (valid_proposals, rejected_records). Proposals carry the unmasked prompt."""
    masked, originals = protect.mask(prompt_text)
    rule = ""
    if originals:
        rule = PROTECTED_RULE.format(tokens=", ".join(protect.token(n) for n in originals))

    def _payload(extra: str = "") -> str:
        return MUTATION_PROMPT.format(
            task_block=_task_block(cfg),
            positive_label=cfg.task.positive_label,
            negative_label=cfg.task.negative_label,
            objective_block=_objective_block(cfg),
            phase_block=_phase_block(cfg, phase, bundle["metrics"]),
            prompt_text=masked,
            diagnosis=clustering.diagnosis,
            clusters=_format_clusters(clustering.clusters),
            mutation_history=_format_history(history, cfg.reflection.mutation_history_max),
            n_proposals=n_proposals,
            protected_rule=rule,
            placeholder=cfg.task.input_placeholder,
        ) + extra

    structured = await llm_mod.build_structured(cfg.models.reflection_model_id,
                                                cfg.models.reflection_temperature, MutationOutput)
    invoke = _retrying(cfg.budget.reflection_timeout_seconds)
    response = await invoke(structured, _payload())

    valid: List[PromptProposal] = []
    rejected: List[dict] = []
    to_retry = []
    for p in response.proposals:
        problems = protect.validate_tokens(p.new_prompt, originals) if originals else []
        if problems:
            to_retry.append((p, problems))
        else:
            valid.append(p)

    # One retry per offending proposal, with the specific violation spelled out.
    for p, problems in to_retry:
        extra = (
            f"\n\nYour previous proposal targeting cluster \"{p.target_cluster}\" violated the locked-section rule "
            f"({'; '.join(problems)}). Produce exactly 1 proposal for that same cluster that obeys the rule."
        )
        try:
            again = await invoke(structured, _payload(extra).replace(
                f"Produce exactly {n_proposals} distinct", "Produce exactly 1 distinct"))
            cand = again.proposals[0] if again.proposals else None
        except asyncio.CancelledError:
            raise
        except Exception as e:  # treat as failed retry
            logger.warning("protected-section retry failed: %s", e)
            cand = None
        retry_problems = protect.validate_tokens(cand.new_prompt, originals) if cand else ["no proposal returned"]
        if cand is not None and not retry_problems:
            valid.append(cand)
        else:
            rejected.append({"target_cluster": p.target_cluster, "rationale": p.rationale,
                             "problems": retry_problems})

    for p in valid:
        restored = protect.unmask(p.new_prompt, originals) if originals else p.new_prompt
        p.new_prompt = _ensure_placeholder(cfg, restored)
    return valid, rejected


def choose_error_mix(cfg: RunConfig, phase: str, metrics: Dict) -> Dict[str, int]:
    """How many FP / FN to analyse. Phase A tilts towards the side that violates a constraint."""
    r = cfg.reflection
    total = r.n_false_positives + r.n_false_negatives
    if phase != "A":
        return {"fp": r.n_false_positives, "fn": r.n_false_negatives}
    sides = set(obj.violated_sides(cfg.objective, metrics))
    if sides == {"fn"}:
        n_fn = round(total * r.phase_a_focus_ratio)
        return {"fp": total - n_fn, "fn": n_fn}
    if sides == {"fp"}:
        n_fp = round(total * r.phase_a_focus_ratio)
        return {"fp": n_fp, "fn": total - n_fp}
    return {"fp": r.n_false_positives, "fn": r.n_false_negatives}


async def reflect(cfg: RunConfig, prompt_text: str, bundle: Dict, errors: Dict[str, List[dict]],
                  history: List[dict], phase: str, on_stage: Optional[StageFn] = None) -> ReflectionResult:
    stage = on_stage or (lambda s, st: None)
    n = cfg.budget.proposals_per_round

    stage("error_analysis", "start")
    analyses = await analyze_errors(cfg, prompt_text, errors)
    stage("error_analysis", "done")

    stage("clustering", "start")
    clustering = await cluster_errors(cfg, prompt_text, bundle, analyses, phase, n)
    stage("clustering", "done")

    stage("mutation", "start")
    proposals, rejected = await mutate(cfg, prompt_text, bundle, clustering, history, phase, n)
    stage("mutation", "done")

    return ReflectionResult(diagnosis=clustering.diagnosis, clusters=clustering.clusters,
                            proposals=proposals, rejected=rejected, n_analyses=len(analyses))