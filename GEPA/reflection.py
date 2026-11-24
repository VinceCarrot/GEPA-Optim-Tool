"""Reflection and mutation module for the GEPA genetic evolution loop.

Three-stage reflective prompt evolution:
  Stage 0 (per-error analysis): each sampled FP/FN is analyzed individually
    with the full input text + current prompt + actual model output, so the
    diagnosis is grounded in the exact view the task model had (no truncation).
  Stage 1 (clustering): the per-error summaries are grouped into
    NUM_PROPOSALS_PER_ITERATION clusters, each tied to the responsible prompt
    section(s), so each mutation can target a distinct failure mode.
  Stage 2 (targeted mutation): one revised prompt is proposed per cluster,
    informed by mutation history so previously failed strategies are not
    re-tried.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel, Field
import openai_auth_patch
from safechain.core.model import amodel
from tenacity import retry, stop_after_attempt, wait_exponential

import config

_T = TypeVar("_T", bound=BaseModel)


async def _build_structured_llm(model_id: str, temperature: float, schema: Type[_T]):
    """Build a structured-output LLM once so a stage can reuse it across many calls.

    Mirrors inference_run._get_chain: fetch the token/model a single time and share the
    object across all rows, instead of re-fetching a token per call (which floods the auth
    endpoint and triggers 429s).
    """
    llm = await amodel(model_id, {"temperature": temperature})
    return llm.with_structured_output(schema)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30), reraise=True)
async def _invoke_structured(structured_llm, payload: str):
    """Invoke a pre-built structured LLM with retry on transient failures."""
    # A hung request would otherwise block asyncio.gather forever; a timeout lets tenacity retry it.
    return await asyncio.wait_for(structured_llm.ainvoke(payload), timeout=config.REFLECTION_CALL_TIMEOUT_SECONDS)

# ---------------------------------------------------------------------------
# Stage 0: per-error analysis
# ---------------------------------------------------------------------------

ERROR_ANALYSIS_PROMPT = """\
You are analyzing a single misclassification made by an LLM-based binary classifier.
The classifier reads a customer complaint and must output "Amex Error" or "Non Amex Error".

You are given the EXACT system prompt the classifier used, the FULL input text it saw,
the output it produced, and the correct gold label. Your job is to explain precisely why
the classifier got this example wrong and which specific instruction(s) in the system
prompt led it astray.

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


class ErrorAnalysis(BaseModel):
    responsible_prompt_aspect: str = Field(..., description="The specific prompt instruction/section/gap that caused this error.")
    why_wrong: str = Field(..., description="Concise explanation of why the classifier got this example wrong.")


# ---------------------------------------------------------------------------
# Stage 1: clustering
# ---------------------------------------------------------------------------

CLUSTERING_PROMPT = """\
You are grouping individual error analyses of an LLM-based binary classifier
(Amex Error vs Non Amex Error) into distinct failure clusters.

Below are per-error analyses. Each was produced by examining one misclassified
example with the full input text and the classifier's system prompt. Each notes
the responsible prompt aspect and why the classifier was wrong.

=== CURRENT SYSTEM PROMPT ===
{prompt_text}

=== METRICS ===
Label-based precision: {precision:.3f}
Label-based recall: {recall:.3f}
Optimization score: {optimization_score:.3f}
Sample size: {n} total, {n_positive} actual positives, {n_predicted_positive} predicted positives
Recall target (hard business floor): {recall_target:.2f}

=== PER-ERROR ANALYSES ===
{error_analyses}

Group these errors into exactly {n_clusters} distinct clusters, each capturing a
different root-cause failure mode. Make the clusters as non-overlapping as
possible. For each cluster provide:
- a short name/theme
- the responsible prompt section(s) or missing guidance
- how many false positives vs false negatives fall in it
- a root-cause summary
- whether fixing it risks lowering recall below the floor
"""


class ErrorCluster(BaseModel):
    name: str = Field(..., description="Short theme/name for this failure cluster.")
    responsible_prompt_sections: str = Field(..., description="Prompt section(s) or missing guidance responsible.")
    fp_vs_fn: str = Field(..., description="Rough count of false positives vs false negatives in this cluster.")
    root_cause: str = Field(..., description="Root-cause summary for this cluster.")
    recall_risk: str = Field(..., description="Whether fixing this risks lowering recall below the floor.")


class ClusteringOutput(BaseModel):
    diagnosis: str = Field(..., description="Overall natural-language diagnosis across all clusters.")
    clusters: List[ErrorCluster] = Field(..., description="Distinct failure clusters, one per intended mutation.")


# ---------------------------------------------------------------------------
# Stage 2: targeted mutation
# ---------------------------------------------------------------------------

MUTATION_PROMPT = """\
You are optimizing the system prompt of an LLM-based binary classifier.
Task: classify a customer complaint as Amex Error (1) or Non Amex Error (0).
The output schema is fixed and enforced by the calling code (issue_classification
is constrained to exactly "Amex Error" or "Non Amex Error", plus a risk_score
field) -- do not invent a different output format or field names.

HARD BUSINESS CONSTRAINT: recall must stay at or above {recall_target:.2f}.
A candidate whose recall falls below this floor is nearly worthless regardless
of its precision. Prioritize keeping (or improving) recall while improving precision.

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
- Do NOT reduce recall below the floor to chase precision.
- Each proposal must be self-contained and usable as-is.

CRITICAL REQUIREMENT: Every proposed prompt MUST end with the exact placeholder
line {{NEW_QUERY_POTENTIAL_ISSUE}} (including the curly braces). This is the
template variable where the actual complaint text is injected at runtime. If you
omit it, the model will never see the input and cannot classify anything.
"""


class PromptProposal(BaseModel):
    target_cluster: str = Field(..., description="The error cluster name this proposal targets.")
    rationale: str = Field(..., description="What this revision changes and why, based on the targeted cluster.")
    new_prompt: str = Field(..., description="The full revised system prompt text.")


class MutationOutput(BaseModel):
    proposals: List[PromptProposal] = Field(..., description="Proposed revised prompts, one per cluster.")


# Preserved for callers/consumers that expect the combined reflection result shape.
class ReflectionOutput(BaseModel):
    diagnosis: str = Field(..., description="Natural-language diagnosis of why the current prompt is failing.")
    proposals: List[PromptProposal] = Field(..., description="Proposed revised prompts.")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_analyses(analyses: List[dict]) -> str:
    if not analyses:
        return "(none)"
    lines = []
    for a in analyses:
        lines.append(
            f"- id={a['id']} [{a['error_type']}] risk_score={a['risk_score']}\n"
            f"  responsible_prompt_aspect: {a['responsible_prompt_aspect']}\n"
            f"  why_wrong: {a['why_wrong']}"
        )
    return "\n".join(lines)


def _format_clusters(clusters: List[ErrorCluster]) -> str:
    lines = []
    for idx, c in enumerate(clusters, 1):
        lines.append(
            f"Cluster {idx}: {c.name}\n"
            f"  responsible_prompt_sections: {c.responsible_prompt_sections}\n"
            f"  fp_vs_fn: {c.fp_vs_fn}\n"
            f"  root_cause: {c.root_cause}\n"
            f"  recall_risk: {c.recall_risk}"
        )
    return "\n".join(lines)


def _format_mutation_history(history: Optional[List[dict]]) -> str:
    if not history:
        return "(no prior mutations)"
    max_n = config.MUTATION_HISTORY_MAX
    entries = history if max_n <= 0 else history[-max_n:]
    lines = []
    for h in entries:
        lines.append(
            f"- iter {h['iteration']}: \"{h['rationale']}\" -> "
            f"precision {h['parent_precision']:.2f}->{h['child_precision']:.2f}, "
            f"recall {h['parent_recall']:.2f}->{h['child_recall']:.2f} "
            f"(opt {h['parent_score']:.3f}->{h['child_score']:.3f}, "
            f"{'ACCEPTED' if h['accepted'] else 'REJECTED'})"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

STAGE0_MAX_CONCURRENT = 50
STAGE0_REQUEST_DELAY = 0.5


async def _analyze_one_error(prompt_text: str, ex: dict, error_type: str,
                             structured_llm, fallback_llm=None,
                             semaphore: Optional[asyncio.Semaphore] = None,
                             delay: float = 0.0) -> dict:
    if delay > 0:
        await asyncio.sleep(delay)
    payload = ERROR_ANALYSIS_PROMPT.format(
        prompt_text=prompt_text,
        input_text=str(ex.get(config.TEXT_COL, "")),
        predicted=ex.get("issue_classification"),
        risk_score=ex.get("risk_score"),
        model_reason=str(ex.get("issue_classification_reason", "")),
        gold_label="Amex Error" if ex.get(config.TARGET_COL) == 1 else "Non Amex Error",
        error_type=error_type,
    )

    async def _call():
        try:
            return await _invoke_structured(structured_llm, payload)
        except Exception:
            if fallback_llm is None:
                raise
            return await _invoke_structured(fallback_llm, payload)

    if semaphore:
        async with semaphore:
            result = await _call()
    else:
        result = await _call()
    return {
        "id": ex.get(config.ID_COL),
        "error_type": error_type,
        "risk_score": ex.get("risk_score"),
        "responsible_prompt_aspect": result.responsible_prompt_aspect,
        "why_wrong": result.why_wrong,
    }


async def analyze_errors_async(prompt_text: str, analysis_errors: Dict[str, List[dict]],
                               model_id: str = config.ERROR_ANALYSIS_MODEL_ID,
                               fallback_model_id: Optional[str] = config.TASK_MODEL_ID) -> List[dict]:
    """Stage 0: analyze each sampled FP/FN individually, in parallel, with full text."""
    semaphore = asyncio.Semaphore(STAGE0_MAX_CONCURRENT)
    # Build the model(s) once and share across all per-error calls (one token fetch, not one per call).
    structured_llm = await _build_structured_llm(model_id, 1.0, ErrorAnalysis)
    fallback_llm = None
    if fallback_model_id and fallback_model_id != model_id:
        fallback_llm = await _build_structured_llm(fallback_model_id, 1.0, ErrorAnalysis)
    tasks = []
    for idx, ex in enumerate(analysis_errors.get("false_positives", [])):
        tasks.append(_analyze_one_error(prompt_text, ex, "false_positive", structured_llm, fallback_llm,
                                        semaphore, delay=idx * STAGE0_REQUEST_DELAY))
    for idx, ex in enumerate(analysis_errors.get("false_negatives", [])):
        tasks.append(_analyze_one_error(prompt_text, ex, "false_negative", structured_llm, fallback_llm,
                                        semaphore, delay=idx * STAGE0_REQUEST_DELAY))
    if not tasks:
        return []
    return await asyncio.gather(*tasks)


async def cluster_errors_async(prompt_text: str, metrics: Dict, analyses: List[dict],
                               recall_target: float, n_clusters: int,
                               model_id: str = config.REFLECTION_MODEL_ID) -> ClusteringOutput:
    """Stage 1: group per-error analyses into n_clusters distinct failure modes."""
    label_metrics = metrics["label_metrics"]
    payload = CLUSTERING_PROMPT.format(
        prompt_text=prompt_text,
        precision=label_metrics["precision"],
        recall=label_metrics["recall"],
        optimization_score=metrics["optimization_score"],
        n=label_metrics["n"],
        n_positive=label_metrics["n_positive"],
        n_predicted_positive=label_metrics["n_predicted_positive"],
        recall_target=recall_target,
        error_analyses=_format_analyses(analyses),
        n_clusters=n_clusters,
    )
    structured_llm = await _build_structured_llm(model_id, 1.0, ClusteringOutput)
    return await _invoke_structured(structured_llm, payload)


async def mutate_from_clusters_async(prompt_text: str, clustering: ClusteringOutput,
                                     mutation_history: Optional[List[dict]], recall_target: float,
                                     n_proposals: int,
                                     model_id: str = config.REFLECTION_MODEL_ID) -> MutationOutput:
    """Stage 2: propose one targeted revised prompt per error cluster."""
    payload = MUTATION_PROMPT.format(
        prompt_text=prompt_text,
        diagnosis=clustering.diagnosis,
        clusters=_format_clusters(clustering.clusters),
        mutation_history=_format_mutation_history(mutation_history),
        recall_target=recall_target,
        n_proposals=n_proposals,
    )
    structured_llm = await _build_structured_llm(model_id, 1.0, MutationOutput)
    response = await _invoke_structured(structured_llm, payload)
    placeholder = "{NEW_QUERY_POTENTIAL_ISSUE}"
    for proposal in response.proposals:
        if placeholder not in proposal.new_prompt:
            proposal.new_prompt = proposal.new_prompt.rstrip() + f"\n\n---\n\nNow analyze the following complaint:\n\n{placeholder}"
    return response


async def propose_mutations_async(
    prompt_text: str,
    metrics: Dict,
    recall_target: float = config.RECALL_TARGET,
    n_proposals: int = config.NUM_PROPOSALS_PER_ITERATION,
    analysis_errors: Optional[Dict[str, List[dict]]] = None,
    mutation_history: Optional[List[dict]] = None,
    analysis_model_id: str = config.ERROR_ANALYSIS_MODEL_ID,
    model_id: str = config.REFLECTION_MODEL_ID,
) -> ReflectionOutput:
    """Run the full three-stage pipeline and return the combined reflection result."""
    analyses = await analyze_errors_async(prompt_text, analysis_errors or {}, analysis_model_id)
    clustering = await cluster_errors_async(
        prompt_text, metrics, analyses, recall_target, n_clusters=n_proposals, model_id=model_id
    )
    mutation = await mutate_from_clusters_async(
        prompt_text, clustering, mutation_history, recall_target, n_proposals, model_id=model_id
    )
    return ReflectionOutput(diagnosis=clustering.diagnosis, proposals=mutation.proposals)


def propose_mutations(
    prompt_text: str,
    metrics: Dict,
    recall_target: float = config.RECALL_TARGET,
    n_proposals: int = config.NUM_PROPOSALS_PER_ITERATION,
    analysis_errors: Optional[Dict[str, List[dict]]] = None,
    mutation_history: Optional[List[dict]] = None,
    analysis_model_id: str = config.ERROR_ANALYSIS_MODEL_ID,
    model_id: str = config.REFLECTION_MODEL_ID,
) -> ReflectionOutput:
    """Synchronous wrapper around the three-stage propose_mutations_async."""
    return asyncio.run(
        propose_mutations_async(
            prompt_text, metrics, recall_target, n_proposals,
            analysis_errors, mutation_history, analysis_model_id, model_id,
        )
    )