"""LLM access (SafeChain) and the generic binary-classification output schema."""

from __future__ import annotations

import asyncio
import os
from typing import Literal, Optional, Type

from pydantic import BaseModel, Field, create_model

from .spec import ModelSpec, TaskSpec

_initialized_for: Optional[str] = None


def init_models(models: ModelSpec) -> None:
    """Point SafeChain at the model config and load credentials (idempotent)."""
    global _initialized_for
    if _initialized_for == models.model_config_path:
        return
    if models.env_file and os.path.exists(models.env_file):
        from dotenv import load_dotenv
        load_dotenv(models.env_file, override=True)
    os.environ["CONFIG_PATH"] = models.model_config_path
    try:  # Amex SafeChain 2.0.3 Azure auth workaround (no-op elsewhere)
        from . import openai_auth_patch  # noqa: F401
    except Exception:
        pass
    _initialized_for = models.model_config_path


async def build_llm(model_id: str, temperature: float):
    from safechain.core.model import amodel
    return await amodel(model_id, {"temperature": temperature})


async def build_structured(model_id: str, temperature: float, schema: Type[BaseModel]):
    llm = await build_llm(model_id, temperature)
    return llm.with_structured_output(schema)


async def invoke_with_timeout(runnable, payload, timeout: float, **kwargs):
    return await asyncio.wait_for(runnable.ainvoke(payload, **kwargs), timeout=timeout)


def classification_schema(task: TaskSpec) -> Type[BaseModel]:
    """Structured output the task model must return for every row.

    reasoning -> risk_score -> classification ordering mirrors the legacy CaseNoteOutput
    (justify first, then score, then decide).
    """
    pos, neg = task.positive_label, task.negative_label
    label_type = Literal[pos, neg]  # type: ignore[valid-type]
    return create_model(
        "ClassificationOutput",
        supporting_evidence=(list[str], Field(..., description=(
            "Short paraphrases or brief quotes from the input that support the classification. "
            "Must not contain personally identifiable information."))),
        classification_reason=(str, Field(..., description=(
            "Clear justification for the classification, considered before assigning the risk score "
            "and the final classification."))),
        risk_score=(float, Field(..., ge=0.0, le=1.0, description=(
            f"Confidence that the input belongs to '{pos}', from 0.0 to 1.0."))),
        classification=(label_type, Field(..., description=(
            f"Final binary classification, decided holistically from the evidence and reasoning "
            f"(not a mechanical threshold on risk_score). Must be exactly '{pos}' or '{neg}'."))),
    )