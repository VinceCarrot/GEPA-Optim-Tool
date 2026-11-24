"""LLM access and the generic binary-classification output schema.

Default backend: SafeChain (`amodel(model_id)`), configured by ModelSpec.model_config_path.

Optional local backend: a module that replaces SafeChain on a given machine. SafeChain is
always the default. A local backend is used only when it is asked for explicitly:

  * command-line flag `--local-backend` (or an alias the module lists in CLI_FLAGS) on
    `python -m gepa_tool.server` / `python -m gepa_tool.engine`, which selects
    `gepa_tool.engine.local_backend`; see apply_backend_flag(), or
  * the GEPA_LLM_BACKEND environment variable: a dotted module path ("safechain" = default).

The server passes its environment to the engine subprocesses, so they use the same backend.
The module must provide:

    async build_structured(model_id, temperature, schema) -> object with `async ainvoke(str)`
                                                             returning a `schema` instance
    list_models() -> [{"id": str, "name": str}, ...]                 (for the UI)
    ui_defaults() -> dict of suggested RunConfig defaults             (optional)
    NAME, CLI_FLAGS                                                   (optional)
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import logging
import os
from typing import Any, List, Literal, Optional, Type

from pydantic import BaseModel, Field, create_model

from .spec import ModelSpec, TaskSpec

logger = logging.getLogger("gepa_tool.llm")

_initialized_for: Optional[str] = None
_backend: Any = None
_backend_loaded = False
LOCAL_BACKEND_MODULE = "gepa_tool.engine.local_backend"
LOCAL_BACKEND_FLAG = "--local-backend"


def apply_backend_flag(argv: List[str]) -> List[str]:
    """Pick the LLM backend from the command line; returns argv without the backend flag.

    With `--local-backend` (or one of the local module's CLI_FLAGS) GEPA_LLM_BACKEND is set to the
    local module; otherwise, unless GEPA_LLM_BACKEND is already set, it is set to "safechain".
    Must run before anything calls backend().
    """
    flags = {LOCAL_BACKEND_FLAG}
    has_local = importlib.util.find_spec(LOCAL_BACKEND_MODULE) is not None
    if has_local:
        flags.update(getattr(importlib.import_module(LOCAL_BACKEND_MODULE), "CLI_FLAGS", ()))
    rest = [a for a in argv if a not in flags]
    if len(rest) != len(argv):
        if not has_local:
            raise SystemExit(f"{LOCAL_BACKEND_FLAG}: no local backend on this machine "
                             f"({LOCAL_BACKEND_MODULE.replace('.', '/')}.py not found)")
        os.environ["GEPA_LLM_BACKEND"] = LOCAL_BACKEND_MODULE
    else:
        os.environ.setdefault("GEPA_LLM_BACKEND", "safechain")
    return rest


def backend_name() -> str:
    be = backend()
    return "SafeChain" if be is None else getattr(be, "NAME", be.__name__)


def backend():
    """The local backend module, or None when SafeChain should be used."""
    global _backend, _backend_loaded
    if _backend_loaded:
        return _backend
    name = os.environ.get("GEPA_LLM_BACKEND", "").strip()
    _backend = None if not name or name.lower() == "safechain" else importlib.import_module(name)
    if _backend is not None:
        logger.info("using local LLM backend %s", _backend.__name__)
    _backend_loaded = True
    return _backend


def init_models(models: ModelSpec) -> None:
    """Point SafeChain at the model config and load credentials (idempotent)."""
    global _initialized_for
    if _initialized_for == models.model_config_path:
        return
    if models.env_file and os.path.exists(models.env_file):
        from dotenv import load_dotenv
        load_dotenv(models.env_file, override=True)
    if backend() is None:
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
    be = backend()
    if be is not None:
        return await be.build_structured(model_id, temperature, schema)
    llm = await build_llm(model_id, temperature)
    return llm.with_structured_output(schema)


class TemplateChain:
    """`template | structured_llm` for a single-variable prompt, without LangChain.

    `template` is already brace-escaped ({{ }} for literals, {var} for the input), so
    str.format restores literal braces exactly like the SafeChain prompt template does.
    """

    def __init__(self, template: str, var: str, structured):
        self.template, self.var, self.structured = template, var, structured

    async def ainvoke(self, inputs: dict, config: Optional[dict] = None):
        return await self.structured.ainvoke(self.template.format(**{self.var: inputs[self.var]}))


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