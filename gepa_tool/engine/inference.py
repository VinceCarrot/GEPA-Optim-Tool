"""Task-model inference: run one or more candidate prompts over a dataframe.

All rows of all prompts share one semaphore (max_workers), so evaluating a round's
children in parallel never exceeds the configured concurrency. A progress callback
reports (prompt_index, done, total, n_errors) so the UI can show live progress.
Cancellation (Stop) propagates through asyncio: every in-flight request is cancelled.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from . import llm as llm_mod
from . import protect
from .dataset import INPUT_COL
from .spec import RunConfig

logger = logging.getLogger("gepa_tool.inference")

ERROR_LABEL = "ERROR"
ProgressFn = Callable[[int, int, int, int], None]


def _escape_braces(prompt_text: str, placeholder_name: str) -> str:
    """Escape literal { } so only {placeholder} is a template variable (see legacy inference_run)."""
    ph = "{" + placeholder_name + "}"
    escaped = prompt_text.replace("{", "{{").replace("}", "}}")
    return escaped.replace("{{" + placeholder_name + "}}", ph)


async def build_chain(cfg: RunConfig, prompt_text: str):
    from safechain.prompts import ValidPromptTemplate

    var = cfg.task.input_placeholder
    text = protect.strip_markers(prompt_text)
    escaped = _escape_braces(text, var)
    if "{" + var + "}" not in escaped:
        raise ValueError(f"Prompt is missing the required {{{var}}} placeholder")
    template = ValidPromptTemplate(template=escaped, input_variables=[var])
    llm = await llm_mod.build_llm(cfg.models.task_model_id, cfg.models.task_temperature)
    return template | llm.with_structured_output(llm_mod.classification_schema(cfg.task))


def _error_row(row_id, err: str) -> Dict[str, Any]:
    return {
        "row_id": row_id,
        "classification": ERROR_LABEL,
        "classification_reason": ERROR_LABEL,
        "risk_score": 0.0,
        "supporting_evidence": [err[:500]],
    }


def _is_auth_error(e: Exception) -> bool:
    s = str(e).lower()
    return any(x in s for x in ("401", "unauthorized", "invalid api key", "authentication", "expired"))


class _ChainHolder:
    """Holds one prompt's chain and rebuilds it when its IDaaS token may have expired.

    SafeChain bakes the access token into the client at build time, so a chain built at the start
    of a long evaluation would start failing with 401 part-way through. Chains are rebuilt every
    CHAIN_MAX_AGE seconds and immediately after an auth error (at most once per AUTH_REBUILD_GAP).
    """
    CHAIN_MAX_AGE = 240.0
    AUTH_REBUILD_GAP = 5.0

    def __init__(self, cfg: RunConfig, prompt_text: str):
        self.cfg, self.prompt_text = cfg, prompt_text
        self.chain = None
        self.built_at = 0.0
        self.lock = asyncio.Lock()

    async def get(self, force: bool = False):
        now = time.time()
        stale = self.chain is None or now - self.built_at > self.CHAIN_MAX_AGE
        if not (stale or force):
            return self.chain
        async with self.lock:
            now = time.time()
            if self.chain is not None and not (now - self.built_at > self.CHAIN_MAX_AGE) \
                    and (not force or now - self.built_at < self.AUTH_REBUILD_GAP):
                return self.chain
            self.chain = await build_chain(self.cfg, self.prompt_text)
            self.built_at = time.time()
            return self.chain


async def run_many_async(
    cfg: RunConfig,
    prompt_texts: List[str],
    df: pd.DataFrame,
    progress: Optional[ProgressFn] = None,
    progress_every: int = 50,
) -> List[pd.DataFrame]:
    """Evaluate each prompt on every row of df; returns one predictions frame per prompt.

    Rows are interleaved across prompts (row 0 of every prompt, then row 1, ...) so all
    candidates progress together under the shared concurrency limit.
    """
    b = cfg.budget
    id_col = cfg.data.id_col
    var = cfg.task.input_placeholder
    semaphore = asyncio.Semaphore(b.max_workers)
    holders = [_ChainHolder(cfg, p) for p in prompt_texts]
    for h in holders:  # fail fast on a broken prompt before queueing thousands of rows
        await h.get()

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30),
           retry=retry_if_exception_type(Exception), reraise=True)
    async def _call(holder: _ChainHolder, text, index):
        chain = await holder.get()
        try:
            resp = await asyncio.wait_for(
                chain.ainvoke({var: text}, config={"metadata": {"batch_index": index}}),
                timeout=b.row_timeout_seconds,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if _is_auth_error(e):
                logger.warning("auth error on row %s; refreshing token", index)
                await holder.get(force=True)
            raise
        return resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)

    ids = df[id_col].tolist()
    texts = df[INPUT_COL].tolist()
    total = len(df)
    done = [0] * len(prompt_texts)
    errors = [0] * len(prompt_texts)
    last_emit = [0.0] * len(prompt_texts)

    def _tick(pi: int, failed: bool):
        done[pi] += 1
        if failed:
            errors[pi] += 1
        if progress is None:
            return
        now = time.time()
        if done[pi] == total or done[pi] % progress_every == 0 or now - last_emit[pi] > 5:
            last_emit[pi] = now
            progress(pi, done[pi], total, errors[pi])

    async def _row(pi: int, i: int):
        await asyncio.sleep(b.request_delay)
        try:
            async with semaphore:
                res = await _call(holders[pi], texts[i], i)
            res["row_id"] = ids[i]
            _tick(pi, False)
            return res
        except asyncio.CancelledError:
            raise
        except Exception as e:  # permanent failure after retries
            logger.error("prompt %d row %s failed permanently: %s", pi, ids[i], e)
            _tick(pi, True)
            return _error_row(ids[i], str(e))

    n_prompts = len(prompt_texts)
    order = [(pi, i) for i in range(total) for pi in range(n_prompts)]  # interleaved
    tasks = [asyncio.ensure_future(_row(pi, i)) for pi, i in order]
    try:
        results = await asyncio.gather(*tasks)
    except BaseException:
        for t in tasks:
            t.cancel()
        raise
    per_prompt: List[List[Dict[str, Any]]] = [[] for _ in range(n_prompts)]
    for (pi, _i), res in zip(order, results):
        per_prompt[pi].append(res)
    return [pd.DataFrame(rows).rename(columns={"row_id": id_col}) for rows in per_prompt]


async def run_one_async(cfg: RunConfig, prompt_text: str, df: pd.DataFrame,
                        progress: Optional[ProgressFn] = None) -> pd.DataFrame:
    return (await run_many_async(cfg, [prompt_text], df, progress))[0]