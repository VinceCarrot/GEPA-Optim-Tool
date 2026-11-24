"""Reusable model-inference runner for the GEPA optimization loop.

Given a system prompt string and a dataframe of (id, text) rows, runs the
task LLM over all rows and returns structured predictions. This module owns
model invocation only -- metric computation lives in evaluator.py and the
optimization loop lives in optimizer.py.
"""

import warnings

warnings.filterwarnings("ignore")

import os
import math
import asyncio
import logging
from typing import Any, Dict, List, Literal

import pandas as pd
import nest_asyncio
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from pydantic import BaseModel, Field
import openai_auth_patch
from safechain.core.model import amodel
from safechain.prompts import ValidPromptTemplate
from tqdm.asyncio import tqdm as atqdm

import config

nest_asyncio.apply()
load_dotenv(config.ENV_FILE, override=True)
os.environ["CONFIG_PATH"] = config.MODEL_CONFIG_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(config.GEPA_DIR, "run_sample.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ----------- Runtime knobs --------------

MAX_WORKERS = 200
REQUEST_DELAY = 0.5  # lowered from 3s; MAX_WORKERS already throttles concurrency
CHUNK_SIZE = 1800 #1000

INPUT_VAR = "NEW_QUERY_POTENTIAL_ISSUE"

# ----------- Schema --------------


class CaseNoteOutput(BaseModel):
    case_note: List[str] = Field(
        ...,
        description=(
            "Concise, professional case notes written in third person. "
            "The notes should summarize the issue, identify the primary issue category "
            "(Amex Error or Non Amex Error), and reference repeated complaints "
            "by the same cardmember only if supported by evidence."
        ),
    )
    supporting_evidence_classification: List[str] = Field(
        ...,
        description=(
            "Short paraphrases or brief, redacted quotes from the call transcript or "
            "case description that directly support the case note and issue "
            "classification. Must not contain personally identifiable information (PII)."
        ),
    )
    issue_classification_reason: str = Field(
        ...,
        description=(
            "Clear justification for the issue classification explaining the "
            "evidence or reasoning, considered before assigning a risk score or a "
            "final classification."
        ),
    )
    risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model risk score in the Amex Error classification decision, 0.0 to 1.0.",
    )
    issue_classification: Literal["Amex Error", "Non Amex Error"] = Field(
        ...,
        description=(
            "Final binary issue classification, decided holistically from the evidence and "
            "reasoning above (not a mechanical threshold on risk_score). Must be exactly "
            "'Amex Error' if the primary cause is an Amex system/process failure or Amex "
            "agent human error, or 'Non Amex Error' otherwise (merchant/third-party cause, "
            "customer misunderstanding, or no reported issue)."
        ),
    )
    is_repeated_complaint: Literal["yes", "no", "cannot_be_determined"] = Field(
        ...,
        description=(
            "Indicates whether the complaint is a repeated complaint by the same "
            "cardmember. 'yes' = explicit evidence of a prior similar complaint; "
            "'no' = explicit evidence this is the first complaint for this issue; "
            "'cannot_be_determined' = insufficient or ambiguous information."
        ),
    )
    supporting_evidence_repeated_complaints: List[str] = Field(
        ...,
        description=(
            "Short paraphrases or brief, redacted quotes supporting the repeated "
            "complaint determination. Must not contain PII."
        ),
    )


def _escape_literal_braces(prompt_text: str) -> str:
    """Escape `{`/`}` that aren't the {INPUT_VAR} placeholder.

    Reflection-proposed prompts sometimes ask the model to output literal JSON
    like {"label": 0 or 1}. ValidPromptTemplate treats any {...} as an f-string
    template variable, so unescaped JSON braces raise a missing-variable error.
    Doubling braces here escapes them for the template engine while leaving the
    real {INPUT_VAR} placeholder intact.
    """
    placeholder = "{" + INPUT_VAR + "}"
    escaped = prompt_text.replace("{", "{{").replace("}", "}}")
    return escaped.replace("{{" + INPUT_VAR + "}}", placeholder)


async def _get_chain(prompt_text: str, model_id: str = config.TASK_MODEL_ID):
    """Build a fresh chain for the given candidate prompt text."""
    escaped = _escape_literal_braces(prompt_text)
    placeholder = "{" + INPUT_VAR + "}"
    if placeholder not in escaped:
        raise ValueError(
            f"Prompt text is missing the required {placeholder} placeholder. "
            f"The model will never see the input text without it."
        )
    template = ValidPromptTemplate(template=escaped, input_variables=[INPUT_VAR])
    llm = await amodel(model_id, {"temperature": 1.0})
    return template | llm.with_structured_output(CaseNoteOutput)





# ----------- Retry-wrapped call --------------

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def _call_llm_with_retry(chain: Any, payload: Dict[str, str], index: int) -> Dict[str, Any]:
    try:
        response = await asyncio.wait_for(
            chain.ainvoke(payload, config={"metadata": {"batch_index": index}}),
            timeout=config.ROW_TIMEOUT_SECONDS,
        )
        return response.model_dump() if hasattr(response, "model_dump") else response
    except Exception as e:
        is_auth_error = any(
            x in str(e).lower() for x in ["401", "unauthorized", "invalid api key", "authentication"]
        )
        if is_auth_error:
            logger.warning(f"[AUTH ERROR] Authentication failure on index {index}; retrying SafeChain request.")
        raise


async def _process_row_async(index: int, text: str, row_id: str, chain: Any, semaphore: asyncio.Semaphore,
                              request_delay: float) -> Dict[str, Any]:
    await asyncio.sleep(request_delay)
    try:
        async with semaphore:
            res = await _call_llm_with_retry(chain, {INPUT_VAR: text}, index)
        res[config.ID_COL] = row_id
        return res
    except Exception as e:
        logger.error(f"Index {index} (id={row_id}) failed permanently after retries. Error: {str(e)}")
        return {
            config.ID_COL: row_id,
            "case_note": ["ERROR"],
            "issue_classification": "ERROR",
            "issue_classification_reason": "ERROR",
            "is_repeated_complaint": "ERROR",
            "risk_score": 0.0,
            "supporting_evidence_classification": [str(e)],
            "supporting_evidence_repeated_complaints": [],
        }


async def run_inference_async(
    prompt_text: str,
    df: pd.DataFrame,
    id_col: str = config.ID_COL,
    text_col: str = config.TEXT_COL,
    model_id: str = config.TASK_MODEL_ID,
    max_workers: int = MAX_WORKERS,
    request_delay: float = REQUEST_DELAY,
    chunk_size: int = CHUNK_SIZE,
) -> pd.DataFrame:
    """Run the task LLM with `prompt_text` over every row of `df`. Returns a predictions dataframe."""
    semaphore = asyncio.Semaphore(max_workers)

    ids = df[id_col].tolist()
    texts = df[text_col].tolist()
    n_chunks = math.ceil(len(df) / chunk_size)
    all_results: List[Dict[str, Any]] = []

    for chunk_idx in range(n_chunks):
        chain = await _get_chain(prompt_text, model_id=model_id)

        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(df))
        tasks = [
            _process_row_async(start + i, text, row_id, chain, semaphore, request_delay)
            for i, (text, row_id) in enumerate(zip(texts[start:end], ids[start:end]))
        ]
        chunk_results = await atqdm.gather(*tasks, desc=f"Chunk {chunk_idx + 1}/{n_chunks}")
        all_results.extend(chunk_results)

    return pd.DataFrame(all_results)


def run_inference(
    prompt_text: str,
    df: pd.DataFrame,
    id_col: str = config.ID_COL,
    text_col: str = config.TEXT_COL,
    model_id: str = config.TASK_MODEL_ID,
    **kwargs,
) -> pd.DataFrame:
    """Synchronous wrapper around run_inference_async for callers outside an event loop."""
    return asyncio.run(run_inference_async(prompt_text, df, id_col, text_col, model_id, **kwargs))


async def run_inference_many_async(
    prompt_texts: List[str],
    df: pd.DataFrame,
    id_col: str = config.ID_COL,
    text_col: str = config.TEXT_COL,
    model_id: str = config.TASK_MODEL_ID,
    max_workers: int = MAX_WORKERS,
    request_delay: float = REQUEST_DELAY,
    chunk_size: int = CHUNK_SIZE,
) -> List[pd.DataFrame]:
    """Run several candidate prompts over the same df concurrently, sharing one worker pool/semaphore."""
    semaphore = asyncio.Semaphore(max_workers)

    ids = df[id_col].tolist()
    texts = df[text_col].tolist()
    n_chunks = math.ceil(len(df) / chunk_size)
    # One result list per candidate prompt, filled chunk by chunk.
    all_results: List[List[Dict[str, Any]]] = [[] for _ in prompt_texts]

    for chunk_idx in range(n_chunks):
        chains = [await _get_chain(p, model_id=model_id) for p in prompt_texts]

        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(df))
        chunk_texts = texts[start:end]
        chunk_ids = ids[start:end]

        tasks = []
        task_owner: List[int] = []  # candidate index each task belongs to, for routing results back
        for cand_idx, chain in enumerate(chains):
            for i, (text, row_id) in enumerate(zip(chunk_texts, chunk_ids)):
                tasks.append(_process_row_async(start + i, text, row_id, chain, semaphore, request_delay))
                task_owner.append(cand_idx)

        desc = f"Chunk {chunk_idx + 1}/{n_chunks} ({len(prompt_texts)} candidates)"
        chunk_results = await atqdm.gather(*tasks, desc=desc)

        for cand_idx, res in zip(task_owner, chunk_results):
            all_results[cand_idx].append(res)

    return [pd.DataFrame(r) for r in all_results]


def run_inference_many(
    prompt_texts: List[str],
    df: pd.DataFrame,
    id_col: str = config.ID_COL,
    text_col: str = config.TEXT_COL,
    model_id: str = config.TASK_MODEL_ID,
    **kwargs,
) -> List[pd.DataFrame]:
    """Synchronous wrapper: evaluates multiple candidate prompts on the same df concurrently."""
    return asyncio.run(run_inference_many_async(prompt_texts, df, id_col, text_col, model_id, **kwargs))



async def run_inference_multi_df_async(
    prompt_text: str,
    dfs: List[pd.DataFrame],
    id_col: str = config.ID_COL,
    text_col: str = config.TEXT_COL,
    model_id: str = config.TASK_MODEL_ID,
    max_workers: int = MAX_WORKERS,
    request_delay: float = REQUEST_DELAY,
) -> List[pd.DataFrame]:
    """Run one candidate prompt over several dataframes in a single batched call, sharing one worker pool.

    Useful for merging e.g. a minibatch eval and a Pareto-set eval for the same candidate into one
    round-trip instead of two sequential calls. Assumes the combined row count fits in one chunk.
    """
    semaphore = asyncio.Semaphore(max_workers)

    chain = await _get_chain(prompt_text, model_id=model_id)

    tasks = []
    task_owner: List[int] = []  # source dataframe index each task belongs to
    global_idx = 0
    for df_idx, df in enumerate(dfs):
        for text, row_id in zip(df[text_col].tolist(), df[id_col].tolist()):
            tasks.append(_process_row_async(global_idx, text, row_id, chain, semaphore, request_delay))
            task_owner.append(df_idx)
            global_idx += 1

    desc = f"Multi-df eval ({len(dfs)} sets, {len(tasks)} rows)"
    results = await atqdm.gather(*tasks, desc=desc)

    all_results: List[List[Dict[str, Any]]] = [[] for _ in dfs]
    for df_idx, res in zip(task_owner, results):
        all_results[df_idx].append(res)

    return [pd.DataFrame(r) for r in all_results]


def run_inference_multi_df(
    prompt_text: str,
    dfs: List[pd.DataFrame],
    id_col: str = config.ID_COL,
    text_col: str = config.TEXT_COL,
    model_id: str = config.TASK_MODEL_ID,
    **kwargs,
) -> List[pd.DataFrame]:
    """Synchronous wrapper: evaluates one candidate prompt on several dataframes in one batched call."""
    return asyncio.run(run_inference_multi_df_async(prompt_text, dfs, id_col, text_col, model_id, **kwargs))


if __name__ == "__main__":
    import dataset as ds

    with open(config.SEED_PROMPT_PATH, "r") as f:
        seed_prompt = f.read()

    raw_df = ds.load_raw_dataset()
    start_time = time.time()
    preds = run_inference(seed_prompt, raw_df)
    print(f"Time Elapsed: {round(time.time() - start_time, 2)}s")

    os.makedirs(config.BASELINE_DIR, exist_ok=True)
    out_path = os.path.join(config.BASELINE_DIR, "seed_prompt_predictions.parquet")
    preds.to_parquet(out_path, index=False)
    print(f"Saved baseline predictions to {out_path}")