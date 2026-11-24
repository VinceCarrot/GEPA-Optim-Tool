"""Clean the final text of reflection-stage LLM calls before it is sent.

Applied to the fully formatted error-analysis and clustering payloads (task text, prompt,
input, model reasoning, analyses -- everything in the request):
  1. PII redaction with ee_utils.redaction (mode="masking")
  2. the run's profanity word list (data.profanity_wordlist), matched words -> "**"

Redaction fails closed: if ee_utils is not installed the call raises instead of sending
unredacted text.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

from .dataset import _profanity_pattern
from .spec import RunConfig


def _redact_text(value: Any) -> str:
    from ee_utils.redaction import RedactionConfig, redact_text

    text = "" if value is None else str(value)
    return redact_text(text, RedactionConfig(mode="masking")).text


@lru_cache(maxsize=8)
def _cached_pattern(wordlist: Optional[str]):
    return _profanity_pattern(wordlist)


def mask_profanity(text: str, wordlist: Optional[str]) -> str:
    pattern = _cached_pattern(wordlist)
    return pattern.sub("**", text) if pattern is not None else text


def sanitize_payload(cfg: RunConfig, text: str) -> str:
    """Redact PII, then mask words from the run's profanity list."""
    return mask_profanity(_redact_text(text), cfg.data.profanity_wordlist)
