"""Protected prompt sections.

Users mark sections the optimizer must never change:

    <!-- PROTECTED:output_format -->
    ...text...
    <!-- /PROTECTED -->

Before mutation each block is replaced with a token `[[PROTECTED:name]]`; the mutation LLM
must keep every token exactly once and in order. Afterwards the tokens are swapped back
for the original text, so protected wording is byte-for-byte preserved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

_BLOCK_RE = re.compile(
    r"<!--\s*PROTECTED:(?P<name>[A-Za-z0-9_\-]+)\s*-->(?P<body>.*?)<!--\s*/PROTECTED\s*-->",
    re.DOTALL,
)
_OPEN_RE = re.compile(r"<!--\s*PROTECTED:([A-Za-z0-9_\-]+)\s*-->")
_CLOSE_RE = re.compile(r"<!--\s*/PROTECTED\s*-->")
_TOKEN_RE = re.compile(r"\[\[PROTECTED:([A-Za-z0-9_\-]+)\]\]")


class ProtectedSectionError(ValueError):
    pass


@dataclass
class ProtectedBlock:
    name: str
    full: str   # including the comment markers, restored verbatim
    body: str


def token(name: str) -> str:
    return f"[[PROTECTED:{name}]]"


def parse(prompt: str) -> List[ProtectedBlock]:
    """Return protected blocks in order. Raises on malformed / duplicate / nested markers."""
    opens = _OPEN_RE.findall(prompt)
    closes = _CLOSE_RE.findall(prompt)
    if len(opens) != len(closes):
        raise ProtectedSectionError(
            f"Unbalanced PROTECTED markers: {len(opens)} opening vs {len(closes)} closing"
        )
    blocks = [ProtectedBlock(m.group("name"), m.group(0), m.group("body")) for m in _BLOCK_RE.finditer(prompt)]
    if len(blocks) != len(opens):
        raise ProtectedSectionError("PROTECTED markers are nested or out of order")
    names = [b.name for b in blocks]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ProtectedSectionError(f"Duplicate PROTECTED section names: {sorted(dupes)}")
    for b in blocks:
        if _OPEN_RE.search(b.body):
            raise ProtectedSectionError(f"Nested PROTECTED marker inside section {b.name}")
    return blocks


def mask(prompt: str) -> Tuple[str, Dict[str, str]]:
    """Replace each protected block with its token. Returns (masked_prompt, {name: original_block})."""
    blocks = parse(prompt)
    masked = prompt
    originals: Dict[str, str] = {}
    for b in blocks:
        masked = masked.replace(b.full, token(b.name), 1)
        originals[b.name] = b.full
    return masked, originals


def validate_tokens(masked_candidate: str, originals: Dict[str, str]) -> List[str]:
    """Return a list of problems (empty == valid): each token exactly once, same order, no extras."""
    problems = []
    expected = list(originals.keys())
    found = _TOKEN_RE.findall(masked_candidate)
    for name in expected:
        cnt = found.count(name)
        if cnt == 0:
            problems.append(f"missing protected section {name}")
        elif cnt > 1:
            problems.append(f"protected section {name} appears {cnt} times")
    extras = sorted(set(found) - set(expected))
    if extras:
        problems.append(f"unknown protected tokens {extras}")
    if not problems and found != expected:
        problems.append("protected sections were reordered")
    if _OPEN_RE.search(masked_candidate) or _CLOSE_RE.search(masked_candidate):
        problems.append("candidate introduced raw PROTECTED markers")
    return problems


def unmask(masked_candidate: str, originals: Dict[str, str]) -> str:
    out = masked_candidate
    for name, full in originals.items():
        out = out.replace(token(name), full, 1)
    return out


def verify_preserved(original_prompt: str, candidate_prompt: str) -> bool:
    """True if every protected block of the original appears verbatim, in order, in the candidate."""
    orig = [b.full for b in parse(original_prompt)]
    try:
        cand = [b.full for b in parse(candidate_prompt)]
    except ProtectedSectionError:
        return False
    return orig == cand


def strip_markers(prompt: str) -> str:
    """Remove the HTML comment markers (keep the text) before sending a prompt to the task model."""
    return _CLOSE_RE.sub("", _OPEN_RE.sub("", prompt))