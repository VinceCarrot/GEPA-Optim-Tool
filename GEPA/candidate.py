"""Prompt candidate representation used by the genetic evolution loop."""

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional

import config


@dataclass
class PromptCandidate:
    prompt_text: str
    parent_id: Optional[str] = None
    generation: int = 0
    mutation_rationale: str = ""
    candidate_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "PromptCandidate":
        return PromptCandidate(**d)


class CandidateStore:
    """Persists candidate prompt text and metadata for one optimizer run."""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.candidates_dir = os.path.join(run_dir, "candidates")
        self.history_path = os.path.join(run_dir, "candidate_history.jsonl")
        os.makedirs(self.candidates_dir, exist_ok=True)

    def save(self, candidate: PromptCandidate) -> None:
        prompt_path = os.path.join(self.candidates_dir, f"{candidate.candidate_id}.md")
        with open(prompt_path, "w") as f:
            f.write(candidate.prompt_text)
        with open(self.history_path, "a") as f:
            f.write(json.dumps(candidate.to_dict()) + "\n")

    def load_all(self):
        candidates = []
        if not os.path.exists(self.history_path):
            return candidates
        with open(self.history_path, "r") as f:
            for line in f:
                if line.strip():
                    candidates.append(PromptCandidate.from_dict(json.loads(line)))
        return candidates


def seed_candidate() -> PromptCandidate:
    with open(config.SEED_PROMPT_PATH, "r") as f:
        prompt_text = f.read()
    return PromptCandidate(
        prompt_text=prompt_text,
        parent_id=None,
        generation=0,
        mutation_rationale="seed prompt",
        candidate_id="seed",
    )