"""Run-directory persistence: event log, candidate store, Pareto archive."""

from __future__ import annotations

import json
import os
import random
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Event log (events.jsonl) -- the single source of truth the UI replays/tails
# ---------------------------------------------------------------------------

class EventLog:
    def __init__(self, run_dir: str):
        self.path = os.path.join(run_dir, "events.jsonl")
        self._lock = threading.Lock()
        self._seq = 0
        if os.path.exists(self.path):
            with open(self.path) as f:
                for line in f:
                    if line.strip():
                        try:
                            self._seq = max(self._seq, json.loads(line).get("seq", 0))
                        except json.JSONDecodeError:
                            pass

    def emit(self, type_: str, **fields) -> dict:
        with self._lock:
            self._seq += 1
            ev = {"seq": self._seq, "ts": time.time(), "type": type_, **fields}
            with open(self.path, "a") as f:
                f.write(json.dumps(ev, default=_json_default) + "\n")
                f.flush()
        return ev


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
    except ImportError:
        pass
    return str(o)


def read_events(run_dir: str, after_seq: int = 0) -> List[dict]:
    path = os.path.join(run_dir, "events.jsonl")
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue  # partially written last line
            if ev.get("seq", 0) > after_seq:
                out.append(ev)
    return out


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    prompt_text: str
    parent_id: Optional[str] = None
    generation: int = 0
    round: int = 0
    target_cluster: str = ""
    rationale: str = ""
    candidate_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    created_at: float = field(default_factory=time.time)


class CandidateStore:
    """Every proposed candidate (accepted or not) is saved so the UI can show its prompt."""

    def __init__(self, run_dir: str):
        self.dir = os.path.join(run_dir, "candidates")
        self.index = os.path.join(run_dir, "candidates.jsonl")
        os.makedirs(self.dir, exist_ok=True)

    def save(self, c: Candidate) -> str:
        path = os.path.join(self.dir, f"{c.candidate_id}.md")
        with open(path, "w") as f:
            f.write(c.prompt_text)
        meta = asdict(c)
        meta.pop("prompt_text")
        with open(self.index, "a") as f:
            f.write(json.dumps(meta) + "\n")
        return path

    def prompt(self, candidate_id: str) -> Optional[str]:
        path = os.path.join(self.dir, f"{candidate_id}.md")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return f.read()


# ---------------------------------------------------------------------------
# Pareto archive (same semantics as GEPA/pareto.py)
# ---------------------------------------------------------------------------

class ParetoArchive:
    def __init__(self, run_dir: str, seed: int):
        self.state_path = os.path.join(run_dir, "pareto_state.json")
        self.scores: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.mutation_counts: Dict[str, int] = defaultdict(int)
        self.rng = random.Random(seed)

    def add(self, candidate_id: str, instance_scores: Dict[str, float]) -> None:
        for iid, s in instance_scores.items():
            self.scores[iid][candidate_id] = s

    def frontier(self) -> Dict[str, int]:
        wins: Dict[str, int] = defaultdict(int)
        for cand_scores in self.scores.values():
            if not cand_scores:
                continue
            best = max(cand_scores.values())
            for cid, s in cand_scores.items():
                if s == best:
                    wins[cid] += 1
        return dict(wins)

    def record_mutation(self, candidate_id: str) -> None:
        self.mutation_counts[candidate_id] += 1

    def sample(self, fallback: str, valid_ids: Optional[set] = None) -> str:
        wins = self.frontier()
        if valid_ids is not None:
            wins = {c: w for c, w in wins.items() if c in valid_ids}
        if not wins:
            return fallback
        cands = list(wins)
        weights = [max(1.0, wins[c] / (1 + self.mutation_counts.get(c, 0))) for c in cands]
        return self.rng.choices(cands, weights=weights, k=1)[0]

    def save(self) -> None:
        with open(self.state_path, "w") as f:
            json.dump({"scores": self.scores, "mutation_counts": self.mutation_counts}, f)