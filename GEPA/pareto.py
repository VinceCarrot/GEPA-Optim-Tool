"""Pareto-frontier bookkeeping for the GEPA genetic evolution loop.

Tracks a candidate x instance score matrix over a fixed "Pareto set" of
training examples. A candidate is on the frontier if it achieves the best
known score on at least one instance. Sampling a parent candidate to mutate
is weighted toward frontier members that win on the most instances, matching
the paper's idea of favoring complementary strong candidates rather than
always mutating a single running best.
"""

import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional


class ParetoArchive:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.state_path = os.path.join(run_dir, "pareto_state.json")
        # scores[instance_id][candidate_id] = score (higher is better, e.g. 1.0 correct / 0.0 wrong)
        self.scores: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._mutation_counts: Dict[str, int] = defaultdict(int)

    def add_candidate_scores(self, candidate_id: str, instance_scores: Dict[str, float]) -> None:
        for instance_id, score in instance_scores.items():
            self.scores[instance_id][candidate_id] = score

    def compute_frontier(self) -> Dict[str, int]:
        """Return {candidate_id: num_instances_won} for candidates on the Pareto frontier."""
        wins = defaultdict(int)
        for instance_id, candidate_scores in self.scores.items():
            if not candidate_scores:
                continue
            best_score = max(candidate_scores.values())
            for candidate_id, score in candidate_scores.items():
                if score == best_score:
                    wins[candidate_id] += 1
        return dict(wins)

    def record_mutation(self, candidate_id: str) -> None:
        self._mutation_counts[candidate_id] += 1

    def sample_parent(self, fallback_candidate_id: str, valid_ids: Optional[set] = None) -> str:
        """Sample a candidate id from the frontier, weighted by instances won, penalizing over-mutated parents."""
        wins = self.compute_frontier()
        if valid_ids is not None:
            wins = {c: w for c, w in wins.items() if c in valid_ids}
        if not wins:
            return fallback_candidate_id
        candidates: List[str] = list(wins.keys())
        # Halve weight for each prior mutation to encourage exploring other frontier members
        weights: List[float] = [
            max(1.0, wins[c] / (1 + self._mutation_counts.get(c, 0)))
            for c in candidates
        ]
        return random.choices(candidates, weights=weights, k=1)[0]

    def save(self) -> None:
        with open(self.state_path, "w") as f:
            json.dump({"scores": self.scores, "mutation_counts": self._mutation_counts}, f)

    def load(self) -> None:
        if os.path.exists(self.state_path):
            with open(self.state_path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict) and "scores" in data:
                self.scores = defaultdict(dict, data["scores"])
                self._mutation_counts = defaultdict(int, data.get("mutation_counts", {}))
            else:
                # backward compat: old format stored scores directly
                self.scores = defaultdict(dict, data)