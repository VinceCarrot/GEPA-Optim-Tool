"""GEPA optimization loop (generic binary classification).

One round = select parent -> error analysis -> clustering -> mutation -> evaluate children
-> accept/reject. Everything the UI shows is emitted to <run_dir>/events.jsonl.

Phases (when the objective has constraints):
  A: no candidate in the pool satisfies every constraint yet. Parent = least-violating
     candidate; reflection focuses on the violating error side.
  B: at least one feasible candidate exists. Parent = Pareto sample over feasible pool.
strategy="legacy" keeps the old behaviour (Pareto sampling over the whole pool, legacy rule).
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
import traceback
from typing import Dict, List, Optional

from . import evaluator as ev
from . import inference as infer
from . import llm as llm_mod
from . import objective as obj
from . import protect
from . import reflection as refl
from .dataset import load_splits, sample_rows
from .spec import RunConfig, dump_config
from .store import Candidate, CandidateStore, EventLog, ParetoArchive

logger = logging.getLogger("gepa_tool.optimizer")


class Optimizer:
    def __init__(self, cfg: RunConfig, run_dir: str):
        self.cfg = cfg
        self.run_dir = run_dir
        os.makedirs(os.path.join(run_dir, "predictions"), exist_ok=True)
        self.events = EventLog(run_dir)
        self.store = CandidateStore(run_dir)
        self.archive = ParetoArchive(run_dir, cfg.random_seed)
        self.pool: Dict[str, Candidate] = {}
        self.bundles: Dict[str, Dict] = {}
        self.history: List[dict] = []
        self.stopped = False

    # ------------------------------------------------------------------ helpers

    def _pred_path(self, cid: str, tag: str) -> str:
        return os.path.join(self.run_dir, "predictions", f"{cid}_{tag}.parquet")

    def _rank_key(self, cid: str):
        b = self.bundles[cid]
        return (b["feasible"], -b["violation"] if not b["feasible"] else 0.0, b["metrics"][self.cfg.objective.maximize])

    def best_id(self) -> str:
        return max(self.pool, key=self._rank_key)

    def any_feasible(self) -> bool:
        return any(self.bundles[c]["feasible"] for c in self.pool)

    def phase(self) -> str:
        if not self.cfg.objective.constraints:
            return "B"
        return "B" if self.any_feasible() else "A"

    def select_parent(self, phase: str, seed_id: str) -> str:
        if self.cfg.acceptance.strategy == "legacy":
            return self.archive.sample(seed_id, set(self.pool))
        if phase == "A":
            # least violation first, then higher objective; light penalty for repeatedly mutated parents
            def key(cid):
                b = self.bundles[cid]
                return (-b["violation"], b["metrics"][self.cfg.objective.maximize],
                        -self.archive.mutation_counts.get(cid, 0))
            return max(self.pool, key=key)
        feasible = {c for c in self.pool if self.bundles[c]["feasible"]}
        return self.archive.sample(seed_id, feasible or set(self.pool))

    def _progress_cb(self, ids: List[str], round_no: int):
        def cb(pi: int, done: int, total: int, n_err: int):
            self.events.emit("eval_progress", round=round_no, candidate_id=ids[pi],
                             done=done, total=total, n_errors=n_err)
        return cb

    def _best_summary(self) -> dict:
        bid = self.best_id()
        return {"best_id": bid, "best": ev.summarize(self.bundles[bid])}

    # ------------------------------------------------------------------ main

    async def run(self) -> None:
        cfg = self.cfg
        t0 = time.time()
        llm_mod.init_models(cfg.models)
        protect.parse(cfg.seed_prompt)  # fail fast on malformed markers

        train_df, test_df = load_splits(cfg.data, out_dir=os.path.join(self.run_dir, "splits"))
        scoring = sample_rows(train_df, cfg.budget.scoring_set_size, seed=cfg.random_seed)
        self.events.emit(
            "run_start", name=cfg.name, total_rounds=cfg.budget.max_rounds,
            proposals_per_round=cfg.budget.proposals_per_round,
            objective=cfg.objective.model_dump(), objective_text=obj.describe_objective(cfg.objective),
            acceptance=cfg.acceptance.model_dump(), patience=cfg.patience,
            n_train=len(train_df), n_test=len(test_df), n_scoring=len(scoring),
            n_scoring_positive=int(scoring["target"].sum()),
            positive_label=cfg.task.positive_label, negative_label=cfg.task.negative_label,
            pid=os.getpid(),
        )

        # ---- seed
        seed = Candidate(prompt_text=cfg.seed_prompt, candidate_id="seed", rationale="seed prompt")
        self.store.save(seed)
        self.events.emit("candidate_proposed", round=0, candidate_id="seed", parent_id=None,
                         target_cluster="", rationale="seed prompt", generation=0)
        self.events.emit("stage", round=0, stage="evaluate", status="start")
        preds = await infer.run_one_async(cfg, seed.prompt_text, scoring, self._progress_cb(["seed"], 0))
        bundle = ev.evaluate(cfg, preds, scoring, save_path=self._pred_path("seed", "scoring_round0"))
        self.events.emit("stage", round=0, stage="evaluate", status="done")
        self.bundles["seed"] = bundle
        self.pool["seed"] = seed
        self.archive.add("seed", ev.per_instance_correct(cfg, bundle["merged"]))
        self.archive.save()
        self.events.emit("candidate_decision", round=0, candidate_id="seed", parent_id=None,
                         accepted=True, reason_code="seed", metrics=ev.summarize(bundle))

        feasible_alerted = False
        current_phase = self.phase()
        self.events.emit("phase", round=0, phase=current_phase,
                         note="constraints already met by seed" if current_phase == "B" else "seed misses a constraint")

        for r in range(1, cfg.budget.max_rounds + 1):
            rt = time.time()
            phase = self.phase()
            if phase != current_phase:
                self.events.emit("phase", round=r, phase=phase, note="first feasible candidate found",
                                 candidate_id=self.best_id())
                current_phase = phase
            parent_id = self.select_parent(phase, "seed")
            self.archive.record_mutation(parent_id)
            parent = self.pool[parent_id]
            pb = self.bundles[parent_id]
            self.events.emit("round_start", round=r, total_rounds=cfg.budget.max_rounds, phase=phase,
                             parent_id=parent_id, parent_metrics=ev.summarize(pb))

            mix = refl.choose_error_mix(cfg, phase, pb["metrics"])
            errors = ev.sample_errors(cfg, pb["merged"], mix["fp"], mix["fn"], seed=cfg.random_seed + r - 1)
            self.events.emit("error_sample", round=r, n_fp=len(errors["false_positives"]),
                             n_fn=len(errors["false_negatives"]))

            def on_stage(stage, status, _r=r):
                self.events.emit("stage", round=_r, stage=stage, status=status)

            try:
                result = await refl.reflect(cfg, parent.prompt_text, pb, errors, self.history, phase, on_stage)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("reflection failed in round %d: %s", r, e)
                self.events.emit("round_error", round=r, error=str(e)[:2000])
                self.events.emit("round_end", round=r, duration=time.time() - rt, failed=True,
                                 **self._best_summary())
                await asyncio.sleep(30)
                continue

            self.events.emit("reflection", round=r, parent_id=parent_id, diagnosis=result.diagnosis,
                             clusters=[c.model_dump() for c in result.clusters], n_analyses=result.n_analyses)
            for rej in result.rejected:
                self.events.emit("proposal_rejected", round=r, reason_code="protected_violation", **rej)

            children: List[Candidate] = []
            for p in result.proposals:
                if not protect.verify_preserved(parent.prompt_text, p.new_prompt):
                    self.events.emit("proposal_rejected", round=r, reason_code="protected_violation",
                                     target_cluster=p.target_cluster, rationale=p.rationale,
                                     problems=["protected text changed after restore"])
                    continue
                c = Candidate(prompt_text=p.new_prompt, parent_id=parent_id, generation=parent.generation + 1,
                              round=r, target_cluster=p.target_cluster, rationale=p.rationale)
                self.store.save(c)
                children.append(c)
                self.events.emit("candidate_proposed", round=r, candidate_id=c.candidate_id, parent_id=parent_id,
                                 target_cluster=c.target_cluster, rationale=c.rationale, generation=c.generation)

            if not children:
                self.events.emit("round_end", round=r, duration=time.time() - rt, failed=True,
                                 **self._best_summary())
                continue

            # ---- evaluate children
            ids = [c.candidate_id for c in children]
            self.events.emit("stage", round=r, stage="evaluate", status="start")
            prog = self._progress_cb(ids, r)
            if cfg.budget.parallel_children:
                preds_list = await infer.run_many_async(cfg, [c.prompt_text for c in children], scoring, prog)
            else:
                preds_list = []
                for k, c in enumerate(children):
                    one = await infer.run_one_async(
                        cfg, c.prompt_text, scoring,
                        lambda _pi, d, t, e, _k=k: prog(_k, d, t, e))
                    preds_list.append(one)
            self.events.emit("stage", round=r, stage="evaluate", status="done")

            # ---- decide
            self.events.emit("stage", round=r, stage="decision", status="start")
            for c, preds in zip(children, preds_list):
                cb = ev.evaluate(cfg, preds, scoring, save_path=self._pred_path(c.candidate_id, f"scoring_round{r}"))
                self.bundles[c.candidate_id] = cb
                accepted, reason = obj.decide(cfg.objective, cfg.acceptance, pb["metrics"], cb["metrics"])
                if cb["metrics"]["n"] == 0:
                    accepted, reason = False, "no_valid_predictions"
                pm, cm = pb["metrics"], cb["metrics"]
                self.history.append({
                    "round": r, "rationale": c.rationale,
                    "parent_precision": pm["precision"], "child_precision": cm["precision"],
                    "parent_recall": pm["recall"], "child_recall": cm["recall"],
                    "parent_score": pb["optimization_score"], "child_score": cb["optimization_score"],
                    "accepted": accepted,
                })
                if accepted:
                    self.pool[c.candidate_id] = c
                    self.archive.add(c.candidate_id, ev.per_instance_correct(cfg, cb["merged"]))
                self.events.emit("candidate_decision", round=r, candidate_id=c.candidate_id, parent_id=parent_id,
                                 accepted=accepted, reason_code=reason, reason=obj.REASONS.get(reason, reason),
                                 metrics=ev.summarize(cb), parent_metrics=ev.summarize(pb))
            self.archive.save()
            self.events.emit("stage", round=r, stage="decision", status="done")

            if cfg.objective.constraints and not self.any_feasible() and r >= cfg.patience and not feasible_alerted:
                feasible_alerted = True
                b = self.bundles[self.best_id()]
                self.events.emit("feasibility_alert", round=r, patience=cfg.patience,
                                 best_id=self.best_id(), best=ev.summarize(b),
                                 message=f"No candidate meets all constraints after {r} rounds; the run continues.")

            self.events.emit("round_end", round=r, duration=time.time() - rt, failed=False, **self._best_summary())

        await self._finish(test_df, t0, status="completed")

    async def _finish(self, test_df, t0: float, status: str) -> None:
        cfg = self.cfg
        bid = self.best_id()
        best = self.pool[bid]
        with open(os.path.join(self.run_dir, "best_prompt.md"), "w") as f:
            f.write(best.prompt_text)
        result = {"best_candidate_id": bid, "scoring": ev.summarize(self.bundles[bid]),
                  "seed_scoring": ev.summarize(self.bundles["seed"]), "status": status}
        if status == "completed" and cfg.budget.final_test_eval and len(test_df):
            self.events.emit("stage", round=cfg.budget.max_rounds + 1, stage="final_test", status="start")
            cands = [bid] if bid == "seed" else [bid, "seed"]
            preds_list = await infer.run_many_async(
                cfg, [self.pool[c].prompt_text for c in cands], test_df,
                self._progress_cb([f"{c}@test" for c in cands], cfg.budget.max_rounds + 1))
            for c, p in zip(cands, preds_list):
                tb = ev.evaluate(cfg, p, test_df, save_path=self._pred_path(c, "final_test"))
                result["test" if c == bid else "seed_test"] = ev.summarize(tb)
            if bid == "seed":
                result["seed_test"] = result["test"]
            self.events.emit("stage", round=cfg.budget.max_rounds + 1, stage="final_test", status="done")
        import json
        with open(os.path.join(self.run_dir, "final_result.json"), "w") as f:
            json.dump(result, f, indent=2, default=str)
        self.events.emit("run_end", duration=time.time() - t0, **result)

    def write_stopped(self, t0: float) -> None:
        """Called after cancellation: persist best-so-far without any further LLM calls."""
        try:
            if self.pool:
                bid = self.best_id()
                with open(os.path.join(self.run_dir, "best_prompt.md"), "w") as f:
                    f.write(self.pool[bid].prompt_text)
                import json
                result = {"best_candidate_id": bid, "scoring": ev.summarize(self.bundles[bid]),
                          "seed_scoring": ev.summarize(self.bundles["seed"]), "status": "stopped"}
                with open(os.path.join(self.run_dir, "final_result.json"), "w") as f:
                    json.dump(result, f, indent=2, default=str)
                self.events.emit("run_end", duration=time.time() - t0, **result)
            else:
                self.events.emit("run_end", status="stopped", duration=time.time() - t0)
            self.archive.save()
        except Exception:
            logger.exception("failed to write stopped state")


def _setup_logging(run_dir: str) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(run_dir, "engine.log"))
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.handlers = [fh, sh]
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def run(cfg: RunConfig, run_dir: str) -> int:
    """Entry point used by the CLI. Returns a process exit code."""
    os.makedirs(run_dir, exist_ok=True)
    _setup_logging(run_dir)
    dump_config(cfg, os.path.join(run_dir, "run_config.yaml"))
    opt = Optimizer(cfg, run_dir)
    t0 = time.time()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    main = loop.create_task(opt.run())

    def _on_signal(signum):
        logger.warning("received signal %s, cancelling all work", signum)
        opt.stopped = True
        main.cancel()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _on_signal, sig)
        except NotImplementedError:  # pragma: no cover (non-POSIX)
            pass

    code = 0
    try:
        loop.run_until_complete(main)
    except asyncio.CancelledError:
        opt.write_stopped(t0)
        code = 130
    except Exception as e:
        logger.error("run failed: %s", e)
        opt.events.emit("run_end", status="failed", error=str(e)[:4000], traceback=traceback.format_exc()[-8000:],
                        duration=time.time() - t0)
        code = 1
    finally:
        pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
        for t in pending:
            t.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()
    return code          