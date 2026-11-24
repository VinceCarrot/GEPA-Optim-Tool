"""Main genetic evolution loop for GEPA-style system-prompt optimization.

Implements, at lean scope, the paper's core cycle:
  1. select a candidate from the Pareto frontier
  2. execute it on a minibatch, capturing full traces
  3. reflect on the traces to diagnose failures
  4. mutate: propose revised prompts
  5. accept improved candidates into the pool and update the Pareto front
Repeats until the evaluation budget (MAX_ITERATIONS) is exhausted, then
locks in the best candidate and reports final test metrics.
"""

import argparse
import glob
import json
import os
import time
from datetime import datetime

import pandas as pd

import config
import dataset as ds
import evaluator as ev
import inference_run as infer
import reflection as refl
from candidate import CandidateStore, PromptCandidate, seed_candidate
from pareto import ParetoArchive


def predictions_path(run_dir: str, candidate_id: str, tag: str) -> str:
    preds_dir = os.path.join(run_dir, "predictions")
    os.makedirs(preds_dir, exist_ok=True)
    return os.path.join(preds_dir, f"{candidate_id}_{tag}.parquet")


def evaluate_candidate_on_df(candidate: PromptCandidate, df: pd.DataFrame, recall_target: float,
                              save_path: str = None) -> dict:
    preds = infer.run_inference(candidate.prompt_text, df)
    return ev.evaluate(preds, df, recall_target=recall_target, save_path=save_path)


def scores_from_preds(preds: pd.DataFrame, raw_df: pd.DataFrame) -> dict:
    merged = ev.merge_predictions(preds, raw_df)
    correct = (merged["llm_pred_flag"] == merged[config.TARGET_COL]).astype(int)
    return dict(zip(merged[config.ID_COL].astype(str), correct))


def per_instance_scores(candidate: PromptCandidate, df: pd.DataFrame) -> dict:
    preds = infer.run_inference(candidate.prompt_text, df)
    return scores_from_preds(preds, df)


def log_event(run_dir: str, event: dict) -> None:
    event["ts"] = time.time()
    with open(os.path.join(run_dir, "log.jsonl"), "a") as f:
        f.write(json.dumps(event, default=str) + "\n")


def load_log_events(run_dir: str) -> list:
    path = os.path.join(run_dir, "log.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_saved_scoring_preds(run_dir: str, candidate_id: str) -> pd.DataFrame:
    """Load a candidate's saved scoring-set predictions, stripped back to raw prediction columns."""
    files = sorted(glob.glob(os.path.join(run_dir, "predictions", f"{candidate_id}_scoring_iter*.parquet")))
    if not files:
        raise FileNotFoundError(f"No saved scoring predictions for candidate {candidate_id} in {run_dir}")
    saved = pd.read_parquet(files[-1])
    # evaluate() re-merges text and gold target from the scoring set; drop the saved copies to avoid duplicate columns.
    return saved.drop(columns=[c for c in (config.TEXT_COL, config.TARGET_COL) if c in saved.columns])


def run_optimization(resume_dir: str = None):
    train_df, dev_df, test_df = ds.load_splits()

    if resume_dir:
        run_dir = os.path.abspath(resume_dir)
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"Cannot resume: {run_dir} does not exist")
        print(f"Resuming run directory: {run_dir}")
    else:
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir = os.path.join(config.RUNS_DIR, run_id)
        os.makedirs(run_dir, exist_ok=True)
        print(f"Run directory: {run_dir}")

    store = CandidateStore(run_dir)
    archive = ParetoArchive(run_dir)

    pool = {}
    if resume_dir:
        for c in store.load_all():
            pool[c.candidate_id] = c
        archive.load()
        seed = next((c for c in pool.values() if c.parent_id is None), None)
        if seed is None:
            raise RuntimeError(f"Cannot resume: no seed candidate found in {run_dir}")
    else:
        seed = seed_candidate()
        pool[seed.candidate_id] = seed
        store.save(seed)

    # Single fixed scoring set drives every accept/reject decision, the Pareto per-instance
    # tracking, and the reflection error examples. Scoring each candidate once on the identical
    # set (and caching it) removes the cross-iteration minibatch variance that made parent/child
    # comparisons unstable.
    scoring_set = ds.sample_minibatch(train_df, config.SCORING_SET_SIZE, seed=config.RANDOM_SEED)
    score_cache = {}    # candidate_id -> full metrics bundle on the fixed scoring set
    merged_cache = {}   # candidate_id -> merged predictions, reused to rotate reflection error examples
    correct_cache = {}  # candidate_id -> per-instance correctness vector, pushed to the archive only on acceptance

    def score_candidate(candidate, preds, tag):
        metrics = ev.evaluate(
            preds, scoring_set, recall_target=config.RECALL_TARGET,
            save_path=predictions_path(run_dir, candidate.candidate_id, tag) if tag else None,
        )
        merged = ev.merge_predictions(preds, scoring_set)
        score_cache[candidate.candidate_id] = metrics
        merged_cache[candidate.candidate_id] = merged
        correct = (merged["llm_pred_flag"] == merged[config.TARGET_COL]).astype(int)
        correct_cache[candidate.candidate_id] = dict(zip(merged[config.ID_COL].astype(str), correct))
        return metrics

    # The archive is the evolutionary population eligible for exploration: only pool members enter it.
    def add_to_archive(candidate_id):
        archive.add_candidate_scores(candidate_id, correct_cache[candidate_id])

    dev_scores = {}  # candidate_id -> optimization_score on dev
    mutation_history = []  # compact per-child record fed to Stage 2 so failed strategies are not re-tried
    start_iteration = 0

    if resume_dir:
        # Rebuild per-candidate caches from saved predictions (no LLM calls); only pool members can be parents.
        for cid in pool:
            score_candidate(pool[cid], load_saved_scoring_preds(run_dir, cid), None)
        events = load_log_events(run_dir)
        eval_events = [e for e in events if e["event"] == "candidate_eval"]
        for e in events:
            if e["event"] == "dev_eval":
                dev_scores[e["candidate_id"]] = e["dev_score"]
        rationale_by_id = {cid: c.mutation_rationale for cid, c in pool.items()}
        for e in eval_events:
            parent_m = score_cache.get(e["parent_id"])
            mutation_history.append({
                "iteration": e["iteration"],
                "rationale": e.get("rationale") or rationale_by_id.get(e["candidate_id"])
                             or "(rationale not recorded before resume)",
                "parent_precision": parent_m["label_metrics"]["precision"] if parent_m else float("nan"),
                "child_precision": e["precision"],
                "parent_recall": parent_m["label_metrics"]["recall"] if parent_m else float("nan"),
                "child_recall": e["recall"],
                "parent_score": e["parent_score"],
                "child_score": e["optimization_score"],
                "accepted": e["accepted"],
            })
        start_iteration = max((e["iteration"] for e in eval_events), default=-1) + 1
        print(f"Restored {len(pool)} pool candidates and {len(mutation_history)} mutation-history entries; "
              f"continuing from iteration {start_iteration + 1}/{config.MAX_ITERATIONS}.")
    else:
        print("Evaluating seed candidate on the fixed scoring set...")
        seed_preds = infer.run_inference(seed.prompt_text, scoring_set)
        score_candidate(seed, seed_preds, "scoring_iter0")
        add_to_archive(seed.candidate_id)
        archive.save()

    for i in range(start_iteration, config.MAX_ITERATIONS):
        print(f"\n=== Iteration {i + 1}/{config.MAX_ITERATIONS} ===")
        parent_id = archive.sample_parent(fallback_candidate_id=seed.candidate_id, valid_ids=set(pool.keys()))
        archive.record_mutation(parent_id)
        parent = pool[parent_id]

        # Cached fixed-set score; parents are never re-evaluated, so comparisons stay stable.
        parent_metrics = score_cache[parent_id]
        log_event(run_dir, {
            "event": "parent_eval", "iteration": i, "candidate_id": parent_id,
            "optimization_score": parent_metrics["optimization_score"],
            "precision": parent_metrics["label_metrics"]["precision"],
            "recall": parent_metrics["label_metrics"]["recall"],
            "n": parent_metrics["label_metrics"]["n"],
            "n_positive": parent_metrics["label_metrics"]["n_positive"],
            "n_predicted_positive": parent_metrics["label_metrics"]["n_predicted_positive"],
        })

        try:
            # Stage 0 sees full-text FP/FN samples (rotated per iteration); Stage 2 sees mutation history.
            analysis_errors = ev.get_analysis_errors(
                merged_cache[parent_id], seed=config.RANDOM_SEED + i
            )
            reflection_output = refl.propose_mutations(
                parent.prompt_text, parent_metrics, recall_target=config.RECALL_TARGET,
                analysis_errors=analysis_errors, mutation_history=mutation_history,
            )
        except Exception as e:
            print(f"Reflection failed: {e}")
            log_event(run_dir, {"event": "reflection_error", "iteration": i, "error": str(e)})
            time.sleep(30)
            continue

        log_event(run_dir, {
            "event": "reflection", "iteration": i, "parent_id": parent_id,
            "diagnosis": reflection_output.diagnosis,
            "n_proposals": len(reflection_output.proposals),
        })

        children = [
            PromptCandidate(
                prompt_text=proposal.new_prompt,
                parent_id=parent_id,
                generation=parent.generation + 1,
                mutation_rationale=proposal.rationale,
            )
            for proposal in reflection_output.proposals
        ]
        proposal_by_child = {c.candidate_id: p for c, p in zip(children, reflection_output.proposals)}

        # Evaluate all proposed children on the fixed scoring set
        preds_list = [infer.run_inference(c.prompt_text, scoring_set) for c in children]

        for child, preds in zip(children, preds_list):
            # Caches the child's fixed-set metrics; its per-instance scores enter the archive only if accepted.
            child_metrics = score_candidate(child, preds, f"scoring_iter{i}")

            # Feasibility-first acceptance: RECALL_TARGET is a hard constraint and precision is
            # optimized within the feasible region. Epsilon only guards precision, never recall.
            parent_p = parent_metrics["label_metrics"]["precision"]
            parent_r = parent_metrics["label_metrics"]["recall"]
            child_p = child_metrics["label_metrics"]["precision"]
            child_r = child_metrics["label_metrics"]["recall"]
            target = config.RECALL_TARGET

            parent_feasible = parent_r >= target
            child_feasible = child_r >= target

            if parent_feasible:
                # Once feasible, never leave the feasible region; accept only precision-preserving moves.
                accepted = child_feasible and child_p >= parent_p - config.ACCEPTANCE_EPSILON
            elif child_feasible:
                # Crossing into the feasible region is always valuable.
                accepted = True
            elif child_r > parent_r:
                # Still infeasible, but moving toward the recall constraint.
                accepted = True
            elif abs(child_r - parent_r) < 1e-9:
                # Same recall: allow precision improvement.
                accepted = child_p >= parent_p
            else:
                # Moving farther from the constraint; reject.
                accepted = False
            mutation_history.append({
                "iteration": i,
                "rationale": proposal_by_child[child.candidate_id].rationale,
                "parent_precision": parent_metrics["label_metrics"]["precision"],
                "child_precision": child_metrics["label_metrics"]["precision"],
                "parent_recall": parent_metrics["label_metrics"]["recall"],
                "child_recall": child_metrics["label_metrics"]["recall"],
                "parent_score": parent_metrics["optimization_score"],
                "child_score": child_metrics["optimization_score"],
                "accepted": accepted,
            })
            log_event(run_dir, {
                "event": "candidate_eval", "iteration": i, "candidate_id": child.candidate_id,
                "parent_id": parent_id, "rationale": proposal_by_child[child.candidate_id].rationale,
                "optimization_score": child_metrics["optimization_score"],
                "parent_score": parent_metrics["optimization_score"], "accepted": accepted,
                "precision": child_metrics["label_metrics"]["precision"],
                "recall": child_metrics["label_metrics"]["recall"],
                "n": child_metrics["label_metrics"]["n"],
                "n_positive": child_metrics["label_metrics"]["n_positive"],
                "n_predicted_positive": child_metrics["label_metrics"]["n_predicted_positive"],
            })

            if accepted:
                pool[child.candidate_id] = child
                store.save(child)
                add_to_archive(child.candidate_id)
                print(f"Accepted candidate {child.candidate_id} "
                      f"(score {child_metrics['optimization_score']:.3f} vs parent {parent_metrics['optimization_score']:.3f}, eps {config.ACCEPTANCE_EPSILON})")
            else:
                print(f"Rejected candidate {child.candidate_id} "
                      f"(score {child_metrics['optimization_score']:.3f} < parent {parent_metrics['optimization_score']:.3f} - eps)")

        archive.save()

        if (config.ENABLE_DEV_SCORING and (i + 1) % config.DEV_EVAL_EVERY == 0):
            frontier = archive.compute_frontier()
            to_eval = [cid for cid in frontier if cid not in dev_scores and cid in pool]
            if to_eval:
                min_n = int(len(dev_df) * 0.9)
                dev_preds_list = infer.run_inference_many([pool[cid].prompt_text for cid in to_eval], dev_df)
                for cid, preds in zip(to_eval, dev_preds_list):
                    dev_metrics = ev.evaluate(
                        preds, dev_df, recall_target=config.RECALL_TARGET,
                        save_path=predictions_path(run_dir, cid, f"dev_eval_iter{i}"),
                    )
                    if dev_metrics["label_metrics"]["n"] < min_n:
                        print(f"Dev eval for {cid}: only {dev_metrics['label_metrics']['n']}/{len(dev_df)} rows survived, retrying after 30s...")
                        time.sleep(30)
                        retry_preds = infer.run_inference(pool[cid].prompt_text, dev_df)
                        dev_metrics = ev.evaluate(
                            retry_preds, dev_df, recall_target=config.RECALL_TARGET,
                            save_path=predictions_path(run_dir, cid, f"dev_eval_iter{i}_retry"),
                        )
                    if dev_metrics["label_metrics"]["n"] < min_n:
                        print(f"Dev eval for {cid}: still only {dev_metrics['label_metrics']['n']} rows after retry, skipping.")
                        log_event(run_dir, {"event": "dev_eval_skipped", "iteration": i, "candidate_id": cid,
                                             "n": dev_metrics["label_metrics"]["n"], "reason": "too_few_rows"})
                        continue
                    dev_scores[cid] = dev_metrics["optimization_score"]
                    log_event(run_dir, {"event": "dev_eval", "iteration": i, "candidate_id": cid,
                                         "dev_score": dev_scores[cid],
                                         "precision": dev_metrics["label_metrics"]["precision"],
                                         "recall": dev_metrics["label_metrics"]["recall"],
                                         "n": dev_metrics["label_metrics"]["n"],
                                         "n_positive": dev_metrics["label_metrics"]["n_positive"],
                                         "n_predicted_positive": dev_metrics["label_metrics"]["n_predicted_positive"]})

    # # Final dev evaluation of every candidate in the pool not yet scored on dev.
    # unscored = [cid for cid in pool if cid not in dev_scores]
    # if unscored:
    #     min_n = int(len(dev_df) * 0.9)
    #     dev_preds_list = infer.run_inference_many([pool[cid].prompt_text for cid in unscored], dev_df)
    #     for cid, preds in zip(unscored, dev_preds_list):
    #         dev_metrics = ev.evaluate(
    #             preds, dev_df, recall_target=config.RECALL_TARGET,
    #             save_path=predictions_path(run_dir, cid, "dev_eval_final"),
    #         )
    #         if dev_metrics["label_metrics"]["n"] < min_n:
    #             print(f"Final dev eval for {cid}: only {dev_metrics['label_metrics']['n']}/{len(dev_df)} rows, retrying after 30s...")
    #             time.sleep(30)
    #             retry_preds = infer.run_inference(pool[cid].prompt_text, dev_df)
    #             dev_metrics = ev.evaluate(
    #                 retry_preds, dev_df, recall_target=config.RECALL_TARGET,
    #                 save_path=predictions_path(run_dir, cid, "dev_eval_final_retry"),
    #             )
    #         dev_scores[cid] = dev_metrics["optimization_score"]

    # best_id = max(dev_scores, key=dev_scores.get)
    # best_candidate = pool[best_id]
    # print(f"\nBest candidate on dev: {best_id} (score={dev_scores[best_id]:.3f})")
# ---------------------------------------------------------------------------
# Final candidate selection
# ---------------------------------------------------------------------------

    if config.ENABLE_DEV_SCORING:
        # Final dev evaluation of every candidate in the pool not yet scored on dev.
        unscored = [cid for cid in pool if cid not in dev_scores]

        if unscored:
            min_n = int(len(dev_df) * 0.9)

            dev_preds_list = infer.run_inference_many(
                [pool[cid].prompt_text for cid in unscored],
                dev_df,
            )

            for cid, preds in zip(unscored, dev_preds_list):
                dev_metrics = ev.evaluate(
                    preds,
                    dev_df,
                    recall_target=config.RECALL_TARGET,
                    save_path=predictions_path(
                        run_dir,
                        cid,
                        "dev_eval_final",
                    ),
                )

                # Retry if too many dev rows failed inference.
                if dev_metrics["label_metrics"]["n"] < min_n:
                    print(
                        f"Final dev eval for {cid}: only "
                        f"{dev_metrics['label_metrics']['n']}/{len(dev_df)} rows, "
                        f"retrying after 30s..."
                    )

                    time.sleep(30)

                    retry_preds = infer.run_inference(
                        pool[cid].prompt_text,
                        dev_df,
                    )

                    dev_metrics = ev.evaluate(
                        retry_preds,
                        dev_df,
                        recall_target=config.RECALL_TARGET,
                        save_path=predictions_path(
                            run_dir,
                            cid,
                            "dev_eval_final_retry",
                        ),
                    )

                dev_scores[cid] = dev_metrics["optimization_score"]  

        # Safety check: there must be at least one valid dev score.
        if not dev_scores:
            raise RuntimeError(
                "ENABLE_DEV_SCORING=True, but no valid dev scores were produced."
            )

        # Select the best candidate using dev score.
        best_id = max(
            dev_scores,
            key=dev_scores.get,
        )

        best_candidate = pool[best_id]

        print(
            f"\nBest candidate on dev: {best_id} "
            f"(score={dev_scores[best_id]:.3f})"
        )

    else:
        # Dev scoring is disabled.
        #
        # Select the best candidate using its score on the fixed
        # training/scoring set. These scores have already been computed
        # during optimization, so this requires NO additional LLM calls.

        eligible_ids = [
            cid
            for cid in pool
            if cid in score_cache
        ]

        if not eligible_ids:
            raise RuntimeError(
                "ENABLE_DEV_SCORING=False, but no candidates were found "
                "in score_cache for final selection."
            )

        best_id = max(
            eligible_ids,
            key=lambda cid: score_cache[cid]["optimization_score"],
        )

        best_candidate = pool[best_id]

        best_metrics = score_cache[best_id]
        best_score = best_metrics["optimization_score"]

        label_metrics = best_metrics["label_metrics"]
        best_precision = label_metrics["precision"]
        best_recall = label_metrics["recall"]

        print(
            f"\nBest candidate on fixed scoring set: {best_id}\n"
            f"  score     = {best_score:.3f}\n"
            f"  precision = {best_precision:.3f}\n"
            f"  recall    = {best_recall:.3f}\n"
            f"  n         = {label_metrics['n']}"
        )

    # Locked final evaluation on the untouched test split.
    final_metrics = evaluate_candidate_on_df(
        best_candidate, test_df, config.RECALL_TARGET,
        save_path=predictions_path(run_dir, best_id, "final_test"),
    )

    with open(os.path.join(run_dir, "best_prompt.md"), "w") as f:
        f.write(best_candidate.prompt_text)
    with open(os.path.join(run_dir, "final_result.json"), "w") as f:
        json.dump({
            "best_candidate_id": best_id,
            "dev_scores": dev_scores,
            "final_test_metrics": {
                "label_metrics": final_metrics["label_metrics"],
                "threshold_metrics": final_metrics["threshold_metrics"],
                "optimization_score": final_metrics["optimization_score"],
            },
        }, f, indent=2, default=str)

    print(f"Final test optimization score: {final_metrics['optimization_score']:.3f}")
    print(f"Results saved under: {run_dir}")
    return run_dir, best_candidate, final_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GEPA prompt optimization.")
    parser.add_argument("--resume", metavar="RUN_DIR", default=None,
                        help="Continue an interrupted run from its saved state instead of starting a new one.")
    run_optimization(resume_dir=parser.parse_args().resume)