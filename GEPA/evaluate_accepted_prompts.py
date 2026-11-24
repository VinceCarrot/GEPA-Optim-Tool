"""Evaluate accepted GEPA candidates on one frozen test split with checkpoints."""

import argparse
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

import config
from evaluator import compute_metrics, merge_predictions


def load_candidates(run_dirs, seeds_only=False):
    candidates = []
    seen = set()
    for run_dir in run_dirs:
        events = [json.loads(line) for line in (run_dir / "log.jsonl").read_text().splitlines() if line.strip()]
        rounds = {event["iteration"] for event in events if event["event"] == "candidate_eval"}
        if rounds != set(range(20)):
            raise ValueError(f"Expected 20 evaluated rounds in {run_dir}, got {sorted(rounds)}")
        selected = ([{"candidate_id": "seed", "iteration": -1, "parent_id": None}] if seeds_only else [
            event for event in events if event["event"] == "candidate_eval" and event.get("accepted") is True
        ])
        for event in selected:
            key = (run_dir.name, event["candidate_id"])
            if key in seen:
                raise ValueError(f"Duplicate accepted candidate: {key}")
            seen.add(key)
            prompt_path = run_dir / "candidates" / f"{event['candidate_id']}.md"
            prompt_text = prompt_path.read_text()
            if not prompt_text.strip() or "{NEW_QUERY_POTENTIAL_ISSUE}" not in prompt_text:
                raise ValueError(f"Empty prompt or missing input placeholder: {prompt_path}")
            candidates.append({
                "run": run_dir.name,
                "candidate_id": event["candidate_id"],
                "round": event["iteration"] + 1,
                "parent_id": event["parent_id"],
                "prompt_path": str(prompt_path.resolve()),
                "prompt_sha256": hashlib.sha256(prompt_text.encode()).hexdigest(),
                "prompt_text": prompt_text,
            })
    return candidates


def summarize(predictions, test_df):
    if predictions[config.ID_COL].duplicated().any():
        raise ValueError("Duplicate prediction IDs")
    if set(predictions[config.ID_COL]) != set(test_df[config.ID_COL]):
        raise ValueError("Prediction IDs do not match the frozen test set")
    allowed = {"Amex Error", "Non Amex Error", "ERROR"}
    if not predictions["issue_classification"].isin(allowed).all():
        raise ValueError("Unexpected prediction labels")
    merged = merge_predictions(predictions, test_df)
    metrics = compute_metrics(merged) if len(merged) else {
        "precision": None, "recall": None, "f1": None,
        "n": 0, "n_positive": 0, "n_predicted_positive": 0,
    }
    for name, gold, predicted in (("TP", 1, 1), ("FP", 0, 1), ("FN", 1, 0), ("TN", 0, 0)):
        metrics[name] = int(((merged[config.TARGET_COL] == gold) & (merged["llm_pred_flag"] == predicted)).sum())
    adjusted_tp = int(metrics["TP"] * 1.35)
    adjusted_fp = int(metrics["FP"] * 0.65)
    adjusted_fn = int(metrics["FN"] * 0.89)
    metrics["AFTER_REVIEW_precision"] = (
        adjusted_tp / (adjusted_tp + adjusted_fp) if adjusted_tp + adjusted_fp else None
    )
    metrics["AFTER_REVIEW_recall"] = (
        adjusted_tp / (adjusted_tp + adjusted_fn) if adjusted_tp + adjusted_fn else None
    )
    metrics.update({
        "n_test": len(test_df),
        "n_test_positive": int(test_df[config.TARGET_COL].sum()),
        "n_errors": len(test_df) - len(merged),
        "coverage": len(merged) / len(test_df),
        "status": "complete" if len(merged) == len(test_df) else "partial",
    })
    return metrics


def write_comparison(output_dir, compare_with):
    own_manifest = json.loads((output_dir / "manifest.json").read_text())
    other_manifest = json.loads((compare_with / "manifest.json").read_text())
    for key in ("test_sha256", "n_test", "n_test_positive", "model_id", "model_config_sha256", "metrics"):
        if own_manifest[key] != other_manifest[key]:
            raise ValueError(f"Cannot compare evaluations with different {key}")
    own_results = json.loads((output_dir / "metrics.json").read_text())
    other_path = compare_with / "metrics.json"
    other_results = json.loads(other_path.read_text()) if other_path.exists() else []
    results = other_results + own_results
    baselines = {result["run"]: result for result in results if result["candidate_id"] == "seed"}
    comparison = []
    for result in results:
        row = dict(result)
        row["prompt_type"] = "seed" if row["candidate_id"] == "seed" else "accepted"
        baseline = baselines.get(row["run"])
        row["seed_comparison_complete"] = bool(
            baseline and baseline["status"] == "complete" and row["status"] == "complete"
        )
        for metric in ("precision", "recall", "AFTER_REVIEW_precision", "AFTER_REVIEW_recall"):
            row[f"seed_{metric}"] = baseline[metric] if baseline else None
            row[f"delta_{metric}_pp"] = (
                100 * (row[metric] - baseline[metric])
                if row["seed_comparison_complete"] and row[metric] is not None and baseline[metric] is not None
                else None
            )
        comparison.append(row)
    pd.DataFrame(comparison).to_csv(output_dir / "comparison.csv", index=False)
    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(f"Comparison updated: {len(other_results)} accepted prompts + {len(own_results)} seeds", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--test-set", type=Path, default=Path(config.SPLITS_DIR) / "test.parquet")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=50)
    parser.add_argument("--retry-passes", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds-only", action="store_true")
    parser.add_argument("--compare-with", type=Path)
    args = parser.parse_args()
    if args.max_workers < 1 or args.retry_passes < 0:
        parser.error("max-workers must be positive and retry-passes must be nonnegative")
    if args.compare_with and not args.seeds_only:
        parser.error("--compare-with requires --seeds-only")
    candidates = load_candidates(args.run_dir, seeds_only=args.seeds_only)
    test_df = pd.read_parquet(args.test_set)[[config.ID_COL, config.TEXT_COL, config.TARGET_COL]]
    if test_df.empty or test_df.isna().any().any() or test_df[config.ID_COL].duplicated().any():
        raise ValueError("Test split must be nonempty with unique IDs and no null values")
    if not test_df[config.TARGET_COL].isin([0, 1]).all():
        raise ValueError("Test targets must be binary")
    manifest = {
        "test_set": str(args.test_set.resolve()),
        "test_sha256": hashlib.sha256(args.test_set.read_bytes()).hexdigest(),
        "n_test": len(test_df),
        "n_test_positive": int(test_df[config.TARGET_COL].sum()),
        "model_id": config.TASK_MODEL_ID,
        "model_config_sha256": hashlib.sha256(Path(config.MODEL_CONFIG_PATH).read_bytes()).hexdigest(),
        "metrics": {
            "label_mapping": {"Amex Error": "Closed Amex Error", "Non Amex Error": "Closed Non Amex Error"},
            "positive_label": "Closed Amex Error",
            "error_rows": "Excluded; partial coverage explicitly reported",
            "after_review_factors": {"TP": 1.35, "FP": 0.65, "FN": 0.89},
            "after_review_rounding": "Integer truncation before calculating ratios",
        },
        "candidates": [{key: value for key, value in candidate.items() if key != "prompt_text"} for candidate in candidates],
    }
    print(f"Candidates: {len(candidates)}; test rows: {len(test_df)}; positives: {manifest['n_test_positive']}", flush=True)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing manifest differs: use a new output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    snapshot_path = args.output_dir / "test.parquet"
    if not snapshot_path.exists():
        test_df.to_parquet(snapshot_path, index=False)

    from inference_run import run_inference

    logging.getLogger("httpx").setLevel(logging.WARNING)
    results = []
    for index, candidate in enumerate(candidates, start=1):
        prediction_dir = args.output_dir / "predictions" / candidate["run"]
        prediction_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = prediction_dir / f"{candidate['candidate_id']}.parquet"
        print(f"[{index}/{len(candidates)}] {candidate['run']}/{candidate['candidate_id']}", flush=True)
        shared_seed = next((
            result for result in results
            if args.seeds_only and result["prompt_sha256"] == candidate["prompt_sha256"]
            and result["status"] == "complete"
        ), None)
        if prediction_path.exists():
            predictions = pd.read_parquet(prediction_path)
            summarize(predictions, test_df)
        elif shared_seed:
            predictions = pd.read_parquet(shared_seed["prediction_path"])
            summarize(predictions, test_df)
            predictions.to_parquet(prediction_path, index=False)
            print("Reusing identical seed predictions from the other run", flush=True)
        else:
            predictions = run_inference(candidate["prompt_text"], test_df, max_workers=args.max_workers)
            summarize(predictions, test_df)
            predictions.to_parquet(prediction_path, index=False)
        for retry_pass in range(args.retry_passes):
            failed_ids = predictions.loc[predictions["issue_classification"] == "ERROR", config.ID_COL]
            if failed_ids.empty:
                break
            print(f"Retry pass {retry_pass + 1}: {len(failed_ids)} failed rows", flush=True)
            retries = run_inference(
                candidate["prompt_text"], test_df[test_df[config.ID_COL].isin(failed_ids)],
                max_workers=min(args.max_workers, 10),
            )
            predictions = pd.concat([
                predictions[~predictions[config.ID_COL].isin(failed_ids)], retries,
            ], ignore_index=True)
            summarize(predictions, test_df)
            predictions.to_parquet(prediction_path, index=False)
        metrics = summarize(predictions, test_df)
        result = {key: value for key, value in candidate.items() if key != "prompt_text"}
        result.update(metrics)
        result["prediction_path"] = str(prediction_path.resolve())
        if shared_seed:
            result["shared_seed_prediction_path"] = shared_seed["prediction_path"]
        result["evaluated_at"] = datetime.now().isoformat(timespec="seconds")
        results.append(result)
        pd.DataFrame(results).to_csv(args.output_dir / "metrics.csv", index=False)
        (args.output_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
        if args.compare_with:
            write_comparison(args.output_dir, args.compare_with)
        print(f"precision={metrics['precision']} recall={metrics['recall']} coverage={metrics['coverage']:.1%} status={metrics['status']}", flush=True)
        if metrics["n"] == 0:
            raise RuntimeError("No successful predictions after retries; stopping to avoid repeated failed batches")
    complete = sum(result["status"] == "complete" for result in results)
    print(f"Finished: {len(results)} prompts, {complete} complete, {len(results) - complete} partial. Results: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()