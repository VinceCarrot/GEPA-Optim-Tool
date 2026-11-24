# GEPA tool — local UI for clustering-based GEPA prompt optimization (binary classification)

## Start
```bash
source ~/.venvs/optim/bin/activate          # needs fastapi, uvicorn, python-multipart (installed)
cd <repo root>
python -m gepa_tool.server                  # http://127.0.0.1:8765  (serves the built UI)
```
Frontend development (hot reload, proxies /api to :8765):
```bash
cd gepa_tool/web
NODE_EXTRA_CA_CERTS=~/.config/gepa_tool/ca.pem npm install --cafile ~/.config/gepa_tool/ca.pem   # Artifactory, see .npmrc
npm run dev        # http://localhost:5173
npm run build      # -> web/dist, served by the FastAPI app
```

Windows (conda): `conda activate optim` then `python -m gepa_tool.server`. Stop sends CTRL_BREAK to the
engine's process group (graceful: best prompt so far is written), then kills the process tree.

## LLM backend
SafeChain by default, for the server and the engine CLI. A machine can add another backend without code
changes: put a module at `gepa_tool/engine/local_backend.py` (gitignored) and start with
`python -m gepa_tool.server --local-backend` (the module may define extra alias flags in `CLI_FLAGS`).
`GEPA_LLM_BACKEND=<module>` does the same through the environment. Engine processes started by the server
inherit the choice. The module provides `build_structured(model_id, temperature, schema)`, `list_models()`
and optionally `ui_defaults()`; see the docstring in `engine/llm.py`. Saved settings are kept per mode
(`workspace/settings.yaml` / `settings_local.yaml`), and the setup wizard resets model ids and budget
when the mode changes.

## Headless (no UI)
```bash
python -m gepa_tool.engine validate gepa_tool/configs/amex_error_20261117.yaml
python -m gepa_tool.engine trial    gepa_tool/configs/amex_error_20261117.yaml --out /tmp/trial --n 50
python -m gepa_tool.engine run      gepa_tool/configs/amex_error_20261117.yaml --run-dir gepa_tool/workspace/runs/my_run
```
`configs/amex_error_20261117.yaml` reproduces the settings of `GEPA/logs/optimizer_run_20261117_172200.log`
(same data, split, seed prompt, models 3/8/8, 20 rounds × 3 children, scoring set 1800, precision s.t. recall ≥ 0.85).

## Layout
| path | role |
|---|---|
| `engine/spec.py` | `RunConfig` — the single contract between UI and engine (YAML) |
| `engine/objective.py` | metrics, constraints, feasibility, accept/reject (`feasibility_first` and `legacy`) |
| `engine/protect.py` | locked prompt sections: parse → mask as `[[PROTECTED:name]]` → validate → restore |
| `engine/reflection.py` | 3-stage reflection (per-error analysis → clustering → targeted mutation), task-agnostic |
| `engine/optimizer.py` | the loop; Phase A (reach constraints) / Phase B (optimize); writes `events.jsonl` |
| `engine/inference.py` | task-model calls; shared concurrency across children; token refresh; progress callbacks |
| `engine/trial.py` | seed trial on a balanced sample |
| `engine/nl2spec.py` | natural language → ObjectiveSpec (whitelisted metrics, clarifications, unsupported) |
| `server/app.py` | FastAPI: datasets, validation, NL→spec, trials, runs (single-run lock), SSE, Stop All |
| `web/` | React UI: Setup wizard, live Monitor, Results |
| `workspace/` | uploads, trials, runs (local data — do not commit) |

## Run directory
`events.jsonl` (everything the UI shows) · `candidates/<id>.md` (every proposed prompt, accepted or not) ·
`predictions/<id>_scoring_round<r>.parquet` · `run_config.yaml` · `best_prompt.md` · `final_result.json` · `engine.log`

## Protected sections
```
<!-- PROTECTED:output_format -->
...text the optimizer must never change...
<!-- /PROTECTED -->
```
Markers are stripped before the prompt is sent to the task model.

## Acceptance (feasibility_first)
* parent misses a constraint: accept if the total constraint gap shrinks and the objective drops ≤ δ (0.05), or the gap is unchanged and the objective does not drop; entering the feasible region is always accepted.
* parent meets all constraints: child must also meet them and lose ≤ ε (0.01) of the objective.
* inference errors: when more than `budget.max_error_rate` (1%) of a candidate's rows failed after the per-call
  retries, only the failed rows are re-run, up to `budget.error_retry_passes` (3) passes. A child still above the
  limit gets status `error` (shown in amber): it is excluded like a rejected child (not in the pool, the Pareto
  archive or the mutation history) and the run continues. A seed above the limit only raises an alert.
* `legacy` reproduces GEPA/optimizer.py exactly (used for regression checks).