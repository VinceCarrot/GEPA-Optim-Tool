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
* degenerate children (predict-positive rate > 3× the true rate and > 50%, or no positives) are rejected.
* `legacy` reproduces GEPA/optimizer.py exactly (used for regression checks).