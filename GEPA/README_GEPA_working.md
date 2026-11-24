# GEPA — Genetic Evolution Prompt Optimizer

Automatically improves an LLM system prompt for binary complaint classification
(`Amex Error` / `Non Amex Error`) by running a genetic evolution loop: score
→ reflect → mutate → accept. The optimization objective is label-based precision
subject to a minimum recall floor (`RECALL_TARGET = 0.75`).

---

## Folder layout

```
GEPA/
├── config.py               # All hyperparameters and paths — single place to tune a run
├── dataset.py              # Data loading, profanity scrubbing, stratified splits
├── evaluator.py            # Precision/recall metrics, scoring function, error samplers
├── inference_run.py        # Async LLM inference runner (task model, structured output)
├── reflection.py           # Three-stage reflection pipeline (per-error → cluster → mutate)
├── optimizer.py            # Main genetic evolution loop
├── candidate.py            # PromptCandidate dataclass + CandidateStore persistence
├── pareto.py               # Pareto frontier tracking for parent selection
├── plot_candidate_tree.py  # Visualize the candidate lineage tree as a PNG
├── run_gepa_optimizer.sh   # Shell script: activate env, rebuild splits, run optimizer
├── seed_prompt.md          # Starting system prompt fed to the task model
├── logs/                   # Per-run stdout/stderr log files from the shell script
└── results/
    ├── baseline/           # Seed-prompt predictions and metrics
    └── runs/
        └── run_YYYYMMDD_HHMMSS/
            ├── log.jsonl             # Structured event log (parent_eval, candidate_eval, …)
            ├── candidate_history.jsonl  # Accepted candidates in insertion order
            ├── best_prompt.md        # Best prompt selected by dev score
            ├── final_result.json     # Dev scores for all candidates + final test metrics
            ├── pareto_state.json     # Pareto archive snapshot
            ├── candidate_tree.png    # Lineage visualisation
            ├── candidates/           # One .md file per candidate prompt text
            └── predictions/          # Parquet files with raw LLM predictions per candidate
```

---

## How a run works

### 1. Data (`dataset.py`)

`load_splits()` regenerates stratified train / dev / test splits on every call
(fractions set in `config.py`). A fixed random seed ensures the same partition
across runs. Profanity in the complaint text is masked with `**` before any
model sees it.

A **fixed scoring set** of `SCORING_SET_SIZE` rows is sampled once from train at
the start of each run. Every candidate is evaluated on this identical set so
accept/reject comparisons are stable across iterations.

### 2. Scoring (`evaluator.py`)

`evaluate()` returns a metrics bundle:

| Key | What it is |
|-----|-----------|
| `label_metrics` | Precision / recall / F1 computed from the discrete `issue_classification` field |
| `threshold_metrics` | Best precision achievable at or above the recall floor via `risk_score` thresholding |
| `optimization_score` | The value the optimizer maximises — see below |

**Optimization score formula** (smoothed, bounded penalty for sub-floor candidates):

```
if recall >= RECALL_TARGET:
    opt_score = precision                                        # above floor: raw precision
else:
    opt_score = 0.1 × precision × (recall / RECALL_TARGET)²    # below floor: compressed into [0, 0.1)
```

The `0.1` weight guarantees that no sub-floor candidate can ever outscore an
above-floor one, while the quadratic term still provides gradient signal for the
optimizer when all candidates are below the recall floor.

### 3. Inference (`inference_run.py`)

All task-model calls are async with a semaphore (`MAX_WORKERS = 100`). The
structured output schema `CaseNoteOutput` enforces `issue_classification` as a
`Literal["Amex Error", "Non Amex Error"]`. Key helpers:

- `run_inference(prompt, df)` — single prompt over a dataframe
- `run_inference_many(prompts, df)` — multiple prompts over the same dataframe
  in one shared worker pool (used to evaluate all children in one batched call)
- `run_inference_multi_df(prompt, dfs)` — one prompt over several dataframes
- Token refresh every 15 minutes; reactive refresh on 401 errors

### 4. Pareto archive (`pareto.py`)

Tracks a `candidate × instance` correctness matrix. A candidate is on the
**Pareto frontier** if it is correct on at least one instance that no other
candidate gets right. Parent selection for mutation is weighted by instances won,
with an over-mutation penalty that halves a candidate's weight for each time it
has already been mutated. This encourages exploring diverse lineages instead of
repeatedly mutating a single best candidate.

### 5. Three-stage reflection pipeline (`reflection.py`)

Called once per iteration to produce `NUM_PROPOSALS_PER_ITERATION` revised
prompts. Stages run sequentially; Stages 0 calls are fully parallelised.

#### Stage 0 — Per-error analysis (parallelised, ~20 calls)

Each sampled FP and FN is analyzed individually. The model receives:
- The **full current system prompt**
- The **complete, untruncated complaint text** (no character limit)
- The classifier's actual output: predicted label, risk score, stated reasoning
- The gold label

Output per example: `responsible_prompt_aspect` + `why_wrong` (2–4 sentences).
Model: `ERROR_ANALYSIS_MODEL_ID` (default: strong model, configurable in
`config.py`). Temperature: 0.3 (analytical task).

#### Stage 1 — Clustering (1 call, strong model)

All per-error summaries (~50 tokens each) are grouped into exactly
`NUM_PROPOSALS_PER_ITERATION` distinct clusters. Each cluster captures a
different root-cause failure mode and identifies the responsible prompt
section(s), the FP/FN breakdown, and whether fixing it risks lowering recall
below the floor. Temperature: 0.7.

#### Stage 2 — Targeted mutation (1 call, strong model)

One surgical prompt revision is proposed per cluster. The model receives:
- The current prompt
- The overall diagnosis and cluster descriptions from Stage 1
- The **full mutation history** (all prior iterations): what was tried, precision
  / recall before→after, accepted or rejected — so previously failed strategies
  are not re-tried
- The recall floor stated as a hard constraint

Proposals prefer surgical edits (rewrite only the failing section, keep the rest
verbatim) unless the failure is structural. Temperature: 1.0 (creative task).

### 6. Main loop (`optimizer.py`)

```
for each iteration:
    1. Sample parent from Pareto frontier
    2. Build analysis_errors (10 FP + 10 FN, full text, rotated by seed)
    3. Run three-stage reflection → N proposals
    4. Evaluate all N children on the fixed scoring set (one batched call)
    5. For each child:
         - Append to mutation_history (rationale, P/R before→after, accepted/rejected)
         - Accept if child_score >= parent_score - ACCEPTANCE_EPSILON (0.02)
         - Add accepted children to pool and Pareto archive
    6. Every DEV_EVAL_EVERY iterations: evaluate Pareto frontier on dev set
after all iterations:
    - Evaluate remaining pool members on dev (if not already scored)
    - Select best by dev score
    - Run final locked evaluation on test set
    - Save best_prompt.md and final_result.json
```

**Acceptance rule**: a child within `ACCEPTANCE_EPSILON = 0.02` of the parent
score is accepted (annealing-style slack for sideways moves). The smoothed
scoring function means sub-floor children very rarely cross this threshold
against an above-floor parent.

**Mutation history** (`mutation_history` list in optimizer) is accumulated
across all iterations of the run and passed to Stage 2 each iteration.
`MUTATION_HISTORY_MAX = 0` means the full history is always shown
(`> 0` limits to the most recent N entries).

### 7. Visualisation (`plot_candidate_tree.py`)

Renders the candidate lineage as a tree PNG with one circle per candidate:

- **O** = train optimization score × 100
- **P** = precision × 100
- **R** = recall × 100
- **D** = dev score × 100 (blue, when available)
- White circles = accepted; grey = rejected; cyan = best (by dev score)
- Red edges = winning lineage from seed to best candidate

Run: `python plot_candidate_tree.py [run_dir]` (defaults to most recent run).

---

## Configuration (`config.py`)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `TASK_MODEL_ID` | `"5"` | Cheaper model used for all candidate evaluations |
| `REFLECTION_MODEL_ID` | `"54"` | Strong model for Stage 1 clustering + Stage 2 mutation |
| `ERROR_ANALYSIS_MODEL_ID` | `"54"` | Model for Stage 0 per-error analysis (can be cheaper) |
| `MAX_ITERATIONS` | 5 (prod: 40+) | Total reflection/mutation iterations |
| `SCORING_SET_SIZE` | 250 | Fixed train subset for all accept/reject decisions |
| `NUM_PROPOSALS_PER_ITERATION` | 3 | Children per iteration; also = number of error clusters |
| `ACCEPTANCE_EPSILON` | 0.02 | Slack for sideways/near-tie acceptance |
| `RECALL_TARGET` | 0.75 | Hard business floor; sub-floor candidates scored × 0.1 |
| `N_FALSE_POSITIVES_FOR_ANALYSIS` | 10 | FPs analyzed individually in Stage 0 |
| `N_FALSE_NEGATIVES_FOR_ANALYSIS` | 10 | FNs analyzed individually in Stage 0 |
| `MUTATION_HISTORY_MAX` | 0 | 0 = full history; >0 = most recent N entries shown to Stage 2 |
| `DEV_EVAL_EVERY` | 5 | Evaluate frontier on dev every N iterations |

---

## Running

```bash
# Full run (activates conda env, rebuilds splits, tees output to logs/)
bash run_gepa_optimizer.sh

# Visualise the latest run
python plot_candidate_tree.py

# Visualise a specific run
python plot_candidate_tree.py results/runs/run_20260817_120000
```