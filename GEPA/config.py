"""Central configuration for the GEPA prompt-optimization framework.

All paths and hyperparameters used by dataset.py, inference_run.py, evaluator.py,
reflection.py, pareto.py and optimizer.py are defined here so a single run can be
tuned without touching the individual modules.
"""

import os

PROJECT_ROOT = "/Users/bhu10/Library/CloudStorage/OneDrive-AmericanExpress/Documents/Work/Projects/PromptOptim2026/agentic_prompt_optimization"
GEPA_DIR = os.path.join(PROJECT_ROOT, "GEPA")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")

ENV_FILE = os.path.join(PROJECT_ROOT, ".env")
MODEL_CONFIG_PATH = os.path.join(CONFIG_DIR, "config_gpt.yml")

# ------------------ raw data / dataset contract ------------------
RAW_DATA_PATH = os.path.join(DATA_DIR, "apr_wk_1_to_4_corr.parquet")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")

ID_COL = "case_no"
TEXT_COL = "new_query_potential_issue_redacted"
LABEL_SOURCE_COL = "latest_rslv_optn"
POSITIVE_LABEL_VALUE = "Closed Amex Error"
TARGET_COL = "target"

TRAIN_FRAC = 0.75
DEV_FRAC = 0.15
TEST_FRAC = 0.10
RANDOM_SEED = 42

ENABLE_DEV_SCORING = False

# ------------------ seed prompt ------------------
# SEED_PROMPT_PATH = os.path.join(GEPA_DIR, "seed_prompt.md")
SEED_PROMPT_PATH = os.path.join(GEPA_DIR, "seed_prompt_bare_minimum.md")

# ------------------ models ------------------
# Model ids reference the `models` block in config/config_gpt.yml

# TODO
# try with GPT 4.1 --> faster runs
# beam search for best model for reflection and error analysis

TASK_MODEL_ID = "3"        # cheaper/faster model used for many candidate evaluations
REFLECTION_MODEL_ID = "8"   # stronger model used to diagnose failures and propose mutations
ERROR_ANALYSIS_MODEL_ID = "8"   # model for per-error analysis (Stage 0); can differ from task/reflection models

# ------------------ optimizer hyperparameters ------------------
MAX_ITERATIONS = 20 #40             # optimization budget: reflection/mutation iterations (each proposes NUM_PROPOSALS_PER_ITERATION candidates)
SCORING_SET_SIZE = 1800         # fixed train subset scored once per candidate (cached) and reused for
                               # every accept/reject decision, Pareto tracking, and reflection examples
NUM_PROPOSALS_PER_ITERATION = 3  # candidate prompts proposed per reflection call
ACCEPTANCE_EPSILON = 0.01      # annealing-style slack: accept children within this margin below the parent score
DEV_EVAL_EVERY = 5             # evaluate frontier candidates on dev every N iterations
OPTIMIZATION_METRIC = "label_precision"  # "threshold_precision" or "label_precision"

RECALL_TARGET = 0.85           # business floor: chosen threshold must achieve >= this recall

REFLECTION_CALL_TIMEOUT_SECONDS = 180  # hard cap per reflection/error-analysis LLM call (retried on timeout)
ROW_TIMEOUT_SECONDS = 90        # hard cap per-row LLM call duration, to bound stuck/hanging requests

N_FALSE_POSITIVES_FOR_REFLECTION = 5
N_FALSE_NEGATIVES_FOR_REFLECTION = 5
N_UNCERTAIN_FOR_REFLECTION = 5

# ------------------ three-stage reflection pipeline ------------------
N_FALSE_POSITIVES_FOR_ANALYSIS = 50   # per-error analysis (Stage 0): FPs analyzed individually with full text
N_FALSE_NEGATIVES_FOR_ANALYSIS = 50   # per-error analysis (Stage 0): FNs analyzed individually with full text
MUTATION_HISTORY_MAX = 10              # 0 = keep full history in Stage 2 prompt; >0 = most recent N entries

# ------------------ results ------------------
RESULTS_DIR = os.path.join(GEPA_DIR, "results")
BASELINE_DIR = os.path.join(RESULTS_DIR, "baseline")
RUNS_DIR = os.path.join(RESULTS_DIR, "runs")

os.environ.setdefault("CONFIG_PATH", MODEL_CONFIG_PATH)