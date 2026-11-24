"""FastAPI backend for the GEPA tool UI.

Run lifecycle:
  * At most one optimization run at a time (workspace/active_run.json lock).
  * The engine runs as a separate process in its own session/process group, so Stop All
    can signal the whole group (engine + any children) and SIGKILL stragglers.
  * The UI reads <run_dir>/events.jsonl through an SSE stream (with Last-Event-ID replay).

Start:  python -m gepa_tool.server  (serves the built React app from gepa_tool/web/dist too)
"""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from ..engine import dataset as ds
from ..engine import llm as llm_mod
from ..engine import objective as ob
from ..engine import protect
from ..engine.spec import DataSpec, ModelSpec, ObjectiveSpec, RunConfig, dump_config, load_config
from ..engine.store import read_events

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PKG_DIR)
WORKSPACE = os.environ.get("GEPA_WORKSPACE", os.path.join(PKG_DIR, "workspace"))
UPLOADS = os.path.join(WORKSPACE, "uploads")
RUNS = os.path.join(WORKSPACE, "runs")
TRIALS = os.path.join(WORKSPACE, "trials")
LOCK = os.path.join(WORKSPACE, "active_run.json")
# Saved settings are kept per backend so a local-backend model id never leaks into SafeChain mode.
SETTINGS = os.path.join(WORKSPACE, "settings.yaml" if llm_mod.backend() is None else "settings_local.yaml")
WEB_DIST = os.path.join(PKG_DIR, "web", "dist")
STOP_GRACE_SECONDS = 10

for d in (UPLOADS, RUNS, TRIALS):
    os.makedirs(d, exist_ok=True)

app = FastAPI(title="GEPA Prompt Optimizer")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])

_lock = threading.Lock()
_trial_procs: Dict[str, subprocess.Popen] = {}
_run_procs: Dict[str, subprocess.Popen] = {}


# ---------------------------------------------------------------------------
# Settings (model config defaults, NL->spec model id)
# ---------------------------------------------------------------------------

def _default_settings() -> dict:
    return {
        "model_config_path": os.path.join(REPO_ROOT, "config", "config_gpt.yml"),
        "env_file": os.path.join(REPO_ROOT, ".env"),
        "nl2spec_model_id": "3",  # GPT-5 in config_gpt.yml
        "python": sys.executable,
    }


def _backend_defaults() -> dict:
    be = llm_mod.backend()
    if be is None or not hasattr(be, "ui_defaults"):
        return {}
    return be.ui_defaults() or {}


def get_settings() -> dict:
    s = _default_settings()
    nl_default = (_backend_defaults().get("models") or {}).get("nl2spec_model_id")
    if nl_default:
        s["nl2spec_model_id"] = nl_default
    if os.path.exists(SETTINGS):
        with open(SETTINGS, encoding="utf-8") as f:
            s.update(yaml.safe_load(f) or {})
    return s


@app.get("/api/settings")
def api_settings():
    s = get_settings()
    be = llm_mod.backend()
    s["backend_mode"] = "safechain" if be is None else "local"
    if be is not None:
        # A local LLM backend replaces SafeChain on this machine; its models replace config_gpt.yml.
        return {**s, "models": be.list_models(), "backend": getattr(be, "NAME", be.__name__),
                "ui_defaults": _backend_defaults()}
    models = []
    try:
        with open(s["model_config_path"], encoding="utf-8") as f:
            mc = yaml.safe_load(f) or {}
        for mid, m in (mc.get("models") or {}).items():
            if (m or {}).get("type", "chat") == "chat":
                models.append({"id": str(mid), "name": m.get("model_name") or m.get("deployment_name") or str(mid)})
    except Exception as e:
        return {**s, "models": [], "model_config_error": str(e)}
    return {**s, "models": models}


class SettingsIn(BaseModel):
    model_config_path: Optional[str] = None
    env_file: Optional[str] = None
    nl2spec_model_id: Optional[str] = None


@app.put("/api/settings")
def api_put_settings(body: SettingsIn):
    s = get_settings()
    s.update({k: v for k, v in body.model_dump().items() if v is not None})
    with open(SETTINGS, "w", encoding="utf-8") as f:
        yaml.safe_dump({k: s[k] for k in ("model_config_path", "env_file", "nl2spec_model_id")}, f)
    return api_settings()


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def _dataset_path(dataset_id: str) -> str:
    for f in os.listdir(UPLOADS):
        if f.startswith(dataset_id + "__"):
            return os.path.join(UPLOADS, f)
    raise HTTPException(404, "dataset not found")


def _json_safe_records(df) -> list:
    return json.loads(df.to_json(orient="records", date_format="iso", default_handler=str))


@app.post("/api/datasets")
async def upload_dataset(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in (".parquet", ".csv", ".xlsx", ".xls"):
        raise HTTPException(400, "Upload a parquet, csv or xlsx file")
    did = uuid.uuid4().hex[:10]
    safe = os.path.basename(file.filename).replace(" ", "_")
    path = os.path.join(UPLOADS, f"{did}__{safe}")
    with open(path, "wb") as out:
        shutil.copyfileobj(file.file, out)
    try:
        df = ds.read_table(path)
    except Exception as e:
        os.remove(path)
        raise HTTPException(400, f"Could not read file: {e}")
    return {"dataset_id": did, "filename": file.filename, "path": path, "n_rows": len(df),
            "columns": [{"name": c, "dtype": str(df[c].dtype), "n_unique": int(df[c].nunique()),
                         "n_null": int(df[c].isna().sum())} for c in df.columns],
            "preview": _json_safe_records(df.head(20).astype(str))}


class LocalPathIn(BaseModel):
    path: str


@app.post("/api/datasets/local")
def register_local_dataset(body: LocalPathIn):
    """Use a file already on disk (large datasets) without copying it."""
    if not os.path.isfile(body.path):
        raise HTTPException(400, "file not found")
    try:
        df = ds.read_table(body.path)
    except Exception as e:
        raise HTTPException(400, f"Could not read file: {e}")
    did = uuid.uuid4().hex[:10]
    link = os.path.join(UPLOADS, f"{did}__{os.path.basename(body.path)}")
    try:
        os.symlink(body.path, link)
    except OSError:
        shutil.copy(body.path, link)
    return {"dataset_id": did, "filename": os.path.basename(body.path), "path": link, "n_rows": len(df),
            "columns": [{"name": c, "dtype": str(df[c].dtype), "n_unique": int(df[c].nunique()),
                         "n_null": int(df[c].isna().sum())} for c in df.columns],
            "preview": _json_safe_records(df.head(20).astype(str))}


class ProfileIn(BaseModel):
    data: Dict[str, Any]
    scoring_set_size: Optional[int] = None


@app.post("/api/datasets/profile")
def profile_dataset(body: ProfileIn):
    try:
        spec = DataSpec.model_validate(body.data)
    except ValidationError as e:
        raise HTTPException(422, _pyd_errors(e))
    try:
        check = ds.check_data(spec, body.scoring_set_size)
        # Column/length/label profile of the main file: the full dataset, else the train file.
        main = spec.path if spec.path else spec.split.train_path
        prof: Dict[str, Any] = {}
        if main and ("full" in check["files"] or "train" in check["files"]):
            prof = ds.profile(ds.read_table(main), spec)
        prof["check"] = check
        return prof
    except ValueError as e:
        raise HTTPException(400, str(e))
    except KeyError as e:
        raise HTTPException(400, f"Missing field in input_template: {e}")


# ---------------------------------------------------------------------------
# Config validation, protected sections, NL -> spec
# ---------------------------------------------------------------------------

def _pyd_errors(e: ValidationError) -> list:
    return [{"loc": ".".join(str(x) for x in err["loc"]), "msg": err["msg"]} for err in e.errors()]


@app.post("/api/config/validate")
def validate_config(body: Dict[str, Any]):
    try:
        cfg = RunConfig.model_validate(body)
    except ValidationError as e:
        return {"ok": False, "errors": _pyd_errors(e)}
    try:
        blocks = protect.parse(cfg.seed_prompt)
    except protect.ProtectedSectionError as e:
        return {"ok": False, "errors": [{"loc": "seed_prompt", "msg": str(e)}]}
    try:
        check = ds.check_data(cfg.data, cfg.budget.scoring_set_size)
    except Exception as e:
        check = {"ok": False, "errors": [str(e)]}
    if not check["ok"]:
        return {"ok": False, "errors": [{"loc": "data", "msg": m} for m in check["errors"]]}
    return {"ok": True, "errors": [], "objective_text": ob.describe_objective(cfg.objective),
            "protected_sections": [{"name": b.name, "chars": len(b.body)} for b in blocks],
            "patience": cfg.patience}


class PromptIn(BaseModel):
    prompt: str
    placeholder: Optional[str] = None


@app.post("/api/prompt/inspect")
def inspect_prompt(body: PromptIn):
    out: Dict[str, Any] = {"ok": True, "errors": []}
    try:
        out["protected_sections"] = [{"name": b.name, "chars": len(b.body), "preview": b.body.strip()[:200]}
                                     for b in protect.parse(body.prompt)]
    except protect.ProtectedSectionError as e:
        out["ok"] = False
        out["errors"].append(str(e))
        out["protected_sections"] = []
    if body.placeholder and "{" + body.placeholder + "}" not in body.prompt:
        out["ok"] = False
        out["errors"].append(f"Prompt must contain the input placeholder {{{body.placeholder}}}")
    return out


class NLIn(BaseModel):
    text: str
    positive_label: str = "Positive"
    negative_label: str = "Negative"


@app.post("/api/objective/from_text")
async def objective_from_text(body: NLIn):
    from ..engine import nl2spec
    s = get_settings()
    models = ModelSpec(model_config_path=s["model_config_path"], env_file=s["env_file"],
                       task_model_id=s["nl2spec_model_id"], reflection_model_id=s["nl2spec_model_id"],
                       error_analysis_model_id=s["nl2spec_model_id"])
    try:
        return await nl2spec.translate_async(body.text, models, s["nl2spec_model_id"],
                                             body.positive_label, body.negative_label)
    except Exception as e:
        raise HTTPException(502, f"Translation failed: {e}")


@app.post("/api/objective/describe")
def describe_objective(body: Dict[str, Any]):
    try:
        spec = ObjectiveSpec.model_validate(body)
    except ValidationError as e:
        return {"ok": False, "errors": _pyd_errors(e)}
    return {"ok": True, "text": ob.describe_objective(spec),
            "constraints": [ob.describe_constraint(c) for c in spec.constraints]}


# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------

IS_WINDOWS = os.name == "nt"


def _spawn(args: List[str], log_path: str) -> subprocess.Popen:
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    logf = open(log_path, "ab")
    # Own process group, so Stop can signal the engine and everything it started.
    group = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if IS_WINDOWS
             else {"start_new_session": True})
    return subprocess.Popen([get_settings()["python"], "-m", "gepa_tool.engine", *args], cwd=REPO_ROOT,
                            stdout=logf, stderr=subprocess.STDOUT, env=env, **group)


# Windows: a "group" is the engine process plus its descendants (tracked with psutil).
# "term" sends CTRL_BREAK_EVENT to the engine's process group (the engine turns it into a
# graceful stop); "kill" terminates the whole process tree.

def _win_alive(pid: int) -> bool:
    import psutil
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return psutil.pid_exists(pid)


def _win_signal(pid: int, sig: str) -> bool:
    import psutil
    if not _win_alive(pid):
        return False
    if sig == "term":
        try:
            os.kill(pid, signal.CTRL_BREAK_EVENT)
            return True
        except OSError:
            pass  # no shared console: fall through to a hard kill
    try:
        root = psutil.Process(pid)
        procs = root.children(recursive=True) + [root]
    except psutil.NoSuchProcess:
        return False
    for p in procs:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(procs, timeout=5)
    return True


def _group_alive(pgid: int) -> bool:
    if IS_WINDOWS:
        return _win_alive(pgid)
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # EPERM on macOS when only zombies remain; check whether any live member is left
        try:
            out = subprocess.run(["ps", "-g", str(pgid), "-o", "stat="], capture_output=True, text=True, timeout=5).stdout
            return any(s.strip() and not s.strip().startswith("Z") for s in out.splitlines())
        except Exception:
            return False


def _reap(pid: int) -> None:
    """Collect an exited child so it does not linger as a zombie (zombie groups look 'alive')."""
    if IS_WINDOWS:
        for p in list(_run_procs.values()) + list(_trial_procs.values()):
            if p.pid == pid:
                p.poll()
        return
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass  # not our child (e.g. server restarted) -- init reaps it
    except OSError:
        pass


def _signal_group(pgid: int, sig: str) -> bool:
    """Send sig ('term' | 'kill') to the group. False if the group no longer exists (or only zombies remain)."""
    if IS_WINDOWS:
        return _win_signal(pgid, sig)
    try:
        os.killpg(pgid, signal.SIGTERM if sig == "term" else signal.SIGKILL)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # macOS returns EPERM when every member of the group is a zombie
        _reap(pgid)
        return _group_alive(pgid)


def _kill_group(pgid: int, grace: float = STOP_GRACE_SECONDS) -> str:
    """SIGTERM the process group, wait, then SIGKILL. Returns 'terminated' | 'killed' | 'not_running'."""
    _reap(pgid)
    if not _group_alive(pgid) or not _signal_group(pgid, "term"):
        return "not_running"
    deadline = time.time() + grace
    while time.time() < deadline:
        _reap(pgid)
        if not _group_alive(pgid):
            return "terminated"
        time.sleep(0.25)
    if not _signal_group(pgid, "kill"):
        return "terminated"
    for _ in range(20):
        _reap(pgid)
        if not _group_alive(pgid):
            break
        time.sleep(0.1)
    return "killed"


def _read_lock() -> Optional[dict]:
    if not os.path.exists(LOCK):
        return None
    try:
        with open(LOCK, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def active_run() -> Optional[dict]:
    """The lock if its process group is still alive; stale locks are cleared."""
    with _lock:
        lk = _read_lock()
        if not lk:
            return None
        _reap(lk["pgid"])
        if _group_alive(lk["pgid"]):
            return lk
        _finalize_dead_run(lk["run_id"])
        os.remove(LOCK)
        return None


def _finalize_dead_run(run_id: str, status: str = "failed", note: str = "engine process exited unexpectedly") -> None:
    run_dir = os.path.join(RUNS, run_id)
    evs = read_events(run_dir)
    if evs and any(e["type"] == "run_end" for e in evs):
        return
    from ..engine.store import EventLog
    EventLog(run_dir).emit("run_end", status=status, error=note)


# ---------------------------------------------------------------------------
# Seed trial
# ---------------------------------------------------------------------------

class TrialIn(BaseModel):
    config: Dict[str, Any]
    n: int = 50


@app.post("/api/trials")
def start_trial(body: TrialIn):
    try:
        cfg = RunConfig.model_validate(body.config)
        protect.parse(cfg.seed_prompt)
    except ValidationError as e:
        raise HTTPException(422, _pyd_errors(e))
    except protect.ProtectedSectionError as e:
        raise HTTPException(422, [{"loc": "seed_prompt", "msg": str(e)}])
    tid = datetime.now().strftime("trial_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:4]
    out = os.path.join(TRIALS, tid)
    os.makedirs(out)
    cfg_path = os.path.join(out, "run_config.yaml")
    dump_config(cfg, cfg_path)
    _trial_procs[tid] = _spawn(["trial", cfg_path, "--out", out, "--n", str(max(4, min(body.n, 500)))],
                               os.path.join(out, "process.log"))
    return {"trial_id": tid}


@app.get("/api/trials/{tid}")
def get_trial(tid: str):
    out = os.path.join(TRIALS, tid)
    if not os.path.isdir(out):
        raise HTTPException(404, "trial not found")
    evs = read_events(out)
    prog = [e for e in evs if e["type"] == "eval_progress"]
    res_path = os.path.join(out, "trial_result.json")
    end = next((e for e in evs if e["type"] == "trial_end"), None)
    p = _trial_procs.get(tid)
    running = p is not None and p.poll() is None
    if end is None and not running and os.path.exists(res_path):
        end = {"status": "completed"}
    if end is None and not running:
        log = ""
        lp = os.path.join(out, "process.log")
        if os.path.exists(lp):
            with open(lp, encoding="utf-8") as f:
                log = f.read()[-3000:]
        return {"status": "failed", "error": "trial process exited", "log": log}
    if end is None:
        return {"status": "running", "progress": prog[-1] if prog else None}
    result = None
    if os.path.exists(res_path):
        with open(res_path, encoding="utf-8") as f:
            result = json.load(f)
    return {"status": end.get("status", "completed"), "result": result, "error": (result or {}).get("error")}


@app.post("/api/trials/{tid}/stop")
def stop_trial(tid: str):
    p = _trial_procs.get(tid)
    if not p:
        raise HTTPException(404, "trial not running")
    return {"result": _kill_group(p.pid, grace=3)}


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

class RunIn(BaseModel):
    config: Dict[str, Any]


@app.post("/api/runs")
def start_run(body: RunIn):
    try:
        cfg = RunConfig.model_validate(body.config)
        protect.parse(cfg.seed_prompt)
    except ValidationError as e:
        raise HTTPException(422, _pyd_errors(e))
    except protect.ProtectedSectionError as e:
        raise HTTPException(422, [{"loc": "seed_prompt", "msg": str(e)}])
    if active_run():
        raise HTTPException(409, "Another run is already in progress. Stop it first.")
    with _lock:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in cfg.name)[:40] or "run"
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + safe
        run_dir = os.path.join(RUNS, run_id)
        os.makedirs(run_dir)
        cfg_path = os.path.join(run_dir, "submitted_config.yaml")
        dump_config(cfg, cfg_path)
        p = _spawn(["run", cfg_path, "--run-dir", run_dir], os.path.join(run_dir, "process.log"))
        _run_procs[run_id] = p
        lock = {"run_id": run_id, "pid": p.pid, "pgid": p.pid, "started_at": time.time(), "name": cfg.name}
        with open(LOCK, "w", encoding="utf-8") as f:
            json.dump(lock, f)
    return lock


@app.get("/api/runs/active")
def get_active():
    return {"active": active_run()}


def _run_summary(run_id: str) -> dict:
    run_dir = os.path.join(RUNS, run_id)
    evs = read_events(run_dir)
    start = next((e for e in evs if e["type"] == "run_start"), {})
    end = next((e for e in reversed(evs) if e["type"] == "run_end"), None)
    rounds = [e for e in evs if e["type"] == "round_end"]
    lk = _read_lock()
    status = end["status"] if end else ("running" if lk and lk.get("run_id") == run_id else "unknown")
    best = rounds[-1].get("best") if rounds else None
    return {"run_id": run_id, "name": start.get("name", run_id), "status": status,
            "started_at": evs[0]["ts"] if evs else os.path.getmtime(run_dir),
            "rounds_done": len(rounds), "total_rounds": start.get("total_rounds"),
            "objective_text": start.get("objective_text"), "best": best,
            "best_id": rounds[-1].get("best_id") if rounds else None}


@app.get("/api/runs")
def list_runs():
    active_run()
    ids = sorted((d for d in os.listdir(RUNS) if os.path.isdir(os.path.join(RUNS, d))), reverse=True)
    return {"runs": [_run_summary(i) for i in ids]}


def _run_dir(run_id: str) -> str:
    d = os.path.join(RUNS, os.path.basename(run_id))
    if not os.path.isdir(d):
        raise HTTPException(404, "run not found")
    return d


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    d = _run_dir(run_id)
    out = _run_summary(run_id)
    for name in ("run_config.yaml", "submitted_config.yaml"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                out["config"] = yaml.safe_load(f)
            break
    fr = os.path.join(d, "final_result.json")
    if os.path.exists(fr):
        with open(fr, encoding="utf-8") as f:
            out["final_result"] = json.load(f)
    return out


@app.get("/api/runs/{run_id}/events")
async def stream_events(run_id: str, request: Request):
    d = _run_dir(run_id)
    last = request.headers.get("last-event-id") or request.query_params.get("after") or "0"
    try:
        after = int(last)
    except ValueError:
        after = 0

    async def gen():
        nonlocal after
        idle = 0
        while True:
            if await request.is_disconnected():
                break
            evs = read_events(d, after)
            for e in evs:
                after = e["seq"]
                yield f"id: {e['seq']}\nevent: message\ndata: {json.dumps(e)}\n\n"
            ended = any(e["type"] == "run_end" for e in evs)
            if ended:
                yield "event: end\ndata: {}\n\n"
                break
            if not evs:
                idle += 1
                if idle % 15 == 0:
                    yield ": keepalive\n\n"
                    # engine died without writing run_end -> close the stream with a synthetic end
                    lk = _read_lock()
                    if not lk or lk.get("run_id") != run_id:
                        if not any(e["type"] == "run_end" for e in read_events(d)):
                            active_run()
            else:
                idle = 0
            await asyncio.sleep(1.0)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/runs/{run_id}/candidates/{cid}")
def get_candidate(run_id: str, cid: str):
    d = _run_dir(run_id)
    path = os.path.join(d, "candidates", os.path.basename(cid) + ".md")
    if not os.path.exists(path):
        raise HTTPException(404, "candidate not found")
    with open(path, encoding="utf-8") as f:
        prompt = f.read()
    meta = {}
    idx = os.path.join(d, "candidates.jsonl")
    if os.path.exists(idx):
        with open(idx, encoding="utf-8") as f:
            for line in f:
                m = json.loads(line)
                if m.get("candidate_id") == cid:
                    meta = m
    return {"candidate_id": cid, "prompt": prompt, **meta}


@app.get("/api/runs/{run_id}/candidates/{cid}/diff")
def get_diff(run_id: str, cid: str, against: Optional[str] = None):
    c = get_candidate(run_id, cid)
    other_id = against or c.get("parent_id")
    if not other_id:
        return {"candidate_id": cid, "against": None, "lines": []}
    o = get_candidate(run_id, other_id)
    a, b = o["prompt"].splitlines(), c["prompt"].splitlines()
    lines = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            lines += [{"op": " ", "text": t} for t in a[i1:i2]]
        if tag in ("replace", "delete"):
            lines += [{"op": "-", "text": t} for t in a[i1:i2]]
        if tag in ("replace", "insert"):
            lines += [{"op": "+", "text": t} for t in b[j1:j2]]
    return {"candidate_id": cid, "against": other_id, "lines": lines}


@app.get("/api/runs/{run_id}/candidates/{cid}/errors")
def get_candidate_errors(run_id: str, cid: str, kind: str = "FP", limit: int = 50, split: str = "scoring"):
    import glob
    import pandas as pd
    d = _run_dir(run_id)
    pat = "final_test" if split == "test" else "scoring_round*"
    files = sorted(glob.glob(os.path.join(d, "predictions", f"{os.path.basename(cid)}_{pat}.parquet")))
    if not files:
        raise HTTPException(404, "no predictions for candidate")
    df = pd.read_parquet(files[-1])
    if kind == "FP":
        df = df[(df["pred_flag"] == 1) & (df["target"] == 0)]
    elif kind == "FN":
        df = df[(df["pred_flag"] == 0) & (df["target"] == 1)]
    cfg = get_run(run_id).get("config", {})
    id_col = (cfg.get("data") or {}).get("id_col", df.columns[0])
    rows = [{"id": str(r.get(id_col)), "input": str(r.get("__input__", ""))[:6000],
             "predicted": r.get("classification"), "risk_score": r.get("risk_score"),
             "reason": r.get("classification_reason")} for r in df.head(limit).to_dict(orient="records")]
    return {"kind": kind, "total": int(len(df)), "rows": rows}


@app.get("/api/runs/{run_id}/best_prompt")
def best_prompt(run_id: str):
    p = os.path.join(_run_dir(run_id), "best_prompt.md")
    if not os.path.exists(p):
        raise HTTPException(404, "best prompt not written yet")
    return FileResponse(p, media_type="text/markdown", filename=f"{run_id}_best_prompt.md")


@app.post("/api/stop_all")
def stop_all():
    """Stop every optimization run and seed trial started by this tool (whole process groups)."""
    results = []
    lk = _read_lock()
    if lk:
        r = _kill_group(lk["pgid"])
        _finalize_dead_run(lk["run_id"], status="stopped",
                           note="stopped by user" + (" (force killed)" if r == "killed" else ""))
        results.append({"run_id": lk["run_id"], "result": r})
        with _lock:
            if os.path.exists(LOCK):
                os.remove(LOCK)
    for rid, p in list(_run_procs.items()):
        if p.poll() is None:
            results.append({"run_id": rid, "result": _kill_group(p.pid)})
    for tid, p in list(_trial_procs.items()):
        if p.poll() is None:
            results.append({"trial_id": tid, "result": _kill_group(p.pid, grace=3)})
    return {"stopped": results}


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

if os.path.isdir(WEB_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(WEB_DIST, "assets")), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(404)
        return FileResponse(os.path.join(WEB_DIST, "index.html"))