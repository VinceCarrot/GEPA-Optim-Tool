import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import { classNames } from "../util.js";
import StepData from "../wizard/StepData.jsx";
import StepTask from "../wizard/StepTask.jsx";
import StepSeed from "../wizard/StepSeed.jsx";
import StepObjective from "../wizard/StepObjective.jsx";
import StepModels from "../wizard/StepModels.jsx";
import StepTrial from "../wizard/StepTrial.jsx";

const STORE_KEY = "gepa_wizard_v1";

export const DEFAULT_CONFIG = {
  name: "gepa_run",
  seed_prompt: "",
  data: {
    path: null, id_col: "", label_col: "", positive_value: "", input_columns: [], input_template: "",
    profanity_wordlist: null,
    split: { mode: "auto", train_frac: 0.75, dev_frac: 0.15, test_frac: 0.1, seed: 42, train_path: null, test_path: null },
  },
  task: {
    positive_label: "Positive", negative_label: "Negative", description: "", positive_definition: "",
    negative_definition: "", fp_cost: "", fn_cost: "", input_placeholder: "INPUT",
  },
  objective: { maximize: "precision", beta: null, scoring_mode: "label", constraints: [{ metric: "recall", op: ">=", value: 0.8 }] },
  acceptance: { strategy: "feasibility_first", epsilon: 0.01, infeasible_delta: 0.05, feasibility_patience: null },
  models: {
    model_config_path: "", env_file: "", task_model_id: "", reflection_model_id: "", error_analysis_model_id: "",
    task_temperature: 1.0, reflection_temperature: 1.0,
  },
  budget: {
    max_rounds: 20, scoring_set_size: 1800, proposals_per_round: 3, max_workers: 200, request_delay: 0.5,
    row_timeout_seconds: 90, reflection_timeout_seconds: 180, parallel_children: true, final_test_eval: true,
    max_error_rate: 0.01, error_retry_passes: 3,
  },
  reflection: {
    n_false_positives: 50, n_false_negatives: 50, phase_a_focus_ratio: 0.7, mutation_history_max: 10,
    analysis_concurrency: 50, analysis_request_delay: 0.5,
  },
  random_seed: 42,
};

const STEPS = ["Data", "Task", "Seed prompt", "Objective", "Models & budget", "Trial & start"];

function fresh() {
  return { cfg: JSON.parse(JSON.stringify(DEFAULT_CONFIG)), step: 0, maxStep: 0, dataset: null, trainDataset: null,
           testDataset: null, dataCheck: null, autoTemplate: true, defaultsApplied: false, trial: null };
}

// Older stored setups used the main file as the training file in "provided" mode.
function migrate(s) {
  const d = s.cfg.data;
  if (d.split.mode === "provided" && d.path && d.path === d.split.train_path && !s.trainDataset) {
    return { ...s, trainDataset: s.dataset, dataset: null, cfg: { ...s.cfg, data: { ...d, path: null } } };
  }
  return s;
}

// The Data step passes when "Check data" succeeded for exactly the current data settings.
export const dataKey = (data) => JSON.stringify(data);
export const dataChecked = (st) => !!st.dataCheck && st.dataCheck.ok && st.dataCheck.key === dataKey(st.cfg.data);

function load() {
  try {
    const s = JSON.parse(localStorage.getItem(STORE_KEY));
    // Stored state may predate a field: fill anything missing from the defaults.
    if (s && s.cfg) return migrate({ ...fresh(), ...s, cfg: deepMerge(DEFAULT_CONFIG, s.cfg) });
  } catch { /* ignore */ }
  return fresh();
}

export function setIn(obj, path, value) {
  const keys = path.split(".");
  const out = Array.isArray(obj) ? [...obj] : { ...obj };
  let cur = out;
  for (let i = 0; i < keys.length - 1; i++) {
    const k = keys[i];
    cur[k] = Array.isArray(cur[k]) ? [...cur[k]] : { ...(cur[k] || {}) };
    cur = cur[k];
  }
  cur[keys[keys.length - 1]] = value;
  return out;
}

function deepMerge(base, extra) {
  const out = { ...base };
  for (const [k, v] of Object.entries(extra || {})) {
    out[k] = v && typeof v === "object" && !Array.isArray(v) ? deepMerge(base[k] || {}, v) : v;
  }
  return out;
}

export default function NewRun() {
  const [st, setSt] = useState(load);
  const [settings, setSettings] = useState(null);

  useEffect(() => {
    try { localStorage.setItem(STORE_KEY, JSON.stringify(st)); } catch { /* quota */ }
  }, [st]);

  useEffect(() => {
    api.get("/api/settings").then(setSettings).catch((e) => setSettings({ models: [], model_config_error: e.message }));
  }, []);

  // Fill machine-specific defaults (model config paths, model ids, local-backend budget) once per
  // backend mode. When the server runs in a different mode than the one the stored setup was made
  // for (SafeChain vs local backend), model ids and budget go back to that mode's defaults, so the
  // other mode's models never show up here.
  useEffect(() => {
    if (!settings) return;
    setSt((s) => {
      let cfg = s.cfg;
      const mode = `${settings.backend_mode || "safechain"}:${settings.backend || ""}`;
      const switched = s.defaultsFor !== undefined ? s.defaultsFor !== mode : !!s.defaultsApplied && !settings.backend;
      if (switched) {
        cfg = {
          ...cfg,
          models: { ...cfg.models, model_config_path: "", env_file: "", task_model_id: "", reflection_model_id: "", error_analysis_model_id: "" },
          budget: { ...DEFAULT_CONFIG.budget },
          reflection: { ...DEFAULT_CONFIG.reflection },
        };
      }
      const m = cfg.models;
      const ids = (settings.models || []).map((x) => x.id);
      const ui = settings.ui_defaults || {};
      const dm = ui.models || {};
      const pick = (cur, pref) => (cur && (ids.length === 0 || ids.includes(cur)) ? cur : (pref && ids.includes(pref) ? pref : ids[0] || cur || ""));
      cfg = {
        ...cfg,
        models: {
          ...m,
          model_config_path: m.model_config_path || settings.model_config_path || "",
          env_file: m.env_file || settings.env_file || "",
          task_model_id: pick(m.task_model_id, dm.task_model_id),
          reflection_model_id: pick(m.reflection_model_id, dm.reflection_model_id),
          error_analysis_model_id: pick(m.error_analysis_model_id, dm.error_analysis_model_id),
        },
      };
      if ((!s.defaultsApplied || switched) && (ui.budget || ui.reflection)) {
        cfg = { ...cfg, budget: deepMerge(cfg.budget, ui.budget), reflection: deepMerge(cfg.reflection, ui.reflection) };
      }
      return { ...s, cfg, defaultsApplied: true, defaultsFor: mode };
    });
  }, [settings]);

  const set = (path, value) => setSt((s) => ({ ...s, cfg: setIn(s.cfg, path, value) }));
  const patch = (fields) => setSt((s) => ({ ...s, ...fields }));
  const go = (i) => setSt((s) => ({ ...s, step: i, maxStep: Math.max(s.maxStep, i) }));

  function reset() {
    if (!window.confirm("Clear every field of this setup?")) return;
    localStorage.removeItem(STORE_KEY);
    setSt({ ...fresh() });
    setSettings((x) => (x ? { ...x } : x)); // re-apply machine defaults
  }

  const props = { cfg: st.cfg, set, st, patch, settings };
  const dataOk = dataChecked(st);
  const locked = (i) => i > 0 && !dataOk;
  const Step = [StepData, StepTask, StepSeed, StepObjective, StepModels, StepTrial][st.step];

  return (
    <div>
      <div className="steps">
        {STEPS.map((name, i) => (
          <button key={name} className={classNames("step", i === st.step && "on", i !== st.step && i <= st.maxStep && "done")}
                  disabled={locked(i)} title={locked(i) ? "Run “Check data” on step 1 first" : undefined}
                  onClick={() => go(i)}>
            {i + 1}. {name}
          </button>
        ))}
        <button className="linkbtn reset" onClick={reset}>reset</button>
      </div>
      <Step {...props} />
      <div className="navrow">
        <button className="btn" disabled={st.step === 0} onClick={() => go(st.step - 1)}>← Back</button>
        {st.step < STEPS.length - 1 && (
          <div className="row">
            {locked(st.step + 1) && <span className="muted small">Run “Check data” and fix any errors to continue</span>}
            <button className="btn primary" disabled={locked(st.step + 1)} onClick={() => go(st.step + 1)}>Next →</button>
          </div>
        )}
      </div>
    </div>
  );
}
