import React from "react";
import { num } from "../util.js";

function Num({ value, onChange, step = "1" }) {
  return <input type="number" step={step} value={value ?? ""}
                onChange={(e) => onChange(e.target.value === "" ? null : Number(e.target.value))} />;
}

function ModelSelect({ value, onChange, models }) {
  const list = models.some((m) => m.id === value) || !value ? models : [...models, { id: value, name: value }];
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)}>
      {list.length === 0 && <option value="">(no models found)</option>}
      {list.map((m) => <option key={m.id} value={m.id}>{m.name} (id {m.id})</option>)}
    </select>
  );
}

export function estimateCalls(cfg) {
  const b = cfg.budget;
  const r = cfg.reflection;
  return {
    task: b.scoring_set_size * (1 + b.max_rounds * b.proposals_per_round),
    reflection: b.max_rounds * (r.n_false_positives + r.n_false_negatives + 2),
  };
}

export default function StepModels({ cfg, set, settings }) {
  const m = cfg.models;
  const b = cfg.budget;
  const r = cfg.reflection;
  const models = settings?.models || [];
  const est = estimateCalls(cfg);
  const field = (label, path, value, step) => (
    <label className="field"><span className="lbl">{label}</span><Num value={value} step={step} onChange={(v) => set(path, v)} /></label>
  );

  return (
    <div className="card">
      <div className="card-title">5. Models and budget</div>
      {settings?.backend && (
        <div className="infobox small mb8">
          LLM backend on this machine: <b>{settings.backend}</b>. Model config and .env are not used by it; every call is a separate CLI process, so keep concurrency modest.
        </div>
      )}
      <div className="grid g5 mt8">
        <label className="field"><span className="lbl">Model config (SafeChain yml)</span>
          <input value={m.model_config_path} onChange={(e) => set("models.model_config_path", e.target.value)} /></label>
        <label className="field"><span className="lbl">.env file</span>
          <input value={m.env_file || ""} onChange={(e) => set("models.env_file", e.target.value || null)} /></label>
        <label className="field"><span className="lbl">Task model<span className="hint">scores every row; called most</span></span>
          <ModelSelect value={m.task_model_id} models={models} onChange={(v) => set("models.task_model_id", v)} /></label>
        <label className="field"><span className="lbl">Reflection model<span className="hint">clustering + rewriting</span></span>
          <ModelSelect value={m.reflection_model_id} models={models} onChange={(v) => set("models.reflection_model_id", v)} /></label>
        <label className="field"><span className="lbl">Error-analysis model<span className="hint">per-error diagnosis</span></span>
          <ModelSelect value={m.error_analysis_model_id} models={models} onChange={(v) => set("models.error_analysis_model_id", v)} /></label>
      </div>
      {settings?.model_config_error && !settings?.backend && (
        <div className="errtext small mt8">Could not read the model config: {settings.model_config_error}</div>
      )}

      <div className="card-title mt20">Budget</div>
      <div className="grid g5">
        {field("Rounds", "budget.max_rounds", b.max_rounds)}
        {field("Children per round", "budget.proposals_per_round", b.proposals_per_round)}
        {field("Scoring set size (train rows)", "budget.scoring_set_size", b.scoring_set_size)}
        {field("Max concurrent requests", "budget.max_workers", b.max_workers)}
        {field("Per-row timeout (s)", "budget.row_timeout_seconds", b.row_timeout_seconds)}
        {field("Reflection call timeout (s)", "budget.reflection_timeout_seconds", b.reflection_timeout_seconds)}
        {field("Max API-error rate per candidate", "budget.max_error_rate", b.max_error_rate, "0.005")}
        {field("Retry passes for errored rows", "budget.error_retry_passes", b.error_retry_passes)}
        <label className="row" style={{ gridColumn: "span 1", alignItems: "flex-start", flexWrap: "nowrap", paddingTop: 24 }}>
          <input type="checkbox" checked={b.parallel_children} onChange={(e) => set("budget.parallel_children", e.target.checked)} />
          <span>Evaluate a round's children in parallel (shared concurrency limit)</span>
        </label>
        <label className="row" style={{ gridColumn: "span 1", alignItems: "flex-start", flexWrap: "nowrap", paddingTop: 24 }}>
          <input type="checkbox" checked={b.final_test_eval} onChange={(e) => set("budget.final_test_eval", e.target.checked)} />
          <span>Evaluate best and seed on the test split at the end</span>
        </label>
      </div>

      <details className="adv mt12">
        <summary>Reflection settings (advanced)</summary>
        <div className="grid g5">
          {field("False positives analysed per round", "reflection.n_false_positives", r.n_false_positives)}
          {field("False negatives analysed per round", "reflection.n_false_negatives", r.n_false_negatives)}
          {field("Constraint-phase focus ratio", "reflection.phase_a_focus_ratio", r.phase_a_focus_ratio, "0.05")}
          {field("Mutation history shown (0 = all)", "reflection.mutation_history_max", r.mutation_history_max)}
          {field("Error-analysis concurrency", "reflection.analysis_concurrency", r.analysis_concurrency)}
          {field("Error-analysis stagger (s)", "reflection.analysis_request_delay", r.analysis_request_delay, "0.1")}
          {field("Per-row request delay (s)", "budget.request_delay", b.request_delay, "0.1")}
          {field("Task temperature", "models.task_temperature", m.task_temperature, "0.1")}
          {field("Reflection temperature", "models.reflection_temperature", m.reflection_temperature, "0.1")}
          {field("Random seed", "random_seed", cfg.random_seed)}
        </div>
      </details>

      <div className="infobox mt12">
        ≈ {num(est.task)} task-model calls (scoring set × (1 + rounds × children)) plus ~{num(est.reflection)} reflection calls. The trial in the next step measures real latency and gives a time estimate.
      </div>
    </div>
  );
}
