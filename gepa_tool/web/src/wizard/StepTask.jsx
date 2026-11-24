import React from "react";

function Area({ label, hint, value, onChange, rows = 2 }) {
  return (
    <label className="field mt16">
      <span className="lbl">{label}{hint && <span className="hint">— {hint}</span>}</span>
      <textarea rows={rows} value={value} onChange={(e) => onChange(e.target.value)} />
    </label>
  );
}

export default function StepTask({ cfg, set }) {
  const t = cfg.task;
  return (
    <div className="card">
      <div className="card-title">2. Task definition</div>
      <p className="desc">This text is given to the reflection model when it diagnoses errors and rewrites the prompt. Clear definitions give better mutations.</p>
      <div className="grid g3" style={{ maxWidth: 1250 }}>
        <label className="field"><span className="lbl">Run name</span>
          <input value={cfg.name} onChange={(e) => set("name", e.target.value)} /></label>
        <label className="field"><span className="lbl">Positive class label (model output)</span>
          <input value={t.positive_label} onChange={(e) => set("task.positive_label", e.target.value)} /></label>
        <label className="field"><span className="lbl">Negative class label (model output)</span>
          <input value={t.negative_label} onChange={(e) => set("task.negative_label", e.target.value)} /></label>
      </div>
      {t.positive_label.trim() && t.positive_label.trim() === t.negative_label.trim() && (
        <div className="errtext small mt8">The two labels must differ.</div>
      )}
      <Area label="Task description" hint="what is being classified and why" rows={3}
            value={t.description} onChange={(v) => set("task.description", v)} />
      <Area label={`Definition of "${t.positive_label}"`} hint="when an input is positive"
            value={t.positive_definition} onChange={(v) => set("task.positive_definition", v)} />
      <Area label={`Definition of "${t.negative_label}"`} hint="when an input is negative"
            value={t.negative_definition} onChange={(v) => set("task.negative_definition", v)} />
      <Area label="Cost of a false positive" hint="what goes wrong when a negative is flagged"
            value={t.fp_cost} onChange={(v) => set("task.fp_cost", v)} />
      <Area label="Cost of a false negative" hint="what goes wrong when a positive is missed"
            value={t.fn_cost} onChange={(v) => set("task.fn_cost", v)} />
    </div>
  );
}
