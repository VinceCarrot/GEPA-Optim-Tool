import React, { useState } from "react";
import { api } from "../api.js";
import { METRICS, LOWER_IS_BETTER, METRIC_LABEL, summarizeObjective } from "../util.js";

function Num({ value, onChange, step = "0.01", placeholder }) {
  return <input type="number" step={step} value={value ?? ""} placeholder={placeholder}
                onChange={(e) => onChange(e.target.value === "" ? null : Number(e.target.value))} />;
}

export default function StepObjective({ cfg, set }) {
  const o = cfg.objective;
  const a = cfg.acceptance;
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [tr, setTr] = useState(null);
  const [err, setErr] = useState("");

  const setCons = (cons) => set("objective.constraints", cons);
  const used = new Set(o.constraints.map((c) => c.metric));

  function applySpec(spec) {
    set("objective", { ...o, maximize: spec.maximize, beta: spec.beta ?? null, constraints: spec.constraints });
  }

  async function translate() {
    setBusy(true);
    setErr("");
    setTr(null);
    try {
      const res = await api.post("/api/objective/from_text", {
        text, positive_label: cfg.task.positive_label, negative_label: cfg.task.negative_label,
      });
      setTr(res);
      if (res.ok && res.spec) applySpec(res.spec);
    } catch (e) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  }

  const constraintProblems = o.constraints.filter((c) => c.value === null || c.value < 0 || c.value > 1);
  const pLabel = METRIC_LABEL[o.maximize] || o.maximize;

  return (
    <div className="card">
      <div className="card-title">4. Objective and accept / reject rules</div>
      <div className="grid g2 mt12" style={{ gap: 32 }}>
        <div>
          <b>Describe it in words (optional)</b>
          <p className="desc mt8">e.g. “Flagged cases must be as accurate as possible, but we cannot miss more than 15% of real {cfg.task.positive_label} cases, and no more than 20% of all cases should be flagged.”</p>
          <textarea rows={6} value={text} onChange={(e) => setText(e.target.value)} />
          <div className="row mt8">
            <button className="btn" disabled={busy || !text.trim()} onClick={translate}>{busy ? "Translating…" : "Translate to rules"}</button>
            {err && <span className="errtext small">{err}</span>}
          </div>
          {tr && (
            <div className="infobox mt12 small">
              {tr.readable.length > 0 && (
                <>
                  <b>Understood as</b>
                  <ul style={{ margin: "6px 0 0", paddingLeft: 18 }}>
                    {tr.readable.map((r, i) => (
                      <li key={i}>{r.text}{r.source && <span className="muted"> — “{r.source}”</span>}</li>
                    ))}
                  </ul>
                </>
              )}
              {tr.clarifications.length > 0 && (
                <div className="warnbox mt8"><b>Please clarify</b>{tr.clarifications.map((q) => <div key={q}>? {q}</div>)}</div>
              )}
              {tr.unsupported.length > 0 && (
                <div className="errbox mt8"><b>Not supported (ignored unless you rephrase)</b>{tr.unsupported.map((q) => <div key={q}>✕ {q}</div>)}</div>
              )}
              {tr.problems.length > 0 && <div className="errtext mt8">{tr.problems.join("; ")}</div>}
              {tr.ok && <div className="oktext mt8">✔ Applied to the rules on the right.</div>}
              {!tr.ok && tr.spec && (
                <button className="btn small mt8" onClick={() => applySpec(tr.spec)}>Apply these rules anyway</button>
              )}
            </div>
          )}
        </div>

        <div>
          <b>Rules (this is what the optimizer uses)</b>
          <label className="field mt12">
            <span className="lbl">Maximize</span>
            <select value={o.maximize} onChange={(e) => set("objective.maximize", e.target.value)}>
              {METRICS.filter(([k]) => !LOWER_IS_BETTER.has(k)).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
            </select>
          </label>
          {o.maximize === "fbeta" && (
            <label className="field mt8"><span className="lbl">Beta <span className="muted small">(&gt;1 favours recall, &lt;1 favours precision)</span></span>
              <Num value={o.beta} step="0.1" onChange={(v) => set("objective.beta", v)} /></label>
          )}

          <div className="mt12 mb8"><b>Hard constraints</b> (all must hold)</div>
          {o.constraints.map((c, i) => (
            <div className="row mb8" key={i} style={{ flexWrap: "nowrap" }}>
              <select style={{ width: 230 }} value={c.metric} onChange={(e) => setCons(o.constraints.map((x, j) => (j === i ? { ...x, metric: e.target.value } : x)))}>
                {METRICS.map(([k, v]) => <option key={k} value={k} disabled={k !== c.metric && used.has(k)}>{v}</option>)}
              </select>
              <select style={{ width: 62 }} value={c.op} onChange={(e) => setCons(o.constraints.map((x, j) => (j === i ? { ...x, op: e.target.value } : x)))}>
                <option value=">=">≥</option>
                <option value="<=">≤</option>
              </select>
              <div style={{ width: 110 }}>
                <Num value={c.value} onChange={(v) => setCons(o.constraints.map((x, j) => (j === i ? { ...x, value: v } : x)))} />
              </div>
              <button className="linkbtn" onClick={() => setCons(o.constraints.filter((_, j) => j !== i))}>remove</button>
            </div>
          ))}
          <button className="btn" onClick={() => {
            const free = METRICS.map(([k]) => k).find((k) => !used.has(k) && k !== o.maximize) || "recall";
            setCons([...o.constraints, { metric: free, op: LOWER_IS_BETTER.has(free) ? "<=" : ">=", value: 0.8 }]);
          }}>+ add constraint</button>
          {constraintProblems.length > 0 && <div className="errtext small mt8">Constraint values must be between 0 and 1.</div>}

          <label className="field mt16">
            <span className="lbl">Scoring</span>
            <select value={o.scoring_mode} onChange={(e) => set("objective.scoring_mode", e.target.value)}>
              <option value="label">Use the model's labels directly</option>
              <option value="threshold">Best threshold on the model's risk_score</option>
            </select>
          </label>

          <div className="infobox mt12"><b>Summary:</b> {summarizeObjective(o)}</div>

          <details className="adv mt16">
            <summary>Acceptance settings (advanced)</summary>
            <div className="grid g2">
              <label className="field"><span className="lbl">Strategy</span>
                <select value={a.strategy} onChange={(e) => set("acceptance.strategy", e.target.value)}>
                  <option value="feasibility_first">Feasibility first (recommended)</option>
                  <option value="legacy">Legacy (reproduce GEPA/optimizer.py)</option>
                </select></label>
              <label className="field"><span className="lbl">ε — allowed {pLabel} drop once constraints are met</span>
                <Num value={a.epsilon} onChange={(v) => set("acceptance.epsilon", v)} /></label>
              <label className="field"><span className="lbl">δ — allowed {pLabel} drop while closing a constraint gap</span>
                <Num value={a.infeasible_delta} onChange={(v) => set("acceptance.infeasible_delta", v)} /></label>
              <label className="field"><span className="lbl">Feasibility patience (rounds)</span>
                <Num value={a.feasibility_patience} step="1" placeholder={`auto (30% of rounds = ${Math.max(1, Math.round(0.3 * cfg.budget.max_rounds))})`}
                     onChange={(v) => set("acceptance.feasibility_patience", v)} /></label>
            </div>
          </details>
          <p className="muted small mt8">
            How children are judged: while the parent misses a constraint, a child is accepted if it is closer to meeting the constraints and its {pLabel} drops by at most δ; once constraints are met, a child must keep meeting them and lose at most ε of {pLabel}. A child whose API-error rate is still above the limit (Models & budget) after the retry passes is marked “error” and treated as rejected.
          </p>
        </div>
      </div>
    </div>
  );
}
