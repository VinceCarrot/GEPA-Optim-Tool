import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import { pct, num, classNames } from "../util.js";

const hashCfg = (cfg) => JSON.stringify(cfg);

const OUTCOME_CLASS = { TP: "badge-green", TN: "badge-gray", FP: "badge-red", FN: "badge-amber", error: "badge-red" };

export default function StepTrial({ cfg, st, patch }) {
  const [val, setVal] = useState(null);
  const [rows, setRows] = useState(50);
  const [startErr, setStartErr] = useState("");
  const [starting, setStarting] = useState(false);
  const [filter, setFilter] = useState("all");
  const [openRow, setOpenRow] = useState(null);
  const trial = st.trial;

  useEffect(() => {
    const t = setTimeout(() => {
      api.post("/api/config/validate", cfg).then(setVal).catch((e) => setVal({ ok: false, errors: [{ loc: "", msg: e.message }] }));
    }, 250);
    return () => clearTimeout(t);
  }, [cfg]);

  // Poll a running trial.
  useEffect(() => {
    if (!trial || trial.status !== "running") return;
    let alive = true;
    const tick = async () => {
      try {
        const r = await api.get(`/api/trials/${trial.id}`);
        if (!alive) return;
        patch({ trial: { ...trial, ...r, status: r.status } });
      } catch (e) {
        if (alive) patch({ trial: { ...trial, status: "failed", error: e.message } });
      }
    };
    const h = setInterval(tick, 1500);
    tick();
    return () => { alive = false; clearInterval(h); };
  }, [trial?.id, trial?.status]);

  async function runTrial() {
    try {
      const r = await api.post("/api/trials", { config: cfg, n: Number(rows) || 50 });
      patch({ trial: { id: r.trial_id, status: "running", cfgHash: hashCfg(cfg) } });
    } catch (e) {
      patch({ trial: { status: "failed", error: e.message } });
    }
  }

  async function stopTrial() {
    if (trial?.id) await api.post(`/api/trials/${trial.id}/stop`).catch(() => {});
    patch({ trial: { ...trial, status: "stopped" } });
  }

  async function start() {
    setStarting(true);
    setStartErr("");
    try {
      const r = await api.post("/api/runs", { config: cfg });
      window.location.hash = `#/runs/${encodeURIComponent(r.run_id)}`;
    } catch (e) {
      setStartErr(e.message);
    } finally {
      setStarting(false);
    }
  }

  const res = trial?.result;
  const stale = trial && trial.cfgHash !== hashCfg(cfg);
  const trialOk = trial && trial.status === "completed" && res && !res.error && !res.blocked && !stale;
  const canStart = val?.ok && trialOk;
  const prog = trial?.progress;
  const shown = (res?.rows || []).filter((r) => filter === "all" || r.outcome === filter);

  return (
    <div className="card">
      <div className="card-title">6. Trial run and start</div>
      {!val ? <div className="infobox mt8 muted">Validating…</div> : val.ok ? (
        <div className="infobox mt8">
          ✔ Config valid · {val.objective_text}
          <br />Locked sections: {val.protected_sections.length ? val.protected_sections.map((s) => s.name).join(", ") : "none"} · Patience: {val.patience} rounds
        </div>
      ) : (
        <div className="errbox mt8">
          <b>Fix these before running:</b>
          {val.errors.map((e, i) => <div key={i}>{e.loc ? <span className="mono">{e.loc}</span> : null}{e.loc ? ": " : ""}{e.msg}</div>)}
        </div>
      )}

      <div className="card-title mt20" style={{ fontSize: 15 }}>Seed trial</div>
      <p className="desc">Scores the seed prompt on a small balanced sample to catch format problems and measure speed before spending the full budget.</p>
      <div className="row">
        <span>Rows</span>
        <input type="number" style={{ width: 110 }} value={rows} onChange={(e) => setRows(e.target.value)} />
        <button className="btn primary" disabled={!val?.ok || trial?.status === "running"} onClick={runTrial}>
          {trial?.status === "running" ? "Running…" : "Run trial"}
        </button>
        {trial?.status === "running" && <button className="btn small" onClick={stopTrial}>Stop trial</button>}
      </div>

      {trial?.status === "running" && (
        <div className="mt12" style={{ maxWidth: 520 }}>
          <div className="progress thin"><div className="bar" style={{ width: prog ? `${(100 * prog.done) / Math.max(1, prog.total)}%` : "3%" }} /></div>
          <div className="small muted mt8">{prog ? `${prog.done} / ${prog.total} rows scored${prog.n_errors ? ` · ${prog.n_errors} errors` : ""}` : "Starting…"}</div>
        </div>
      )}
      {trial && (trial.status === "failed" || trial.error) && (
        <div className="errbox mt12">
          Trial failed: {trial.error || res?.error}
          {trial.log && <pre className="small mt8" style={{ whiteSpace: "pre-wrap" }}>{trial.log}</pre>}
        </div>
      )}
      {trial?.status === "stopped" && <div className="warnbox mt12">Trial stopped.</div>}

      {res && !res.error && (
        <div className="mt16">
          {stale && <div className="warnbox mb8 small">The setup changed after this trial. Run the trial again before starting.</div>}
          <div className="row" style={{ gap: 24 }}>
            <div><div className="muted small">Precision</div><b style={{ fontSize: 20 }}>{pct(res.metrics.precision)}</b></div>
            <div><div className="muted small">Recall</div><b style={{ fontSize: 20 }}>{pct(res.metrics.recall)}</b></div>
            <div><div className="muted small">F1</div><b style={{ fontSize: 20 }}>{pct(res.metrics.f1)}</b></div>
            <div><div className="muted small">Predicted positive</div><b style={{ fontSize: 20 }}>{pct(res.metrics.predicted_positive_rate)}</b></div>
            <div><div className="muted small">Unparsed / failed rows</div>
              <b style={{ fontSize: 20 }} className={res.blocked ? "errtext" : ""}>{res.n_errors} / {res.n} ({pct(res.parse_fail_rate)})</b></div>
            <div><div className="muted small">Constraints</div><b style={{ fontSize: 20 }}>{res.feasible ? "met" : "missed"}</b></div>
            <div><div className="muted small">Trial time</div><b style={{ fontSize: 20 }}>{Math.round(res.elapsed_seconds)}s</b></div>
          </div>
          <div className="infobox mt12 small">
            Full run estimate: ≈ {num(res.estimate.task_calls)} task-model calls + ~{num(res.estimate.reflection_calls)} reflection calls,
            roughly <b>{res.estimate.approx_hours.toFixed(1)} h</b> of task-model time at this concurrency.
          </div>
          {res.warnings.length > 0 && <div className="warnbox mt8 small">{res.warnings.map((w) => <div key={w}>⚠ {w}</div>)}</div>}
          <details className="mt12">
            <summary>Trial rows ({res.rows.length})</summary>
            <div className="row mt8">
              {["all", "TP", "FP", "FN", "TN", "error"].map((k) => (
                <button key={k} className={classNames("tab", filter === k && "on")} onClick={() => setFilter(k)}>
                  {k} ({k === "all" ? res.rows.length : res.rows.filter((r) => r.outcome === k).length})
                </button>
              ))}
            </div>
            <div className="scrollbox mt8" style={{ maxHeight: 460 }}>
              <table className="plain">
                <thead><tr><th>id</th><th>gold</th><th>predicted</th><th></th><th>risk</th><th>reason</th></tr></thead>
                <tbody>
                  {shown.map((r) => (
                    <React.Fragment key={r.id}>
                      <tr className="click" onClick={() => setOpenRow(openRow === r.id ? null : r.id)}>
                        <td className="mono small">{r.id}</td><td>{r.gold}</td><td>{r.predicted}</td>
                        <td><span className={`badge ${OUTCOME_CLASS[r.outcome]}`}>{r.outcome}</span></td>
                        <td>{r.risk_score?.toFixed?.(2) ?? "–"}</td>
                        <td className="small">{String(r.reason || "").slice(0, 220)}</td>
                      </tr>
                      {openRow === r.id && (
                        <tr><td colSpan={6}><div className="prompt" style={{ maxHeight: 300 }}>{r.input}</div>
                          <div className="small mt8"><b>Reason:</b> {r.reason}</div></td></tr>
                      )}
                    </React.Fragment>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        </div>
      )}

      <div className="card-title mt20" style={{ fontSize: 15 }}>Start optimization</div>
      <p className="desc">
        {canStart ? "Trial passed. Starting launches the engine in the background; you can close this page and come back from Runs."
          : "Run a successful trial first (fewer than 5% unparsed rows)."}
      </p>
      <button className="btn primary big" disabled={!canStart || starting} onClick={start}>
        {starting ? "Starting…" : `Start ${cfg.budget.max_rounds}-round optimization`}
      </button>
      {startErr && <div className="errtext mt8">{startErr}</div>}
    </div>
  );
}
