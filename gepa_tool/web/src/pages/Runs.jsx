import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import { STATUS_CLASS, pct } from "../util.js";

export default function Runs() {
  const [runs, setRuns] = useState(null);
  const [active, setActive] = useState(null);
  const [error, setError] = useState("");

  async function load() {
    try {
      const [r, a] = await Promise.all([api.get("/api/runs"), api.get("/api/runs/active")]);
      setRuns(r.runs);
      setActive(a.active);
      setError("");
    } catch (e) {
      setError(e.message);
    }
  }
  useEffect(() => {
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, []);

  async function stopAll() {
    if (!window.confirm("Stop every running optimization and trial?")) return;
    await api.post("/api/stop_all");
    load();
  }

  return (
    <div className="card">
      <div className="row">
        <div className="card-title">Runs</div>
        <div className="spacer" />
        {active && (
          <span className="small">
            Active: <a href={`#/runs/${encodeURIComponent(active.run_id)}`}>{active.name || active.run_id}</a>
          </span>
        )}
        <button className="btn danger small" onClick={stopAll}>■ Stop all</button>
        <a className="btn primary small" href="#/new">+ New run</a>
      </div>
      {error && <div className="errbox mt12">{error}</div>}
      {runs === null ? (
        <p className="muted">Loading…</p>
      ) : runs.length === 0 ? (
        <p className="muted mt12">No runs yet. <a href="#/new">Set up a new run</a>.</p>
      ) : (
        <div className="scrollbox mt12">
          <table className="plain">
            <thead>
              <tr>
                <th>Run</th><th>Status</th><th>Started</th><th>Rounds</th><th>Objective</th>
                <th>Best</th><th></th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.run_id} className="click" onClick={() => (window.location.hash = `#/runs/${encodeURIComponent(r.run_id)}`)}>
                  <td><b>{r.name}</b><div className="faint small mono">{r.run_id}</div></td>
                  <td><span className={`badge ${STATUS_CLASS[r.status] || "badge-gray"}`}>{r.status}</span></td>
                  <td className="small">{r.started_at ? new Date(r.started_at * 1000).toLocaleString() : "–"}</td>
                  <td>{r.rounds_done}{r.total_rounds ? ` / ${r.total_rounds}` : ""}</td>
                  <td className="small muted">{r.objective_text || "–"}</td>
                  <td className="small">
                    {r.best ? (
                      <>
                        <span className="mono">{r.best_id}</span> P {pct(r.best.precision)} · R {pct(r.best.recall)}
                        {r.best.feasible === false && <span className="errtext"> · constraints missed</span>}
                      </>
                    ) : "–"}
                  </td>
                  <td onClick={(e) => e.stopPropagation()}>
                    <a href={`#/runs/${encodeURIComponent(r.run_id)}/results`} className="small">results →</a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
