import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import { useRunEvents, ACTIVITY_TEXT } from "../runEvents.js";
import { duration, pct, classNames, STATUS_CLASS } from "../util.js";
import { MetricLine } from "../components/common.jsx";
import CandidateTree from "../components/CandidateTree.jsx";
import RoundPanel from "../components/RoundPanel.jsx";

function useNow(active) {
  const [now, setNow] = useState(Date.now() / 1000);
  useEffect(() => {
    if (!active) return;
    const t = setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => clearInterval(t);
  }, [active]);
  return now;
}

export default function Monitor({ runId }) {
  const { state: s, connected } = useRunEvents(runId);
  const running = s.status === "running" || s.status === "starting";
  const now = useNow(running);
  const [selRound, setSelRound] = useState(null);
  const [selChild, setSelChild] = useState(null);
  const [stopping, setStopping] = useState(false);

  const total = s.start?.total_rounds ?? 0;
  const roundNos = Object.keys(s.rounds).map(Number).filter((n) => n <= total || total === 0).sort((a, b) => a - b);
  const latest = roundNos.length ? roundNos[roundNos.length - 1] : null;
  const shownRound = selRound ?? latest;
  const round = shownRound !== null ? s.rounds[shownRound] : null;

  async function stopAll() {
    if (!window.confirm("Stop every running optimization and trial? The best prompt so far is kept.")) return;
    setStopping(true);
    try { await api.post("/api/stop_all"); } finally { setStopping(false); }
  }

  function selectNode(id) {
    const c = s.cands[id];
    if (!c) return;
    setSelRound(c.round ?? 0);
    setSelChild(id === "seed" ? null : id);
  }

  const act = s.lastActivity;
  let activity = "";
  if (act) activity = ACTIVITY_TEXT[act.stage] || act.stage;
  const shownNo = act ? Math.min(act.round, total || act.round) : 0;
  const elapsed = s.start ? (s.end ? s.end.ts : now) - s.start.ts : null;
  const seed = s.cands.seed?.metrics;
  const selectedNode = selChild || (shownRound === 0 ? "seed" : null);

  if (!s.start && !s.end) {
    return <div className="card">{connected ? "Waiting for the engine to start…" : "Connecting…"}</div>;
  }
  return (
    <div>
      <div className="card run-head">
        <div className="row top">
          <div>
            <h1>{s.start?.name || runId}</h1>
            <span className={`badge ${STATUS_CLASS[s.status] || "badge-gray"}`}>{s.status}</span>
            <div className="objective">{s.start?.objective_text}</div>
          </div>
          <div className="spacer" />
          <a className="btn" href={`#/runs/${encodeURIComponent(runId)}/results`}>View results →</a>
          <button className="btn danger" onClick={stopAll} disabled={stopping}>■ Stop all</button>
        </div>

        <div className="row mt16">
          <div className="round-title">
            Round {shownNo} / {total} {activity && <span className="sub">· {activity}</span>}
          </div>
          <div className="spacer" />
          <span className="muted small">elapsed {duration(elapsed)}</span>
        </div>
        <div className="progress mt8">
          <div className="bar" style={{ width: `${total ? (100 * s.roundsDone) / total : 0}%` }} />
          <div className="txt" style={{ color: s.roundsDone / Math.max(1, total) > 0.55 ? "#fff" : undefined }}>
            {s.roundsDone} of {total} rounds completed
          </div>
        </div>

        {s.end?.status === "failed" && (
          <div className="errbox mt12" style={{ wordBreak: "break-word" }}>
            Run failed: {s.end.error || "unknown error"}
            {s.end.traceback && <details className="mt8"><summary className="small">traceback</summary><pre className="small">{s.end.traceback}</pre></details>}
          </div>
        )}
        {s.end?.status === "stopped" && <div className="warnbox mt12">Run stopped{s.end.error ? `: ${s.end.error}` : ""}. The best prompt so far was saved.</div>}
        {s.alerts.map((a, i) => <div key={i} className="warnbox mt12 small">⚠ {a.message}</div>)}

        <div className="row mt12 bestline">
          <span>Best <b className="mono">{s.bestId || "–"}</b> <MetricLine m={s.best} /></span>
          <div className="spacer" />
          {seed && <span className="muted small">Seed: P {pct(seed.precision)} · R {pct(seed.recall)}</span>}
        </div>
      </div>

      <div className="chips">
        {roundNos.map((n) => {
          const r = s.rounds[n];
          const accepted = r.children.filter((id) => s.cands[id]?.accepted).length;
          const nErr = r.children.filter((id) => s.cands[id]?.status === "error").length;
          const failed = r.failed || r.errors.length > 0;
          const label = n === 0 ? (r.ended ? "seed" : "…") : !r.ended ? "…" : failed ? "⚠" : `✓${accepted}${nErr ? ` E${nErr}` : ""}`;
          return (
            <button key={n} className={classNames("chip", n === shownRound && "on", n !== shownRound && failed && "warn", n !== shownRound && !r.ended && "running")}
                    onClick={() => { setSelRound(n); setSelChild(null); }}>
              R{n} {label}
            </button>
          );
        })}
        {selRound !== null && running && (
          <button className="linkbtn small" onClick={() => { setSelRound(null); setSelChild(null); }}>follow latest</button>
        )}
      </div>

      <div className="split">
        <div className="card">
          <div className="panel-title">Candidate tree</div>
          <div className="treebox">
            <CandidateTree cands={s.cands} order={s.order} selectedId={selectedNode} bestId={s.bestId} onSelect={selectNode} />
          </div>
          <div className="small muted mt8">
            <span style={{ color: "#2f8a3e" }}>●</span> accepted · <span style={{ color: "#c62828" }}>●</span> rejected ·{" "}
            <span style={{ color: "#d08a1e" }}>●</span> error (API failures, treated as rejected) ·{" "}
            <span style={{ color: "#c3c8d2" }}>●</span> evaluating · ★ best · P/R in % on the scoring set
          </div>
        </div>
        <div className="card">
          <RoundPanel runId={runId} state={s} round={round} selectedChild={selChild} onSelectChild={setSelChild} now={now} />
        </div>
      </div>
    </div>
  );
}
