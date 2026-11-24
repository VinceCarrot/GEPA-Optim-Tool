import React, { useState } from "react";
import { STAGE_LABELS } from "../runEvents.js";
import { duration, pct, pts, classNames, METRIC_LABEL } from "../util.js";
import { MetricLine, DiffView, PromptView } from "./common.jsx";

function StagePills({ round }) {
  if (round.round === 0) {
    const st = round.stages.evaluate;
    return (
      <div className="stagepills">
        <span className={classNames("stagepill", st)}>{st === "done" ? "✔ " : ""}Evaluate seed</span>
      </div>
    );
  }
  return (
    <div className="stagepills">
      {STAGE_LABELS.map(([k, label]) => {
        const st = round.stages[k];
        return (
          <span key={k} className={classNames("stagepill", st)}>
            {st === "done" ? "✔ " : st === "failed" ? "✕ " : st === "running" ? "● " : ""}{label}
          </span>
        );
      })}
    </div>
  );
}

const STATUS_BADGE = { accepted: "badge-green", rejected: "badge-red", error: "badge-amber" };

function StatusBadge({ c }) {
  const st = c.status || (c.accepted ? "accepted" : "rejected");
  return <span className={`badge ${STATUS_BADGE[st] || "badge-gray"}`}>{st}</span>;
}

function DecisionTable({ c, objective }) {
  const m = c.metrics;
  const p = c.parent_metrics;
  const rowsSpec = [["precision", "Precision"], ["recall", "Recall"], ["f1", "F1"]];
  if (objective && !["precision", "recall", "f1"].includes(objective)) rowsSpec.push([objective, METRIC_LABEL[objective] || objective]);
  rowsSpec.push(["predicted_positive_rate", "Pred. positive"]);
  const cls = c.status === "error" ? "err" : c.accepted ? "acc" : "rej";
  return (
    <div className={`decision ${cls}`}>
      <div className="row">
        <StatusBadge c={c} />
        <b>{c.reason}</b>
        <div className="spacer" />
        <span className="code">{c.reason_code}</span>
      </div>
      <table>
        <thead><tr><th style={{ textAlign: "left" }}></th><th>Parent</th><th>Child</th><th>Δ (pts)</th></tr></thead>
        <tbody>
          {rowsSpec.map(([k, label]) => (
            <tr key={k}>
              <td>{label}</td>
              <td className="num">{p ? pct(p[k]) : "–"}</td>
              <td className="num child">{pct(m[k])}</td>
              <td className="num">{p && m[k] !== undefined && p[k] !== undefined ? pts(m[k] - p[k]) : ""}</td>
            </tr>
          ))}
          <tr>
            <td>Score</td>
            <td className="num">{p ? p.optimization_score.toFixed(3) : "–"}</td>
            <td className="num child">{m.optimization_score.toFixed(3)}</td>
            <td />
          </tr>
          <tr>
            <td>Constraints</td>
            <td className="num">{p ? (p.feasible ? <><span className="check">✓</span>met</> : <><span className="cross">✕</span>missed</>) : "–"}</td>
            <td className="num child">{m.feasible ? <><span className="check">✓</span>met</> : <><span className="cross">✕</span>missed</>}</td>
            <td />
          </tr>
          {(m.n_errors > 0 || (p && p.n_errors > 0)) && (
            <tr><td>Errored rows</td><td className="num">{p ? p.n_errors : "–"}</td><td className="num child">{m.n_errors}</td><td /></tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

function ChildDetail({ runId, c, round, objective }) {
  const [tab, setTab] = useState("diff");
  const prog = round.progress[c.id];
  return (
    <div>
      {c.target_cluster && <div className="mt8"><b>Targets cluster:</b> {c.target_cluster}</div>}
      {c.rationale && <div className="mt8"><b>Rationale:</b> {c.rationale}</div>}
      {c.metrics ? <DecisionTable c={c} objective={objective} /> : (
        <div className="decision pend">
          <b>Evaluating…</b>{" "}
          {prog ? <span className="muted">{prog.done} / {prog.total} rows{prog.n_errors ? ` · ${prog.n_errors} errors` : ""}
            {prog.retry ? ` · retry pass ${prog.retry}/${prog.max_retry} (${prog.retry_rows} rows)` : ""}</span>
            : <span className="muted">waiting for the task model</span>}
          {prog && <div className="progress thin mt8"><div className="bar" style={{ width: `${(100 * prog.done) / Math.max(1, prog.total)}%` }} /></div>}
        </div>
      )}
      <div className="tabs mb8">
        <button className={classNames("tab", tab === "diff" && "on")} onClick={() => setTab("diff")}>Diff vs parent</button>
        <button className={classNames("tab", tab === "full" && "on")} onClick={() => setTab("full")}>Full prompt</button>
      </div>
      {tab === "diff" ? <DiffView runId={runId} cid={c.id} /> : <PromptView runId={runId} cid={c.id} />}
    </div>
  );
}

function Clusters({ round }) {
  if (!round.clusters) return null;
  const diag = round.diagnosis || "";
  return (
    <details open className="mt12">
      <summary>
        <b style={{ fontSize: 15 }}>Failure clusters</b>{" "}
        <span className="muted small">— {diag.slice(0, 170)}{diag.length > 170 ? "…" : ""}</span>
      </summary>
      <div className="diagnosis">{diag}</div>
      <ol className="cluster-list">
        {round.clusters.map((cl, i) => (
          <li key={i}>
            <span className="cname">{cl.name}</span> <span className="muted">({cl.fp_vs_fn})</span>
            <div className="mt8" style={{ fontSize: 13.5 }}>{cl.root_cause}</div>
            <div className="small muted mt8"><b>Prompt sections:</b> {cl.responsible_prompt_sections}</div>
            <div className="small muted"><b>Constraint risk:</b> {cl.constraint_risk}</div>
          </li>
        ))}
      </ol>
    </details>
  );
}

export default function RoundPanel({ runId, state, round, selectedChild, onSelectChild, now }) {
  const [showParent, setShowParent] = useState(false);
  if (!round) return <div className="muted">Waiting for the first round…</div>;
  const objective = state.start?.objective?.maximize;
  const took = round.duration ?? (round.start_ts ? (round.ended ? null : now - round.start_ts) : null);

  if (round.round === 0) {
    const seed = state.cands.seed;
    const prog = round.progress.seed;
    return (
      <div>
        <div className="row"><div className="panel-title">Round 0 · Seed evaluation</div><div className="spacer" />
          {took != null && <span className="muted small">{round.ended ? "took" : "running"} {duration(took)}</span>}</div>
        <StagePills round={round} />
        <div className="parentbox">
          <span className="muted">Seed prompt</span> <b>seed</b>{" "}
          {seed?.metrics ? <MetricLine m={seed.metrics} /> : prog ? `${prog.done} / ${prog.total} rows scored` : "evaluating…"}
        </div>
        {state.phases[0] && <div className="muted small mt8">Start phase {state.phases[0].phase}: {state.phases[0].note}.</div>}
        <div className="mt12"><PromptView runId={runId} cid="seed" /></div>
      </div>
    );
  }

  const children = round.children.map((id) => state.cands[id]).filter(Boolean);
  const sel = children.find((c) => c.id === selectedChild) || children[0];
  return (
    <div>
      <div className="row">
        <div className="panel-title">Round {round.round} · Phase {round.phase || "?"}</div>
        <div className="spacer" />
        {took != null && <span className="muted small">{round.ended ? "took" : "running"} {duration(took)}</span>}
      </div>
      <StagePills round={round} />
      {round.parent_id && (
        <div className="parentbox">
          <div className="row">
            <span><span className="muted">Parent (current node)</span> <b>{round.parent_id}</b> <MetricLine m={round.parent_metrics} /></span>
            <div className="spacer" />
            <button className="linkbtn" onClick={() => setShowParent(!showParent)}>{showParent ? "hide prompt ▴" : "show prompt ▾"}</button>
          </div>
          {showParent && <div className="mt8"><PromptView runId={runId} cid={round.parent_id} maxHeight={360} /></div>}
        </div>
      )}
      {round.n_fp !== undefined && (
        <div className="muted mt12">Analysed {round.n_fp} false positives and {round.n_fn} false negatives from the parent.</div>
      )}
      {round.errors.map((e, i) => <div key={i} className="errbox mt12 small">Round failed: {e}</div>)}
      <Clusters round={round} />
      {round.dropped.length > 0 && (
        <div className="warnbox mt12 small">
          {round.dropped.map((d, i) => (
            <div key={i}>Dropped a proposal for “{d.target_cluster}”: {(d.problems || []).join("; ")}</div>
          ))}
        </div>
      )}
      {children.length > 0 && (
        <div className="card mt16" style={{ padding: "14px 16px" }}>
          <div className="panel-title" style={{ marginBottom: 0 }}>Children</div>
          <div className="childcards">
            {children.map((c) => {
              const prog = round.progress[c.id];
              return (
                <div key={c.id} className={classNames("childcard", sel && c.id === sel.id && "on")} onClick={() => onSelectChild(c.id)}>
                  {c.accepted === undefined
                    ? <span className="badge badge-gray">{prog ? `${Math.round((100 * prog.done) / Math.max(1, prog.total))}%` : "pending"}</span>
                    : <StatusBadge c={c} />}
                  <div className="cid">{c.id}</div>
                </div>
              );
            })}
          </div>
          {sel && <ChildDetail key={sel.id} runId={runId} c={sel} round={round} objective={objective} />}
        </div>
      )}
    </div>
  );
}
