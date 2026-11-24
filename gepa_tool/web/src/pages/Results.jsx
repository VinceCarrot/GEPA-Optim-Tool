import React, { useEffect, useMemo, useState } from "react";
import { api } from "../api.js";
import { useRunEvents } from "../runEvents.js";
import { pct, pts, duration, classNames, STATUS_CLASS, METRIC_LABEL } from "../util.js";
import { DiffView, PromptView, Constraints } from "../components/common.jsx";

const ROWS = [
  ["precision", "Precision"], ["recall", "Recall"], ["f1", "F1"], ["specificity", "Specificity"],
  ["accuracy", "Accuracy"], ["predicted_positive_rate", "Pred. positive"],
];

function CompareTable({ fr, objective }) {
  const rows = [...ROWS];
  if (objective && !rows.some(([k]) => k === objective)) rows.push([objective, METRIC_LABEL[objective] || objective]);
  const cols = [
    ["seed_scoring", "Seed (scoring set)"], ["scoring", "Best (scoring set)"],
    ["seed_test", "Seed (test)"], ["test", "Best (test)"],
  ].filter(([k]) => fr[k]);
  const hasTest = !!fr.test;
  const cell = (k, metric) => (fr[k] ? pct(fr[k][metric]) : "–");
  const delta = (a, b, metric) => (fr[a] && fr[b] ? pts(fr[b][metric] - fr[a][metric]) : "");
  return (
    <div className="scrollbox">
      <table className="plain">
        <thead>
          <tr><th></th>
            {cols.slice(0, 2).map(([, l]) => <th key={l}>{l}</th>)}<th>Δ (pts)</th>
            {hasTest && cols.slice(2).map(([, l]) => <th key={l}>{l}</th>)}{hasTest && <th>Δ (pts)</th>}
          </tr>
        </thead>
        <tbody>
          {rows.map(([k, label]) => (
            <tr key={k}>
              <td>{label}{k === objective && <span className="badge badge-blue" style={{ marginLeft: 6 }}>objective</span>}</td>
              <td>{cell("seed_scoring", k)}</td><td><b>{cell("scoring", k)}</b></td><td>{delta("seed_scoring", "scoring", k)}</td>
              {hasTest && <><td>{cell("seed_test", k)}</td><td><b>{cell("test", k)}</b></td><td>{delta("seed_test", "test", k)}</td></>}
            </tr>
          ))}
          <tr>
            <td>Constraints</td>
            <td><Constraints m={fr.seed_scoring} /></td><td><Constraints m={fr.scoring} /></td><td />
            {hasTest && <><td><Constraints m={fr.seed_test} /></td><td><Constraints m={fr.test} /></td><td /></>}
          </tr>
          <tr>
            <td>Rows scored (errored)</td>
            <td>{fr.seed_scoring?.n} ({fr.seed_scoring?.n_errors})</td><td>{fr.scoring?.n} ({fr.scoring?.n_errors})</td><td />
            {hasTest && <><td>{fr.seed_test?.n} ({fr.seed_test?.n_errors})</td><td>{fr.test?.n} ({fr.test?.n_errors})</td><td /></>}
          </tr>
        </tbody>
      </table>
    </div>
  );
}

function ErrorBrowser({ runId, cid, hasTest }) {
  const [kind, setKind] = useState("FP");
  const [split, setSplit] = useState("scoring");
  const [data, setData] = useState(null);
  const [err, setErr] = useState("");
  const [open, setOpen] = useState(null);
  useEffect(() => {
    setData(null);
    setErr("");
    api.get(`/api/runs/${encodeURIComponent(runId)}/candidates/${encodeURIComponent(cid)}/errors?kind=${kind}&split=${split}&limit=200`)
      .then(setData).catch((e) => setErr(e.message));
  }, [runId, cid, kind, split]);
  return (
    <div>
      <div className="row">
        <div className="tabs">
          {["FP", "FN"].map((k) => <button key={k} className={classNames("tab", kind === k && "on")} onClick={() => setKind(k)}>
            {k === "FP" ? "False positives" : "False negatives"}</button>)}
        </div>
        <div className="tabs">
          <button className={classNames("tab", split === "scoring" && "on")} onClick={() => setSplit("scoring")}>Scoring set</button>
          {hasTest && <button className={classNames("tab", split === "test" && "on")} onClick={() => setSplit("test")}>Test set</button>}
        </div>
        {data && <span className="muted small">{data.total} total{data.total > data.rows.length ? `, showing ${data.rows.length}` : ""}</span>}
      </div>
      {err && <div className="errtext small mt8">{err}</div>}
      {data && (
        <div className="scrollbox mt8" style={{ maxHeight: 520 }}>
          <table className="plain">
            <thead><tr><th>id</th><th>predicted</th><th>risk</th><th>model reasoning</th></tr></thead>
            <tbody>
              {data.rows.map((r) => (
                <React.Fragment key={r.id}>
                  <tr className="click" onClick={() => setOpen(open === r.id ? null : r.id)}>
                    <td className="mono small">{r.id}</td><td>{r.predicted}</td>
                    <td>{typeof r.risk_score === "number" ? r.risk_score.toFixed(2) : "–"}</td>
                    <td className="small">{String(r.reason || "").slice(0, 260)}</td>
                  </tr>
                  {open === r.id && <tr><td colSpan={4}><div className="prompt" style={{ maxHeight: 320 }}>{r.input}</div>
                    <div className="small mt8"><b>Reasoning:</b> {r.reason}</div></td></tr>}
                </React.Fragment>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default function Results({ runId }) {
  const [run, setRun] = useState(null);
  const [err, setErr] = useState("");
  const { state: s, ended } = useRunEvents(runId);
  const [cid, setCid] = useState(null);
  const [tab, setTab] = useState("full");

  useEffect(() => {
    api.get(`/api/runs/${encodeURIComponent(runId)}`).then(setRun).catch((e) => setErr(e.message));
  }, [runId, ended]);

  const fr = run?.final_result;
  const bestId = fr?.best_candidate_id || s.bestId;
  const shown = cid || bestId;
  const objective = run?.config?.objective?.maximize || s.start?.objective?.maximize;

  const accepted = useMemo(() => s.order.map((id) => s.cands[id]).filter((c) => c && c.id !== "seed" && c.accepted && c.metrics), [s]);
  const lineage = useMemo(() => {
    const out = [];
    let id = bestId;
    while (id && s.cands[id] && out.length < 200) { out.unshift(s.cands[id]); id = s.cands[id].parent_id; }
    return out;
  }, [s, bestId]);

  if (err) return <div className="errbox">{err}</div>;
  if (!run) return <div className="card muted">Loading…</div>;
  return (
    <div>
      <div className="card run-head">
        <div className="row top">
          <div>
            <h1>{run.name}</h1>
            <span className={`badge ${STATUS_CLASS[run.status] || "badge-gray"}`}>{run.status}</span>
            <div className="objective">{run.objective_text}</div>
          </div>
          <div className="spacer" />
          <a className="btn" href={`#/runs/${encodeURIComponent(runId)}`}>← Monitor</a>
          {fr && <a className="btn primary" href={`/api/runs/${encodeURIComponent(runId)}/best_prompt`}>Download best prompt</a>}
        </div>
        <div className="small muted mt8">
          {run.rounds_done} / {run.total_rounds} rounds · {accepted.length} accepted candidates
          {s.end?.duration ? ` · took ${duration(s.end.duration)}` : ""}
        </div>
      </div>

      <div className="card">
        <div className="card-title">Best vs seed</div>
        {fr ? (
          <>
            <p className="desc">Best candidate <b className="mono">{fr.best_candidate_id}</b> — chosen on the scoring set (feasible first, then {METRIC_LABEL[objective] || objective}).
              {fr.test ? " Test-set numbers are the held-out check." : fr.status === "completed" ? " Test evaluation was disabled." : " The run did not finish, so there is no test-set evaluation."}</p>
            {["scoring", "seed_scoring", "test", "seed_test"].filter((k) => fr[k]?.inference_failed).map((k) => (
              <div key={k} className="warnbox small mb8">⚠ {k.replace("_", " ")}: {fr[k].n_errors} rows ({pct(fr[k].error_rate)}) still
                failed with API errors after the retry passes; its metrics cover only the rows that succeeded.</div>
            ))}
            <CompareTable fr={fr} objective={objective} />
          </>
        ) : (
          <p className="muted">Results are written when the run ends{run.status === "running" ? " — it is still running." : "."}
            {s.best && <> Current best: <b className="mono">{s.bestId}</b> P {pct(s.best.precision)} · R {pct(s.best.recall)}.</>}</p>
        )}
      </div>

      {lineage.length > 1 && (
        <div className="card">
          <div className="card-title">Lineage of the best prompt</div>
          <div className="row" style={{ gap: 6 }}>
            {lineage.map((c, i) => (
              <React.Fragment key={c.id}>
                {i > 0 && <span className="faint">→</span>}
                <button className={classNames("chip", c.id === shown && "on")} onClick={() => setCid(c.id)}>
                  {c.id} <span className="small">{c.metrics ? `P ${pct(c.metrics.precision, 0)} R ${pct(c.metrics.recall, 0)}` : ""}</span>
                </button>
              </React.Fragment>
            ))}
          </div>
          {lineage.slice(1).map((c) => c.rationale && (
            <div key={c.id} className="small mt8"><b className="mono">{c.id}</b> (round {c.round}): {c.rationale}</div>
          ))}
        </div>
      )}

      {shown && (
        <div className="card">
          <div className="row mb8">
            <div className="card-title" style={{ marginBottom: 0 }}>Prompt <span className="mono">{shown}</span>{shown === bestId && " (best)"}</div>
            <div className="spacer" />
            <div className="tabs">
              <button className={classNames("tab", tab === "full" && "on")} onClick={() => setTab("full")}>Full prompt</button>
              <button className={classNames("tab", tab === "seed" && "on")} onClick={() => setTab("seed")}>Diff vs seed</button>
              <button className={classNames("tab", tab === "parent" && "on")} onClick={() => setTab("parent")}>Diff vs parent</button>
            </div>
          </div>
          {tab === "full" && <PromptView runId={runId} cid={shown} />}
          {tab === "seed" && (shown === "seed" ? <div className="muted small">This is the seed.</div> : <DiffView runId={runId} cid={shown} against="seed" />)}
          {tab === "parent" && <DiffView runId={runId} cid={shown} />}
        </div>
      )}

      {shown && (
        <div className="card">
          <div className="card-title">Errors of <span className="mono">{shown}</span></div>
          <ErrorBrowser runId={runId} cid={shown} hasTest={!!fr?.test && (shown === bestId || shown === "seed")} />
        </div>
      )}

      {accepted.length > 0 && (
        <div className="card">
          <div className="card-title">Accepted candidates</div>
          <div className="scrollbox" style={{ maxHeight: 460 }}>
            <table className="plain">
              <thead><tr><th>id</th><th>round</th><th>parent</th><th>Precision</th><th>Recall</th><th>F1</th><th>Score</th><th>Constraints</th><th>rationale</th></tr></thead>
              <tbody>
                {[s.cands.seed, ...accepted].filter((c) => c && c.metrics).map((c) => (
                  <tr key={c.id} className="click" onClick={() => setCid(c.id)} style={c.id === shown ? { background: "#f6f7ff" } : undefined}>
                    <td className="mono">{c.id}{c.id === bestId && " ★"}</td><td>{c.round ?? 0}</td><td className="mono small">{c.parent_id || "–"}</td>
                    <td>{pct(c.metrics.precision)}</td><td>{pct(c.metrics.recall)}</td><td>{pct(c.metrics.f1)}</td>
                    <td>{c.metrics.optimization_score.toFixed(3)}</td><td><Constraints m={c.metrics} /></td>
                    <td className="small">{String(c.rationale || "").slice(0, 160)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
