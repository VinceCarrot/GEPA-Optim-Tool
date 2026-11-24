import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import { pct, classNames } from "../util.js";

export function Constraints({ m }) {
  if (!m || m.feasible === undefined) return null;
  return m.feasible
    ? <span className="oktext"><span className="check">✓</span>constraints met</span>
    : <span className="errtext"><span className="cross">✕</span>constraints missed</span>;
}

export function MetricLine({ m, showConstraints = true }) {
  if (!m) return <span className="muted">–</span>;
  return (
    <span>
      P <b>{pct(m.precision)}</b> · R <b>{pct(m.recall)}</b> · F1 {pct(m.f1)} · n={m.n}
      {m.n_errors ? <span className="errored"> · {m.n_errors} errored</span> : null}
      {showConstraints && <> <Constraints m={m} /></>}
    </span>
  );
}

// Unified line diff from /api/runs/<id>/candidates/<cid>/diff
export function DiffView({ runId, cid, against }) {
  const [diff, setDiff] = useState(null);
  const [err, setErr] = useState("");
  const [compact, setCompact] = useState(true);

  useEffect(() => {
    setDiff(null);
    setErr("");
    const q = against ? `?against=${encodeURIComponent(against)}` : "";
    api.get(`/api/runs/${encodeURIComponent(runId)}/candidates/${encodeURIComponent(cid)}/diff${q}`)
      .then(setDiff).catch((e) => setErr(e.message));
  }, [runId, cid, against]);

  if (err) return <div className="errtext small">{err}</div>;
  if (!diff) return <div className="muted small">Loading diff…</div>;
  if (!diff.against) return <div className="muted small">No parent to compare with.</div>;

  const lines = diff.lines;
  const added = lines.filter((l) => l.op === "+").length;
  const removed = lines.filter((l) => l.op === "-").length;
  const CONTEXT = 2;
  let items = lines.map((l, i) => ({ ...l, i }));
  if (compact) {
    const keep = new Set();
    lines.forEach((l, i) => {
      if (l.op !== " ") for (let k = i - CONTEXT; k <= i + CONTEXT; k++) keep.add(k);
    });
    const out = [];
    let skipped = 0;
    items.forEach((l) => {
      if (keep.has(l.i)) {
        if (skipped) out.push({ gap: skipped, i: `g${l.i}` });
        skipped = 0;
        out.push(l);
      } else skipped++;
    });
    if (skipped) out.push({ gap: skipped, i: "gend" });
    items = out;
  }
  return (
    <div>
      <div className="row small mb8">
        <label className="row" style={{ gap: 6 }}>
          <input type="checkbox" checked={compact} onChange={(e) => setCompact(e.target.checked)} />
          only changed lines (+{CONTEXT} context)
        </label>
        <span className="muted">· {added} added, {removed} removed{diff.against !== undefined ? ` · vs ${diff.against}` : ""}</span>
      </div>
      <div className="diff">
        {added + removed === 0 && <div className="gap">(no changes)</div>}
        {items.map((l) => l.gap ? (
          <div key={l.i} className="gap">⋯ {l.gap} unchanged line{l.gap > 1 ? "s" : ""}</div>
        ) : (
          <div key={l.i} className={classNames("ln", l.op === "+" && "add", l.op === "-" && "del")}>
            {l.op === " " ? "  " : l.op + " "}{l.text}
          </div>
        ))}
      </div>
    </div>
  );
}

export function PromptView({ runId, cid, maxHeight }) {
  const [p, setP] = useState(null);
  const [err, setErr] = useState("");
  useEffect(() => {
    setP(null);
    setErr("");
    api.get(`/api/runs/${encodeURIComponent(runId)}/candidates/${encodeURIComponent(cid)}`)
      .then((r) => setP(r.prompt)).catch((e) => setErr(e.message));
  }, [runId, cid]);
  if (err) return <div className="errtext small">{err}</div>;
  if (p === null) return <div className="muted small">Loading prompt…</div>;
  return <div className="prompt" style={maxHeight ? { maxHeight } : undefined}>{p}</div>;
}
