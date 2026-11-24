// Live run state: tail <run_dir>/events.jsonl over SSE and fold the events into a view model.
import { useEffect, useMemo, useRef, useState } from "react";

export function useRunEvents(runId) {
  const events = useRef([]);
  const seen = useRef(new Set());
  const [version, setVersion] = useState(0);
  const [connected, setConnected] = useState(false);
  const [ended, setEnded] = useState(false);

  useEffect(() => {
    events.current = [];
    seen.current = new Set();
    let pending = false;
    const es = new EventSource(`/api/runs/${encodeURIComponent(runId)}/events`);
    const bump = () => {
      if (pending) return;
      pending = true;
      setTimeout(() => { pending = false; setVersion((v) => v + 1); }, 120);
    };
    es.onopen = () => setConnected(true);
    es.onmessage = (m) => {
      const e = JSON.parse(m.data);
      if (seen.current.has(e.seq)) return;
      seen.current.add(e.seq);
      events.current.push(e);
      bump();
    };
    es.addEventListener("end", () => { setEnded(true); es.close(); bump(); });
    es.onerror = () => setConnected(false);
    return () => es.close();
  }, [runId]);

  const state = useMemo(() => reduceEvents(events.current), [version]);
  return { state, connected, ended };
}

const STAGE_KEYS = ["select", "error_analysis", "clustering", "mutation", "evaluate", "decision"];

function newRound(n) {
  return { round: n, stages: {}, children: [], dropped: [], errors: [], progress: {} };
}

export function reduceEvents(events) {
  const s = {
    start: null, end: null, rounds: {}, cands: {}, order: [], phases: [], alerts: [],
    lastActivity: null, best: null, bestId: null, roundsDone: 0, firstTs: null, lastTs: null,
  };
  const R = (n) => (s.rounds[n] = s.rounds[n] || newRound(n));
  for (const e of events) {
    if (s.firstTs === null) s.firstTs = e.ts;
    s.lastTs = e.ts;
    switch (e.type) {
      case "run_start":
        s.start = e;
        break;
      case "candidate_proposed": {
        const c = s.cands[e.candidate_id] || {};
        s.cands[e.candidate_id] = { ...c, id: e.candidate_id, parent_id: e.parent_id, round: e.round,
          target_cluster: e.target_cluster, rationale: e.rationale, generation: e.generation };
        if (!s.order.includes(e.candidate_id)) s.order.push(e.candidate_id);
        if (e.round > 0) R(e.round).children.push(e.candidate_id);
        break;
      }
      case "candidate_decision": {
        const c = s.cands[e.candidate_id] || { id: e.candidate_id, round: e.round, parent_id: e.parent_id };
        s.cands[e.candidate_id] = { ...c, accepted: e.accepted, reason_code: e.reason_code, reason: e.reason,
          status: e.status || (e.accepted ? "accepted" : "rejected"),
          metrics: e.metrics, parent_metrics: e.parent_metrics };
        if (e.candidate_id === "seed") { s.best = s.best || e.metrics; s.bestId = s.bestId || "seed"; }
        break;
      }
      case "round_start": {
        const r = R(e.round);
        Object.assign(r, { phase: e.phase, parent_id: e.parent_id, parent_metrics: e.parent_metrics, start_ts: e.ts });
        r.stages.select = "done";
        s.lastActivity = { round: e.round, stage: "select" };
        break;
      }
      case "stage": {
        const r = R(e.round);
        r.stages[e.stage] = e.status === "start" ? "running" : "done";
        if (e.round === 0 && e.status === "start") r.start_ts = r.start_ts || e.ts;
        s.lastActivity = { round: e.round, stage: e.stage, status: e.status };
        break;
      }
      case "error_sample":
        Object.assign(R(e.round), { n_fp: e.n_fp, n_fn: e.n_fn });
        break;
      case "reflection":
        Object.assign(R(e.round), { diagnosis: e.diagnosis, clusters: e.clusters, n_analyses: e.n_analyses });
        break;
      case "proposal_rejected":
        R(e.round).dropped.push(e);
        break;
      case "eval_progress": {
        const r = R(e.round ?? 0);
        r.progress[e.candidate_id] = { ...r.progress[e.candidate_id], done: e.done, total: e.total, n_errors: e.n_errors };
        break;
      }
      case "eval_retry": {
        const r = R(e.round ?? 0);
        r.progress[e.candidate_id] = { ...r.progress[e.candidate_id], retry: e.attempt, max_retry: e.max_attempts,
          retry_rows: e.n_rows };
        break;
      }
      case "inference_alert":
        s.alerts.push(e);
        break;
      case "round_error": {
        const r = R(e.round);
        r.errors.push(e.error);
        for (const k of STAGE_KEYS) if (r.stages[k] === "running") r.stages[k] = "failed";
        break;
      }
      case "round_end": {
        const r = R(e.round);
        Object.assign(r, { ended: true, failed: e.failed, duration: e.duration, end_ts: e.ts });
        if (e.failed) {
          for (const k of STAGE_KEYS) if (r.stages[k] === "running") r.stages[k] = "failed";
          if (!r.errors.length && !r.children.length) r.errors.push("No valid child prompts were produced this round.");
        }
        s.best = e.best;
        s.bestId = e.best_id;
        s.roundsDone += 1;
        break;
      }
      case "phase":
        s.phases.push(e);
        break;
      case "feasibility_alert":
        s.alerts.push(e);
        break;
      case "run_end":
        s.end = e;
        break;
      default:
        break;
    }
  }
  if (s.rounds[0]) {
    const seed = s.cands.seed;
    s.rounds[0].ended = !!(seed && seed.metrics);
    s.rounds[0].phase = (s.phases[0] || {}).phase;
    if (s.rounds[0].ended && s.rounds[0].start_ts) {
      const d = events.find((e) => e.type === "candidate_decision" && e.candidate_id === "seed");
      if (d) s.rounds[0].duration = d.ts - s.rounds[0].start_ts;
    }
  }
  s.status = s.end ? s.end.status : (s.start ? "running" : "starting");
  return s;
}

export const STAGE_LABELS = [
  ["select", "Select parent"],
  ["error_analysis", "Error analysis"],
  ["clustering", "Clustering"],
  ["mutation", "Generate children"],
  ["evaluate", "Evaluate children"],
  ["decision", "Accept / reject"],
];

export const ACTIVITY_TEXT = {
  select: "selecting parent",
  error_analysis: "error analysis",
  clustering: "clustering errors",
  mutation: "generating children",
  evaluate: "evaluating children",
  decision: "accept / reject",
  final_test: "final test-set evaluation",
};
