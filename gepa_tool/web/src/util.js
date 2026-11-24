export const METRICS = [
  ["precision", "Precision"],
  ["recall", "Recall"],
  ["f1", "F1"],
  ["fbeta", "F-beta"],
  ["accuracy", "Accuracy"],
  ["specificity", "Specificity (true negative rate)"],
  ["npv", "Negative predictive value"],
  ["fpr", "False positive rate"],
  ["fnr", "False negative (miss) rate"],
  ["predicted_positive_rate", "Share of cases predicted positive"],
];
export const METRIC_LABEL = Object.fromEntries(METRICS);
export const LOWER_IS_BETTER = new Set(["fpr", "fnr", "predicted_positive_rate"]);

export const pct = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "–" : `${(v * 100).toFixed(d)}%`);
export const pts = (v) => (v === null || v === undefined || Number.isNaN(v) ? "–" : (v >= 0 ? "+" : "") + (v * 100).toFixed(1));
export const num = (v) => (v === null || v === undefined ? "–" : Number(v).toLocaleString());
export const short = (v) => (v === null || v === undefined ? "–" : Math.round(v * 100));

export function duration(sec) {
  if (sec === null || sec === undefined || Number.isNaN(sec)) return "–";
  sec = Math.max(0, Math.round(sec));
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${s}s`;
  return `${s}s`;
}

export function describeConstraint(c) {
  return `${METRIC_LABEL[c.metric] || c.metric} ${c.op === ">=" ? "≥" : "≤"} ${Number(c.value).toFixed(2)}`;
}

export function summarizeObjective(obj) {
  const target = obj.maximize === "fbeta" ? `F-beta (beta=${obj.beta ?? "?"})` : METRIC_LABEL[obj.maximize];
  const cons = (obj.constraints || []).map(describeConstraint);
  return cons.length ? `maximize ${target} subject to ${cons.join("; ")}.` : `maximize ${target} (no hard constraints).`;
}

export function classNames(...xs) {
  return xs.filter(Boolean).join(" ");
}

export const STATUS_CLASS = {
  running: "badge-blue",
  completed: "badge-green",
  failed: "badge-red",
  stopped: "badge-amber",
  unknown: "badge-gray",
};
