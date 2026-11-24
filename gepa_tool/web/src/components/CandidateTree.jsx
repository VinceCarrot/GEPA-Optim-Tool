import React, { useMemo } from "react";
import { short } from "../util.js";

const COL_W = 150;
const ROW_H = 44;
const R = 11;
const PAD_X = 26;
const PAD_Y = 24;
const LABEL_W = 110;

// Pre-order layout: a node's first child sits on the node's own row, later children on new rows.
function layout(cands, order) {
  const kids = {};
  const roots = [];
  for (const id of order) {
    const c = cands[id];
    if (!c) continue;
    if (c.parent_id && cands[c.parent_id]) (kids[c.parent_id] = kids[c.parent_id] || []).push(id);
    else roots.push(id);
  }
  const pos = {};
  let maxRow = -1;
  let maxDepth = 0;
  function place(id, row, depth) {
    pos[id] = { row, depth };
    maxRow = Math.max(maxRow, row);
    maxDepth = Math.max(maxDepth, depth);
    (kids[id] || []).forEach((k, i) => place(k, i === 0 ? row : maxRow + 1, depth + 1));
  }
  roots.forEach((r) => place(r, maxRow + 1, 0));
  return { pos, kids, rows: maxRow + 1, depth: maxDepth + 1 };
}

const xy = (p) => ({ x: PAD_X + p.depth * COL_W, y: PAD_Y + p.row * ROW_H });

function nodeColor(c) {
  if (c.id === "seed") return "#1f2533";
  if (c.status === "error") return "#d08a1e";
  if (c.accepted === true) return "#2f8a3e";
  if (c.accepted === false) return "#c62828";
  return "#c3c8d2";
}

export default function CandidateTree({ cands, order, selectedId, onSelect, bestId }) {
  const { pos, rows, depth } = useMemo(() => layout(cands, order), [cands, order]);
  const width = PAD_X * 2 + (depth - 1) * COL_W + LABEL_W + 20;
  const height = PAD_Y * 2 + Math.max(0, rows - 1) * ROW_H + 20;

  const edges = [];
  for (const id of order) {
    const c = cands[id];
    if (!c || !c.parent_id || !pos[c.parent_id] || !pos[id]) continue;
    const a = xy(pos[c.parent_id]);
    const b = xy(pos[id]);
    const rejected = c.accepted === false;
    const errored = c.status === "error";
    const pending = c.accepted === undefined;
    const d = a.y === b.y
      ? `M ${a.x + R} ${a.y} L ${b.x - R} ${b.y}`
      : `M ${a.x} ${a.y + R} C ${a.x} ${b.y - 10}, ${a.x + 30} ${b.y}, ${b.x - R} ${b.y}`;
    edges.push(
      <path key={`e-${id}`} d={d} fill="none"
            stroke={errored ? "#e6b866" : rejected ? "#e08a8a" : pending ? "#c3c8d2" : "#a4abb8"}
            strokeWidth={1.4} strokeDasharray={rejected || pending ? "5 4" : undefined} />,
    );
  }

  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      {edges}
      {order.map((id) => {
        const c = cands[id];
        if (!c || !pos[id]) return null;
        const { x, y } = xy(pos[id]);
        const m = c.metrics;
        const sel = id === selectedId;
        return (
          <g key={id} className="tree-node" onClick={() => onSelect(id)}>
            <title>{`${id}${c.rationale ? "\n" + c.rationale : ""}`}</title>
            {id === "seed" && <circle cx={x} cy={y} r={R + 5} fill="none" stroke="#6573e6" strokeWidth={1.4} strokeDasharray="3 3" />}
            {sel && <circle cx={x} cy={y} r={R + 4} fill="#fff" stroke="#1c2230" strokeWidth={2} />}
            <circle cx={x} cy={y} r={R} fill={nodeColor(c)} />
            {id === bestId && <text x={x - 4} y={y + 4} style={{ fill: "#fff", fontSize: 11 }}>★</text>}
            <text x={x + R + 10} y={y - 1}>{id}</text>
            <text className="sub" x={x + R + 10} y={y + 13}>
              {m ? `P ${short(m.precision)} R ${short(m.recall)}` : "evaluating…"}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
