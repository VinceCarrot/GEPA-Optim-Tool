import React, { useRef, useState } from "react";
import { api } from "../api.js";
import { num, pct, classNames } from "../util.js";

function templateFor(cols) {
  if (cols.length === 0) return "";
  if (cols.length === 1) return `{${cols[0]}}`;
  return cols.map((c) => `${c}: {${c}}`).join("\n");
}

// Best-effort column guesses after a file is loaded.
function guess(info) {
  const cols = info.columns;
  const n = info.n_rows;
  const idCol = (cols.find((c) => /(^id$|_id$|^id_|case|_no$|number)/i.test(c.name) && c.n_unique >= n * 0.95)
    || cols.find((c) => c.n_unique === n) || cols[0] || {}).name || "";
  const labelCol = (cols.find((c) => c.name !== idCol && c.n_unique >= 2 && c.n_unique <= 20) || {}).name || "";
  const avgLen = (name) => {
    const vals = (info.preview || []).map((r) => String(r[name] ?? ""));
    return vals.reduce((a, v) => a + v.length, 0) / Math.max(1, vals.length);
  };
  const textCols = cols.filter((c) => c.name !== idCol && c.name !== labelCol && /object|str|string/.test(c.dtype));
  textCols.sort((a, b) => avgLen(b.name) - avgLen(a.name));
  const inputs = textCols.length ? [textCols[0].name] : [];
  return { idCol, labelCol, inputs };
}

function FileLoader({ onLoaded, busyLabel }) {
  const [path, setPath] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const fileRef = useRef(null);

  async function run(fn) {
    setBusy(true);
    setErr("");
    try {
      onLoaded(await fn());
    } catch (e) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <div className="row">
        <input type="file" ref={fileRef} style={{ display: "none" }} accept=".parquet,.csv,.xlsx,.xls"
               onChange={(e) => e.target.files[0] && run(() => api.upload("/api/datasets", e.target.files[0]))} />
        <button className="btn" disabled={busy} onClick={() => fileRef.current.click()}>Upload file…</button>
        <span className="muted">or use a file on this machine:</span>
        <input style={{ flex: 1, minWidth: 260 }} placeholder="/absolute/path/to/data.parquet" value={path}
               onChange={(e) => setPath(e.target.value)} />
        <button className="btn" disabled={busy || !path.trim()}
                onClick={() => run(() => api.post("/api/datasets/local", { path: path.trim() }))}>Use path</button>
      </div>
      {busy && <div className="muted small mt8">{busyLabel || "Reading file…"}</div>}
      {err && <div className="errtext small mt8">{err}</div>}
    </>
  );
}

export default function StepData({ cfg, set, st, patch }) {
  const d = cfg.data;
  const info = st.dataset;
  const [profile, setProfile] = useState(null);
  const [profErr, setProfErr] = useState("");
  const [checking, setChecking] = useState(false);
  const provided = d.split.mode === "provided";

  function onLoaded(res) {
    const g = guess(res);
    patch({ dataset: res, autoTemplate: true });
    set("data", {
      ...d,
      path: res.path,
      id_col: g.idCol,
      label_col: g.labelCol,
      positive_value: "",
      input_columns: g.inputs,
      input_template: templateFor(g.inputs),
      split: { ...d.split, train_path: provided ? res.path : d.split.train_path },
    });
    setProfile(null);
  }

  function toggleInput(col) {
    const cols = d.input_columns.includes(col) ? d.input_columns.filter((c) => c !== col) : [...d.input_columns, col];
    let data = { ...d, input_columns: cols };
    if (st.autoTemplate) data.input_template = templateFor(cols);
    set("data", data);
  }

  async function check() {
    setChecking(true);
    setProfErr("");
    setProfile(null);
    try {
      setProfile(await api.post("/api/datasets/profile", { data: d }));
    } catch (e) {
      setProfErr(e.message);
    } finally {
      setChecking(false);
    }
  }

  const labelInfo = info && info.columns.find((c) => c.name === d.label_col);
  const labelValues = info && d.label_col
    ? [...new Set((info.preview || []).map((r) => String(r[d.label_col])))].slice(0, 30) : [];

  return (
    <div className="card">
      <div className="card-title">1. Annotated dataset</div>
      <p className="desc">One row per example. You need an id column, a gold label column and one or more input text columns. Parquet, CSV or Excel.</p>
      <FileLoader onLoaded={onLoaded} />
      {info && (
        <div className="mt8">
          Loaded <b>{info.filename}</b> — {num(info.n_rows)} rows, {info.columns.length} columns
          {provided && <span className="muted"> (training file)</span>}
        </div>
      )}

      {info && (
        <>
          <div className="grid g3 mt20" style={{ maxWidth: 1250 }}>
            <label className="field">
              <span className="lbl">ID column</span>
              <select value={d.id_col} onChange={(e) => set("data.id_col", e.target.value)}>
                <option value="">—</option>
                {info.columns.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
              </select>
            </label>
            <label className="field">
              <span className="lbl">Gold label column</span>
              <select value={d.label_col} onChange={(e) => { set("data.label_col", e.target.value); }}>
                <option value="">—</option>
                {info.columns.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
              </select>
            </label>
            <label className="field">
              <span className="lbl">Value that means POSITIVE</span>
              <input list="label-values" value={d.positive_value} onChange={(e) => set("data.positive_value", e.target.value)} />
              <datalist id="label-values">{labelValues.map((v) => <option key={v} value={v} />)}</datalist>
              {labelInfo && <div className="sub">{labelInfo.n_unique} distinct values; everything else counts as negative</div>}
            </label>
          </div>

          <div className="mt20">
            <div className="mb8"><b>Input columns</b> <span className="muted">(what the model reads)</span></div>
            <div className="row" style={{ gap: 8 }}>
              {info.columns.map((c) => (
                <label key={c.name} className={classNames("chk-pill", d.input_columns.includes(c.name) && "on")}>
                  <input type="checkbox" checked={d.input_columns.includes(c.name)} onChange={() => toggleInput(c.name)} />
                  {c.name}
                </label>
              ))}
            </div>
          </div>

          <label className="field mt20">
            <span className="lbl">Input template<span className="hint">— how columns are combined into the text inserted into the prompt</span></span>
            <textarea className="mono" rows={3} value={d.input_template}
                      onChange={(e) => { patch({ autoTemplate: false }); set("data.input_template", e.target.value); }} />
          </label>
        </>
      )}

      <div className="grid g5 mt20">
        <label className="field">
          <span className="lbl">Split</span>
          <select value={d.split.mode} onChange={(e) => {
            const mode = e.target.value;
            set("data.split", { ...d.split, mode, train_path: mode === "provided" ? d.path : d.split.train_path });
          }}>
            <option value="auto">Automatic stratified split</option>
            <option value="provided">I provide train and test files</option>
          </select>
        </label>
        {!provided && (
          <>
            <label className="field"><span className="lbl">Train fraction</span>
              <input type="number" step="0.01" value={d.split.train_frac} onChange={(e) => set("data.split.train_frac", Number(e.target.value))} /></label>
            <label className="field"><span className="lbl">Held-out (unused) fraction</span>
              <input type="number" step="0.01" value={d.split.dev_frac} onChange={(e) => set("data.split.dev_frac", Number(e.target.value))} /></label>
            <label className="field"><span className="lbl">Test fraction</span>
              <input type="number" step="0.01" value={d.split.test_frac} onChange={(e) => set("data.split.test_frac", Number(e.target.value))} /></label>
          </>
        )}
        {provided && (
          <div style={{ gridColumn: "span 3" }} className="field">
            <span className="lbl">Test file <span className="muted small">(the file loaded above is the training file)</span></span>
            <FileLoader busyLabel="Reading test file…" onLoaded={(res) => {
              patch({ testDataset: res });
              set("data.split", { ...d.split, train_path: d.path, test_path: res.path });
            }} />
            {st.testDataset && <div className="small mt8">Test: <b>{st.testDataset.filename}</b> — {num(st.testDataset.n_rows)} rows</div>}
          </div>
        )}
        <label className="field">
          <span className="lbl">Profanity word list (optional)</span>
          <input value={d.profanity_wordlist || ""} placeholder="path to a word list"
                 onChange={(e) => set("data.profanity_wordlist", e.target.value || null)} />
        </label>
      </div>

      <div className="row mt20">
        <button className="btn" onClick={check} disabled={checking || !d.path}>{checking ? "Checking…" : "Check data"}</button>
        {profErr && <span className="errtext small">{profErr}</span>}
      </div>

      {profile && (
        <div className="infobox mt12">
          <div className="kv">
            <span className="k">Usable rows</span><span>{num(profile.n_usable)} of {num(profile.n_raw)}</span>
            <span className="k">Positives</span><span>{num(profile.n_positive)} ({pct(profile.positive_rate)})</span>
            {profile.split && (<><span className="k">Split</span>
              <span>train {num(profile.split.train)} ({num(profile.split.train_pos)} positive) · test {num(profile.split.test)} ({num(profile.split.test_pos)} positive)</span></>)}
            <span className="k">Input length (chars)</span>
            <span>min {num(profile.input_chars.min)} · median {num(profile.input_chars.median)} · p95 {num(profile.input_chars.p95)} · max {num(profile.input_chars.max)}</span>
            <span className="k">Label values</span>
            <span className="small">{Object.entries(profile.label_values).map(([k, v]) => `${k} (${v})`).join(" · ")}</span>
          </div>
          {profile.issues.length > 0 && (
            <div className="warnbox mt12 small">{profile.issues.map((x) => <div key={x}>⚠ {x}</div>)}</div>
          )}
          {profile.issues.length === 0 && <div className="oktext small mt8">✔ No data issues found.</div>}
          <details className="mt12">
            <summary className="small">Example rendered input</summary>
            <div className="prompt mt8" style={{ maxHeight: 240 }}>{profile.example_input}</div>
          </details>
        </div>
      )}

      {info && (
        <details className="mt16">
          <summary>Preview (first 20 rows)</summary>
          <div className="scrollbox mt8" style={{ maxHeight: 380 }}>
            <table className="plain">
              <thead><tr>{info.columns.map((c) => <th key={c.name}>{c.name}<div className="faint small">{c.dtype}</div></th>)}</tr></thead>
              <tbody>
                {info.preview.map((r, i) => (
                  <tr key={i}>{info.columns.map((c) => (
                    <td key={c.name} className="small">{String(r[c.name] ?? "").slice(0, 300)}</td>
                  ))}</tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </div>
  );
}
