import React, { useRef, useState } from "react";
import { api } from "../api.js";
import { num, pct, classNames } from "../util.js";
import { dataKey, dataChecked } from "../pages/NewRun.jsx";

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

function WordList({ path, info, onChange }) {
  const ref = useRef(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  async function upload(file) {
    setBusy(true);
    setErr("");
    try {
      onChange(await api.upload("/api/wordlists", file));
    } catch (e) {
      setErr(e.message);
    } finally {
      setBusy(false);
      ref.current.value = "";
    }
  }
  const name = info?.filename || (path ? path.split(/[\\/]/).pop().replace(/^[0-9a-f]{10}__wordlist__/, "") : "");
  return (
    <div className="field" style={{ gridColumn: "span 2" }}>
      <span className="lbl">Profanity word list (optional)
        <span className="hint">plain text, one word per line; matching words in the inputs are replaced with **</span></span>
      <input type="file" ref={ref} style={{ display: "none" }} accept=".txt,.lst,.csv,text/plain"
             onChange={(e) => e.target.files[0] && upload(e.target.files[0])} />
      {path ? (
        <div className="row">
          <span title={info?.preview ? `e.g. ${info.preview.join(", ")}` : path}>
            ✔ <b>{name}</b>{info ? ` — ${num(info.n_words)} words` : ""}</span>
          <button className="linkbtn small" disabled={busy} onClick={() => ref.current.click()}>replace</button>
          <button className="linkbtn small" onClick={() => onChange(null)}>remove</button>
        </div>
      ) : (
        <div className="row">
          <button className="btn" disabled={busy} onClick={() => ref.current.click()}>{busy ? "Uploading…" : "Upload word list…"}</button>
          <span className="muted small">none — inputs are used as is</span>
        </div>
      )}
      {err && <div className="errtext small mt8">{err}</div>}
    </div>
  );
}

function FileSlot({ title, hint, info, onLoaded, onClear, clearLabel, busyLabel }) {
  return (
    <div className="field mt12">
      <span className="lbl">{title}{hint && <span className="hint">{hint}</span>}</span>
      {info ? (
        <div className="row">
          <span>✔ <b>{info.filename}</b> — {num(info.n_rows)} rows, {info.columns.length} columns</span>
          <button className="linkbtn small" onClick={onClear}>{clearLabel || "replace"}</button>
        </div>
      ) : <FileLoader onLoaded={onLoaded} busyLabel={busyLabel} />}
    </div>
  );
}

function FilesTable({ files }) {
  const order = ["full", "train", "test"].filter((k) => files[k]);
  if (!order.length) return null;
  return (
    <table className="plain mt8" style={{ maxWidth: 760 }}>
      <thead><tr><th>file</th><th>rows</th><th>usable</th><th>positive</th><th>positive rate</th><th>duplicate ids</th></tr></thead>
      <tbody>
        {order.map((k) => {
          const f = files[k];
          return (
            <tr key={k}>
              <td><b>{k}</b></td><td>{num(f.n_raw)}</td><td>{num(f.n_usable)}</td><td>{num(f.n_positive)}</td>
              <td>{pct(f.positive_rate)}</td><td className={f.duplicate_ids ? "errtext" : ""}>{num(f.duplicate_ids)}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

export default function StepData({ cfg, set, st, patch }) {
  const d = cfg.data;
  const provided = d.split.mode === "provided";
  // The file whose columns drive the pickers: the full dataset, else the train file, else the test file.
  const valid = (x) => (x && Array.isArray(x.columns) ? x : null);
  const info = valid(st.dataset) || (provided ? valid(st.trainDataset) || valid(st.testDataset) : null);
  const [profile, setProfile] = useState(null);
  const [profErr, setProfErr] = useState("");
  const [checking, setChecking] = useState(false);
  const key = dataKey(d);
  const stale = profile && st.dataCheck && st.dataCheck.key !== key;

  // Pick columns from a newly loaded file. Only the first file guesses; later files keep the choices.
  function columnsFrom(res, data) {
    if (data.id_col && info) return data;
    const g = guess(res);
    patch({ autoTemplate: true });
    return { ...data, id_col: g.idCol, label_col: g.labelCol, positive_value: "", input_columns: g.inputs,
             input_template: templateFor(g.inputs) };
  }

  function onFull(res) {
    patch({ dataset: res });
    set("data", columnsFrom(res, { ...d, path: res.path }));
    setProfile(null);
  }
  function onTrain(res) {
    patch({ trainDataset: res });
    set("data", columnsFrom(res, { ...d, split: { ...d.split, train_path: res.path } }));
    setProfile(null);
  }
  function onTest(res) {
    patch({ testDataset: res });
    set("data", columnsFrom(res, { ...d, split: { ...d.split, test_path: res.path } }));
    setProfile(null);
  }
  const clearFull = () => { patch({ dataset: null }); set("data.path", null); setProfile(null); };
  const clearTrain = () => { patch({ trainDataset: null }); set("data.split.train_path", null); setProfile(null); };
  const clearTest = () => { patch({ testDataset: null }); set("data.split.test_path", null); setProfile(null); };

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
      const res = await api.post("/api/datasets/profile", { data: d, scoring_set_size: cfg.budget.scoring_set_size });
      setProfile(res);
      patch({ dataCheck: { key, ok: res.check.ok } });
    } catch (e) {
      setProfErr(e.message);
      patch({ dataCheck: { key, ok: false } });
    } finally {
      setChecking(false);
    }
  }

  const labelInfo = info && info.columns.find((c) => c.name === d.label_col);
  const labelValues = info && d.label_col
    ? [...new Set((info.preview || []).map((r) => String(r[d.label_col])))].slice(0, 30) : [];
  const canCheck = provided ? !!(d.split.train_path && d.split.test_path) : !!d.path;
  const chk = profile && profile.check;
  const passed = dataChecked(st);

  return (
    <div className="card">
      <div className="card-title">1. Annotated dataset</div>
      <p className="desc">One row per example. You need an id column, a gold label column and one or more input text columns. Parquet, CSV or Excel.</p>

      <div className="grid g5">
        <label className="field">
          <span className="lbl">Split</span>
          <select value={d.split.mode} onChange={(e) => { set("data.split.mode", e.target.value); setProfile(null); }}>
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
        <WordList path={d.profanity_wordlist} info={st.wordlist}
                  onChange={(res) => { patch({ wordlist: res }); set("data.profanity_wordlist", res ? res.path : null); setProfile(null); }} />
      </div>

      {!provided && (
        <FileSlot title="Dataset" hint="split automatically into train and test (stratified by label)"
                  info={st.dataset} onLoaded={onFull} onClear={clearFull} />
      )}
      {provided && (
        <>
          <FileSlot title="Train file (required)" hint="the optimizer scores and learns from these rows"
                    info={st.trainDataset} onLoaded={onTrain} onClear={clearTrain} busyLabel="Reading train file…" />
          <FileSlot title="Test file (required)" hint="held out; only used for the final evaluation"
                    info={st.testDataset} onLoaded={onTest} onClear={clearTest} busyLabel="Reading test file…" />
          <FileSlot title="Full dataset (optional)"
                    hint="if given, train and test are checked against it: every id must exist with the same label and input"
                    info={st.dataset} onLoaded={onFull} onClear={clearFull} clearLabel="remove" />
        </>
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

      <div className="row mt20">
        <button className="btn primary" onClick={check} disabled={checking || !canCheck}>{checking ? "Checking…" : "Check data"}</button>
        {!canCheck && <span className="muted small">{provided ? "Load a train and a test file first." : "Load a dataset first."}</span>}
        {passed && !profile && <span className="oktext small">✔ Data checked</span>}
        {stale && <span className="warnbox small">Settings changed since the last check — run it again.</span>}
        {profErr && <span className="errtext small">{profErr}</span>}
      </div>

      {profile && (
        <div className="infobox mt12">
          {chk && chk.errors.length > 0 && (
            <div className="errbox mb8 small"><b>Fix before continuing</b>{chk.errors.map((x) => <div key={x}>✕ {x}</div>)}</div>
          )}
          {chk && chk.warnings.length > 0 && (
            <div className="warnbox mb8 small">{chk.warnings.map((x) => <div key={x}>⚠ {x}</div>)}</div>
          )}
          {chk && chk.ok && <div className="oktext small mb8">✔ Data check passed{chk.warnings.length ? " (with warnings)" : ""}.</div>}
          {chk && <FilesTable files={chk.files} />}
          {profile.n_usable !== undefined && (
            <div className="kv mt12">
              <span className="k">Input length (chars)</span>
              <span>min {num(profile.input_chars.min)} · median {num(profile.input_chars.median)} · p95 {num(profile.input_chars.p95)} · max {num(profile.input_chars.max)}</span>
              <span className="k">Label values</span>
              <span className="small">{Object.entries(profile.label_values).map(([k, v]) => `${k} (${v})`).join(" · ")}</span>
            </div>
          )}
          {/* size/length notes from the profile (ids, classes and dropped rows are covered by the check) */}
          {(profile.issues || []).filter((x) => /longer than|fewer than/.test(x)).map((x) => (
            <div key={x} className="warnbox mt12 small">⚠ {x}</div>
          ))}
          {profile.example_input && (
            <details className="mt12">
              <summary className="small">Example rendered input</summary>
              <div className="prompt mt8" style={{ maxHeight: 240 }}>{profile.example_input}</div>
            </details>
          )}
        </div>
      )}

      {info && (
        <details className="mt16">
          <summary>Preview (first 20 rows of {info.filename})</summary>
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
