import React, { useEffect, useRef, useState } from "react";
import { api } from "../api.js";

export default function StepSeed({ cfg, set }) {
  const ph = cfg.task.input_placeholder;
  const prompt = cfg.seed_prompt;
  const ref = useRef(null);
  const fileRef = useRef(null);
  const [lockName, setLockName] = useState("");
  const [inspect, setInspect] = useState(null);
  const [msg, setMsg] = useState("");
  const [phDraft, setPhDraft] = useState(ph);

  useEffect(() => setPhDraft(ph), [ph]);

  useEffect(() => {
    const t = setTimeout(() => {
      api.post("/api/prompt/inspect", { prompt, placeholder: ph }).then(setInspect).catch((e) => setInspect({ ok: false, errors: [e.message], protected_sections: [] }));
    }, 300);
    return () => clearTimeout(t);
  }, [prompt, ph]);

  function commitPlaceholder() {
    const v = phDraft.trim();
    if (!v || v === ph) return setPhDraft(ph);
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(v)) {
      setMsg("Placeholder name must be letters, digits and underscores (not starting with a digit).");
      return setPhDraft(ph);
    }
    setMsg("");
    set("task.input_placeholder", v);
    if (prompt.includes(`{${ph}}`)) set("seed_prompt", prompt.split(`{${ph}}`).join(`{${v}}`));
  }

  function replaceSelection(fn) {
    const el = ref.current;
    const s = el.selectionStart, e = el.selectionEnd;
    const [text, caret] = fn(prompt.slice(0, s), prompt.slice(s, e), prompt.slice(e));
    set("seed_prompt", text);
    requestAnimationFrame(() => { el.focus(); el.setSelectionRange(caret, caret); });
  }

  function insertPlaceholder() {
    replaceSelection((a, sel, b) => [a + `{${ph}}` + b, a.length + ph.length + 2]);
  }

  function lockSelection() {
    const el = ref.current;
    if (el.selectionStart === el.selectionEnd) {
      setMsg("Select the text to lock first.");
      return;
    }
    const existing = new Set((inspect?.protected_sections || []).map((x) => x.name));
    let name = lockName.trim().replace(/[^A-Za-z0-9_-]+/g, "_");
    if (!name) { let i = 1; while (existing.has(`section_${i}`)) i++; name = `section_${i}`; }
    if (existing.has(name)) { setMsg(`A locked section named "${name}" already exists.`); return; }
    const selected = prompt.slice(el.selectionStart, el.selectionEnd);
    if (selected.includes("<!-- PROTECTED") || selected.includes("<!-- /PROTECTED")) {
      setMsg("The selection overlaps an existing locked section.");
      return;
    }
    setMsg("");
    replaceSelection((a, sel, b) => {
      const block = `<!-- PROTECTED:${name} -->\n${sel}\n<!-- /PROTECTED -->`;
      return [a + block + b, a.length + block.length];
    });
    setLockName("");
  }

  function unlock(name) {
    const re = new RegExp(`<!--\\s*PROTECTED:${name}\\s*-->\\n?([\\s\\S]*?)\\n?<!--\\s*/PROTECTED\\s*-->`);
    set("seed_prompt", prompt.replace(re, "$1"));
  }

  function loadFile(f) {
    const r = new FileReader();
    r.onload = () => set("seed_prompt", String(r.result));
    r.readAsText(f, "utf-8");
  }

  const sections = inspect?.protected_sections || [];
  return (
    <div className="card">
      <div className="card-title">3. Seed prompt</div>
      <p className="desc">The starting system prompt. It must contain the input placeholder where each example's text is inserted. Select text and click <b>Lock selection</b> to mark sections the optimizer must never change (e.g. output format, compliance wording).</p>
      <div className="row">
        <span>Input placeholder name</span>
        <input style={{ width: 260 }} value={phDraft} onChange={(e) => setPhDraft(e.target.value)}
               onBlur={commitPlaceholder} onKeyDown={(e) => e.key === "Enter" && commitPlaceholder()} />
        <button className="btn" onClick={insertPlaceholder}>Insert {`{${ph}}`} at cursor</button>
        <input type="file" ref={fileRef} accept=".md,.txt" style={{ display: "none" }}
               onChange={(e) => { e.target.files[0] && loadFile(e.target.files[0]); e.target.value = ""; }} />
        <button className="btn" onClick={() => fileRef.current.click()}>Load .md / .txt</button>
        <div className="spacer" />
        <input style={{ width: 180 }} placeholder="lock name (optional)" value={lockName} onChange={(e) => setLockName(e.target.value)} />
        <button className="btn" onClick={lockSelection}>🔒 Lock selection</button>
      </div>
      <textarea ref={ref} className="mono mt8" rows={20} value={prompt} spellCheck={false}
                onChange={(e) => set("seed_prompt", e.target.value)}
                placeholder={`You are a classifier...\n\n{${ph}}`} />
      <div className="small mt8">
        <b>Locked sections:</b>{" "}
        {sections.length === 0 ? <span className="muted">none</span> : sections.map((s) => (
          <span key={s.name} className="chk-pill on" style={{ marginRight: 6 }} title={s.preview}>
            🔒 {s.name} <span className="faint">({s.chars} chars)</span>
            <button className="linkbtn small" onClick={() => unlock(s.name)}>unlock</button>
          </span>
        ))}
      </div>
      <div className="small mt8">
        {inspect && inspect.ok && !msg && <span className="muted">Prompt looks valid.</span>}
        {inspect && !inspect.ok && inspect.errors.map((e) => <div key={e} className="errtext">{e}</div>)}
        {msg && <div className="errtext">{msg}</div>}
      </div>
    </div>
  );
}
