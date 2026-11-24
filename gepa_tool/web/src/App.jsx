import React, { useEffect, useState } from "react";
import Runs from "./pages/Runs.jsx";
import NewRun from "./pages/NewRun.jsx";
import Monitor from "./pages/Monitor.jsx";
import Results from "./pages/Results.jsx";

// Hash routes: #/runs, #/new, #/runs/<id>, #/runs/<id>/results
function useRoute() {
  const [hash, setHash] = useState(window.location.hash || "#/runs");
  useEffect(() => {
    const on = () => setHash(window.location.hash || "#/runs");
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return hash.replace(/^#/, "").split("/").filter(Boolean);
}

export function navigate(path) {
  window.location.hash = path;
}

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }
  static getDerivedStateFromError(error) {
    return { error };
  }
  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="errbox">
        <b>This page hit an error:</b> {String(this.state.error.message || this.state.error)}
        <div className="row mt12">
          <button className="btn" onClick={() => this.setState({ error: null })}>Try again</button>
          <button className="btn" onClick={() => { localStorage.removeItem("gepa_wizard_v1"); window.location.reload(); }}>
            Reset the setup wizard
          </button>
        </div>
      </div>
    );
  }
}

export default function App() {
  const parts = useRoute();
  let page;
  let active = "runs";
  if (parts[0] === "new") {
    page = <NewRun />;
    active = "new";
  } else if (parts[0] === "runs" && parts[1] && parts[2] === "results") {
    page = <Results runId={decodeURIComponent(parts[1])} key={"r" + parts[1]} />;
  } else if (parts[0] === "runs" && parts[1]) {
    page = <Monitor runId={decodeURIComponent(parts[1])} key={"m" + parts[1]} />;
  } else {
    page = <Runs />;
  }
  return (
    <>
      <nav className="topnav">
        <a className="brand" href="#/runs">GEPA Prompt Optimizer</a>
        <a className={active === "runs" ? "on" : ""} href="#/runs">Runs</a>
        <a className={active === "new" ? "on" : ""} href="#/new">New run</a>
      </nav>
      <main className="page"><ErrorBoundary key={parts.join("/")}>{page}</ErrorBoundary></main>
    </>
  );
}
