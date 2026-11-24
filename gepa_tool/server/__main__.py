"""python -m gepa_tool.server [--port 8765] [--local-backend]

SafeChain is the LLM backend by default. --local-backend (or an alias defined by the gitignored
engine/local_backend.py) switches this server and the engine processes it starts to that module.
"""
import argparse
import sys

import uvicorn

from ..engine import llm as llm_mod


def main():
    argv = llm_mod.apply_backend_flag(sys.argv[1:])
    ap = argparse.ArgumentParser(prog="gepa_tool.server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(argv)
    print(f"LLM backend: {llm_mod.backend_name()}", flush=True)
    uvicorn.run("gepa_tool.server.app:app", host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
