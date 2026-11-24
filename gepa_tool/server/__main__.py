"""python -m gepa_tool.server [--port 8765]"""
import argparse

import uvicorn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    uvicorn.run("gepa_tool.server.app:app", host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()