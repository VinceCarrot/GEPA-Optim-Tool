"""CLI: python -m gepa_tool.engine run CONFIG --run-dir DIR | trial CONFIG --out DIR [--n 50]"""

import argparse
import sys

from .spec import load_config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="gepa_tool.engine")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a full optimization")
    r.add_argument("config")
    r.add_argument("--run-dir", required=True)
    r.add_argument("--max-rounds", type=int, default=None, help="override budget.max_rounds")
    r.add_argument("--scoring-set-size", type=int, default=None)
    r.add_argument("--no-final-test", action="store_true")
    t = sub.add_parser("trial", help="score the seed prompt on a small balanced sample")
    t.add_argument("config")
    t.add_argument("--out", required=True)
    t.add_argument("--n", type=int, default=50)
    v = sub.add_parser("validate", help="validate a config file")
    v.add_argument("config")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.cmd == "validate":
        print("config OK:", cfg.name)
        return 0
    if args.cmd == "trial":
        from .trial import run_trial_sync
        return run_trial_sync(cfg, args.out, args.n)
    if args.max_rounds is not None:
        cfg.budget.max_rounds = args.max_rounds
    if args.scoring_set_size is not None:
        cfg.budget.scoring_set_size = args.scoring_set_size
    if args.no_final_test:
        cfg.budget.final_test_eval = False
    from .optimizer import run
    return run(cfg, args.run_dir)


if __name__ == "__main__":
    sys.exit(main())