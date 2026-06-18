"""Command-line interface for learned-cache-sim.

Subcommands:
    gen-trace   generate a synthetic trace and save it
    run         benchmark all policies on a trace at one capacity
    sweep       benchmark across a range of capacities and chart the result
    demo        end-to-end: generate train/eval traces, run everything, save charts

Run ``cachesim --help`` or ``cachesim <subcommand> --help`` for details.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from cachesim.benchmark import run_benchmark, sweep_capacity
from cachesim.trace import PATTERNS, generate_trace, load_trace, save_trace


def _print_report(report, header: str) -> None:
    print(f"\n{header}")
    print("-" * len(header))
    for r in report.sorted_by_hit_rate():
        marker = "  <- learned" if r.policy.startswith("ML") else ""
        print(
            f"  {r.policy:<18} hit_rate={r.hit_rate:6.2%}  "
            f"hits={r.hits:>7}/{r.accesses}{marker}"
        )
    gain = report.ml_vs_lru_gain()
    if gain is not None:
        verb = "above" if gain >= 0 else "below"
        print(f"\n  ML is {abs(gain):.2f} percentage points {verb} LRU.")


def _cmd_gen_trace(args: argparse.Namespace) -> int:
    trace = generate_trace(
        pattern=args.pattern,
        length=args.length,
        address_space=args.address_space,
        working_set=args.working_set,
        seed=args.seed,
    )
    save_trace(trace, args.out)
    print(f"Wrote {len(trace)} accesses ({args.pattern}) to {args.out}")
    return 0


def _resolve_trace(args: argparse.Namespace) -> List[int]:
    """Load a trace from --trace, or generate one from the pattern flags."""
    if getattr(args, "trace", None):
        return load_trace(args.trace)
    return generate_trace(
        pattern=args.pattern,
        length=args.length,
        address_space=args.address_space,
        working_set=args.working_set,
        seed=args.seed,
    )


def _cmd_run(args: argparse.Namespace) -> int:
    eval_trace = _resolve_trace(args)
    # Train on a different seed so we measure generalization, not memorization.
    train_trace = generate_trace(
        pattern=args.pattern,
        length=args.length,
        address_space=args.address_space,
        working_set=args.working_set,
        seed=args.seed + 1000,
    )
    report = run_benchmark(
        eval_trace,
        capacity=args.capacity,
        train_trace=train_trace,
        ml_model=args.model,
        reuse_window=args.reuse_window,
    )
    _print_report(report, f"Benchmark @ capacity={args.capacity}")

    if args.chart:
        from cachesim.plotting import plot_hit_rates

        out = plot_hit_rates(report, args.chart)
        print(f"\nSaved chart to {out}")
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    eval_trace = _resolve_trace(args)
    train_trace = generate_trace(
        pattern=args.pattern,
        length=args.length,
        address_space=args.address_space,
        working_set=args.working_set,
        seed=args.seed + 1000,
    )
    capacities = [int(c) for c in args.capacities.split(",")]
    sweep = sweep_capacity(
        eval_trace,
        capacities,
        train_trace=train_trace,
        ml_model=args.model,
        reuse_window=args.reuse_window,
    )
    for cap in capacities:
        _print_report(sweep[cap], f"Benchmark @ capacity={cap}")

    if args.chart:
        from cachesim.plotting import plot_capacity_sweep

        out = plot_capacity_sweep(sweep, args.chart)
        print(f"\nSaved chart to {out}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    """One command that produces the artifacts shown in the README."""
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    from cachesim.plotting import (
        plot_capacity_sweep,
        plot_hit_rates,
        plot_pattern_comparison,
    )

    # 1. Headline: compare every policy across several access patterns. Trains and
    #    evaluates on different seeds for each pattern, so the gains are generalization.
    patterns = ["zipfian", "mixed", "looping", "sequential"]
    reports = {}
    for pat in patterns:
        eval_trace = generate_trace(pattern=pat, length=args.length, seed=1)
        train_trace = generate_trace(pattern=pat, length=args.length, seed=2)
        reports[pat] = run_benchmark(
            eval_trace, capacity=args.capacity, train_trace=train_trace, ml_model=args.model
        )
        _print_report(reports[pat], f"[demo] pattern={pat} capacity={args.capacity}")
    plot_pattern_comparison(reports, outdir / "pattern_comparison.png")

    # 2. Single-pattern bar chart on the skewed (zipfian) workload.
    plot_hit_rates(
        reports["zipfian"],
        outdir / "hit_rates.png",
        title=f"Hit rate by policy (zipfian, capacity = {args.capacity})",
    )

    # 3. Capacity sweep on the mixed workload.
    eval_mixed = generate_trace(pattern="mixed", length=args.length, seed=1)
    train_mixed = generate_trace(pattern="mixed", length=args.length, seed=2)
    capacities = [16, 32, 64, 128, 256]
    sweep = sweep_capacity(
        eval_mixed, capacities, train_trace=train_mixed, ml_model=args.model
    )
    plot_capacity_sweep(sweep, outdir / "capacity_sweep.png")

    print(
        f"\nCharts written to {outdir}/: "
        "pattern_comparison.png, hit_rates.png, capacity_sweep.png"
    )
    return 0


def _add_trace_flags(p: argparse.ArgumentParser) -> None:
    """Flags shared by run/sweep for sourcing a trace."""
    p.add_argument("--trace", help="path to a trace file (.txt or .json); overrides generation")
    p.add_argument("--pattern", choices=PATTERNS, default="mixed", help="synthetic pattern")
    p.add_argument("--length", type=int, default=20_000, help="trace length")
    p.add_argument("--address-space", type=int, default=2_000, dest="address_space")
    p.add_argument("--working-set", type=int, default=200, dest="working_set")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for generation")
    p.add_argument("--model", choices=["logistic", "gradient_boosting"], default="logistic")
    p.add_argument("--reuse-window", type=int, default=64, dest="reuse_window")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cachesim",
        description="Benchmark a learned cache-replacement policy against LRU and friends.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("gen-trace", help="generate and save a synthetic trace")
    g.add_argument("--pattern", choices=PATTERNS, default="mixed")
    g.add_argument("--length", type=int, default=20_000)
    g.add_argument("--address-space", type=int, default=2_000, dest="address_space")
    g.add_argument("--working-set", type=int, default=200, dest="working_set")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", required=True, help="output path (.txt or .json)")
    g.set_defaults(func=_cmd_gen_trace)

    r = sub.add_parser("run", help="benchmark all policies at one capacity")
    _add_trace_flags(r)
    r.add_argument("--capacity", type=int, default=64)
    r.add_argument("--chart", help="optional path to save a bar chart (.png)")
    r.set_defaults(func=_cmd_run)

    s = sub.add_parser("sweep", help="benchmark across a range of capacities")
    _add_trace_flags(s)
    s.add_argument("--capacities", default="16,32,64,128,256", help="comma-separated sizes")
    s.add_argument("--chart", help="optional path to save a line chart (.png)")
    s.set_defaults(func=_cmd_sweep)

    d = sub.add_parser("demo", help="end-to-end run that writes the README charts")
    d.add_argument("--length", type=int, default=20_000)
    d.add_argument("--capacity", type=int, default=64)
    d.add_argument("--model", choices=["logistic", "gradient_boosting"], default="logistic")
    d.add_argument("--outdir", default="results")
    d.set_defaults(func=_cmd_demo)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
