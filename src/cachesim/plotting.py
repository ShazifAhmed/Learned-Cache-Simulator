"""Charts for the benchmark results.

Two figures tell the whole story:

  1. bar chart  — hit rate of every policy at a single capacity (who wins right now).
  2. line chart — hit rate vs. cache size for every policy (how the gap behaves as the
                  cache grows; the ML and LRU curves both converge to Belady eventually).

Matplotlib is used with the non-interactive "Agg" backend so charts render in CI and on
headless machines without a display.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402

from cachesim.benchmark import BenchmarkReport  # noqa: E402

# Stable colours so a policy looks the same across both charts.
_COLORS = {
    "Belady": "#444444",
    "LRU": "#1f77b4",
    "LFU": "#ff7f0e",
    "FIFO": "#2ca02c",
}
_ML_COLOR = "#d62728"


def _color_for(policy: str) -> str:
    return _ML_COLOR if policy.startswith("ML") else _COLORS.get(policy, "#888888")


def plot_hit_rates(report: BenchmarkReport, path: str | Path, title: str | None = None) -> Path:
    """Bar chart of hit rate per policy at a single capacity."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ranked = report.sorted_by_hit_rate()
    names = [r.policy for r in ranked]
    rates = [r.hit_rate * 100 for r in ranked]
    colors = [_color_for(n) for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, rates, color=colors)
    ax.set_ylabel("Hit rate (%)")
    ax.set_ylim(0, max(rates) * 1.15 if rates else 1)
    ax.set_title(title or f"Hit rate by policy (cache capacity = {report.capacity})")
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_pattern_comparison(
    reports: Dict[str, BenchmarkReport],
    path: str | Path,
    title: str | None = None,
) -> Path:
    """Grouped bar chart: one group of bars per access pattern, one bar per policy.

    This is the headline figure. It shows the project's thesis at a glance: every
    classic policy has a pattern where it collapses, while the learned policy stays
    near the Bélády ceiling across all of them.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    patterns = list(reports.keys())
    # Stable policy order across groups.
    policies: list[str] = []
    for rep in reports.values():
        for r in rep.results:
            if r.policy not in policies:
                policies.append(r.policy)

    import numpy as np

    x = np.arange(len(patterns))
    width = 0.8 / max(len(policies), 1)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, policy in enumerate(policies):
        heights = []
        for pat in patterns:
            match = reports[pat].by_policy().get(policy)
            heights.append(match.hit_rate * 100 if match else 0.0)
        offset = (i - (len(policies) - 1) / 2) * width
        ax.bar(x + offset, heights, width, label=policy, color=_color_for(policy))

    ax.set_xticks(x)
    ax.set_xticklabels(patterns)
    ax.set_ylabel("Hit rate (%)")
    ax.set_xlabel("Access pattern")
    ax.set_title(title or "Hit rate by policy across access patterns")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=len(policies), fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_capacity_sweep(
    sweep: Dict[int, BenchmarkReport],
    path: str | Path,
    title: str | None = None,
) -> Path:
    """Line chart of hit rate vs. capacity, one line per policy."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    capacities = sorted(sweep.keys())
    # Collect every policy name that appears anywhere in the sweep.
    policies = []
    for cap in capacities:
        for r in sweep[cap].results:
            if r.policy not in policies:
                policies.append(r.policy)

    fig, ax = plt.subplots(figsize=(8, 5))
    for policy in policies:
        ys = []
        for cap in capacities:
            match = sweep[cap].by_policy().get(policy)
            ys.append(match.hit_rate * 100 if match else float("nan"))
        style = "--" if policy == "Belady" else "-"
        ax.plot(
            capacities,
            ys,
            style,
            marker="o",
            markersize=4,
            label=policy,
            color=_color_for(policy),
        )

    ax.set_xlabel("Cache capacity (lines)")
    ax.set_ylabel("Hit rate (%)")
    ax.set_title(title or "Hit rate vs. cache capacity")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
