"""Benchmark harness: run several policies on one trace and collect results.

This is the glue that produces the headline comparison — every classic policy plus the
learned one, all on the identical trace and capacity, reported side by side and ranked
against the Bélády optimal upper bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from cachesim.ml_policy import MLReplacementPolicy
from cachesim.policies import FIFO, LFU, LRU, Belady, Policy
from cachesim.simulator import SimResult, simulate

# Factories so each run gets a fresh, stateless policy instance.
BASELINE_FACTORIES: Dict[str, Callable[[int], Policy]] = {
    "LRU": LRU,
    "LFU": LFU,
    "FIFO": FIFO,
    "Belady": Belady,
}


@dataclass
class BenchmarkReport:
    """All policy results for a single (trace, capacity) configuration."""

    capacity: int
    results: List[SimResult]

    def sorted_by_hit_rate(self) -> List[SimResult]:
        return sorted(self.results, key=lambda r: r.hit_rate, reverse=True)

    def by_policy(self) -> Dict[str, SimResult]:
        return {r.policy: r for r in self.results}

    def ml_vs_lru_gain(self) -> Optional[float]:
        """Percentage-point hit-rate gain of the ML policy over LRU (None if absent)."""
        d = self.by_policy()
        ml = next((r for name, r in d.items() if name.startswith("ML")), None)
        lru = d.get("LRU")
        if ml is None or lru is None:
            return None
        return (ml.hit_rate - lru.hit_rate) * 100.0


def run_benchmark(
    trace: Sequence[int],
    capacity: int,
    train_trace: Optional[Sequence[int]] = None,
    include_ml: bool = True,
    ml_model: str = "logistic",
    reuse_window: int = 64,
) -> BenchmarkReport:
    """Run every baseline (and optionally the ML policy) on ``trace`` at ``capacity``.

    Args:
        trace: the evaluation trace all policies are scored on.
        capacity: cache size.
        train_trace: trace used to fit the ML policy. Defaults to ``trace`` itself;
            pass a *different* trace to demonstrate generalization rather than memorization.
        include_ml: whether to train and evaluate the learned policy.
        ml_model: "logistic" or "gradient_boosting".
        reuse_window: the ML policy's reuse-window label horizon.
    """
    results: List[SimResult] = []

    for factory in BASELINE_FACTORIES.values():
        results.append(simulate(trace, factory(capacity), capacity))

    if include_ml:
        ml = MLReplacementPolicy(capacity, model_kind=ml_model, reuse_window=reuse_window)
        ml.train(train_trace if train_trace is not None else trace)
        results.append(simulate(trace, ml, capacity))

    return BenchmarkReport(capacity=capacity, results=results)


def sweep_capacity(
    trace: Sequence[int],
    capacities: Sequence[int],
    train_trace: Optional[Sequence[int]] = None,
    include_ml: bool = True,
    ml_model: str = "logistic",
    reuse_window: int = 64,
) -> Dict[int, BenchmarkReport]:
    """Run the full benchmark across a range of cache capacities."""
    return {
        cap: run_benchmark(
            trace,
            cap,
            train_trace=train_trace,
            include_ml=include_ml,
            ml_model=ml_model,
            reuse_window=reuse_window,
        )
        for cap in capacities
    }
