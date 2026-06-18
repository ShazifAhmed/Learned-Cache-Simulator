"""Tests for the simulation loop and capacity sweep helper."""

from cachesim.policies import LRU
from cachesim.simulator import hit_rate_vs_capacity, simulate


def test_empty_trace_is_safe():
    result = simulate([], LRU(8))
    assert result.accesses == 0
    assert result.hit_rate == 0.0


def test_single_access_is_one_miss():
    result = simulate([42], LRU(8))
    assert result.misses == 1
    assert result.hits == 0


def test_capacity_override_takes_effect():
    trace = [i % 100 for i in range(3_000)]
    small = simulate(trace, LRU(8), capacity=8)
    large = simulate(trace, LRU(8), capacity=128)
    # A bigger cache can only help (or tie) on hit rate.
    assert large.hit_rate >= small.hit_rate
    assert small.capacity == 8 and large.capacity == 128


def test_hit_rate_is_monotonic_in_capacity():
    trace = [i % 200 for i in range(8_000)]
    results = hit_rate_vs_capacity(trace, LRU, [8, 16, 32, 64, 128])
    rates = [r.hit_rate for r in results]
    assert rates == sorted(rates)  # non-decreasing with capacity
