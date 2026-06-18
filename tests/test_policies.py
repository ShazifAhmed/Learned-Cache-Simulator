"""Tests for the classic replacement policies and their core invariants."""

import pytest

from cachesim.policies import FIFO, LFU, LRU, Belady
from cachesim.simulator import simulate


def test_capacity_must_be_positive():
    with pytest.raises(ValueError):
        LRU(0)


def test_lru_evicts_least_recently_used():
    # Access 1,2,3 into a size-2 cache, then 1 again, then 4.
    # After [1,2,3]: cache holds {2,3} (1 was evicted on the 3rd access).
    # Access 1 -> miss, evict LRU which is 2 -> cache {3,1}.
    # Access 4 -> miss, evict LRU which is 3 -> cache {1,4}.
    trace = [1, 2, 3, 1, 4]
    result = simulate(trace, LRU(2))
    assert result.hits == 0  # nothing is reused while still resident here
    assert result.policy == "LRU"


def test_perfect_locality_gives_full_hit_rate_after_warmup():
    # A working set that fits entirely in the cache should be almost all hits.
    trace = ([0, 1, 2, 3] * 100)
    result = simulate(trace, LRU(4))
    # Only the first 4 accesses are compulsory misses.
    assert result.misses == 4
    assert result.hit_rate > 0.98


@pytest.mark.parametrize("policy_cls", [LRU, LFU, FIFO, Belady])
def test_hits_plus_misses_equals_accesses(policy_cls):
    trace = [i % 50 for i in range(2_000)]
    result = simulate(trace, policy_cls(16))
    assert result.hits + result.misses == result.accesses == len(trace)


@pytest.mark.parametrize("policy_cls", [LRU, LFU, FIFO, Belady])
def test_never_exceeds_capacity(policy_cls):
    # Evictions should only happen once the cache is full.
    trace = [i for i in range(500)]  # all unique -> all compulsory misses
    cap = 32
    result = simulate(trace, policy_cls(cap))
    # 500 unique addresses, 32 slots -> 500 - 32 evictions.
    assert result.evictions == len(trace) - cap


def test_belady_is_optimal_upper_bound():
    # Bélády must have a hit rate >= every online policy on the same trace.
    trace = [i % 80 for i in range(5_000)]
    cap = 32
    belady = simulate(trace, Belady(cap)).hit_rate
    for online in (LRU, LFU, FIFO):
        assert belady >= simulate(trace, online(cap)).hit_rate - 1e-9
