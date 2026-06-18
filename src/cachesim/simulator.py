"""The cache simulator: the single loop every policy is judged by.

The model is a fully-associative cache of fixed capacity (it can hold any ``capacity``
distinct addresses at once). For each access we ask: is this address already resident?

  - Yes  -> a HIT (fast, free).
  - No   -> a MISS. If there's room, just insert it. If the cache is full, ask the
            policy which resident to evict, drop it, then insert the new address.

The headline metric is the **hit rate**: hits / total accesses. Higher is better. Every
policy sees the identical trace and identical capacity, so differences in hit rate are
attributable purely to the eviction decisions — which is the whole point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Set

from cachesim.policies import Policy


@dataclass
class SimResult:
    """Outcome of simulating one policy on one trace."""

    policy: str
    capacity: int
    accesses: int
    hits: int
    misses: int
    evictions: int

    @property
    def hit_rate(self) -> float:
        return self.hits / self.accesses if self.accesses else 0.0

    @property
    def miss_rate(self) -> float:
        return 1.0 - self.hit_rate

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.policy:<16} capacity={self.capacity:<5} "
            f"hit_rate={self.hit_rate:6.2%} "
            f"({self.hits}/{self.accesses}, evictions={self.evictions})"
        )


@dataclass
class _Cache:
    """Tiny container tracking which addresses are currently resident."""

    capacity: int
    resident: Set[int] = field(default_factory=set)

    def __contains__(self, addr: int) -> bool:
        return addr in self.resident

    @property
    def full(self) -> bool:
        return len(self.resident) >= self.capacity


def simulate(trace: Sequence[int], policy: Policy, capacity: int | None = None) -> SimResult:
    """Run ``policy`` over ``trace`` and return hit/miss statistics.

    Args:
        trace: ordered sequence of integer addresses.
        policy: a Policy instance. Its ``capacity`` is used unless ``capacity`` is given.
        capacity: optional override for the cache size.

    Returns:
        A :class:`SimResult` with hit/miss/eviction counts.
    """
    cap = capacity if capacity is not None else policy.capacity
    if cap != policy.capacity:
        # Keep policy and cache in lockstep if the caller overrides capacity.
        policy.capacity = cap

    policy.reset()
    policy.bind_trace(trace)

    cache = _Cache(capacity=cap)
    hits = misses = evictions = 0

    for t, addr in enumerate(trace):
        hit = addr in cache
        if hit:
            hits += 1
        else:
            misses += 1
            if cache.full:
                victim = policy.select_victim(cache.resident, t)
                cache.resident.discard(victim)
                evictions += 1
            cache.resident.add(addr)

        # The policy observes every access *after* the hit/miss is known, so it can
        # update recency/frequency stats (and so FIFO can timestamp insertions).
        policy.record_access(addr, t, hit)

    return SimResult(
        policy=policy.name,
        capacity=cap,
        accesses=len(trace),
        hits=hits,
        misses=misses,
        evictions=evictions,
    )


def hit_rate_vs_capacity(
    trace: Sequence[int],
    policy_factory,
    capacities: Sequence[int],
) -> List[SimResult]:
    """Sweep cache capacity for one policy, returning a result per capacity.

    Args:
        trace: the access trace.
        policy_factory: a callable ``capacity -> Policy`` (so we get a fresh policy
            per capacity rather than reusing stale state).
        capacities: cache sizes to evaluate.
    """
    results = []
    for cap in capacities:
        policy = policy_factory(cap)
        results.append(simulate(trace, policy, cap))
    return results
