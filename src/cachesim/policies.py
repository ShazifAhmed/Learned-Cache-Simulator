"""Cache replacement policies.

A *replacement policy* answers one question: when the cache is full and we need room
for a new address, which resident address do we evict? That single decision is what
separates a good cache from a bad one.

The simulator owns the set of resident addresses. A policy only:
  1. observes every access (to maintain stats like recency or frequency), and
  2. names a victim when asked.

All policies implement the :class:`Policy` interface so the simulator can treat them
interchangeably. This is the Strategy pattern: swap the policy, keep everything else.
"""

from __future__ import annotations

import bisect
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Sequence, Set


class Policy(ABC):
    """Base class for all replacement policies.

    Lifecycle per simulation:
        bind_trace(trace)             # once, before the run (most policies ignore it)
        record_access(addr, t, hit)   # every access, in order
        select_victim(resident, t)    # only when the cache is full and we must evict
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity

    @property
    def name(self) -> str:
        return type(self).__name__

    def bind_trace(self, trace: Sequence[int]) -> None:
        """Hook for policies that need to see the whole trace up front (e.g. Belady).

        Online policies leave this as a no-op — they only ever see the past.
        """
        return None

    @abstractmethod
    def record_access(self, addr: int, t: int, hit: bool) -> None:
        """Update internal bookkeeping for the access to ``addr`` at time ``t``."""

    @abstractmethod
    def select_victim(self, resident: Set[int], t: int) -> int:
        """Return the resident address to evict at time ``t``."""

    def reset(self) -> None:
        """Clear state so the policy instance can be reused for another run."""


class LRU(Policy):
    """Least Recently Used: evict whatever was accessed longest ago.

    Intuition: the recent past predicts the near future. Works great on data with
    temporal locality; fails on streaming/scan patterns larger than the cache, where
    the line you just evicted is the next one you need (the classic "LRU thrash").
    """

    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        self._last_used: Dict[int, int] = {}

    def record_access(self, addr: int, t: int, hit: bool) -> None:
        self._last_used[addr] = t

    def select_victim(self, resident: Set[int], t: int) -> int:
        # Smallest last-used timestamp == least recently used.
        return min(resident, key=lambda a: self._last_used.get(a, -1))

    def reset(self) -> None:
        self._last_used.clear()


class LFU(Policy):
    """Least Frequently Used: evict the address touched the fewest times.

    Intuition: popular data should stay. Weakness: an address that was hot early can
    get "stuck" in the cache long after it stops being used (no aging). We break ties
    by recency so it degrades gracefully toward LRU.
    """

    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        self._freq: Dict[int, int] = defaultdict(int)
        self._last_used: Dict[int, int] = {}

    def record_access(self, addr: int, t: int, hit: bool) -> None:
        self._freq[addr] += 1
        self._last_used[addr] = t

    def select_victim(self, resident: Set[int], t: int) -> int:
        return min(resident, key=lambda a: (self._freq.get(a, 0), self._last_used.get(a, -1)))

    def reset(self) -> None:
        self._freq.clear()
        self._last_used.clear()


class FIFO(Policy):
    """First In, First Out: evict whatever was inserted earliest, ignoring usage.

    The simplest possible policy and a useful floor: any "smart" policy should beat it.
    """

    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        self._inserted_at: Dict[int, int] = {}

    def record_access(self, addr: int, t: int, hit: bool) -> None:
        # Record insertion time only on the first sighting (a miss that brings it in).
        if not hit and addr not in self._inserted_at:
            self._inserted_at[addr] = t

    def select_victim(self, resident: Set[int], t: int) -> int:
        return min(resident, key=lambda a: self._inserted_at.get(a, t))

    def reset(self) -> None:
        self._inserted_at.clear()


class Belady(Policy):
    """Bélády's optimal (OPT/MIN): evict the line reused farthest in the future.

    This is the provably best possible policy — but it cheats by looking at the future,
    so it can't run on a live system. We include it as the *upper bound*: the gap
    between a real policy and Belady tells you how much room is left to improve.

    Implementation: precompute every position each address appears at, then at eviction
    time binary-search for each resident's next future occurrence.
    """

    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        self._positions: Dict[int, List[int]] = defaultdict(list)
        self._trace_len = 0

    def bind_trace(self, trace: Sequence[int]) -> None:
        self._positions.clear()
        for i, addr in enumerate(trace):
            self._positions[addr].append(i)
        self._trace_len = len(trace)

    def record_access(self, addr: int, t: int, hit: bool) -> None:
        return None  # Belady's decision comes purely from the precomputed future.

    def _next_use(self, addr: int, t: int) -> int:
        """Index of the next use of ``addr`` strictly after ``t`` (inf if never again)."""
        occ = self._positions.get(addr)
        if not occ:
            return self._trace_len + 1
        idx = bisect.bisect_right(occ, t)
        if idx == len(occ):
            return self._trace_len + 1  # never used again -> ideal victim
        return occ[idx]

    def select_victim(self, resident: Set[int], t: int) -> int:
        # Evict whichever resident is needed farthest in the future (or never).
        return max(resident, key=lambda a: self._next_use(a, t))

    def reset(self) -> None:
        self._positions.clear()
        self._trace_len = 0
