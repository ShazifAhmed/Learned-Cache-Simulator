"""Online feature extraction for the learned policy.

The learned policy needs to describe each address with numbers a model can reason about.
Crucially, every feature here is computable from the **past only** — the history of
accesses seen so far — so the policy remains a legitimate *online* policy at decision
time. (Only training peeks at the future, to create labels.)

For an address ``a`` at time ``t`` we compute:

    recency        : how long since ``a`` was last touched          (small = hot)
    frequency      : how many times ``a`` has been seen             (large = popular)
    last_reuse_gap : gap between its two most recent uses           (its rhythm)
    mean_reuse_gap : average gap between consecutive uses           (smoothed rhythm)
    age            : how long ``a`` has been known                  (new vs. old)

All values are passed through log1p to compress heavy tails (access gaps span many
orders of magnitude), which keeps the linear model well-conditioned.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

# Order matters: it defines the columns of every feature vector.
FEATURE_NAMES = ("recency", "frequency", "last_reuse_gap", "mean_reuse_gap", "age")
N_FEATURES = len(FEATURE_NAMES)


class FeatureExtractor:
    """Maintains per-address statistics as a trace streams past.

    Usage is always: ``features_for(addr, t)`` to read the current description, then
    ``observe(addr, t)`` to fold that access into the running statistics.
    """

    def __init__(self) -> None:
        self._last_used: Dict[int, int] = {}
        self._freq: Dict[int, int] = {}
        self._first_seen: Dict[int, int] = {}
        self._last_reuse_gap: Dict[int, int] = {}
        self._sum_gap: Dict[int, int] = {}
        self._reuse_count: Dict[int, int] = {}

    def reset(self) -> None:
        self._last_used.clear()
        self._freq.clear()
        self._first_seen.clear()
        self._last_reuse_gap.clear()
        self._sum_gap.clear()
        self._reuse_count.clear()

    def features_for(self, addr: int, t: int) -> np.ndarray:
        """Return the feature vector describing ``addr`` using history up to ``t``."""
        last = self._last_used.get(addr)
        recency = (t - last) if last is not None else t  # unseen -> as old as the trace
        freq = self._freq.get(addr, 0)
        last_gap = self._last_reuse_gap.get(addr, recency)
        reuse_count = self._reuse_count.get(addr, 0)
        mean_gap = (self._sum_gap.get(addr, 0) / reuse_count) if reuse_count else last_gap
        first = self._first_seen.get(addr, t)
        age = t - first

        return np.array(
            [
                np.log1p(recency),
                np.log1p(freq),
                np.log1p(last_gap),
                np.log1p(mean_gap),
                np.log1p(age),
            ],
            dtype=np.float64,
        )

    def observe(self, addr: int, t: int) -> None:
        """Fold the access to ``addr`` at time ``t`` into the running statistics."""
        last = self._last_used.get(addr)
        if last is not None:
            gap = t - last
            self._last_reuse_gap[addr] = gap
            self._sum_gap[addr] = self._sum_gap.get(addr, 0) + gap
            self._reuse_count[addr] = self._reuse_count.get(addr, 0) + 1
        else:
            self._first_seen[addr] = t
        self._last_used[addr] = t
        self._freq[addr] = self._freq.get(addr, 0) + 1
