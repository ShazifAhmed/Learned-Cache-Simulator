"""A learned replacement policy that tries to imitate Bélády's optimal.

The big idea (a simplified version of the "Hawkeye" line of research): Bélády evicts the
line reused farthest in the future, but needs an oracle. We can't see the future at run
time — but we *can* learn, offline, what future-reused lines tend to look like, and then
apply that learned judgement online.

Concretely:

  Training (offline, allowed to see the future of a training trace):
      For every access, build a feature vector (recency, frequency, reuse rhythm, age)
      and a binary label: "was this address reused again within the next W accesses?"
      Fit a classifier to predict that label.

  Inference (online, during simulation, future hidden):
      When the cache is full, score every resident with the model's predicted
      probability of near-future reuse, and evict the one least likely to be reused.

This keeps the ML small and explainable — a logistic-regression or gradient-boosted
classifier over five interpretable features — which is exactly what you want to be able
to defend line-by-line, while still beating LRU on patterned traces.
"""

from __future__ import annotations

from typing import Sequence, Set

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cachesim.features import FeatureExtractor
from cachesim.policies import Policy


def _build_model(kind: str):
    """Create the underlying scikit-learn classifier.

    Two deliberately lightweight choices:
      - "logistic"          : linear, fast, the easy-to-explain default.
      - "gradient_boosting" : trees, captures non-linear feature interactions.
    """
    if kind == "logistic":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        )
    if kind == "gradient_boosting":
        # Imported lazily so the common path doesn't pay for it.
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1)
    raise ValueError(f"unknown model kind {kind!r}; use 'logistic' or 'gradient_boosting'")


class _ConstantModel:
    """Fallback used when training labels are a single class (nothing to learn)."""

    def __init__(self, proba: float) -> None:
        self._proba = float(proba)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = np.full((len(X), 1), self._proba)
        return np.hstack([1.0 - p, p])


class MLReplacementPolicy(Policy):
    """Eviction policy driven by a learned "will this be reused soon?" classifier."""

    def __init__(
        self,
        capacity: int,
        model_kind: str = "logistic",
        reuse_window: int = 64,
        random_state: int = 0,
    ) -> None:
        """
        Args:
            capacity: cache size (lines).
            model_kind: "logistic" or "gradient_boosting".
            reuse_window: W. An access is labelled "reused soon" if the same address
                reappears within the next W accesses. Smaller W = stricter notion of
                "useful to keep". This is the policy's main tuning knob.
            random_state: seed for any stochastic model components.
        """
        super().__init__(capacity)
        self.model_kind = model_kind
        self.reuse_window = reuse_window
        self.random_state = random_state
        self._model = None
        self._extractor = FeatureExtractor()
        self._trained = False

    @property
    def name(self) -> str:
        return f"ML[{self.model_kind}]"

    # ------------------------------------------------------------------ training
    def train(self, training_trace: Sequence[int]) -> "MLReplacementPolicy":
        """Fit the classifier on a training trace (offline, future visible).

        We stream the trace, recording each address's features *before* folding the
        access in, then label it by looking ahead ``reuse_window`` steps. Training is
        the only place the future is used; inference never is.
        """
        extractor = FeatureExtractor()
        X = np.empty((len(training_trace), 0))
        feats = []
        addrs = []
        times = []

        for t, addr in enumerate(training_trace):
            feats.append(extractor.features_for(addr, t))
            addrs.append(addr)
            times.append(t)
            extractor.observe(addr, t)

        X = np.vstack(feats) if feats else np.empty((0, 1))
        y = self._make_labels(training_trace)

        if len(np.unique(y)) < 2:
            # Degenerate trace (e.g. everything reused, or nothing): learn nothing,
            # just predict the majority behaviour so inference stays well-defined.
            self._model = _ConstantModel(proba=float(y.mean()) if len(y) else 0.5)
        else:
            self._model = _build_model(self.model_kind)
            self._model.fit(X, y)

        self._trained = True
        return self

    def _make_labels(self, trace: Sequence[int]) -> np.ndarray:
        """Label[i] = 1 if trace[i] reappears within the next ``reuse_window`` accesses."""
        n = len(trace)
        labels = np.zeros(n, dtype=np.int8)
        # next_pos[addr] = the next index (> i) at which addr occurs; computed by a
        # single right-to-left sweep, so labelling is O(n) rather than O(n*W).
        next_occurrence: dict[int, int] = {}
        for i in range(n - 1, -1, -1):
            addr = trace[i]
            nxt = next_occurrence.get(addr)
            if nxt is not None and (nxt - i) <= self.reuse_window:
                labels[i] = 1
            next_occurrence[addr] = i
        return labels

    # ----------------------------------------------------------------- inference
    def record_access(self, addr: int, t: int, hit: bool) -> None:
        self._extractor.observe(addr, t)

    def select_victim(self, resident: Set[int], t: int) -> int:
        if not self._trained or self._model is None:
            raise RuntimeError("MLReplacementPolicy.train(...) must be called before use")

        residents = list(resident)
        X = np.vstack([self._extractor.features_for(a, t) for a in residents])
        # Probability each resident will be reused soon; evict the least likely.
        reuse_proba = self._model.predict_proba(X)[:, 1]
        victim_idx = int(np.argmin(reuse_proba))
        return residents[victim_idx]

    def reset(self) -> None:
        # Reset only the *online* state; the trained model is preserved across runs.
        self._extractor.reset()
