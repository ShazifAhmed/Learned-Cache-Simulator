"""Tests for the learned replacement policy.

The headline behavioural claim of the whole project — "a small learned model can beat
LRU on patterned traces" — is pinned down here as an executable test.
"""

import pytest

from cachesim.features import N_FEATURES, FeatureExtractor
from cachesim.ml_policy import MLReplacementPolicy
from cachesim.policies import LRU
from cachesim.simulator import simulate
from cachesim.trace import generate_trace


def test_feature_vector_shape():
    fx = FeatureExtractor()
    fx.observe(5, 0)
    fx.observe(5, 3)
    vec = fx.features_for(5, 4)
    assert vec.shape == (N_FEATURES,)
    assert (vec >= 0).all()  # all features are log1p of non-negative quantities


def test_using_before_training_raises():
    policy = MLReplacementPolicy(16)
    with pytest.raises(RuntimeError):
        policy.select_victim({1, 2, 3}, t=10)


def test_train_returns_self_and_marks_trained():
    trace = generate_trace(pattern="mixed", length=3_000, seed=0)
    policy = MLReplacementPolicy(16).train(trace)
    assert policy._trained is True


def test_ml_runs_and_accounts_for_all_accesses():
    trace = generate_trace(pattern="mixed", length=4_000, seed=1)
    policy = MLReplacementPolicy(32).train(trace)
    result = simulate(trace, policy, 32)
    assert result.hits + result.misses == len(trace)
    assert result.policy.startswith("ML")


def test_ml_beats_lru_on_skewed_workload():
    # The project's central claim, as an executable test. On a skewed (zipfian) workload
    # a few addresses are far hotter than the rest. LRU ignores popularity and evicts
    # hot lines just because they were briefly idle; a model that learns frequency keeps
    # them. Trained and evaluated on *different* seeds, so this measures generalization.
    train = generate_trace(pattern="zipfian", length=12_000, seed=2)
    eval_ = generate_trace(pattern="zipfian", length=12_000, seed=3)
    cap = 64

    lru = simulate(eval_, LRU(cap), cap).hit_rate
    ml = simulate(eval_, MLReplacementPolicy(cap).train(train), cap).hit_rate

    assert ml > lru, f"expected ML ({ml:.3f}) to beat LRU ({lru:.3f})"


def test_ml_does_not_collapse_on_scan():
    # On scans (LRU's worst case, ~0% hits) the learned policy should be no worse.
    train = generate_trace(pattern="sequential", length=8_000, seed=2)
    eval_ = generate_trace(pattern="sequential", length=8_000, seed=3)
    cap = 64

    lru = simulate(eval_, LRU(cap), cap).hit_rate
    ml = simulate(eval_, MLReplacementPolicy(cap).train(train), cap).hit_rate

    assert ml >= lru - 1e-3


def test_reset_preserves_trained_model():
    trace = generate_trace(pattern="mixed", length=2_000, seed=4)
    policy = MLReplacementPolicy(16).train(trace)
    policy.reset()
    # Still trained after reset; reset only clears online (per-run) state.
    assert policy._trained is True
    # And it can run again without retraining.
    result = simulate(trace, policy, 16)
    assert result.accesses == len(trace)
