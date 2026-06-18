"""Tests for trace generation and I/O."""

import json

import pytest

from cachesim.trace import PATTERNS, generate_trace, load_trace, save_trace


@pytest.mark.parametrize("pattern", PATTERNS)
def test_generate_length_and_bounds(pattern):
    trace = generate_trace(pattern=pattern, length=5_000, address_space=500, seed=0)
    assert len(trace) == 5_000
    assert all(0 <= a < 500 for a in trace)


def test_generate_is_reproducible():
    a = generate_trace(pattern="mixed", length=2_000, seed=42)
    b = generate_trace(pattern="mixed", length=2_000, seed=42)
    c = generate_trace(pattern="mixed", length=2_000, seed=43)
    assert a == b
    assert a != c


def test_sequential_pattern_is_monotonic_modulo():
    trace = generate_trace(pattern="sequential", length=10, address_space=1000, noise=0.0)
    assert trace == list(range(10))


def test_invalid_pattern_raises():
    with pytest.raises(ValueError):
        generate_trace(pattern="does-not-exist")


def test_invalid_length_raises():
    with pytest.raises(ValueError):
        generate_trace(length=0)


def test_save_and_load_txt_roundtrip(tmp_path):
    trace = generate_trace(pattern="zipfian", length=1_000, seed=7)
    path = tmp_path / "t.txt"
    save_trace(trace, path)
    assert load_trace(path) == trace


def test_save_and_load_json_roundtrip(tmp_path):
    trace = [1, 2, 3, 4, 5]
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"trace": trace}))
    assert load_trace(path) == trace


def test_load_txt_tolerates_comments_and_extra_columns(tmp_path):
    path = tmp_path / "t.txt"
    path.write_text("# header comment\n10 read\n20 write\n\n30\n")
    assert load_trace(path) == [10, 20, 30]
