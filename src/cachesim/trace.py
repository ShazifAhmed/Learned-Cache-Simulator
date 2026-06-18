"""Access-trace generation and I/O.

A *trace* is just an ordered list of integer addresses (think: which memory block,
cache line, or storage page was touched, in the order it was touched). Every policy
in this project is fed the exact same trace so comparisons are apples-to-apples.

We generate traces *synthetically* so the repo runs instantly with no downloads, but
each generator mimics a real-world access pattern that caches actually see:

- "sequential"     : 0, 1, 2, 3, ...        (streaming a large array; LRU-hostile)
- "strided"        : 0, S, 2S, ...          (column-major matrix walks)
- "looping"        : a working set scanned over and over (loops over a table)
- "zipfian"        : a few hot addresses dominate (web/db key popularity)
- "mixed"          : a realistic blend of the above with noise

Real traces can be pulled separately via scripts/download_traces.sh and loaded with
load_trace(); the rest of the codebase does not care where a trace came from.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

import numpy as np

# A trace is a plain list of non-negative integer addresses.
Trace = List[int]

PATTERNS = ("sequential", "strided", "looping", "zipfian", "mixed")


def generate_trace(
    pattern: str = "mixed",
    length: int = 20_000,
    address_space: int = 2_000,
    working_set: int = 200,
    stride: int = 7,
    zipf_s: float = 1.2,
    noise: float = 0.05,
    seed: int = 0,
) -> Trace:
    """Generate a synthetic access trace.

    Args:
        pattern: one of PATTERNS. Controls the underlying access behaviour.
        length: number of accesses (length of the returned list).
        address_space: highest address + 1; the universe of distinct addresses.
        working_set: size of the "hot" region for looping/mixed patterns. This is
            the knob that decides whether a cache can hold the reused data or not.
        stride: step size for the strided pattern.
        zipf_s: skew of the zipfian pattern (higher = a few addresses get hotter).
        noise: fraction of accesses replaced by a uniformly random address. Real
            traces are never perfectly clean, so we sprinkle in randomness.
        seed: RNG seed for reproducibility. Same seed -> identical trace.

    Returns:
        A list of integer addresses of length ``length``.
    """
    if pattern not in PATTERNS:
        raise ValueError(f"unknown pattern {pattern!r}; choose from {PATTERNS}")
    if length <= 0:
        raise ValueError("length must be positive")

    rng = np.random.default_rng(seed)

    if pattern == "sequential":
        base = np.arange(length) % address_space
    elif pattern == "strided":
        base = (np.arange(length) * stride) % address_space
    elif pattern == "looping":
        # Scan a fixed working set repeatedly: a, a+1, ..., a+W-1, a, a+1, ...
        base = np.arange(length) % working_set
    elif pattern == "zipfian":
        base = _zipf(rng, address_space, length, zipf_s)
    else:  # "mixed"
        base = _mixed(rng, length, address_space, working_set, stride, zipf_s)

    trace = base.astype(np.int64)

    # Inject uniform noise so no policy can rely on a perfectly clean signal.
    if noise > 0:
        mask = rng.random(length) < noise
        trace[mask] = rng.integers(0, address_space, size=int(mask.sum()))

    return trace.tolist()


def _zipf(rng: np.random.Generator, address_space: int, length: int, s: float) -> np.ndarray:
    """Draw addresses so that rank-1 is most popular, rank-2 next, etc. (Zipf's law)."""
    ranks = np.arange(1, address_space + 1)
    weights = 1.0 / np.power(ranks, s)
    weights /= weights.sum()
    # Shuffle which address gets which popularity so it isn't always address 0.
    addresses = rng.permutation(address_space)
    return addresses[rng.choice(address_space, size=length, p=weights)]


def _mixed(
    rng: np.random.Generator,
    length: int,
    address_space: int,
    working_set: int,
    stride: int,
    zipf_s: float,
) -> np.ndarray:
    """Concatenate phases of different behaviour, like a real program's lifetime.

    Programs change behaviour over time (init, steady-state, teardown). A policy that
    only handles one regime well will lose here, which is exactly what we want to test.
    """
    phases = []
    remaining = length
    generators = ["looping", "zipfian", "strided", "sequential"]
    i = 0
    while remaining > 0:
        chunk = min(remaining, max(500, length // 6))
        kind = generators[i % len(generators)]
        if kind == "looping":
            phases.append(np.arange(chunk) % working_set)
        elif kind == "zipfian":
            phases.append(_zipf(rng, address_space, chunk, zipf_s))
        elif kind == "strided":
            start = rng.integers(0, address_space)
            phases.append((start + np.arange(chunk) * stride) % address_space)
        else:  # sequential
            start = rng.integers(0, address_space)
            phases.append((start + np.arange(chunk)) % address_space)
        remaining -= chunk
        i += 1
    return np.concatenate(phases)[:length]


def save_trace(trace: Sequence[int], path: str | Path) -> None:
    """Persist a trace as newline-delimited integers (one address per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for addr in trace:
            fh.write(f"{int(addr)}\n")


def load_trace(path: str | Path) -> Trace:
    """Load a trace.

    Supports two formats so external/real traces drop in easily:
      - ``.txt`` : one integer address per line (what save_trace writes)
      - ``.json``: a JSON array of integers, or {"trace": [...]}
    """
    path = Path(path)
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            data = data["trace"]
        return [int(x) for x in data]

    trace: Trace = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Tolerate "addr" or "addr extra columns" (real traces often have more).
            trace.append(int(line.split()[0]))
    return trace
