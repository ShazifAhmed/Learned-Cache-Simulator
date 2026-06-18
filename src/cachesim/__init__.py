"""learned-cache-sim: benchmark a learned cache-replacement policy against classic baselines.

Public API re-exports the pieces most users need so they can write, e.g.:

    from cachesim import simulate, LRU, generate_trace
"""

from cachesim.policies import FIFO, LFU, LRU, Belady, Policy
from cachesim.simulator import SimResult, simulate
from cachesim.trace import generate_trace, load_trace, save_trace

__all__ = [
    "Policy",
    "LRU",
    "LFU",
    "FIFO",
    "Belady",
    "simulate",
    "SimResult",
    "generate_trace",
    "load_trace",
    "save_trace",
]

__version__ = "0.1.0"
