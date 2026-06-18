#!/usr/bin/env bash
#
# Capture real memory-access traces from instrumented C workloads and write them to
# data/traces/ in the project's one-address-per-line format.
#
# Each workload is a real algorithm (see tools/gen_real_trace.c); the trace is the
# genuine sequence of cache lines it touches. Raw line addresses are remapped to a
# dense 0..K id space (first-seen order) so the committed files are compact and stable
# regardless of address-space layout randomisation.
#
# Usage: scripts/capture_traces.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/data/traces"
BIN="$(mktemp -d)/gen_real_trace"
mkdir -p "$DEST"

echo "Compiling tools/gen_real_trace.c ..."
cc -O2 -o "$BIN" "$ROOT/tools/gen_real_trace.c"

# densify: map raw line addresses to dense ids in first-seen order.
densify() { awk '{ if (!($1 in m)) m[$1] = n++; print m[$1] }'; }

capture() { # name workload size max
    local name="$1" workload="$2" size="$3" max="$4"
    echo "  capturing $name ($workload, size=$size, max=$max accesses)"
    "$BIN" "$workload" "$size" "$max" | densify > "$DEST/$name.txt"
    echo "    -> $DEST/$name.txt ($(wc -l < "$DEST/$name.txt") accesses)"
}

echo "Capturing real workloads:"
capture matmul     matmul     48   20000
capture linkedlist linkedlist 1500 20000
capture bst        bst        4000 20000

echo
echo "Done. Benchmark a real trace with, e.g.:"
echo "    cachesim run --trace data/traces/bst.txt --capacity 64"
