#!/usr/bin/env bash
#
# Optional: fetch real-world access traces to run the benchmark on instead of synthetic
# data. The project works fully without this — synthetic traces are the default. Real
# traces are large, so they are intentionally NOT committed to the repo.
#
# After downloading, point the CLI at a trace with:
#     cachesim run --trace data/traces/<file>.txt --capacity 128
#
# Note: the loader expects one integer address per line (extra columns are ignored).
# You may need a tiny converter for a given source's native format; see the README.

set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/data/traces"
mkdir -p "$DEST"

echo "Real cache/memory access traces commonly used for this kind of benchmark:"
echo
echo "  1. ML Prefetching Competition (ChampSim LoadTraces)"
echo "     https://github.com/Quangmire/ChampSim  (see the ML-DPC competition data)"
echo
echo "  2. 2nd Cache Replacement Championship (CRC2) traces"
echo "     https://crc2.ece.tamu.edu/"
echo
echo "  3. SNIA IOTTA storage I/O traces (good for the SSD/page-cache angle)"
echo "     http://iotta.snia.org/traces"
echo
echo "These hosts require accepting terms and/or large downloads, so this script does not"
echo "fetch them automatically. Download the trace you want, convert it to one integer"
echo "address per line, and drop it in: $DEST"
echo
echo "Example conversion (addresses in column 3 of a CSV):"
echo "    awk -F, '{print \$3}' raw_trace.csv > $DEST/my_trace.txt"
