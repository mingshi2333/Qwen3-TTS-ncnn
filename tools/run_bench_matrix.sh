#!/usr/bin/env bash
# Full benchmark matrix: threads x precision. Run on an IDLE machine.
set -e
cd "$(dirname "$0")/.."
MODELS=${1:-models}
DATA=${2:-tests/data}

echo "CPU: $(lscpu | grep 'Model name' | sed 's/.*: *//')"
echo "date: $(date -Iseconds)"
for FP16 in 0 1; do
  for T in 1 4 8 16; do
    echo
    echo "===== threads=$T fp16=$FP16 ====="
    Q3TTS_THREADS=$T Q3TTS_FP16=$FP16 ./build/bench "$MODELS" "$DATA" 2055
  done
done
