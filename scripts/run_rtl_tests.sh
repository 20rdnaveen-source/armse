#!/usr/bin/env bash
# One command to compile + simulate the RTL locally and produce a log you can paste back.
# Usage:  bash scripts/run_rtl_tests.sh
set -u
cd "$(dirname "$0")/.."
if ! command -v iverilog >/dev/null 2>&1; then
  echo "iverilog not found. Install it:"
  echo "  Ubuntu/Debian : sudo apt-get install iverilog"
  echo "  macOS (brew)  : brew install icarus-verilog"
  echo "  Windows       : use WSL (Ubuntu) and the Ubuntu command above"
  exit 2
fi
iverilog -V 2>&1 | head -1
python3 -c "import numpy" 2>/dev/null || { echo "numpy missing: pip install numpy"; exit 2; }
make rtl-test 2>&1 | tee rtl_test_log.txt
status=${PIPESTATUS[0]}
echo
echo "Log saved to rtl_test_log.txt  (exit status $status)"
exit $status
