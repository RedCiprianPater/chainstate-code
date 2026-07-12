#!/usr/bin/env bash
# ASI-Evolve invokes this once per trial with the candidate program path as $1.
# Owner: Ciprian Florin Pater · CHAINSTATE ecosystem
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
python "$HERE/evaluator.py" "$1"
