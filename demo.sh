#!/usr/bin/env bash
# One-keystroke demo launcher for recording. Sets recording-friendly env, prints
# a clean banner, runs unbuffered so the live Cerebras timing streams cleanly.
#
#   ./demo.sh real ["a spoken/typed command"]   # physical rover (default: voice)
#   ./demo.sh sim  ["a command"]                 # MuJoCo simulation
#
# Examples:
#   ./demo.sh real                               # always-on voice on the rover
#   ./demo.sh real "move toward the orange case" # typed, no mic
#   ./demo.sh sim  "go to the orange cup"        # sim, headless-friendly
set -e
cd "$(dirname "$0")"

MODE="${1:-real}"; shift || true
CMD="${*:-voice}"

if [ "$MODE" = "sim" ]; then
  export USE_MJC=1 MAX_RPM=600 MOVE_TIME=0.4 SETTLE_TIME=0.2 CLOSE_STEPS=1
  BODY="MuJoCo simulation (onboard camera)"
else
  unset USE_MJC USE_SIM USE_MOCK
  export CLOSE_STEPS=1
  BODY="Waveshare UGV Beast (real hardware)"
fi

clear
cat <<BANNER
============================================================
  Roro  —  Cerebras x Gemma 4 31B
  voice  ->  Whisper  ->  Gemma intent  ->  3-agent loop
  agents: intent-parser + perception + safety
  body  : $BODY
============================================================
BANNER

exec ./.venv/bin/python -u main.py $CMD
