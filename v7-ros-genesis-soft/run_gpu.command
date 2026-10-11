#!/bin/zsh -l
set -euo pipefail
cd -- "${0:A:h}"
task_uv="$(command -v uv || true)"
if [[ -z "$task_uv" ]]; then task_uv="$HOME/.local/bin/uv"; fi
if [[ ! -f .env ]]; then cp .env.example .env; fi
"$task_uv" sync --locked
./start_ros.command
ROLE=probe "$task_uv" run python src/main.py
mkdir -p output
sim_log="output/simulation-$(TZ=Asia/Tokyo date +%Y%m%d-%H%M%S).log"
print -r -- "Genesisのログ: $PWD/$sim_log"
ROLE=simulation GENESIS_BACKEND=metal DEVICE=mps "$task_uv" run python src/main.py > "$sim_log" 2>&1 &
sim_pid=$!
trap 'kill "$sim_pid" 2>/dev/null || true' EXIT INT TERM
ROLE=controller GENESIS_BACKEND=metal DEVICE=mps "$task_uv" run python src/main.py
wait "$sim_pid"
