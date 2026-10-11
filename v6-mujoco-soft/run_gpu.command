#!/bin/zsh -l
set -euo pipefail
cd -- "${0:A:h}"
task_uv="$(command -v uv || true)"
if [[ -z "$task_uv" ]]; then
    task_uv="$HOME/.local/bin/uv"
fi
DEVICE=mps RENDERER=native "$task_uv" run python src/main.py
