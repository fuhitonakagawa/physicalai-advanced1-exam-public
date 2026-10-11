#!/bin/zsh -l
set -euo pipefail
cd -- "${0:A:h}"
if ! docker info >/dev/null 2>&1; then
  print -u2 'Dockerが停止しています。通常のTerminalで colima start --cpu 4 --memory 4 を実行してください。'
  exit 1
fi
export DOCKER_CONTEXT="$(docker context show)"
docker-compose up -d --build --wait
