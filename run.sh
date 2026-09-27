#!/usr/bin/env bash
# Запуск команды в окружении проекта (код монтируется, датасет — из DATASET_DIR или ./data):
#   ./run.sh python -m src.realtime.stream_submission
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
[ -f "$HERE/.env" ] && DATASET_DIR="${DATASET_DIR:-$(grep -E '^DATASET_DIR=' "$HERE/.env" | cut -d= -f2-)}"
DATASET_DIR="${DATASET_DIR:-$HERE/data}"
case "$DATASET_DIR" in /*) ;; *) DATASET_DIR="$HERE/$DATASET_DIR" ;; esac
exec docker run --rm -i --user "$(id -u):$(id -g)" -v "$HERE:/app" -v "$DATASET_DIR:/app/data:ro" -w /app \
  -e PYTHONPATH=/app -e HOME=/tmp pulse-route:latest "$@"
