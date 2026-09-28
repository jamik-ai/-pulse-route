#!/usr/bin/env bash
# Свой сервер маршрутизации OSRM для путей по дорогам (без лимитов публичных серверов).
#   ./scripts/osrm_prepare.sh                  # скачать карту Москвы (BBBike, ~85 МБ) и подготовить граф
#   docker compose --profile router up -d router   # затем ROUTER_URL=http://router:5000 в .env
set -e
DIR="$(cd "$(dirname "$0")/.." && pwd)/osrm"
PBF_URL="${PBF_URL:-https://download.bbbike.org/osm/bbbike/Moscow/Moscow.osm.pbf}"
IMG=ghcr.io/project-osrm/osrm-backend:v5.27.1
mkdir -p "$DIR" && cd "$DIR"
[ -f map.osm.pbf ] || curl -fL -o map.osm.pbf "$PBF_URL"
run() { docker run --rm --memory=2500m -v "$DIR:/data" "$IMG" "$@"; }
run osrm-extract -p /opt/car.lua -t 2 /data/map.osm.pbf
run osrm-partition /data/map.osrm
run osrm-customize /data/map.osrm
echo "готово: $DIR/map.osrm*"
