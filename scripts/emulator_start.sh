#!/usr/bin/env bash
# Настроить эмулятор: N терминалов из справочника validate шлют пакеты на ndtp-server:9201.
# ./scripts/emulator_start.sh [N=11] [intervalMs=1000];  остановить: ./scripts/emulator_start.sh stop
set -e
API=http://localhost:18080/api/config
if [ "$1" = "stop" ]; then
  curl -s -X POST $API -H 'Content-Type: application/json' -d '{"targetHost":"ndtp-server","targetPort":9201,"units":[]}'; echo; exit
fi
N=${1:-11}; MS=${2:-1000}
HERE="$(cd "$(dirname "$0")/.." && pwd)"
UNITS=$(tail -n +2 "$HERE/data/validate/traffic.csv" | cut -d, -f3 | sort -u | head -n "$N")
BODY=$(printf '%s\n' $UNITS | awk -v ms="$MS" 'BEGIN{printf "{\"targetHost\":\"ndtp-server\",\"targetPort\":9201,\"units\":["}
  {printf "%s{\"unitId\":%s,\"intervalMs\":%s,\"autoGenerate\":true,\"cells\":[]}", (NR>1?",":""), $1, ms}
  END{print "]}"}')
curl -s -X POST $API -H 'Content-Type: application/json' -d "$BODY"; echo
