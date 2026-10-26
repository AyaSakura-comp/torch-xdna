#!/bin/bash
# wait (max 40 min) for the box to be quiet, then benchmark
for i in $(seq 1 80); do
  l=$(cut -d' ' -f1 /proc/loadavg); if awk "BEGIN{exit !($l < 4)}"; then break; fi; sleep 30
done
echo "start load=$(cut -d' ' -f1 /proc/loadavg) at $(date +%T)"
~/src/torch-xdna/results/run_all.sh
echo "end load=$(cut -d' ' -f1 /proc/loadavg) at $(date +%T)"
