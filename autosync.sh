#!/bin/bash
# Sync the export repo to GitHub every SYNC_INTERVAL seconds until the sweep ends.
INTERVAL=${SYNC_INTERVAL:-7200}
EX=/workspace/rql_export
while true; do
  bash "$EX/sync.sh" >> "$EX/autosync.log" 2>&1
  if ! pgrep -f "python launch_sweep.py" > /dev/null; then
    echo "$(date -u '+%F %T UTC') sweep no longer running; final sync done, exiting" >> "$EX/autosync.log"
    break
  fi
  sleep "$INTERVAL"
done
