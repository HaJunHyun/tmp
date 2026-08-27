#!/bin/bash
# Refresh the export tree from the live sweep and commit (and push, if a remote is set).
#   ./sync.sh            -> commit + push
#   ./sync.sh --no-push  -> commit only
set -u
EX="$(cd "$(dirname "$0")" && pwd)"
REPO=/workspace/rql
RESULTS=/workspace/rql_results
VENV=/venv/main/bin/activate
[ -f "$VENV" ] && . "$VENV" >/dev/null 2>&1

# 1) code + patches (they can change as the setup is fixed up)
cp "$REPO"/launch_sweep.py "$REPO"/collect_results.py "$REPO"/download_ogbench_data.py \
   "$REPO"/SETUP_NOTES.md "$REPO"/requirements.lock.txt "$EX/code/" 2>/dev/null
( cd "$REPO" && git diff > "$EX/code/patches/rql-compat.patch" )

# 2) per-run metric files (skip the multi-MB tqdm logs)
mkdir -p "$EX/results/runs"
if [ -d "$RESULTS/runs" ]; then
  for d in "$RESULTS"/runs/*/; do
    rid=$(basename "$d")
    for f in $(find "$d" -name 'eval.csv' -o -name 'train.csv' -o -name 'flags.json' 2>/dev/null); do
      mkdir -p "$EX/results/runs/$rid"
      cp "$f" "$EX/results/runs/$rid/$(basename "$f")"
    done
  done
fi
cp "$RESULTS/status.json" "$EX/results/status.json" 2>/dev/null
grep -v '^\[.*| ETA' "$RESULTS/sweep.log" > "$EX/results/sweep_header.log" 2>/dev/null
tail -200 "$RESULTS/sweep.log" > "$EX/results/sweep_tail.log" 2>/dev/null

# 3) aggregate
python "$REPO/collect_results.py" --results "$RESULTS" --csv "$EX/results/per_run.csv" \
  > "$EX/results/summary_by_domain.txt" 2>&1
python "$REPO/collect_results.py" --results "$RESULTS" --by task \
  > "$EX/results/summary_by_task.txt" 2>&1

# 4) progress line for the README
done_n=$(ls -d "$EX"/results/runs/*/ 2>/dev/null | wc -l)
{
  echo "last sync: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "runs with metric files: $done_n / 320"
  echo
  cat "$EX/results/summary_by_domain.txt"
} > "$EX/PROGRESS.md"

cd "$EX"
git add -A
if git diff --cached --quiet; then echo "sync: nothing changed"; exit 0; fi
git commit -q -m "sweep results: $(date -u '+%Y-%m-%d %H:%M UTC') (${done_n}/320 runs)"
echo "sync: committed (${done_n}/320)"
if [ "${1:-}" != "--no-push" ] && git remote get-url origin >/dev/null 2>&1; then
  git push -q origin HEAD && echo "sync: pushed" || echo "sync: push FAILED"
fi
