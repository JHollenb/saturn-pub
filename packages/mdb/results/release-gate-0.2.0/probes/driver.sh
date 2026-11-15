#!/usr/bin/env bash
# Release-gate 0.2.0 driver: submit jobs one at a time, only when Beast has nothing running or
# queued (shared GPU, never preempt); attach to each job and keep its log.
# Usage: driver.sh <plan-file>   (lines: label|kind|args ; kind = beast (env + run_on_beast.sh) or stage)
set -uo pipefail
RG="$(cd "$(dirname "$0")/.." && pwd)"
EXS="$(cd "$RG/../../examples" && pwd)"
export MRUN_URL="${MRUN_URL:-http://127.0.0.1:19025}"
C=~/domains/common/bin/common
M=~/domains/common/bin/mrun
JOBS="$RG/jobs.tsv"
wait_free() {
  while true; do
    r=$($C mrun jobs --state running 2>/dev/null | python3 -c 'import sys,json; print(len(json.load(sys.stdin)["jobs"]))' 2>/dev/null || echo err)
    q=$($C mrun queue 2>/dev/null | python3 -c 'import sys,json; print(len(json.load(sys.stdin)["jobs"]))' 2>/dev/null || echo err)
    [ "$r" = 0 ] && [ "$q" = 0 ] && return 0
    echo "$(date +%T) beast busy (running=$r queued=$q); waiting" >&2
    sleep 30
  done
}
while IFS='|' read -r label kind skip_if args; do
  [ -z "$label" ] && continue; case "$label" in \#*) continue;; esac
  if [ -n "$skip_if" ] && /usr/bin/grep -q "^$skip_if	.*	failed" "$JOBS" 2>/dev/null; then
    echo "$(date +%T) skip $label: $skip_if failed"; printf '%s\t-\tskipped\n' "$label" >> "$JOBS"; continue
  fi
  wait_free
  echo "$(date +%T) submit $label"
  if [ "$kind" = beast ]; then
    out=$(cd "$EXS" && eval "$args" 2>&1)
  else
    out=$(cd "$RG/probes" && eval "bash stage_submit.sh $args" 2>&1)
  fi
  job=$(printf '%s' "$out" | /usr/bin/grep -oE 'job-[0-9a-f]{12}' | head -1)
  if [ -z "$job" ]; then
    echo "$out" > "$RG/logs/$label.submit-error.txt"; printf '%s\t-\tsubmit-failed\n' "$label" >> "$JOBS"; continue
  fi
  echo "$(date +%T) $label -> $job"
  $M attach "$job" > "$RG/logs/$label-$job.log" 2>&1
  st=$(tail -3 "$RG/logs/$label-$job.log" | /usr/bin/grep -oE 'finished: [a-z_]+' | head -1 | cut -d' ' -f2)
  [ -z "$st" ] && st=unknown
  [ "$st" = succeeded ] || st="failed:$st"
  printf '%s\t%s\t%s\n' "$label" "$job" "$st" >> "$JOBS"
  echo "$(date +%T) $label $job $st"
done < "$1"
echo "$(date +%T) plan done"
