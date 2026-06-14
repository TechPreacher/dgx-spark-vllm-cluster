#!/usr/bin/env bash
#
# mem-watch.sh — stream this host's memory, one compact line per sample.
# Run directly ON the Spark (you're already SSH'd in).
#
# NOTE: because this runs on the Spark, if the host freezes this watcher dies
# with your SSH session. For freeze detection, run the laptop-side version
# instead. This local version is best for live monitoring during a load/test.
#
# Usage:
#   ./mem-watch.sh             # 2s interval, warn when avail < 6 GiB
#   ./mem-watch.sh -i 1 -l 10  # 1s interval, warn under 10 GiB avail
#
# Flags:
#   -i <seconds>   sample interval    (default: 2)
#   -l <GiB>       low-avail warning  (default: 6)
#   -h             help

set -uo pipefail

INTERVAL="${INTERVAL:-2}"
LOW_GIB="${LOW_GIB:-6}"

while getopts "i:l:h" opt; do
  case "$opt" in
    i) INTERVAL="$OPTARG" ;;
    l) LOW_GIB="$OPTARG" ;;
    h) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "try -h"; exit 1 ;;
  esac
done

echo "Watching $(hostname)  every ${INTERVAL}s  (warn when avail < ${LOW_GIB} GiB).  Ctrl-C to stop."
echo "time      used        free        buff/cache  avail       total"

while true; do
  ts=$(date +%H:%M:%S)
  free -m | awk -v ts="$ts" -v low="$LOW_GIB" '/^Mem:/ {
    avail=$7/1024
    flag=(avail<low) ? "  <== LOW" : ""
    printf "%s  %7.1fG    %7.1fG    %7.1fG    %7.1fG    %7.1fG%s\n", \
           ts, $3/1024, $4/1024, $6/1024, $7/1024, $2/1024, flag
  }'
  sleep "$INTERVAL"
done
