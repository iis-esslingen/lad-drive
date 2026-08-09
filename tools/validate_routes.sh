#!/usr/bin/env bash
# validate_routes.sh — validate existing .tar.zst archives (no repack)

set -euo pipefail

OUT="${OUT:-/beegfs/scratch/workspace/es_fschmidt-ad_projects/LMDrive/dataset/sub-0/data_zipped}"
JOBS="${JOBS:-16}"   # parallel jobs

echo "[INFO] OUT=$OUT  JOBS=$JOBS"

bad=0
ok=0

# Run validations in parallel; collect statuses
# status file to aggregate results
STATUS_FILE="$(mktemp)"
trap 'rm -f "$STATUS_FILE"' EXIT

find "$OUT" -type f -name '*.tar.zst' -print0 \
| parallel -0 -j "$JOBS" --no-notice --bar '
  arc="{}"
  name="$(basename "$arc")"

  # 1) zstd CRC/integrity
  if ! zstd -q -t -- "$arc" >/dev/null 2>&1; then
    echo "[BAD-ZSTD]  $name"
    echo "BAD" >> "'"$STATUS_FILE"'"
    exit 0
  fi

  # 2) tar structure
  if ! zstd -q -dc -- "$arc" | tar -tf - >/dev/null 2>&1; then
    echo "[BAD-TAR]   $name"
    echo "BAD" >> "'"$STATUS_FILE"'"
    exit 0
  fi

  echo "[OK]        $name"
  echo "OK" >> "'"$STATUS_FILE"'"
'

# Summary
ok=$(grep -c '^OK$' "$STATUS_FILE" || true)
bad=$(grep -c '^BAD$' "$STATUS_FILE" || true)
total=$((ok + bad))
echo
echo "==== Summary ===="
echo "Total : $total"
echo "OK    : $ok"
echo "BAD   : $bad"

# exit non-zero if any bad
[[ $bad -eq 0 ]]
