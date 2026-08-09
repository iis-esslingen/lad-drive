#!/usr/bin/env bash
set -euo pipefail

###############################################################################
#  clean_rgb.sh
#  Remove rgb_front / rgb_left / rgb_right / rgb_rear directories for every
#  route listed in dataset_root/dataset_index.txt
#
#  usage:  clean_rgb.sh DATASET_ROOT [NUM_WORKERS]
###############################################################################

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 DATASET_ROOT [NUM_WORKERS]" >&2
  exit 1
fi

DATASET_ROOT=$(realpath "$1")
IDX_FILE="$DATASET_ROOT/dataset_index.txt"
WORKERS="${2:-32}"            # default 32 parallel jobs

if [[ ! -f $IDX_FILE ]]; then
  echo "Cannot find $IDX_FILE" >&2
  exit 1
fi

# Function executed per-route
clean_route() {
  local route="$1"
  for cam in rgb_front rgb_left rgb_right rgb_rear; do
    rm -rf "${route}/${cam}"
  done
}

export -f clean_route

###############################################################################
# create a list of absolute route paths (1st column of dataset_index.txt)
###############################################################################
mapfile -t ROUTES < <(cut -d' ' -f1 "$IDX_FILE" | sed "s|^|$DATASET_ROOT/|")

echo "Found ${#ROUTES[@]} routes. Removing camera folders …"

###############################################################################
# Run in parallel if GNU parallel is available; otherwise run sequentially
###############################################################################
if command -v parallel &>/dev/null; then
  printf '%s\n' "${ROUTES[@]}" | parallel -j "$WORKERS" clean_route {}
else
  echo "GNU parallel not found – running sequentially" >&2
  for r in "${ROUTES[@]}"; do
    clean_route "$r"
  done
fi

echo "✅  Done."
