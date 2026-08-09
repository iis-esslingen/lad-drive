#!/bin/bash
set -euo pipefail

SRC="/beegfs/scratch/workspace/es_fschmidt-ad_projects/LMDrive/dataset/sub-0/data_zipped"
DST="/localscratch/tmpdir.30064/dataset/sub-0/data"
mkdir -p "$DST"

# Silence the GNU Parallel citation
parallel --citation >/dev/null 2>&1 || true

# NUL-safe; stream via zstd -dc so tar never needs -f
find "$SRC" -type f -name "*.tar.zst" -print0 \
| head -zn 1 \
| parallel -0 -j1 -v 'zstd -dc -- "{}" | tar -x -C "'"$DST"'"'

echo "Done."
