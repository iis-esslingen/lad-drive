#!/bin/bash
# pack_routes.sh

echo "Pack routes of sub-0"

SRC="/beegfs/scratch/workspace/es_fschmidt-ad_projects/LMDrive/dataset/sub-0/data"
export OUT="/beegfs/scratch/workspace/es_fschmidt-ad_projects/LMDrive/dataset/sub-0/data_zipped"

mkdir -p "$OUT"

# list all routes
find "$SRC" -mindepth 1 -maxdepth 1 -type d > routes.txt

total=$(wc -l < routes.txt)
echo "Found $total routes to pack."

cat routes.txt | parallel --bar -j 16 '
  route_path={};
  route_name=$(basename "$route_path");
  
  out_file="$OUT/${route_name}.tar.zst"

  # Skip if already exists
  if [[ -f "$out_file" ]]; then
    echo "Skipping $route_name (already packed)"
    exit 0
  fi

  # Create tar and compress with zstd
  tar -cf - -C "$(dirname "$route_path")" "$route_name" \
    | zstd -q -T0 -5 -f -o "$out_file"
'
