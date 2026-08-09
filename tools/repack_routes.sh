#!/usr/bin/env bash
# re_pack_routes.sh - Create .tar.zst archives for each route directory
# Skips a route if its .tar.zst is newer than any file in the route dir.
#
# Tunables (env overrides):
#   SRC       - source dir with route folders
#   OUT       - output dir for .tar.zst files
#   ZSTD_LVL  - zstd compression level (default: 19; try 5 or 3 for speed)
#   ZSTD_T    - zstd threads per job (default: 3)
#   J         - GNU parallel jobs (default: auto = CORES / ZSTD_T)
#   LOAD_CAP  - desired load cap passed to parallel (if supported)
#   SIMPLE_MTIME - 1 = compare only dir mtime (faster), 0 = deep tree check (default)

set -euo pipefail

# --- Config (override via environment) ---------------------------------------
SRC="${SRC:-/beegfs/scratch/workspace/es_fschmidt-ad_projects/LMDrive/dataset/sub-0/data}"
OUT="${OUT:-/beegfs/scratch/workspace/es_fschmidt-ad_projects/LMDrive/dataset/sub-0/data_zipped}"
ZSTD_LVL="${ZSTD_LVL:-19}"
ZSTD_T="${ZSTD_T:-3}"
CORES="${SLURM_CPUS_ON_NODE:-$(nproc)}"
if [[ -z "${J:-}" ]]; then
  J=$(( CORES / ZSTD_T ))
  (( J < 2 )) && J=2
fi
LOAD_CAP="${LOAD_CAP:-$CORES}"
SIMPLE_MTIME="${SIMPLE_MTIME:-0}"

echo "[INFO] SRC=$SRC"
echo "[INFO] OUT=$OUT"
echo "[INFO] CORES=$CORES  ZSTD_T=$ZSTD_T  J=$J  ZSTD_LVL=$ZSTD_LVL  LOAD_CAP=$LOAD_CAP"
echo "[INFO] SIMPLE_MTIME=$SIMPLE_MTIME (0=deep check, 1=dir mtime only)"

# Decide load flag for GNU Parallel
PAR_LOAD_ARGS=()
if parallel --help 2>/dev/null | grep -qE '(^| )--load( |$)'; then
  PAR_LOAD_ARGS=(--load "$LOAD_CAP")
else
  echo "[WARN] Your GNU Parallel lacks --load; proceeding without load cap."
fi

# --- Prep --------------------------------------------------------------------
mkdir -p "$OUT"
routes_file="$(mktemp -t routes.XXXX.txt)"
trap 'rm -f "$routes_file"' EXIT

find "$SRC" -mindepth 1 -maxdepth 1 -type d | sort > "$routes_file"
total=$(wc -l < "$routes_file" || echo 0)
echo "[INFO] Found $total routes to pack."

# --- Pack with GNU parallel --------------------------------------------------
export OUT ZSTD_T ZSTD_LVL SIMPLE_MTIME
cat "$routes_file" \
| parallel --bar -j "$J" --no-notice "${PAR_LOAD_ARGS[@]}" '
    route_path="{}"
    route_name="$(basename "$route_path")"
    out_file="$OUT/${route_name}.tar.zst"

    # Decide whether to skip based on mtimes
    need_repack=1
    if [[ -f "$out_file" ]]; then
      if [[ "$SIMPLE_MTIME" == "1" ]]; then
        [[ "$out_file" -nt "$route_path" ]] && need_repack=0
      else
        arc_mtime=$(stat -c %Y -- "$out_file" 2>/dev/null || echo 0)
        newest=$(find "$route_path" -type f -printf "%T@\n" 2>/dev/null | sort -nr | head -n1)
        if [[ -z "$newest" ]]; then
          newest=$(stat -c %Y -- "$route_path" 2>/dev/null || echo 0)
        else
          newest=${newest%.*}
        fi
        [[ "$arc_mtime" -ge "$newest" ]] && need_repack=0
      fi
    fi

    if [[ "$need_repack" -eq 0 ]]; then
      echo "[SKIP] $route_name up-to-date"
      exit 0
    fi

    echo "[PACK] $route_name"
    rm -f -- "$out_file"
    tar -cf - -C "$(dirname "$route_path")" "$route_name" \
      | zstd -q -T'"$ZSTD_T"' -'"$ZSTD_LVL"' -f -o "$out_file"
'

echo "[DONE] Considered $total route(s); outputs in: $OUT"
