#!/usr/bin/env bash
# node_extract_dataset.sh
# Run this ON a compute node. It extracts archives from global storage to node-local storage.

set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 --run-storage <path> --src-tars <dir> --idx-src <file> --nav-src <file> --notice-src <file> [--sg-mode <abstracted|standard|none>]
       [--cores <N>] [--jobs <J>]

Options:
  --run-storage   Node-local dataset root (will create <run-storage>/sub-0/data)
  --src-tars      Global dir containing *.tar.zst route archives
  --idx-src       dataset_index.txt path on global storage
  --nav-src       navigation_instruction_list.txt path on global storage
  --notice-src    notice_instruction_list.json path on global storage
  --sg-mode       scene_graph extraction mode:
                    abstracted  -> only *_abstracted.{json,pt,txt}
                    standard    -> everything under scene_graph/
                    none        -> skip scene_graph entirely
                  (default: abstracted)
  --cores         Override CPU count used to size parallelism (default: \$SLURM_CPUS_ON_NODE or nproc)
  --jobs          Fixed GNU parallel jobs (default: min(max(cores/3,8),16))
EOF
}

# --- Args ---
RUN_STORAGE=""
SRC_TARS=""
IDX_SRC=""
NAV_SRC=""
NOTICE_SRC=""
SG_MODE="abstracted"
CORES="${SLURM_CPUS_ON_NODE:-$(nproc)}"
J=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-storage) RUN_STORAGE="$2"; shift 2;;
    --src-tars)    SRC_TARS="$2";    shift 2;;
    --idx-src)     IDX_SRC="$2";     shift 2;;
    --nav-src)     NAV_SRC="$2";     shift 2;;
    --notice-src)  NOTICE_SRC="$2";  shift 2;;
    --sg-mode)     SG_MODE="$2";     shift 2;;
    --cores)       CORES="$2";       shift 2;;
    --jobs)        J="$2";           shift 2;;
    -h|--help)     usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 2;;
  esac
done

[[ -n "$RUN_STORAGE" && -n "$SRC_TARS" && -n "$IDX_SRC" && -n "$NAV_SRC" && -n "$NOTICE_SRC" ]] || { usage; exit 2; }

# Compute jobs if not set
if [[ -z "$J" ]]; then
  J=$(( CORES / 3 ))
  [[ $J -lt 8 ]] && J=8
  [[ $J -gt 16 ]] && J=16
fi

echo "[NODE $(hostname)] run_storage=$RUN_STORAGE  src_tars=$SRC_TARS  sg_mode=$SG_MODE  cores=$CORES jobs=$J"

# Fresh root
rm -rf -- "$RUN_STORAGE"
mkdir -p "$RUN_STORAGE/sub-0/data"

# Copy aux files
cp -f --preserve=timestamps "$IDX_SRC"    "$RUN_STORAGE/dataset_index.txt"
cp -f --preserve=timestamps "$NAV_SRC"    "$RUN_STORAGE/navigation_instruction_list.txt"
cp -f --preserve=timestamps "$NOTICE_SRC" "$RUN_STORAGE/notice_instruction_list.json"

# Build include/exclude set for scene_graph
EXCLUDES=(
  --exclude="*/2d_bbs_front/*"
  --exclude="*/2d_bbs_left/*"
  --exclude="*/2d_bbs_right/*"
  --exclude="*/3d_bbs/*"
  --exclude="*/actors_data/*"
  --exclude="*/affordances/*"
  --exclude="*/birdview/*"
  --exclude="*/depth_front/*"
  --exclude="*/depth_left/*"
  --exclude="*/depth_right/*"
  --exclude="*/measurements/*"
  --exclude="*/rgb_front/*"
  --exclude="*/rgb_left/*"
  --exclude="*/rgb_rear/*"
  --exclude="*/rgb_right/*"
  --exclude="*/seg_front/*"
  --exclude="*/seg_left/*"
  --exclude="*/seg_right/*"
  --exclude="*/topdown/*"
  --exclude="*/scene_graph/*.pkl"
  --exclude="*/scene_graph/*.json"
)
INCLUDES=( "*/lidar/*" "*/lidar_odd/*" "*/rgb_full/*" "*/measurements_full/*" "*/measurements_all.json" )

case "$SG_MODE" in
  abstracted)
    # only the abstracted triplet(s)
    SG_INCLUDES=( "*/scene_graph/*_abstracted.pt" "*/scene_graph/*_abstracted.txt" )
    ;;
  standard)
    # everything under scene_graph
    SG_INCLUDES=( "*/scene_graph/*.pt" "*/scene_graph/*.txt" )
    ;;
  none)
    # skip scene_graph entirely
    SG_INCLUDES=()
    # additionally exclude heavy native SG artifacts just in case
    EXCLUDES+=( --exclude="*/scene_graph/*" )
    ;;
  *)
    echo "[WARN] Unknown --sg-mode '$SG_MODE', defaulting to 'none'"; SG_MODE="none"
    SG_INCLUDES=( )
    EXCLUDES+=( --exclude="*/scene_graph/*" )
    ;;
esac

# Helper per-archive extractor script (node-local)
JOBSH="$RUN_STORAGE/.extract_one.sh"
cat > "$JOBSH" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
src="$1"; dest_root="$2"; shift 2

# Parse 3 groups: EXCLUDES … :: INCLUDES … :: SG_INCLUDES …
group=0
EXC=(); INC=(); SGI=()
for tok in "$@"; do
  if [[ "$tok" == "::" ]]; then
    ((group++)); continue
  fi
  case "$group" in
    0) EXC+=("$tok");;
    1) INC+=("$tok");;
    2) SGI+=("$tok");;
  esac
done

base="$(basename "$src" .tar.zst)"
out="$dest_root/$base"
mkdir -p "$out"

# shellcheck disable=SC2086
zstd -q -dc -- "$src" \
| env -u TAR_OPTIONS command tar -x -C "$out" -f - \
    --strip-components=1 \
    --wildcards --wildcards-match-slash --no-anchored \
    "${EXC[@]}" \
    "${INC[@]}" \
    "${SGI[@]}" \
    > /dev/null \
|| { echo "[WARN] $(hostname): extract failed or patterns missing for $base" >&2; exit 0; }
EOS
chmod +x "$JOBSH"

# Export arrays for GNU parallel env (serialize to words)
export EXCLUDES INCLUDES SG_INCLUDES

# Run in parallel
find "$SRC_TARS" -maxdepth 1 -type f -name "*.tar.zst" ! -name "._*" -print0 \
| parallel -0 -j "$J" --no-notice --tmpdir /tmp \
    --joblog "$RUN_STORAGE/.parallel_extract_$(hostname).tsv" \
    "$JOBSH" {} "$RUN_STORAGE/sub-0/data" \
    "${EXCLUDES[@]}" :: "${INCLUDES[@]}" :: "${SG_INCLUDES[@]}"

echo "[NODE $(hostname)] done."
