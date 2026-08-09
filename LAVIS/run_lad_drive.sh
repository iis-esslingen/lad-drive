#!/bin/bash
#SBATCH --job-name=train_lad_drive
#SBATCH --nodes=8
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=300G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu1
#SBATCH --output=/beegfs/scratch/workspace/es_kafeit00-test-lad-drive/lad-drive/LAVIS/train_logs/lad_drive_%x_%j.out
#SBATCH --error=/beegfs/scratch/workspace/es_kafeit00-test-lad-drive/lad-drive/LAVIS/train_logs/lad_drive_%x_%j.err

# ---- Job-local scratch paths (no per-user subdir exists) ----
RUN_ROOT="/localscratch/tmpdir.${SLURM_JOB_ID}"
RUN_STORAGE="$RUN_ROOT/dataset"

# Ensure TMPDIR exists for temp files later
: "${TMPDIR:=/tmp}"
mkdir -p "$TMPDIR"

# Cleanup *this job's* dir on exit (tolerant)
trap 'set +e;
  if [[ -d "'"$RUN_ROOT"'" ]]; then
    shopt -s nullglob dotglob
    rm -rf --one-file-system "'"$RUN_ROOT"'"/* "'"$RUN_ROOT"'"/.[!.]* "'"$RUN_ROOT"'"/..?* 2>/dev/null || true
    rmdir "'"$RUN_ROOT"'" 2>/dev/null || true
  fi
' EXIT TERM INT HUP

export PYTHONPATH=$PYTHONPATH:/beegfs/scratch/workspace/es_kafeit00-lang-traj/research-project/LADDrive/vision_encoder # TODO adjust
CONFIG_BASE_PATH=/beegfs/scratch/workspace/es_kafeit00-test-lad-drive/lad-drive/LAVIS/lavis/projects/lad_drive # TODO adjust
CONFIG_PATH=$CONFIG_BASE_PATH/lad_drive.yaml
echo "CONFIG_BASE_PATH: $CONFIG_BASE_PATH"
echo "CONFIG_PATH: $CONFIG_PATH"
echo "SLURM_JOB_ID: $SLURM_JOB_ID  NODES: $SLURM_JOB_NUM_NODES"

echo "[INFO] Staging at $(date)"

SRC_TARS="/beegfs/scratch/workspace/es_fschmidt-ad_projects/LMDrive/dataset/sub-0/data_zipped"
IDX_SRC="/beegfs/scratch/workspace/es_fschmidt-ad_projects/LMDrive/dataset/dataset_index.txt"
NAV_SRC="/beegfs/scratch/workspace/es_fschmidt-ad_projects/LMDrive/dataset/navigation_instruction_list.txt"
NOTICE_SRC="/beegfs/scratch/workspace/es_fschmidt-ad_projects/LMDrive/dataset/notice_instruction_list.json"

# ---- Per-node: free space safely, then extract ----
srun -N "$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 \
  --export=ALL,RUN_ROOT="$RUN_ROOT",RUN_STORAGE="$RUN_STORAGE",SRC_TARS="$SRC_TARS",IDX_SRC="$IDX_SRC",NAV_SRC="$NAV_SRC",NOTICE_SRC="$NOTICE_SRC" \
  bash -s <<'NODE'
set -euo pipefail

echo "[NODE $(hostname)] /localscratch before:"
df -h /localscratch || true
du -sh /localscratch/* 2>/dev/null | sed "s|^|[NODE $(hostname)] |" || true

# Make sure our job-local dir exists
mkdir -p "$RUN_STORAGE/sub-0/data"

# SAFE CLEANUP: remove only *your* old tmpdir.* (owned by you), older than 12h
find /localscratch -mindepth 1 -maxdepth 1 -type d -name "tmpdir.*" \
     -user "$USER" ! -path "$RUN_ROOT" \
     -exec rm -rf --one-file-system {} + 2>/dev/null || true

echo "[NODE $(hostname)] /localscratch after cleanup:"
df -h /localscratch || true

# Copy auxiliary files to dataset root (quiet, overwrite)
cp -f --preserve=timestamps "$IDX_SRC"    "$RUN_STORAGE/dataset_index.txt"
cp -f --preserve=timestamps "$NAV_SRC"    "$RUN_STORAGE/navigation_instruction_list.txt"
cp -f --preserve=timestamps "$NOTICE_SRC" "$RUN_STORAGE/notice_instruction_list.json"

# Parallelism
CORES="${SLURM_CPUS_ON_NODE:-48}"
J=$(( CORES / 3 )); [ $J -lt 8 ] && J=8; [ $J -gt 16 ] && J=16

# Helper extractor
JOBSH="$RUN_STORAGE/.extract_one.sh"
cat > "$JOBSH" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
src="$1"; dest_root="$2"

base="$(basename "$src" .tar.zst)"
out="$dest_root/$base"
mkdir -p "$out"

zstd -q -dc -- "$src" \
| env -u TAR_OPTIONS tar -x -C "$out" -f - \
    --strip-components=1 \
    --wildcards --wildcards-match-slash --no-anchored \
    --exclude="*/scene_graph/*" \
    --exclude="*/2d_bbs_front/*" \
    --exclude="*/2d_bbs_left/*" \
    --exclude="*/2d_bbs_right/*" \
    --exclude="*/3d_bbs/*" \
    --exclude="*/actors_data/*" \
    --exclude="*/affordances/*" \
    --exclude="*/birdview/*" \
    --exclude="*/depth_front/*" \
    --exclude="*/depth_left/*" \
    --exclude="*/depth_right/*" \
    --exclude="*/measurements/*" \
    --exclude="*/rgb_front/*" \
    --exclude="*/rgb_left/*" \
    --exclude="*/rgb_rear/*" \
    --exclude="*/rgb_right/*" \
    --exclude="*/seg_front/*" \
    --exclude="*/seg_left/*" \
    --exclude="*/seg_right/*" \
    --exclude="*/topdown/*" \
    "*/lidar/*" "*/lidar_odd/*" "*/rgb_full/*" \
    "*/measurements_full/*" "*/measurements_all.json" > /dev/null \
|| { echo "[WARN] $(hostname): extract failed or patterns missing for $base" >&2; exit 0; }
EOS
chmod +x "$JOBSH"

# Extract all relevant archives (skip resource forks and scene_graph-only packs)
find "$SRC_TARS" -maxdepth 1 -type f -name "*.tar.zst" ! -name "._*" ! -name "*scene_graphs*" -print0 \
| parallel -0 -j "$J" --no-notice --tmpdir /tmp \
    --joblog "$RUN_STORAGE/.parallel_extract_$(hostname).tsv" \
    "$JOBSH" {} "$RUN_STORAGE/sub-0/data"

echo "[NODE $(hostname)] Extraction done."
df -h /localscratch || true
NODE

# ---- Activate env for yaml override & training ----
source ~/.bashrc
conda activate lad_drive

# ---- Dynamic config override (write to SHARED CONFIG_BASE_PATH) ----
TMP_CFG="$CONFIG_BASE_PATH/lad_drive_${SLURM_JOB_ID}.yaml"
python - "$CONFIG_PATH" "$TMP_CFG" "$RUN_STORAGE" <<'PY'
import sys, yaml, os
cfg_in, cfg_out, run_storage = sys.argv[1:4]
with open(cfg_in) as f:
    cfg = yaml.safe_load(f)

# Point both train/val dataset storage to node-local RUN_STORAGE
for split in ('train','val'):
    try:
        cfg['datasets']['carla_voice']['build_info']['annotations'][split]['storage'] = run_storage
    except Exception:
        pass

with open(cfg_out, 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print("Wrote", cfg_out)
PY
echo "[INFO] Using shared cfg: $TMP_CFG"

start_ts=$(date +%s)
echo "[INFO] Job $SLURM_JOB_ID started at $(date)"

srun python -m torch.distributed.run \
    --nnodes="$SLURM_JOB_NUM_NODES" \
    --nproc_per_node=1 \
    --node_rank="$SLURM_NODEID" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1):29500" \
    train.py --cfg-path "$TMP_CFG"

time rm -fr "$TMPDIR"/* || true

end_ts=$(date +%s)
elapsed=$(( end_ts - start_ts ))

printf "[INFO] Job %s finished at %s\n"    "$SLURM_JOB_ID" "$(date)"
printf "[INFO] Elapsed wall time: %02dh:%02dm:%02ds\n" \
        $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
