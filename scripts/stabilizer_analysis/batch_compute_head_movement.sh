#!/bin/bash
# HPC-specific: the #SBATCH directives below target this lab's SLURM cluster
# and are opt-in -- SLURM ignores this file entirely unless submitted via
# `sbatch`. Running it directly with `bash` skips scheduling and just
# executes the body once (SLURM_ARRAY_TASK_ID will be unset). Adjust or drop
# the partition line for a different cluster.
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=15

#SBATCH -p workq
#SBATCH --job-name=headmove
#SBATCH --error=error_head_movement_%A_%a.err
#SBATCH --output=output_head_movement_%A_%a.out
#SBATCH --array=1-50  # 5 subjects x 10 sessions = 50 jobs
#SBATCH --requeue

echo "Running in shell: $SHELL"
export NCCL_SOCKET_IFNAME=lo

# Load required modules
spack load miniconda3
eval "$(conda shell.bash hook)"
conda activate avs

# Base paths. script_path resolves to wherever this repo checkout lives, so
# it is portable across machines/users. data_path requires `pyavs configure
# --data-path /path/to/avs-public` to have been run once. output_dir defaults
# to this lab's scratch location but can be overridden with the
# PYAVS_OUTPUT_DIR env var.
# Resolve this script's directory in the repo checkout.
#
# BASH_SOURCE alone is not enough under sbatch: SLURM copies the batch script
# to a node-local spool dir (/var/spool/slurm/...), so BASH_SOURCE resolves
# there and sibling files are not found. Recover the submitted path from
# SLURM, keeping BASH_SOURCE for plain `bash` invocation. SCRIPT_PATH
# overrides both.
if [[ -n "${SCRIPT_PATH:-}" ]]; then
    script_path="$(cd "$SCRIPT_PATH" && pwd)"
elif [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol >/dev/null 2>&1; then
    _submitted="$(scontrol show job "$SLURM_JOB_ID" \
        | awk -F= '/^[[:space:]]*Command=/{print $2; exit}')"
    script_path="$(cd "$(dirname "$_submitted")" && pwd)"
else
    script_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
data_path="$(pyavs configure --show)"
output_dir="${PYAVS_OUTPUT_DIR:-/share/klab/psulewski/psulewski/pyavs/stabilizer}"

# Calculate subject_id and session from array task ID
# Formula: subject = (TASK_ID - 1) / 10 + 1, session = (TASK_ID - 1) % 10 + 1
subject_id=$(( ($SLURM_ARRAY_TASK_ID - 1) / 10 + 1 ))
session_num=$(( ($SLURM_ARRAY_TASK_ID - 1) % 10 + 1 ))

echo "==================================================="
echo "Computing head movement for subject $subject_id, session $session_num"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "==================================================="

# Run the head movement computation with joblib parallelization over runs
python ${script_path}/compute_head_movement.py \
    --subjects $subject_id \
    --sessions $session_num \
    --data-path $data_path \
    --output-dir $output_dir \
    --n-jobs 15 \
    --verbose

echo "==================================================="
echo "Subject $subject_id, session $session_num processing complete"
echo "==================================================="
