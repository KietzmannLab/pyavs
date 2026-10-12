#!/bin/bash
# HPC-specific: the #SBATCH directives below target this lab's SLURM cluster
# and are opt-in -- SLURM ignores this file entirely unless submitted via
# `sbatch`. Running it directly with `bash` skips scheduling and just
# executes the body once (SLURM_ARRAY_TASK_ID will be unset). Adjust or drop
# the partition line for a different cluster.
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=10

#SBATCH -p klab-cpu
#SBATCH --job-name=ica
#SBATCH --error=error_ica_%A_%a.err
#SBATCH --output=output_ica_%A_%a.out
#SBATCH --array=1-50
#SBATCH --requeue

echo "Running in shell: $SHELL"
export NCCL_SOCKET_IFNAME=lo

# Load required modules
spack load miniconda3
eval "$(conda shell.bash hook)"
conda activate avs

# Base paths. script_path resolves to wherever this repo checkout lives, so
# it is portable across machines/users. data_path requires `pyavs configure
# --data-path /path/to/avs-public` to have been run once.
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

# Decode subject (1-5) and session (1-10) from array task ID (1-50)
# task 1-10 -> subject 1, task 11-20 -> subject 2, ..., task 41-50 -> subject 5
subject=$(( (SLURM_ARRAY_TASK_ID - 1) / 10 + 1 ))
session=$(( (SLURM_ARRAY_TASK_ID - 1) % 10 + 1 ))

echo "==================================================="
echo "Running ICA pipeline for subject $subject, session $session"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Top fraction (eye): 0.04"
echo "==================================================="

python ${script_path}/compute_ica.py \
    --subject $subject \
    --session $session \
    --data-path $data_path \
    --top-fraction 0.04 \
    --verbose

echo ""
echo "==================================================="
echo "Subject $subject session $session ICA complete"
echo "==================================================="
