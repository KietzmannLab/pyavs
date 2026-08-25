#!/bin/bash
# HPC-specific: the #SBATCH directives below target this lab's SLURM cluster
# (partition names, array size, etc.) and are opt-in -- SLURM ignores this
# file entirely unless submitted via `sbatch`. Running it directly with
# `bash` skips scheduling and just executes the body once (SLURM_ARRAY_TASK_ID
# will be unset, so pass subject/session another way in that case). Adjust or
# drop the partition line for a different cluster.
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4

#SBATCH -p klab-cpu
#SBATCH --job-name=fix_epochs
#SBATCH --error=error_fix_epochs_%A_%a.err
#SBATCH --output=output_fix_epochs_%A_%a.out
#SBATCH --array=1-50
#SBATCH --requeue

set -e

echo "Running in shell: $SHELL"
set -e

export NCCL_SOCKET_IFNAME=lo

# Load required modules
source /etc/profile.d/spack.sh
spack load miniconda3
eval "$(conda shell.bash hook)"
conda activate avs

# Base paths. script_path resolves to wherever this repo checkout lives, so
# it is portable across machines/users. data_path requires `pyavs configure
# --data-path /path/to/avs-public` to have been run once (or the
# PYAVS_DATA_PATH env var to be set -- pyavs configure --show reads that
# cascade automatically).
script_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_path="$(pyavs configure --show)"

# Decode subject (1-5) and session (1-10) from array task ID (1-50)
# task 1-10 -> subject 1, task 11-20 -> subject 2, ..., task 41-50 -> subject 5
subject=$(( (SLURM_ARRAY_TASK_ID - 1) / 10 + 1 ))
session=$(( (SLURM_ARRAY_TASK_ID - 1) % 10 + 1 ))

echo "==================================================="
echo "Computing fixation/saccade epochs for subject $subject, session $session"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "ICA: BIDS derivatives (new ET-based ICA)"
echo "==================================================="

python ${script_path}/compute_fixation_epochs.py \
    --subject $subject \
    --session $session \
    --data-path $data_path \
    --skip-empty-room \
    --verbose

echo ""
echo "==================================================="
echo "Subject $subject session $session epoch computation complete"
echo "==================================================="
