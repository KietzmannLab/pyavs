#!/bin/bash
# HPC-specific: the #SBATCH directives below target this lab's SLURM cluster
# and are opt-in -- SLURM ignores this file entirely unless submitted via
# `sbatch`. Running it directly with `bash` skips scheduling and just
# executes the body once (SLURM_ARRAY_TASK_ID will be unset). Adjust or drop
# the partition line for a different cluster.
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --mem=300G
#SBATCH --cpus-per-task=50

#SBATCH -p klab-cpu
#SBATCH --job-name=encoding
#SBATCH --error=error_encoding_%A_%a.err
#SBATCH --output=output_encoding_%A_%a.out
#SBATCH --array=1-5  # Run for subjects 1-5
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
script_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_path="$(pyavs configure --show)"

# Get subject ID from array task ID
subject=$SLURM_ARRAY_TASK_ID

echo "==================================================="
echo "Running MEG encoding analysis for subject $subject"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Model: resnet50_ecoset_crop, Layer: avgpool"
echo "Sessions: 1-10 (all sessions combined)"
echo "==================================================="

# Run the encoding analysis
# Note: The script will use all 30 CPUs with n_jobs=-1
python ${script_path}/compute_encoding.py \
    --data-path $data_path \
    --subjects $subject \
    --sessions 1 2 3 4 5 6 7 8 9 10 \
    --model resnet50_ecoset_crop \
    --layer avgpool \
    #--n-jobs 1

echo "==================================================="
echo "Subject $subject encoding analysis complete"
echo "==================================================="
