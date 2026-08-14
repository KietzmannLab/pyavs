#!/bin/bash
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

# Base paths
script_path="/home/student/p/psulewski/pyAVS/scripts/ica"
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
