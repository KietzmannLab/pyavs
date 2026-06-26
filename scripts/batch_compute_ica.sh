#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

#SBATCH -p klab-cpu
#SBATCH --job-name=ica
#SBATCH --error=error_ica_%A_%a.err
#SBATCH --output=output_ica_%A_%a.out
#SBATCH --array=1-5
#SBATCH --requeue

echo "Running in shell: $SHELL"
export NCCL_SOCKET_IFNAME=lo

# Load required modules
spack load miniconda3
eval "$(conda shell.bash hook)"
conda activate avs

# Base paths
script_path="/home/student/p/psulewski/pyAVS/scripts"
data_path="/share/klab/datasets/avs/"

# Get subject ID from array task ID
subject=$SLURM_ARRAY_TASK_ID

echo "==================================================="
echo "Running ICA pipeline for subject $subject"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Sessions: 1 2"
echo "Top fraction (eye): 0.04"
echo "==================================================="

python ${script_path}/compute_ica.py \
    --data-path $data_path \
    --subject $subject \
    --sessions 1 2 \
    --top-fraction 0.04 \
    --skip-existing \
    --verbose

echo ""
echo "==================================================="
echo "Subject $subject ICA pipeline complete"
echo "==================================================="
