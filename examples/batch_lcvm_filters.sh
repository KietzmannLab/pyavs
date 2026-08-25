#!/bin/bash
#SBATCH --partition=klab-cpu
#SBATCH --cpus-per-task=30
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --job-name=lcmv_filters
#SBATCH --output=lcmv_sub%a_%j.out
#SBATCH --error=lcmv_sub%a_%j.err
#SBATCH --array=1-5
echo "Running in shell: $SHELL"
echo "Processing subject $SLURM_ARRAY_TASK_ID as part of array job $SLURM_ARRAY_JOB_ID"

export NCCL_SOCKET_IFNAME=lo


spack load miniconda3
eval "$(conda shell.bash hook)"
conda activate avs

python /home/student/p/psulewski/pyAVS/examples/compute_cross_session_filters.py --subject-id $SLURM_ARRAY_TASK_ID --verbose