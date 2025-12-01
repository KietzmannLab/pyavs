#!/bin/bash
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

# Base paths
script_path="/home/student/p/psulewski/pyAVS/scripts/stabilizer_analysis/"
data_path="/share/klab/datasets/avs/"
output_dir="/share/klab/psulewski/psulewski/pyavs/stabilizer"

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
