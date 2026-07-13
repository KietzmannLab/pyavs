#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --mem=300G
#SBATCH --cpus-per-task=50

#SBATCH -p klab-cpu
#SBATCH --job-name=decoding
#SBATCH --error=error_decoding_%A_%a.err
#SBATCH --output=output_decoding_%A_%a.out
#SBATCH --array=1-5  # One task per subject (subjects 1-5); adjust if more subjects
#SBATCH --requeue

echo "Running in shell: $SHELL"
export NCCL_SOCKET_IFNAME=lo

# Load required modules
spack load miniconda3
eval "$(conda shell.bash hook)"
conda activate avs

# Base paths
script_path="/home/student/p/psulewski/pyAVS/scripts/decoding"
data_path="/share/klab/datasets/avs/"

# Get subject ID from array task ID
subject=$SLURM_ARRAY_TASK_ID

echo "==================================================="
echo "Running MEG object-category decoding for subject $subject"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Channels: grad, window: 50-200 ms, min occurrences: 200"
echo "Sessions: 1-10 (all sessions combined)"
echo "==================================================="

python ${script_path}/compute_decoding.py \
    --data-path $data_path \
    --subjects $subject \
    --sessions 1 2 3 4 5 6 7 8 9 10 \
    --channels grad \
    --time-window 50 200 \
    --min-occurrences 200 \
    --n-splits 5

echo "==================================================="
echo "Subject $subject decoding analysis complete"
echo "==================================================="

# ---------------------------------------------------------------------------
# Non-SLURM fallback (run locally over all subjects, no scheduler):
#
#   for s in 1 2 3 4 5; do
#       python compute_decoding.py \
#           --data-path /share/klab/datasets/avs/ \
#           --subjects $s --sessions 1 2 3 4 5 6 7 8 9 10 \
#           --channels grad --time-window 50 200 --min-occurrences 200 --n-splits 5
#   done
#
# Then aggregate the group figure across all subjects:
#   python plot_decoding_results.py --results-dir /share/klab/datasets/avs/decoding_results
# ---------------------------------------------------------------------------
