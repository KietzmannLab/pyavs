#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --mem=300G
#SBATCH --cpus-per-task=50

#SBATCH -p klab-cpu
#SBATCH --job-name=decoding_reg
#SBATCH --error=error_decoding_reg_%A_%a.err
#SBATCH --output=output_decoding_reg_%A_%a.out
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
data_path="$(pyavs configure --show)"

# Get subject ID from array task ID
subject=$SLURM_ARRAY_TASK_ID

echo "==================================================="
echo "Running MEG -> embedding-PC regression decoding for subject $subject"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Model: resnet50_ecoset_crop, layers: layer1-4 + avgpool, first 3 PCs"
echo "Channels: grad, window: 50-200 ms"
echo "Sessions: 1-10 (all sessions combined)"
echo "==================================================="

python ${script_path}/compute_decoding_regression.py \
    --data-path $data_path \
    --subjects $subject \
    --sessions 1 2 3 4 5 6 7 8 9 10 \
    --channels grad \
    --time-window 50 200 \
    --model resnet50_ecoset_crop \
    --layers layer1 layer2 layer3 layer4 avgpool \
    --n-pcs 3 \
    --n-splits 5

echo "==================================================="
echo "Subject $subject regression decoding complete"
echo "==================================================="

# ---------------------------------------------------------------------------
# Non-SLURM fallback (run locally over all subjects):
#
#   for s in 1 2 3 4 5; do
#       python compute_decoding_regression.py \
#           --data-path /share/klab/datasets/avs/ \
#           --subjects $s --sessions 1 2 3 4 5 6 7 8 9 10 \
#           --channels grad --time-window 50 200 \
#           --model resnet50_ecoset_crop --layers layer1 layer2 layer3 layer4 avgpool --n-pcs 3
#   done
#
# Then the group figure:
#   python plot_decoding_regression_results.py --results-dir /share/klab/datasets/avs/decoding_regression_results
# ---------------------------------------------------------------------------
