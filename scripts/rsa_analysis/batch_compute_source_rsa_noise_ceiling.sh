#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=800G
#SBATCH --cpus-per-task=200

#SBATCH -p klab-cpu
#SBATCH --job-name=source_rsa_nc
#SBATCH --error=error_source_rsa_nc_%j.err
#SBATCH --output=output_source_rsa_nc_%j.out
#SBATCH --requeue

# Run after all per-subject array jobs have completed, e.g.:
#   sbatch --dependency=afterok:<array_job_id> batch_compute_source_rsa_noise_ceiling.sh

echo "Running in shell: $SHELL"
export NCCL_SOCKET_IFNAME=lo

# Load required modules
spack load miniconda3
eval "$(conda shell.bash hook)"
conda activate avs

# Base paths
script_path="/home/student/p/psulewski/pyAVS/scripts/rsa_analysis"
output_dir="/share/klab/psulewski/psulewski/pyavs/source_rsa"

echo "==================================================="
echo "Source-space RSA — group noise ceiling"
echo "Subjects: 1 2 3 4 5"
echo "Model: resnet50_ecoset_crop"
echo "Layer: layer3"
echo "==================================================="

python ${script_path}/compute_source_rsa_noise_ceiling.py \
    --subjects 1 2 3 4 5 \
    --models resnet50_ecoset_crop \
    --layers layer3 \
    --output-dir ${output_dir} \
    --n-jobs 200

echo "==================================================="
echo "Noise ceiling complete"
echo "==================================================="
