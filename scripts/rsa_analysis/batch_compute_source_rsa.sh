#!/bin/bash
# NOTE: --time raised from 5h to 12h for the RDM-permutation (shuffle) control,
# which adds an N-times searchlight sweep per subject (--n-permutations below).
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --mem=500G
#SBATCH --cpus-per-task=50
#SBATCH --array=1-5
#SBATCH -p klab-cpu
#SBATCH --job-name=src_rsa
#SBATCH --error=error_source_rsa_%A_%a.err
#SBATCH --output=output_source_rsa_%A_%a.out
#SBATCH --requeue

echo "Running in shell: $SHELL"
export NCCL_SOCKET_IFNAME=lo

# Load required modules
spack load miniconda3
eval "$(conda shell.bash hook)"
conda activate avs

# Base paths
script_path="/home/student/p/psulewski/pyAVS/scripts/rsa_analysis"
data_path="/share/klab/datasets/avs/"
rsa_results_dir="/share/klab/psulewski/psulewski/pyavs/rsa"
output_dir="/share/klab/psulewski/psulewski/pyavs/source_rsa"
layer="layer3"
# One subject per array task
subject=${SLURM_ARRAY_TASK_ID}

echo "==================================================="
echo "Source-space RSA — array task ${SLURM_ARRAY_TASK_ID}"
echo "Subject: ${subject}"
echo "Model: resnet50_ecoset_crop"
echo "Layer: ${layer}"
echo "Sessions: 1-10"
echo "Noise ceiling: skipped (run separately)"
echo "Permutations: 1000 (RDM shuffle control)"
echo "==================================================="

python ${script_path}/compute_source_rsa.py \
    --data-path ${data_path} \
    --subjects ${subject} \
    --sessions 1 2 3 4 5 6 7 8 9 10 \
    --models resnet50_ecoset_crop \
    --layers ${layer} \
    --rsa-results-dir ${rsa_results_dir} \
    --output-dir ${output_dir} \
    --n-jobs 50 \
    --n-permutations 1000 \
    --perm-seed 0 \
    --skip-noise-ceiling

echo "==================================================="
echo "Subject ${subject} complete"
echo "==================================================="
