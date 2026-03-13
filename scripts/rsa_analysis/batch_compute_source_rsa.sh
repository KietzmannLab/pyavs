#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=500G
#SBATCH --cpus-per-task=50

#SBATCH -p klab-cpu
#SBATCH --job-name=source_rsa
#SBATCH --error=error_source_rsa_%j.err
#SBATCH --output=output_source_rsa_%j.out
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

layers="layer1"
models="resnet50_ecoset_crop"
subjects="1 2 3 4 5"

echo "==================================================="
echo "Running source-space RSA analysis"
echo "Model: resnet50_ecoset_crop"
echo "Layer: $layers"
echo "Subjects: 1 2 3 4 5"
echo "Sessions: 1-10 (all sessions combined)"
echo "Noise ceiling: yes (computed after all subjects)"
echo "==================================================="

# Loop through subjects and compute source-space RSA
for sub in $subjects; do
    echo ""
    echo "==================================================="
    echo "Processing Subject $sub"
    echo "==================================================="  

    python ${script_path}/compute_source_rsa.py \
        --data-path $data_path \
        --subjects $sub \
        --sessions 1 2 3 4 5 6 7 8 9 10 \
        --models resnet50_ecoset_crop \
        --layers $layers \
        --rsa-results-dir $rsa_results_dir \
        --output-dir $output_dir \
        --n-jobs 50 \
done

echo ""
echo "==================================================="
echo "Source-space RSA complete for all subjects"
echo "==================================================="
