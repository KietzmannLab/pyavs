#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --mem=400G
#SBATCH --cpus-per-task=50

#SBATCH -p klab-cpu
#SBATCH --job-name=rsa
#SBATCH --error=error_rsa_%A_%a.err
#SBATCH --output=output_rsa_%A_%a.out
#SBATCH --array=1-5  # Run for subjects 1-5
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

# Get subject ID from array task ID
subject=$SLURM_ARRAY_TASK_ID

# Layers to process
layers=("layer1" "layer2" "layer3" "layer4" "avgpool")

echo "==================================================="
echo "Running MEG RSA analysis for subject $subject"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Model: resnet50_ecoset_crop"
echo "Layers: ${layers[@]}"
echo "Sessions: 1-10 (all sessions combined)"
echo "==================================================="

# Run RSA analysis for each layer
for layer in "${layers[@]}"; do
    echo ""
    echo "---------------------------------------------------"
    echo "Processing layer: $layer"
    echo "---------------------------------------------------"

    python ${script_path}/compute_rsa.py \
        --data-path $data_path \
        --subjects $subject \
        --sessions 1 2 3 4 5 6 7 8 9 10 \
        --models resnet50_ecoset_crop \
        --layers $layer
done

echo ""
echo "==================================================="
echo "Subject $subject RSA analysis complete for all layers"
echo "==================================================="
