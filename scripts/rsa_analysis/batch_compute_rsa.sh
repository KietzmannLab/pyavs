#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --mem=400G
#SBATCH --cpus-per-task=50

#SBATCH -p klab-cpu
#SBATCH --job-name=rsa
#SBATCH --error=error_rsa_%A_%a.err
#SBATCH --output=output_rsa_%A_%a.out
#SBATCH --array=1-25  # Run for subjects 1-5 X 5 layers (total 25 tasks)
#SBATCH --requeue

echo "Running in shell: $SHELL"
export NCCL_SOCKET_IFNAME=lo

# Load required modules
spack load miniconda3
eval "$(conda shell.bash hook)"
conda activate avs

# Base paths
script_path="/home/student/p/psulewski/pyAVS/scripts/rsa_analysis"
data_path="$(pyavs configure --show)"

# Subjects and layers to process
subjects=(1 2 3 4 5)
layers=("layer1" "layer2" "layer3" "layer4" "avgpool")

# Map array task ID to one subject/layer pair
task_index=$((SLURM_ARRAY_TASK_ID - 1))
layer_count=${#layers[@]}
subject_index=$((task_index / layer_count))
layer_index=$((task_index % layer_count))

subject=${subjects[$subject_index]}
layer=${layers[$layer_index]}

#

echo "==================================================="
echo "Running MEG RSA analysis for subject $subject"
echo "Layer: $layer"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Model: resnet50_ecoset_crop"
echo "Subjects: ${subjects[@]}"
echo "Layers: ${layers[@]}"
echo "Sessions: 1-10 (all sessions combined)"
echo "==================================================="

echo ""
echo "---------------------------------------------------"
echo "Processing subject: $subject"
echo "Processing layer: $layer"
echo "---------------------------------------------------"

python ${script_path}/compute_rsa.py \
    --data-path $data_path \
    --subjects $subject \
    --sessions 1 2 3 4 5 6 7 8 9 10 \
    --models resnet50_ecoset_crop \
    --layers $layer

echo ""
echo "==================================================="
echo "Subject $subject, layer $layer RSA analysis complete"
echo "==================================================="
