#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=3

#SBATCH -p workq
#SBATCH --job-name=coco_licenses
#SBATCH --error=error_coco_licenses_%j.err
#SBATCH --output=output_coco_licenses_%j.out

echo "Running in shell: $SHELL"

# Load required modules
spack load miniconda3
eval "$(conda shell.bash hook)"
conda activate avs

# Paths
repo_path="/home/student/p/psulewski/pyAVS"
coco_dir="/share/klab/datasets/avs/input/annotations"
#avs_scenes_dir="/share/klab/datasets/avs/AVS-UTILS/avs_scenes"
output_dir="/share/klab/psulewski/psulewski/pyavs/coco_licenses"
output_file="${output_dir}/permissive_images.csv"

mkdir -p "$output_dir"

echo "==================================================="
echo "Extracting COCO permissive license metadata"
echo "COCO dir:       $coco_dir"
#echo "AVS scenes dir: $avs_scenes_dir"
echo "Output:         $output_file"
echo "Flickr enrichment: $([ -n "$FLICKR_API_KEY" ] && echo enabled || echo disabled)"
echo "==================================================="

cd "$repo_path"

python -m pyavs.scenes.coco_licenses \
    --coco-dir "$coco_dir" \
    --output "$output_file"

echo "==================================================="
echo "Done. Output written to $output_file"
echo "==================================================="
