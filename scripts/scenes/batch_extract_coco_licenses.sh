#!/bin/bash
# HPC-specific: the #SBATCH directives below target this lab's SLURM cluster
# and are opt-in -- SLURM ignores this file entirely unless submitted via
# `sbatch`. Running it directly with `bash` skips scheduling. Adjust or drop
# the partition line for a different cluster.
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

# Paths. repo_path resolves to wherever this repo checkout lives, so it is
# portable across machines/users. output_dir defaults to this lab's scratch
# location but can be overridden with the PYAVS_OUTPUT_DIR env var.
repo_path="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
coco_dir="$(pyavs configure --show)/input/annotations"
#avs_scenes_dir="$(pyavs configure --show)/AVS-UTILS/avs_scenes"
output_dir="${PYAVS_OUTPUT_DIR:-/share/klab/psulewski/psulewski/pyavs/coco_licenses}"
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
