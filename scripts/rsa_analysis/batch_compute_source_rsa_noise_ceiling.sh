#!/bin/bash
# HPC-specific: the #SBATCH directives below target this lab's SLURM cluster
# and are opt-in -- SLURM ignores this file entirely unless submitted via
# `sbatch`. Running it directly with `bash` skips scheduling. Adjust or drop
# the partition line for a different cluster.
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

# Base paths. script_path resolves to wherever this repo checkout lives, so
# it is portable across machines/users. output_dir defaults to this lab's
# scratch location but can be overridden with the PYAVS_OUTPUT_DIR env var.
# Resolve this script's directory in the repo checkout.
#
# BASH_SOURCE alone is not enough under sbatch: SLURM copies the batch script
# to a node-local spool dir (/var/spool/slurm/...), so BASH_SOURCE resolves
# there and sibling files are not found. Recover the submitted path from
# SLURM, keeping BASH_SOURCE for plain `bash` invocation. SCRIPT_PATH
# overrides both.
if [[ -n "${SCRIPT_PATH:-}" ]]; then
    script_path="$(cd "$SCRIPT_PATH" && pwd)"
elif [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol >/dev/null 2>&1; then
    _submitted="$(scontrol show job "$SLURM_JOB_ID" \
        | awk -F= '/^[[:space:]]*Command=/{print $2; exit}')"
    script_path="$(cd "$(dirname "$_submitted")" && pwd)"
else
    script_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
output_dir="${PYAVS_OUTPUT_DIR:-/share/klab/psulewski/psulewski/pyavs/source_rsa}"
# FreeSurfer SUBJECTS_DIR (for the fsaverage ico-5 source space used by the
# searchlight). Without it, subjects_dir is None and the script would fall back
# to mne.datasets.fetch_fsaverage(), which needs internet — klab-cpu compute
# nodes are offline, so pass it explicitly. derivatives/freesurfer is the
# avs-public layout (matches batch_compute_source_rsa.sh; the previous
# .../rawdir path was an internal-only layout that doesn't exist in avs-public).
subjects_dir="$(pyavs configure --show)/derivatives/freesurfer"

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
    --subjects-dir ${subjects_dir} \
    --n-jobs 200

echo "==================================================="
echo "Noise ceiling complete"
echo "==================================================="
