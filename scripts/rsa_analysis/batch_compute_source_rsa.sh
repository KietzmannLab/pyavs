#!/bin/bash
# NOTE: --time raised from 5h to 12h for the RDM-permutation (shuffle) control,
# which adds an N-times searchlight sweep per subject (--n-permutations below).
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --mem=500G
#SBATCH --cpus-per-task=50
#SBATCH --array=1-5
#SBATCH -p klab-cpu
# HPC-specific: the #SBATCH directives above/below target this lab's SLURM
# cluster and are opt-in -- SLURM ignores this file entirely unless submitted
# via `sbatch`. Running it directly with `bash` skips scheduling and just
# executes the body once (SLURM_ARRAY_TASK_ID will be unset). Adjust or drop
# the partition line for a different cluster.
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

# Base paths. script_path resolves to wherever this repo checkout lives, so
# it is portable across machines/users. data_path requires `pyavs configure
# --data-path /path/to/avs-public` to have been run once. rsa_results_dir/
# output_dir default to this lab's scratch location but can be overridden
# with the PYAVS_RSA_RESULTS_DIR/PYAVS_OUTPUT_DIR env vars.
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
data_path="$(pyavs configure --show)"
# FreeSurfer SUBJECTS_DIR (individual anatomies sub-0X + fsaverage). Needed for
# morphing to fsaverage and loading the ico-5 source space. Without it,
# subjects_dir is None and the run crashes in _load_fsaverage_src / morphing.
# The pre-computed forward model itself (sub-0X-fwd.fif) is found automatically
# by load_forward_model() under the same derivatives/freesurfer/ tree.
subjects_dir="${data_path}/derivatives/freesurfer"
rsa_results_dir="${PYAVS_RSA_RESULTS_DIR:-/share/klab/psulewski/psulewski/pyavs/rsa}"
output_dir="${PYAVS_OUTPUT_DIR:-/share/klab/psulewski/psulewski/pyavs/source_rsa}"
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
    --subjects-dir ${subjects_dir} \
    --n-jobs 50 \
    --n-permutations 100 \
    --perm-seed 0 \
    --skip-noise-ceiling

echo "==================================================="
echo "Subject ${subject} complete"
echo "==================================================="
