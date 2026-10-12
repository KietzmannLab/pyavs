#!/bin/bash
# HPC-specific: the #SBATCH directives below target this lab's SLURM cluster
# (partition names, log paths, array size) and are opt-in -- SLURM ignores this
# file unless submitted via `sbatch`. Running it with `bash` executes the body
# once (SLURM_ARRAY_TASK_ID unset -> pass SUBJECT/SESSION as env vars instead).
# Adjust the partition and log paths for a different cluster.
#
# Unlike the older batch scripts in this repo, --output/--error are ABSOLUTE.
# A relative log path resolves against the submit CWD, which repeatedly dumped
# job logs into the cluster home directory.
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH -p klab-cpu
#SBATCH --job-name=nsd_roi_proj
#SBATCH --output=/share/klab/psulewski/psulewski/pyavs/logs/nsd_roi_proj_%A_%a.out
#SBATCH --error=/share/klab/psulewski/psulewski/pyavs/logs/nsd_roi_proj_%A_%a.err

set -euo pipefail

export TMPDIR=/share/klab/psulewski/psulewski/pyavs/tmp

source /etc/profile.d/spack.sh
spack load miniconda3
eval "$(conda shell.bash hook)"
conda activate avs
export LD_LIBRARY_PATH=$HOME/.conda/envs/avs/lib:$LD_LIBRARY_PATH

# Locate this script's directory in the repo checkout.
#
# BASH_SOURCE alone is NOT enough under sbatch: SLURM copies the batch script
# to a node-local spool dir (/var/spool/slurm/d/job<id>/), so BASH_SOURCE
# resolves there and the sibling .py is not found. Recover the submitted path
# from SLURM instead, and keep BASH_SOURCE for plain `bash` invocation.
# (The other batch_*.sh scripts in this repo still have the unfixed idiom.)
if [[ -n "${SCRIPT_PATH:-}" ]]; then
    script_path="$SCRIPT_PATH"
elif [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol >/dev/null 2>&1; then
    submitted="$(scontrol show job "$SLURM_JOB_ID" \
        | awk -F= '/^[[:space:]]*Command=/{print $2; exit}')"
    script_path="$(cd "$(dirname "$submitted")" && pwd)"
else
    script_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

if [[ ! -f "${script_path}/project_fixations_to_nsd_rois.py" ]]; then
    echo "ERROR: project_fixations_to_nsd_rois.py not found in ${script_path}" >&2
    echo "Set SCRIPT_PATH to the repo's scripts/source directory." >&2
    exit 1
fi

# The preprocessed raws and FreeSurfer labels live in the public tree; the LCMV
# filters live in the internal derivatives tree. Two roots, deliberately.
DATA_PATH="${DATA_PATH:-/share/klab/datasets/avs/avs-public}"
LCMV_ROOT="${LCMV_ROOT:-/share/klab/datasets/avs}"
SUBJECTS_DIR_ARG="${SUBJECTS_DIR_ARG:-/share/klab/datasets/avs/avs-public/derivatives/freesurfer}"
OUTPUT_DIR="${OUTPUT_DIR:-/share/klab/psulewski/psulewski/pyavs/nsd_roi_timecourses}"
ROI_MODE="${ROI_MODE:-abs_mean}"

# Decode subject (1-5) and session (1-10) from array task ID (1-50).
# With no array (single validation run), fall back to SUBJECT/SESSION env vars.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    subject=$(( (SLURM_ARRAY_TASK_ID - 1) / 10 + 1 ))
    session=$(( (SLURM_ARRAY_TASK_ID - 1) % 10 + 1 ))
else
    subject="${SUBJECT:?set SUBJECT when not running as an array job}"
    session="${SESSION:?set SESSION when not running as an array job}"
fi

echo "==================================================="
echo "NSD stream source projection"
echo "  subject     : $subject"
echo "  session     : $session"
echo "  roi mode    : $ROI_MODE"
echo "  data path   : $DATA_PATH"
echo "  lcmv root   : $LCMV_ROOT"
echo "  subjects dir: $SUBJECTS_DIR_ARG"
echo "  output dir  : $OUTPUT_DIR"
echo "  array task  : ${SLURM_ARRAY_TASK_ID:-<none>}"
echo "==================================================="

python "${script_path}/project_fixations_to_nsd_rois.py" \
    --subjects "$subject" \
    --sessions "$session" \
    --data-path "$DATA_PATH" \
    --lcmv-root "$LCMV_ROOT" \
    --subjects-dir "$SUBJECTS_DIR_ARG" \
    --output-dir "$OUTPUT_DIR" \
    --roi-mode "$ROI_MODE" \
    ${MAX_EPOCHS:+--max-epochs $MAX_EPOCHS} \
    ${EXTRA_ARGS:-}

echo ""
echo "==================================================="
echo "subject $subject session $session complete"
echo "==================================================="
