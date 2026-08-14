#!/bin/bash
#SBATCH --partition=klab-cpu
#SBATCH --cpus-per-task=15
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --job-name=save_annotated_raws
#SBATCH --output=save_raws_sub%a_%j.out
#SBATCH --error=save_raws_sub%a_%j.err
#SBATCH --array=1-5

echo "Running in shell: $SHELL"
echo "Processing subject $SLURM_ARRAY_TASK_ID as part of array job $SLURM_ARRAY_JOB_ID"

export NCCL_SOCKET_IFNAME=lo

# Load conda environment
spack load miniconda3
eval "$(conda shell.bash hook)"
conda activate avs

# Set subject ID from SLURM array task ID
SUBJECT_ID=$SLURM_ARRAY_TASK_ID

echo "Starting processing for Subject $SUBJECT_ID"
echo "Time: $(date)"

# Process all sessions (1-10) for this subject
for SESSION in {1..10}; do
    echo "----------------------------------------"
    echo "Processing Subject $SUBJECT_ID, Session $SESSION"
    echo "Time: $(date)"
    
    # Run the save annotated raw data script
    python /home/student/p/psulewski/pyAVS/scripts/save_annotated_raw_data.py \
        --subject $SUBJECT_ID \
        --session $SESSION \
        --resample_freq 500 \
        --l_freq 0.2 \
        --h_freq 200 \
        --event_types saccade fixation blink \
        --overwrite \

    
    # Check exit status
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed Subject $SUBJECT_ID, Session $SESSION"
    else
        echo "✗ Error processing Subject $SUBJECT_ID, Session $SESSION"
        # Continue with next session rather than failing completely
    fi
    
    echo "Finished Subject $SUBJECT_ID, Session $SESSION at $(date)"
done

echo "----------------------------------------"
echo "Completed all sessions for Subject $SUBJECT_ID"
echo "Time: $(date)"