#!/bin/bash

# Usage: ./submit_job.sh <MAX_TASK_ID> <EXPERIMENT_NAME>
# Example: ./submit_job.sh 599 learning

VALID="learning | generalization_tests | mean_prediction | flanker_pretrain | flanker | rotation_slips | curriculum"

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <MAX_TASK_ID> <EXPERIMENT_NAME>"
    echo "EXPERIMENT_NAME: $VALID"
    echo ""
    echo "Flanker sweep, in this order:"
    echo "  ./submit_job.sh 39 flanker_pretrain   # train the model sets first"
    echo "  ./submit_job.sh 99 flanker            # then the test sessions"
    exit 1
fi

MAX_TASK_ID=$1
EXPERIMENT_NAME=$2
MAX_PARALLEL=100  # Max concurrent jobs

# Per-experiment wall clock: one task of most sweeps is a single ~3 min training run, but a
# curriculum task runs a whole 19-run stage tree (~25 min). Keep the short limit for the others
# rather than raising it globally — a longer request queues behind more.
TIME_LIMIT="0-00:20:00"

if [ "$EXPERIMENT_NAME" = "learning" ]; then
    PYTHON_FILE="cst_correlated_noise_sweep.py"
elif [ "$EXPERIMENT_NAME" = "generalization_tests" ]; then
    PYTHON_FILE="cst_run_generalization.py"
elif [ "$EXPERIMENT_NAME" = "mean_prediction" ]; then
    PYTHON_FILE="mean_prediction_sweep.py"
elif [ "$EXPERIMENT_NAME" = "flanker_pretrain" ]; then
    PYTHON_FILE="flanker_sweep.py"
    PYTHON_ARGS="pretrain"
elif [ "$EXPERIMENT_NAME" = "flanker" ]; then
    PYTHON_FILE="flanker_sweep.py"
elif [ "$EXPERIMENT_NAME" = "rotation_slips" ]; then
    PYTHON_FILE="rotation_slips_perseveration_sweep.py"
elif [ "$EXPERIMENT_NAME" = "curriculum" ]; then
    PYTHON_FILE="rotation_curriculum_sweep.py"
    # TIME_LIMIT="0-02:00:00"
else
    echo "Invalid experiment name: $EXPERIMENT_NAME"
    echo "Valid options: $VALID"
    exit 1
fi

# SLURM cancels a task at 0 seconds if it cannot open its --output file, and reports only
# ExitCode 0:53 with no log to explain it. `mkdir -p ./slurm` does NOT cover that case: when
# ./slurm is a symlink to a missing directory it fails with "File exists" and, with no set -e,
# the script would happily submit an array that is guaranteed to die. Resolve the link and
# verify the target is writable before submitting anything.
LOG_DIR="$(readlink -f ./slurm 2>/dev/null || echo ./slurm)"
mkdir -p "$LOG_DIR"
if [ ! -w "$LOG_DIR" ]; then
    echo "ERROR: SLURM log directory '$LOG_DIR' does not exist or is not writable."
    echo "       (./slurm resolves there; a dangling symlink makes every array task fail"
    echo "        instantly with ExitCode 0:53 and no log file.)"
    exit 1
fi

sbatch --array=0-$MAX_TASK_ID%$MAX_PARALLEL <<EOF
#!/bin/bash
#SBATCH --job-name=neuragem
#SBATCH -n 1
#SBATCH --partition=batch
#SBATCH --output=./slurm/slurm-%A_%a.out
#SBATCH --error=./slurm/slurm-%A_%a.err
#SBATCH --time=$TIME_LIMIT
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hummosa@live.com

# Activate env and run
source $HOME/load_python_venv.sh

python $PYTHON_FILE $PYTHON_ARGS
EOF

echo "Submitted array jobs 0..$MAX_TASK_ID for '$EXPERIMENT_NAME' with max parallelism $MAX_PARALLEL."
