#!/bin/bash

# Usage: ./submit_job.sh <MAX_TASK_ID> <EXPERIMENT_NAME>
# Example: ./submit_job.sh 599 learning

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <MAX_TASK_ID> <EXPERIMENT_NAME>"
    echo "EXPERIMENT_NAME: learning | generalization_tests"
    exit 1
fi

MAX_TASK_ID=$1
EXPERIMENT_NAME=$2
MAX_PARALLEL=100  # Max concurrent jobs

if [ "$EXPERIMENT_NAME" = "learning" ]; then
    PYTHON_FILE="cst_correlated_noise_sweep.py"
elif [ "$EXPERIMENT_NAME" = "generalization_tests" ]; then
    PYTHON_FILE="cst_run_generalization.py"
elif [ "$EXPERIMENT_NAME" = "mean_prediction" ]; then
    PYTHON_FILE="mean_prediction_sweep.py"
else
    echo "Invalid experiment name: $EXPERIMENT_NAME"
    echo "Valid options: learning | generalization_tests | mean_prediction"
    exit 1
fi

mkdir -p ./slurm

sbatch --array=0-$MAX_TASK_ID%$MAX_PARALLEL <<EOF
#!/bin/bash
#SBATCH --job-name=neuragem
#SBATCH -n 1
#SBATCH --partition=batch
#SBATCH --output=./slurm/slurm-%A_%a.out
#SBATCH --error=./slurm/slurm-%A_%a.err
#SBATCH --time=0-00:20:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hummosa@live.com

# Activate env and run
source $HOME/load_python_venv.sh

python $PYTHON_FILE
EOF

echo "Submitted array jobs 0..$MAX_TASK_ID for '$EXPERIMENT_NAME' with max parallelism $MAX_PARALLEL."
