#!/bin/bash
# Z-clamp grids as a SLURM array:  ./hier_switch/run_clamp.sh [RANGE] [AFTER_JOBID]
# One task per (seed, activation) — see `hier_switch_perturb.py list`. Each cell is a test
# session with the latent update off and Z held at a chosen gate. AFTER_JOBID makes the
# array wait for the session array, whose sessions it uses as the context-axis reference.
# HIER_SWITCH_LEVEL picks the cue-noise level; it is read here to size the array and
# written into the submitted script so the two cannot disagree.
set -e
cd "$(dirname "$0")/.."
LEVEL=${HIER_SWITCH_LEVEL:-n05}
N=$(HIER_SWITCH_LEVEL=$LEVEL .venv/bin/python hier_switch/hier_switch_perturb.py list | wc -l)
RANGE=${1:-0-$((N - 1))}
DEP=${2:+--dependency=afterok:$2}
echo "level $LEVEL: $N clamp tasks, array $RANGE"
sbatch --parsable $DEP --array=$RANGE <<EOS
#!/bin/bash
#SBATCH --job-name=hsw_clamp
#SBATCH -n 1
#SBATCH --partition=batch
#SBATCH --output=./slurm/hsw-%A_%a.out
#SBATCH --error=./slurm/hsw-%A_%a.err
#SBATCH --time=0-02:00:00
export OMP_NUM_THREADS=1
export HIER_SWITCH_LEVEL=$LEVEL
source \$HOME/load_python_venv.sh
cd $(pwd)
python hier_switch/hier_switch_perturb.py task
EOS
