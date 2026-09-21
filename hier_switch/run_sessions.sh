#!/bin/bash
# Record and analyse every group session as a SLURM array:
#     ./hier_switch/run_sessions.sh [RANGE] [AFTER_JOBID]
# One task per model (hier_switch_group.py list): it records that model's conditions with
# hidden states and pulses, then writes a results.json beside each session. AFTER_JOBID
# makes the array wait for a training array (e.g. run_tune.sh v17) — with afterany, not
# afterok: a task whose model never appeared just exits with "no model at ...", whereas
# afterok would leave the whole array pending forever if one training task failed.
set -e
cd "$(dirname "$0")/.."
N=$(.venv/bin/python hier_switch/hier_switch_group.py list | wc -l)
RANGE=${1:-0-$((N - 1))}
DEP=${2:+--dependency=afterany:$2}
sbatch --parsable $DEP --array=$RANGE <<EOS
#!/bin/bash
#SBATCH --job-name=hsw_sess
#SBATCH -n 1
#SBATCH --partition=batch
#SBATCH --output=./slurm/hsw-%A_%a.out
#SBATCH --error=./slurm/hsw-%A_%a.err
#SBATCH --time=0-01:30:00
export OMP_NUM_THREADS=1
source \$HOME/load_python_venv.sh
cd $(pwd)
python hier_switch/hier_switch_group.py task
EOS
