#!/bin/bash
# Record and analyse the manipulation sessions (hier_switch_hooks) as a SLURM array:
#     ./hier_switch/run_manipulations.sh [RANGE]
# One task per model, as run_sessions.sh does, but with HIER_SWITCH_MANIP=1 so each NG task
# also records the manipulation conditions. A condition whose session.npz already exists is
# skipped, so this fills gaps rather than redoing work; HIER_SWITCH_FORCE=1 re-records.
set -e
cd "$(dirname "$0")/.."
N=$(HIER_SWITCH_MANIP=1 .venv/bin/python hier_switch/hier_switch_group.py list | wc -l)
RANGE=${1:-0-$((N - 1))}
sbatch --parsable --array=$RANGE <<EOS
#!/bin/bash
#SBATCH --job-name=hsw_manip
#SBATCH -n 1
#SBATCH --partition=batch
#SBATCH --output=./slurm/hsw-manip-%A_%a.out
#SBATCH --error=./slurm/hsw-manip-%A_%a.err
#SBATCH --time=0-03:00:00
export OMP_NUM_THREADS=1
export HIER_SWITCH_MANIP=1
source \$HOME/load_python_venv.sh
cd $(pwd)
python hier_switch/hier_switch_group.py task
EOS
