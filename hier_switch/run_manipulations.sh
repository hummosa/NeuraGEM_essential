#!/bin/bash
# Record and analyse the manipulation sessions (hier_switch_hooks) as a SLURM array:
#     ./hier_switch/run_manipulations.sh [RANGE]
#
# One task per NeuraGEM seed — the RNN has no latent update, so there is nothing to
# perturb, and its tasks are left out of the default range. Each task records that seed's
# 15 manipulation conditions (2000 trials each, hidden states and pulses saved) and then
# re-analyses all of its sessions. A condition whose session.npz already exists is skipped,
# so re-running fills gaps rather than redoing work; HIER_SWITCH_FORCE=1 re-records.
#
# Cost: ~15 sessions x 6 seeds x ~2 min recording, plus ~21 analyses x ~30 s per task, so
# roughly 40-55 min per task inside a 3 h limit. Each session is ~6.4 MB, so this adds
# ~0.6 GB under exports/hier_switch (git-ignored).
#
# HIER_SWITCH_LEVEL picks the cue-noise level; it is read here to size the array and
# written into the submitted script so the two cannot disagree.
set -e
cd "$(dirname "$0")/.."
LEVEL=${HIER_SWITCH_LEVEL:-n05}
# models() lists the NG seeds first, then the RNN baselines; default to the NG block.
NG=$(HIER_SWITCH_MANIP=1 HIER_SWITCH_LEVEL=$LEVEL \
     .venv/bin/python hier_switch/hier_switch_group.py list | awk '$2 == "NG"' | wc -l)
RANGE=${1:-0-$((NG - 1))}
echo "level $LEVEL: $NG manipulation tasks, array $RANGE"
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
export HIER_SWITCH_LEVEL=$LEVEL
source \$HOME/load_python_venv.sh
cd $(pwd)
python hier_switch/hier_switch_group.py task
EOS
