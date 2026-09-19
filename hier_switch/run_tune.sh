#!/bin/bash
# Submit one tuning grid as a SLURM array: ./hier_switch/run_tune.sh <TAG>
# The array size is read from the grid itself, so it cannot drift from GRIDS.
set -e
cd "$(dirname "$0")/.."
TAG=$1
N=$(.venv/bin/python hier_switch/hier_switch_tune.py list "$TAG" | wc -l)
sbatch --parsable --array=0-$((N - 1)) <<EOS
#!/bin/bash
#SBATCH --job-name=hsw_$TAG
#SBATCH -n 1
#SBATCH --partition=batch
#SBATCH --output=./slurm/hsw-%A_%a.out
#SBATCH --error=./slurm/hsw-%A_%a.err
#SBATCH --time=0-01:00:00
source \$HOME/load_python_venv.sh
cd $(pwd)
python hier_switch/hier_switch_tune.py run $TAG
EOS
