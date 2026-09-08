#!/bin/bash
#
# Submit the flanker sweep on SLURM: the noise ladder and the target-onset delay ladder,
# both defined in flanker_sweep_config.py. This replaced run_flanker_factorial.sh, which
# submitted four arms of a 2x2 over oracle gate jitter x p_corr_by_distance[2]; that
# factorial is retired (its result is recorded in the config's docstring) and the sweep
# now runs a single configuration — the one run_flanker.py runs.
#
#   ./run_flanker_sweep.sh check     # parity + job counts, submits nothing
#   ./run_flanker_sweep.sh submit    # pretrain array -> test array (dependency wired)
#   ./run_flanker_sweep.sh figures   # once they finish: build the group figures
#
# Array sizes are computed from the config rather than hard-coded, so adding or removing a
# ladder rung or a seed cannot leave a stale number behind.

set -euo pipefail

MODE="${1:-}"
PYTHON="${PYTHON:-.venv/bin/python}"

read -r MAX_PRETRAIN MAX_TEST RUN <<< "$($PYTHON - <<'PY'
import flanker_sweep_config as C
# Same rule as flanker_sweep.pretrain_tag: a variant with Stage-1 overrides gets its own
# model cache, so it is its own pretrain tag. The delay rungs have none, so they all
# collapse onto 'shared' and add no pretraining.
tags = {v if s.get('pretrain_overrides') else 'shared' for v, s in C.VARIANTS.items()}
print(C.SEEDS * len(tags) - 1, C.SEEDS * len(C.VARIANTS) - 1, C.RUN_NAME)
PY
)"

case "$MODE" in
check)
    # Parity first: a sweep that has drifted from run_flanker.py is not worth submitting.
    $PYTHON flanker_sweep.py parity
    echo
    echo "run: $RUN"
    echo "arrays: pretrain 0..$MAX_PRETRAIN  ($((MAX_PRETRAIN+1)) jobs)"
    echo "        test     0..$MAX_TEST  ($((MAX_TEST+1)) jobs)"
    ;;
submit)
    $PYTHON flanker_sweep.py parity
    echo
    echo "run: $RUN   arrays: pretrain 0..$MAX_PRETRAIN, test 0..$MAX_TEST"
    pre=$(./submit_job.sh "$MAX_PRETRAIN" flanker_pretrain | tail -1)
    tst=$(./submit_job.sh "$MAX_TEST" flanker "$pre" | tail -1)
    echo "  pretrain=$pre  test=$tst (afterany:$pre)"
    ;;
figures)
    $PYTHON flanker_sweep_figures.py --run "$RUN"
    echo
    echo "Figures under exports/flanker_random/sweeps/$RUN"
    ;;
*)
    echo "Usage: $0 {check|submit|figures}"
    exit 1
    ;;
esac
