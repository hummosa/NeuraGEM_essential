#!/bin/bash
#
# The 2x2 flanker factorial: oracle gate jitter x p_corr_by_distance[2], each arm run as
# its own sweep over the 5-level noise ladder. Arms are defined in
# flanker_sweep_config.ARMS and selected by the FLANKER_ARM environment variable, so all
# four submissions read one unedited config file and no arm can race another's edit.
#
#   ./run_flanker_factorial.sh submit    # 4 arms x (pretrain array -> test array)
#   ./run_flanker_factorial.sh figures   # once they finish: build and collect the PDFs
#
# Array sizes are computed from the config rather than hard-coded, so adding or removing a
# noise level or a seed cannot leave a stale number behind.

set -euo pipefail

MODE="${1:-}"
ARMS="nojit_pc52 jit_pc52 nojit_pc58 jit_pc58"
PYTHON="${PYTHON:-.venv/bin/python}"
FIG_DIR="exports/flanker_random/factorial_corr_jitter"
SCORECARD_VARIANT="${SCORECARD_VARIANT:-noise09}"   # the 0.9 noise level

read -r MAX_PRETRAIN MAX_TEST <<< "$($PYTHON - <<'PY'
import flanker_sweep_config as C
# Same rule as flanker_sweep.pretrain_tag: a variant with Stage-1 overrides gets its own
# model cache, so it is its own pretrain tag.
tags = {v if s.get('pretrain_overrides') else 'shared' for v, s in C.VARIANTS.items()}
print(C.SEEDS * len(tags) - 1, C.SEEDS * len(C.VARIANTS) - 1)
PY
)"

case "$MODE" in
submit)
    echo "Arrays per arm: pretrain 0..$MAX_PRETRAIN, test 0..$MAX_TEST"
    for arm in $ARMS; do
        echo
        echo "── $arm ──"
        pre=$(FLANKER_ARM=$arm ./submit_job.sh "$MAX_PRETRAIN" flanker_pretrain | tail -1)
        tst=$(FLANKER_ARM=$arm ./submit_job.sh "$MAX_TEST" flanker "$pre" | tail -1)
        echo "  pretrain=$pre  test=$tst (afterany:$pre)"
    done
    ;;
figures)
    mkdir -p "$FIG_DIR"
    for arm in $ARMS; do
        echo
        echo "── $arm ──"
        FLANKER_ARM=$arm $PYTHON flanker_sweep_figures.py --run "factorial_$arm"
        src="exports/flanker_random/sweeps/factorial_$arm"
        cp "$src/$SCORECARD_VARIANT/group_7_scorecard.pdf" \
           "$FIG_DIR/${arm}__group_7_scorecard_${SCORECARD_VARIANT}.pdf"
        cp "$src/group_8_noise_series.pdf" \
           "$FIG_DIR/${arm}__group_8_noise_series.pdf"
    done
    echo
    echo "Collected into $FIG_DIR:"
    ls -1 "$FIG_DIR"
    ;;
*)
    echo "Usage: $0 {submit|figures}"
    exit 1
    ;;
esac
