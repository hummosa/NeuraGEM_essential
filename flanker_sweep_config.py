"""
Configuration for the flanker noise sweep.

Edit this file to change what flanker_sweep.py runs — seeds, session lengths, the noise
levels, export paths. The runner itself holds no settings.

Design
──────
Each seed is a synthetic subject with its own Stage-1 pretraining, i.e. its own weights.
`arrow_noise_std` is a stimulus parameter, so it has to be the same in training and test:
every noise level therefore gets its own pretrained model per seed, and the comparison
across noise is between-subject rather than within.

What is crossed
───────────────
Three axes: the noise ladder below (5 levels), and two Stage-1 knobs — oracle gate jitter
and p_corr_by_distance[2] — crossed 2x2 into four separate sweep runs, one per ARM. The
noise ladder runs inside every arm.

Everything else is held at exactly what run_flanker.py, the single-session workbench,
runs; the two scripts' configs were diffed attribute by attribute and only bg_noise_std
differed (see PRETRAIN_OVERRIDES). Keep it that way — an axis that is not in ARMS or
VARIANTS is a silent departure from the simulation the workbench figures describe.

Why noise
─────────
The model reproduces the flanker fingerprint and conflict adaptation but fails the
post-error signatures (PIA, PERI). The diagnosis: `arrow_noise_std` is large enough that
the *target slot's own samples* often point the wrong way, so most errors are bad luck
rather than too little control. The latent update minimises this trial's prediction error,
so on those trials it correctly attends the target *less* — locally right, globally
anti-adaptive. Lowering the noise should shrink the share of such errors and restore the
signatures. That single prediction is what this sweep tests.

The axes that used to be here are gone: p_congruent (the proportion-congruent effect is
established and does not need re-running), Z_decay (its role as the control-magnitude knob
is understood), and the spatial gradient (the steep setting is now the default in
configs.py, so it is the model rather than a variant).

Running the four arms
─────────────────────
    FLANKER_ARM=nojit_pc52 python flanker_sweep.py pretrain     # then without `pretrain`

On SLURM, `./run_flanker_factorial.sh` submits all four arms with the right array sizes
and pretrain -> test dependencies, and collects the two figures per arm.
"""

import os

# ── Subjects and session length ───────────────────────────────────────────────
# 20 rather than 10: a split-half decomposition showed 39–100% of the across-seed spread
# in the sequential and post-error measures is within-session noise at n=10, which is what
# made PIA and PERI flip between runs. Seeds are the cheap fix — one job each.
SEEDS             = 20      # one pretrained model per seed per noise level
N_PRETRAIN_TRIALS = 4000    # Stage 1, weights plastic, oracle Z
                            # (matches FlankerTaskConfig.n_pretrain_trials; the
                            #  sweep sets it explicitly so a config change cannot
                            #  silently alter what a named run means)
N_TEST_TRIALS     = 5000    # Stage 2, weights frozen, Z inferred, random trials

# Timesteps per trial, applied through config.set_arrows_duration() in both stages. Stated
# here for the same reason N_PRETRAIN_TRIALS is: a later change to the class default must
# not silently alter what a named run on disk means. It was 5 (4 response steps); 10 gives
# 9, which is what makes room for a delayed target onset and unbumps the RT density.
ARROWS_DURATION   = 10

# ── Manipulation ──────────────────────────────────────────────────────────────
# One congruency level, matching the human task. It is a scalar, not a list: nothing
# sweeps over it any more.
P_CONGRUENT = 0.5

# ── Model settings ────────────────────────────────────────────────────────────
GATING = 'post'             # 'pre' or 'post' multiplicative gating
Z_INIT_SCALE = 0.2          # Z re-seed before the test session

# ── The 2x2: oracle gate jitter x p_corr_by_distance[2] ───────────────────────
#
# Two Stage-1 knobs are crossed, one sweep run per cell. The cell is chosen by the
# FLANKER_ARM environment variable, so all four arms read this file unedited and no
# submission can race a mid-flight edit:
#
#     FLANKER_ARM=nojit_pc52 python flanker_sweep.py        # the baseline arm
#
# jitter   Redraws the sharpness of the oracle gate on every training trial. None is the
#          fixed oracle every run before this used; (0.5, 1.5) is the jittered arm.
#          Stage 1 varies WHICH slot the oracle points at (the target rotates) but never
#          HOW SHARPLY: softmax(one-hot / softmax_temp) is peak 0.405 on every training
#          trial at the default temperature. The weights are therefore calibrated to emit
#          +-1 at that one sharpness, and at a sharper gate they overshoot. Stage 2 infers
#          Z freely and runs sharper than 0.405 on roughly three quarters of trials.
#          Because the latent update descends squared error, on a congruent trial (whose
#          sign is already right at every gate) the only remaining gradient is "make the
#          output smaller", and the update obeys it by flattening the gate. Half the list
#          therefore teaches the controller to stop attending, it drifts onto slots that
#          carry no arrow, and near-flanker trials — which leave the OUTER slots empty,
#          the damaging case — pay for it as a spurious near-vs-far accuracy difference on
#          CONGRUENT trials. Full diagnosis: flanker_near_cong_diagnostic.py.
#
#          WHAT THIS FACTORIAL FOUND (20 seeds, arrow_noise_std 0.9). Jitter is not as
#          critical as that story implies. It was adopted against runs with
#          bg_noise_std = 0.1; at 0 the near-vs-far congruent artifact is already gone
#          without it (+0.011, matches humans, in the no-jitter baseline). What jitter
#          then does is mostly cost: it REVERSES PERI, +0.073 -> -0.188, and pushes the
#          incongruent RT distance effect out of significance, +0.137 -> +0.097. It also
#          raises the control state, mean focus 0.342 -> 0.391 — Z runs sharper — which is
#          the mechanism to suspect for both. Its one real gain is post-error slowing,
#          pes_BI 0.108 (n.s.) -> 0.375. Net: 9 of 11 human signatures matched without
#          jitter, 8 with. The baseline arm is the one to build on.
#
#          Across the whole ladder jitter never matches MORE signatures than the baseline
#          at any level — ties at 1.3 and 0.7 (cleaner at 0.7: 0 opposite vs 1), loses at
#          1.0, 0.9 and 0.4. Its one consistent effect is raising mean focus ~0.05 at every
#          level. The PERI reversal is mid-ladder: at arrow_noise_std 0.4 jitter roughly
#          doubles PERI instead, 0.397 -> 0.936.
#
# p_corr2  p_corr_by_distance[2] — the probability that a companion two slots from the
#          target matches it. 0.52 is barely above chance, 0.58 is the stronger coupling.
#          The rest of the profile is held at [1.0, 0.75, _, 0.51, 0.5]. Nothing in it may
#          dip below 0.5: a companion that predicted the OPPOSITE direction taught the
#          model negative read-out weights at that distance (see configs.py).
ARMS = {
    'nojit_pc52': dict(jitter=None,       p_corr2=0.52),   # baseline: neither knob on
    'jit_pc52':   dict(jitter=(0.5, 1.5), p_corr2=0.52),   # jitter only
    'nojit_pc58': dict(jitter=None,       p_corr2=0.58),   # correlation only
    'jit_pc58':   dict(jitter=(0.5, 1.5), p_corr2=0.58),   # both
}

ARM = os.environ.get('FLANKER_ARM', 'nojit_pc52')
if ARM not in ARMS:
    raise ValueError(f'FLANKER_ARM={ARM!r} is not one of {sorted(ARMS)}')

ORACLE_GATE_JITTER = ARMS[ARM]['jitter']
P_CORR_BY_DISTANCE = [1.0, 0.75, ARMS[ARM]['p_corr2'], 0.51, 0.5]

# bg_noise_std is 0 in BOTH stages, which is what run_flanker.py's update_config() does.
# The class default is 0.1 and every sweep before this one silently inherited it, so the
# sweep and the single-session workbench were not running the same simulation. An
# attribute-by-attribute diff of the two scripts' configs says this was the ONLY setting
# that differed; keep it that way, and change run_flanker.py alongside it if it moves.
# The stimulus noise a variant gets when it does NOT carry its own pretrain_overrides —
# i.e. the delay ladder, whose whole point is to reuse one pretrained model set. The noise
# ladder below overrides it per rung ("Variant Stage-1 overrides last, so they win").
#
# 1.35 is 0.9 x 1.5, the old working point carried across the retiming: evidence
# accumulates over the response window, so SNR grows as sqrt(n_response_steps) and 4 -> 9
# steps is a factor of 1.5. That is a first-order estimate, not a calibration — read the
# real working point off the scorecard once the ladder has run.
DELAY_BASE_NOISE = 1.35

PRETRAIN_OVERRIDES = {                      # every variant's Stage 1
    'oracle_gate_jitter': ORACLE_GATE_JITTER,
    'p_corr_by_distance': P_CORR_BY_DISTANCE,
    'bg_noise_std':       0,
    'arrow_noise_std':    DELAY_BASE_NOISE,
}
TEST_OVERRIDES = {
    'no_of_steps_in_latent_space': 1,
    'bg_noise_std':                0,
}

# ── Variants: the noise ladder ────────────────────────────────────────────────
# `arrow_noise_std` is the SD of the per-timestep noise on each arrow, against a signal of
# 1.0. At 1.3 the target slot misleads on a large minority of trials; at 0.4 it almost
# never does. Everything else is held fixed.
#
# These are `pretrain_overrides` because the stimulus has to match across stages —
# flanker_sweep applies them to the test config too, and gives each level its own model
# cache so a level can never accidentally read another's weights.
# 0.6, 0.8 and 0.9 fill in the interesting stretch: the post-error signatures and the
# sign of the latent's response to a bad-luck error both turn over between 1.0 and 0.7,
# and four points were too coarse to locate that crossing.
#
# The retiming to 10 timesteps raised the SNR by ~1.5x, which slides the whole ladder
# toward "too easy": the old top rung 1.3 is now worth about 0.87 of the old scale, i.e.
# roughly where the old working point already was. Two higher rungs are ADDED rather than
# the existing ones rescaled, because the rung names are also the directory names of the
# 400 five-timestep result pickles already on disk, and several scripts default to them
# (flanker_regression, flanker_near_cong_diagnostic). Renaming would strand all of that
# for no gain; RUN_NAME already keeps the two worlds apart.
VARIANTS = {
    'noise19': dict(pretrain_overrides={'arrow_noise_std': 1.9}),   # added for ad=10
    'noise16': dict(pretrain_overrides={'arrow_noise_std': 1.6}),   # added for ad=10
    'noise13': dict(pretrain_overrides={'arrow_noise_std': 1.3}),   # the original setting
    'noise10': dict(pretrain_overrides={'arrow_noise_std': 1.0}),
    'noise09': dict(pretrain_overrides={'arrow_noise_std': 0.9}),
    # 'noise08': dict(pretrain_overrides={'arrow_noise_std': 0.8}),
    'noise07': dict(pretrain_overrides={'arrow_noise_std': 0.7}),
    # 'noise06': dict(pretrain_overrides={'arrow_noise_std': 0.6}),
    'noise04': dict(pretrain_overrides={'arrow_noise_std': 0.4}),   # near-clean target
}

#: Ordered (variant, noise level) for the figures that plot against noise.
NOISE_LADDER = [('noise19', 1.9), ('noise16', 1.6), ('noise13', 1.3),
                ('noise10', 1.0), ('noise09', 0.9),
                ('noise07', 0.7), ('noise04', 0.4)]

# ── The target-onset delay ladder ─────────────────────────────────────────────
#
# "Flankers first": the flankers are on screen from frame 0 and the TARGET's onset is
# delayed. The question is whether the response is delayed with it — and whether congruent
# trials are held up less, because during the delay the flankers alone already point at
# the answer.
#
# These are TEST-stage `overrides`, not `pretrain_overrides`, and that is the whole
# economy of this axis: the stimulus the weights were trained on is unchanged, so
# `pretrain_tag` resolves every rung to the 'shared' model set and all three levels reuse
# ONE pretrained model per seed. Giving them pretrain_overrides would hand each level its
# own cache tag and triple the pretraining bill for identical Stage-1 stimuli.
#
# Nothing here touches response_start_timestep or temporal_loss_weights. Speed pressure is
# identical at every rung and RT is measured from trial start, so a delayed response shows
# up as a larger RT rather than being defined away. See FlankerTaskConfig.target_delay.
DELAY_LEVELS = [0, 2, 4]        # 9 response steps, so 4 still leaves 5 post-onset

VARIANTS.update({
    f'delay{d}': dict(overrides={'target_delay': d}) for d in DELAY_LEVELS
})

#: Ordered (variant, delay) for the delay-series figure.
DELAY_LADDER = [(f'delay{d}', d) for d in DELAY_LEVELS]

# ── I/O ───────────────────────────────────────────────────────────────────────
# RUN_NAME is the single switch: it decides where a sweep writes AND which sweep every
# analysis and figure script reads. Change it here and everything follows; every entry
# point prints the run it used, and `flanker_sweep.describe_runs()` lists what is on disk
# with the parameters read from the stored configs.
#
# Never reuse a run name for different settings. The latent optimizer is baked into the
# pretrained model at construction and `mirror_to_model` can only patch lr/decay, so
# reusing a cache across optimizers would silently run the old one.
# One run per 2x2 cell, named after the arm, so the four never share a cache and every
# figure script can be pointed at one of them with --run factorial_<arm>.
# ad10_ prefix, not factorial_: arrows_duration moved, so these results are not comparable
# with the 400 pickles under factorial_* and must not land beside them.
RUN_NAME      = f'ad10_{ARM}'
EXPORT_ROOT   = './exports/flanker_random/sweeps'
SKIP_EXISTING = True        # resume: skip jobs whose result pickle already exists

# ── Analysis ──────────────────────────────────────────────────────────────────
RT_THRESHOLD = 0.2
