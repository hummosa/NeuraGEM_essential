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

Why noise is the only axis
──────────────────────────
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
"""

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

# ── Manipulation ──────────────────────────────────────────────────────────────
# One congruency level, matching the human task. It is a scalar, not a list: nothing
# sweeps over it any more.
P_CONGRUENT = 0.5

# ── Model settings ────────────────────────────────────────────────────────────
GATING = 'post'             # 'pre' or 'post' multiplicative gating
Z_INIT_SCALE = 0.2          # Z re-seed before the test session

PRETRAIN_OVERRIDES = {}     # applied to every variant's Stage 1
TEST_OVERRIDES     = {'no_of_steps_in_latent_space': 1}

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
VARIANTS = {
    'noise13': dict(pretrain_overrides={'arrow_noise_std': 1.3}),   # the original setting
    'noise10': dict(pretrain_overrides={'arrow_noise_std': 1.0}),
    'noise09': dict(pretrain_overrides={'arrow_noise_std': 0.9}),
    'noise08': dict(pretrain_overrides={'arrow_noise_std': 0.8}),
    'noise07': dict(pretrain_overrides={'arrow_noise_std': 0.7}),
    'noise06': dict(pretrain_overrides={'arrow_noise_std': 0.6}),
    'noise04': dict(pretrain_overrides={'arrow_noise_std': 0.4}),   # near-clean target
}

#: Ordered (variant, noise level) for the figures that plot against noise.
NOISE_LADDER = [('noise13', 1.3), ('noise10', 1.0), ('noise09', 0.9), ('noise08', 0.8),
                ('noise07', 0.7), ('noise06', 0.6), ('noise04', 0.4)]

# ── I/O ───────────────────────────────────────────────────────────────────────
# RUN_NAME is the single switch: it decides where a sweep writes AND which sweep every
# analysis and figure script reads. Change it here and everything follows; every entry
# point prints the run it used, and `flanker_sweep.describe_runs()` lists what is on disk
# with the parameters read from the stored configs.
#
# Never reuse a run name for different settings. The latent optimizer is baked into the
# pretrained model at construction and `mirror_to_model` can only patch lr/decay, so
# reusing a cache across optimizers would silently run the old one.
RUN_NAME      = 'sweep_noise_pt4k'
EXPORT_ROOT   = './exports/flanker_random/sweeps'
SKIP_EXISTING = True        # resume: skip jobs whose result pickle already exists

# ── Analysis ──────────────────────────────────────────────────────────────────
RT_THRESHOLD = 0.5
