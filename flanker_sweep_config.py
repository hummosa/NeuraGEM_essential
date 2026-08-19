"""
Configuration for the flanker seed x p_congruent sweep.

Edit this file to change what flanker_sweep.py runs — seeds, session lengths,
congruency proportions, export paths. The runner itself holds no settings.

Design
──────
Each seed is a synthetic subject: its own Stage-1 pretraining, i.e. its own weights.
Pretraining is done once per seed and cached, then the same frozen model is tested at
every p_congruent level. That makes congruency proportion a *within-subject*
manipulation, exactly as in the human design, so a shift in the congruency effect
across levels cannot be attributed to a different network.
"""

# ── Subjects and session length ───────────────────────────────────────────────
SEEDS             = 10      # one pretrained model per seed
N_PRETRAIN_TRIALS = 2000    # Stage 1, weights plastic, oracle Z
N_TEST_TRIALS     = 5000    # Stage 2, weights frozen, Z inferred, random trials

# ── Manipulation ──────────────────────────────────────────────────────────────
# 0.5 is the reference condition and matches the human task. The other levels test
# the list-wide proportion-congruent prediction: congruency effects should shrink as
# congruent trials become more frequent.
P_CONGRUENT_LEVELS = [0.2, 0.5, 0.8]

# ── Model settings ────────────────────────────────────────────────────────────
GATING = 'post'             # 'pre' or 'post' multiplicative gating
Z_INIT_SCALE = 0.2          # Z re-seed before the test session

PRETRAIN_OVERRIDES = {}     # e.g. {'arrow_noise_std': 1.3}
TEST_OVERRIDES     = {'no_of_steps_in_latent_space': 1}

# ── Variants ──────────────────────────────────────────────────────────────────
# Extra test-stage config overrides, each run as its own set of sessions. Only the
# test session changes, so every variant reuses the cached pretrained models.
#
# Z_decay is the knob that sets *where* the control state settles: it pulls raw Z
# toward zero, and zero is a uniform softmax, i.e. fully diffuse attention. Z_lr
# would change how fast Z moves and how much it jitters, but not its operating point
# — with Adam the stationary Z is where the task gradient balances the decay term,
# which is independent of the learning rate.
#
# 'baseline' keeps the original export paths; every other variant gets its own folder.
VARIANTS = {
    # ── reference ────────────────────────────────────────────────────────────
    # p_corr_by_distance is pinned here rather than inherited from FlankerTaskConfig.
    # The class default was changed to the steep gradient (0.75/0.52) so that
    # run_flanker.py trains the reference model; without this pin, re-running the
    # sweep's pretraining would silently make 'baseline' identical to 'spatial_steep'
    # and invalidate every comparison against it. This value is what the cached
    # models in exports/.../pretrained/ were actually trained with.
    'baseline':   dict(overrides={},
                       pretrain_overrides={'p_corr_by_distance': [1.0, 0.65, 0.55, 0.25, 0.1]},
                       p_levels=P_CONGRUENT_LEVELS),

    # ── how strongly the control state is pulled back to diffuse attention ───
    # 1x is the baseline (Z_decay = 1e-4). 2x fills the large gap between 1x and 5x,
    # and should land near the control level the mostly-congruent list produces, which
    # makes it the matched-focus comparison against p_congruent=0.8.
    'decay2x':    dict(overrides={'Z_decay': 2e-4},     p_levels=[0.5]),
    'decay5x':    dict(overrides={'Z_decay': 5e-4},     p_levels=[0.5]),
    'decay25x':   dict(overrides={'Z_decay': 2.5e-3},   p_levels=[0.5]),
    'decay100x':  dict(overrides={'Z_decay': 1e-2},     p_levels=[0.5]),

    # ── how steep the spatial gradient is during TRAINING ────────────────────
    # These change Stage 1, so each gets its own pretrained models.
    #
    # In the test stage the target is always the centre slot, so near flankers sit at
    # distance 1 and far flankers at distance 2. The only two numbers that teach the
    # model anything about near vs far are therefore p_corr_by_distance[1] and [2].
    # At the default 0.65 vs 0.55 that gradient is very shallow, which is the likely
    # reason the behavioural near/far effect is small and inconsistent across seeds.
    #
    # Both entries stay above 0.5 so a far companion remains weakly informative rather
    # than becoming anti-correlated, which would be a different manipulation.
    #                                                    gap at [1] vs [2]
    'spatial_flat':    dict(pretrain_overrides={'p_corr_by_distance': [1.0, 0.60, 0.60, 0.25, 0.1]},
                            p_levels=[0.5]),             # 0.00 - predicts no distance effect
    # Run at every congruency level: with a training gradient steep enough to produce a
    # reliable distance effect, does the list manipulation still shift it on top?
    'spatial_steep':   dict(pretrain_overrides={'p_corr_by_distance': [1.0, 0.75, 0.52, 0.25, 0.1]},
                            p_levels=P_CONGRUENT_LEVELS),  # 0.23
    'spatial_steeper': dict(pretrain_overrides={'p_corr_by_distance': [1.0, 0.85, 0.51, 0.25, 0.1]},
                            p_levels=[0.5]),             # 0.34
}

# ── I/O ───────────────────────────────────────────────────────────────────────
RUN_NAME      = 'sweep_v1'
EXPORT_ROOT   = './exports/flanker_random/sweeps'
SKIP_EXISTING = True        # resume: skip jobs whose result pickle already exists

# ── Analysis ──────────────────────────────────────────────────────────────────
RT_THRESHOLD = 0.5
