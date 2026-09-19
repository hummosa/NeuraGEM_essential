"""
Configuration for the flanker sweep.

Edit this file to change what flanker_sweep.py runs — seeds, session lengths, the ladders,
export paths. The runner itself holds no settings.

The parity rule
───────────────
**This sweep must run the same simulation as flanker_run_one_network.py, the single-session
workbench.** That is the whole contract: the workbench is where parameters get tuned by
eye, and the sweep is where the same model is run across seeds for statistics. If they
drift, the group figures stop describing the sessions the workbench figures show.

Parity is achieved by *inheriting* the class defaults in configs.FlankerTaskConfig rather
than restating them here. Anything this file pins is something flanker_run_one_network.py also pins
explicitly; anything it stays silent about (`p_corr_by_distance`, `arrow_noise_std`,
`bg_noise_std`, `latent_activation`, `temporal_decay_factor`, the Z optimizer settings)
comes from the class, so editing configs.py moves both scripts together.

That is a deliberate reversal of this file's old policy of restating every value "so a
config change cannot silently alter what a named run means". Reproducibility is preserved
a better way: `stage1_fingerprint` records the values a run actually used into a sidecar
beside every model, `check_pretrain_fingerprint` refuses a cache trained under different
settings, and `flanker_sweep.describe_runs()` reports what is on disk. Those read the real
config, so they cannot go stale the way a duplicated constant can.

`flanker_sweep.check_parity()` asserts the rule, and is worth running after any edit to
configs.py or to this file.

What is crossed
───────────────
Two ladders, and they are cheap in very different ways:

  NOISE_LADDER  `arrow_noise_std` is a stimulus parameter, so it must match across Stage 1
                and Stage 2. Every rung therefore needs its OWN pretrained model per seed,
                and the comparison across noise is between-subject.

  DELAY_LADDER  `target_delay` is Stage-2 only — the weights never see it — so every rung
                carries test-stage `overrides` and no `pretrain_overrides`, which resolves
                them all to the 'shared' pretrain tag. All four delays reuse ONE model set
                per seed and cost no extra pretraining.

Axes that used to be here and are gone: the 2x2 ARMS factorial over oracle gate jitter x
p_corr_by_distance[2] (its result is recorded — 9 of 11 human signatures matched without
jitter against 8 with — and it hard-coded a p_corr profile that no longer matches
configs.py, which is exactly the drift the parity rule exists to prevent); p_congruent (the
proportion-congruent effect is established); and Z_decay.

Why noise
─────────
The model reproduces the flanker fingerprint and conflict adaptation but fails the
post-error signatures (PIA, PERI). The diagnosis: `arrow_noise_std` is large enough that
the *target slot's own samples* often point the wrong way, so most errors are bad luck
rather than too little control. The latent update minimises this trial's prediction error,
so on those trials it correctly attends the target *less* — locally right, globally
anti-adaptive. Lowering the noise should shrink the share of such errors and restore the
signatures.

Why delay
─────────
"Flankers first": the flankers are on from frame 0 and the TARGET's onset is delayed. Does
a later target mean a later response, and do the flankers alone let a congruent trial
commit early? Nothing is compensated for the delay — speed pressure and the RT origin are
unchanged. See FlankerTaskConfig.target_delay and docs/flanker_task.md.

Running it
──────────
    python flanker_sweep.py pretrain     # populate the model cache first
    python flanker_sweep.py              # then the test sessions

On SLURM, `./run_flanker_sweep.sh submit` sizes both arrays from this file and wires the
pretrain -> test dependency.
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

# Redraws the sharpness of the Stage-1 oracle gate every training trial. Pinned here
# because flanker_run_one_network.py pins it too (and to the same value) — it is one of the few
# settings the workbench sets explicitly rather than inheriting.
#
# NOTE, and worth revisiting: the case for jitter was built when `latent_activation` was
# 'softmax', where the oracle gate is the same vector on every training trial (peak 0.405
# at Z_dim=5) and the read-out is only ever calibrated at that one sharpness. With
# `latent_activation = 'none'` the gate is the raw one-hot instead, so jitter is now a
# plain gain knob on a peak of 1.0 rather than a fix for a degenerate softmax. The 20-seed
# factorial that measured jitter (9 of 11 signatures without it, 8 with) predates that
# change and does not describe the current model.
ORACLE_GATE_JITTER = (0.5, 1.5)

# The stimulus noise a variant gets when it does NOT carry its own pretrain_overrides —
# i.e. the delay ladder, whose whole point is to reuse one pretrained model set. The noise
# ladder below overrides it per rung ("Variant Stage-1 overrides last, so they win").
#
# 1.35 is 0.9 x 1.5, the old working point carried across the retiming: evidence
# accumulates over the response window, so SNR grows as sqrt(n_response_steps) and 4 -> 9
# steps is a factor of 1.5. That is a first-order estimate, not a calibration — read the
# real working point off the scorecard once the ladder has run.
# ── What the sweep pins, and what it inherits ─────────────────────────────────
#
# Only `oracle_gate_jitter`, because flanker_run_one_network.py pins that one explicitly too. Every
# other stimulus and model parameter — p_corr_by_distance, arrow_noise_std, bg_noise_std,
# latent_activation, temporal_decay_factor, the Z optimizer settings — is inherited from
# configs.FlankerTaskConfig so that editing configs.py moves the workbench and the sweep
# together. That is the parity rule; see the module docstring, and
# flanker_sweep.check_parity() which asserts it.
PRETRAIN_OVERRIDES = {                      # every variant's Stage 1
    'oracle_gate_jitter': ORACLE_GATE_JITTER,
}
TEST_OVERRIDES = {
    # flanker_run_one_network.py sets this on its test config explicitly; the class default is also 1,
    # but the workbench states it, so the sweep does too.
    'no_of_steps_in_latent_space': 1,
}

# ── Ladder 1: stimulus noise ──────────────────────────────────────────────────
# `arrow_noise_std` is the SD of the per-timestep noise on each arrow, against a signal of
# 1.0. High values mean the target slot's own samples mislead on a large minority of
# trials; low values mean it almost never does.
#
# These are `pretrain_overrides` because the stimulus must match across stages —
# flanker_sweep applies them to the test config too, and gives each rung its own model
# cache so a rung can never accidentally read another's weights.
#
# The rungs are named for their value and bracket the class default (1.35). The retiming
# to 10 timesteps raised SNR by ~1.5x, so these sit higher than the five-timestep ladder
# did; `RUN_NAME` keeps the two worlds in separate directories, and the old rung names
# (noise13/10/09/07/04) still address the 400 five-timestep pickles under factorial_*.
VARIANTS = {
    'noise19':  dict(pretrain_overrides={'arrow_noise_std': 1.9}),
    'noise135': dict(pretrain_overrides={'arrow_noise_std': 1.35}),  # the class default
    'noise10':  dict(pretrain_overrides={'arrow_noise_std': 1.0}),
    'noise06':  dict(pretrain_overrides={'arrow_noise_std': 0.6}),   # near-clean target
}

#: Ordered (variant, noise level) for the figures that plot against noise.
NOISE_LADDER = [('noise19', 1.9), ('noise135', 1.35), ('noise10', 1.0), ('noise06', 0.6)]

# ── Ladder 2: the target-onset delay ──────────────────────────────────────────
#
# "Flankers first": the flankers are on screen from frame 0 and the TARGET's onset is
# delayed. Does a later target mean a later response, and do the flankers alone let a
# congruent trial commit before the target exists?
#
# TEST-stage `overrides`, not `pretrain_overrides`, and that is the whole economy of this
# axis: the stimulus the weights were trained on is unchanged, so `pretrain_tag` resolves
# every rung to the 'shared' model set and all four delays reuse ONE pretrained model per
# seed. Giving them pretrain_overrides would hand each rung its own cache tag and
# quadruple the pretraining bill for identical Stage-1 stimuli.
#
# The 'shared' set trains at the class default `arrow_noise_std`, so the delay ladder runs
# at whatever flanker_run_one_network.py runs at — parity again.
#
# Nothing here touches response_start_timestep or temporal_loss_weights. Speed pressure is
# identical at every rung and RT is measured from trial start, so a delayed response shows
# up as a larger RT rather than being defined away. See FlankerTaskConfig.target_delay.
DELAY_LEVELS = [0, 1, 2, 4]     # 1 is what flanker_run_one_network.py currently runs; 9 response
                                # steps, so even 4 leaves 5 post-onset

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
#
# ad10_ prefix: arrows_duration moved from 5 to 10, so these results are not comparable
# with the 400 pickles under factorial_* and must not land beside them.
RUN_NAME      = 'ad10_delay'

#: The variant every entry point reads when none is named. 'delay1' is the rung
#: flanker_run_one_network.py currently runs, so the workbench figures and a no-argument group run
#: describe the same condition. Without this the two disagreed: the figure script had its
#: own DEFAULT_VARIANT while the analysis script fell through to next(iter(VARIANTS)),
#: which is whichever rung happens to be declared first.
DEFAULT_VARIANT = 'delay1'
EXPORT_ROOT   = './exports/flanker_random/sweeps'
SKIP_EXISTING = True        # resume: skip jobs whose result pickle already exists

# ── Analysis ──────────────────────────────────────────────────────────────────
# Matches flanker_run_one_network.py's extract_trials(rt_threshold=0.5) — parity applies to how
# sessions are READ as well as how they are run, since the threshold sets
# correct_at_decision and therefore every one of the 11 signatures.
#
# It is a post-hoc parameter: changing it re-reads the pickles and needs no re-running.
# Worth re-sweeping at the new trial length — 10 timesteps give the decision variable far
# longer to accumulate, and a pilot showed ~37% of trials undecided at 0.5 against ~10% at
# five timesteps. See docs/rt_threshold_experiment.md.
RT_THRESHOLD = 0.5
