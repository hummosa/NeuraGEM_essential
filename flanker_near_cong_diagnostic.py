"""
flanker_near_cong_diagnostic.py — why is near-congruent WORSE than far-congruent?

The anomaly (single session, run_flanker.py, arrow_noise_std = 0.4):

    accuracy   near-cong  <  far-cong          (backwards: near flankers agree with the
    RT         near-cong  >  far-cong           target and are the ones the model was
                                                taught to weight most)
    P(target)  near-cong spikes highest at t=1, then falls below far-cong

Near flankers carry p_corr_by_distance[1] = 0.75 in Stage 1 and far ones 0.52, so the
weights should read near slots as strong evidence and far slots as almost none. A
congruent near pair therefore adds evidence pointing at the right answer, and every
linear intuition says near-cong should be the *easiest* cell. It is not.

The tests below are ordered so each one removes a candidate explanation:

  T0  reproduce           the four-cell table + the within-trial P(target) curves
  T1  transfer function   flanker amplitude ladder, Z frozen, no LU, noise off.
                          A monotone rising curve = evidence integration works and the
                          anomaly is elsewhere. A curve that turns over = the model
                          saturates / rings on drive it never saw in Stage 1, where at
                          most TWO slots were ever active at once.
  T2  slot count          0 / 1 / 2 congruent flankers at each distance. Stage 1 only
                          ever showed 1 companion; Stage 2 always shows 2.
  T3  matched centre      near vs far on identical centre-slot samples (paired), so no
                          stimulus-RNG difference can carry the effect.
  T4  Z source            same stimuli under oracle centre one-hot Z / uniform Z /
                          session-mean self-Z / the trial's own inherited Z.
                          Separates "the weights do this" from "Z inference does this".
  T5  Z drivers           where each condition pushes Z (delta_z, dL/dZ per slot).
                          Congruent trials are ambiguous about which slot is the target;
                          near-cong may be actively teaching Z to attend the flankers.
  T6  within-trial        is the late decline a drift or a sign reversal? overshoot at
                          t=1 vs flip probability.
  T7  train-vs-test drift what the Stage-1 input distribution looked like next to each
                          Stage-2 cell (total drive, per-slot drive).

Run:
    .venv/bin/python flanker_near_cong_diagnostic.py                      # T0..T9 + A..E
    .venv/bin/python flanker_near_cong_diagnostic.py --tests 0,4,A,D,E    # pick tests
    .venv/bin/python flanker_near_cong_diagnostic.py --seeds 0,1,2,3,4,5  # T13, across seeds
    .venv/bin/python flanker_near_cong_diagnostic.py --seeds 0,1,2 --noise-sweep 0.4,0.7,1.0,1.3
    .venv/bin/python flanker_near_cong_diagnostic.py --lr-sweep           # T14, gate stability
    .venv/bin/python flanker_near_cong_diagnostic.py --figure             # 3-panel summary

Everything prints tables; the figure goes to exports/flanker_random/<run>/diag_*.pdf.
Stage-1 models and the Stage-2 trial dict are cached, so re-running a test is seconds.


═══════════════════════════════════════════════════════════════════════════════
WHAT THESE TESTS FOUND  (seed 42 unless stated; arrow_noise_std = 0.4)
═══════════════════════════════════════════════════════════════════════════════

It is not a bug in the flanker code. It is an artefact of the gate wandering onto slots
that the near display leaves empty, and it is not a distance effect at all.

**1. It is not evidence integration.** The 2x2 is perfectly additive: near flankers cost
   -0.087 accuracy when they AGREE with the target and -0.088 when they DISAGREE
   (interaction +0.001). Evidence would have opposite signs in the two cells. Something
   direction-independent is doing the damage.

**2. At any fixed Z the effect is gone** (T4, replay validated against the logged outputs
   at r = 0.9998). Substituting one Z for every trial — the session mean, the Stage-1
   oracle one-hot, a sharp centre gate — puts every cell at 0.99+ and near-cong minus
   far-cong at +0.002. Shuffling the inherited Z across trials keeps the effect (-0.083).
   So it is carried by the *distribution* of Z, not by the weights and not by which trial
   inherited which state.

**3. The gate is off centre on 37% of trials, and that is where all of it lives** (T12).
   Split by the slot the inherited gate points at:

       gate on   near-cong  far-cong    near-cong - far-cong
       slot 0      0.371      0.948            -0.577
       slot 1      1.000      0.845            +0.155
       slot 2      0.996      0.995            +0.001     <- 63% of trials, no effect
       slot 3      1.000      0.773            +0.227
       slot 4      0.257      0.969            -0.712

   Slots 0 and 4 carry no arrow on a NEAR display; slots 1 and 3 carry none on a FAR one.
   The imposed-Z grid (T10) reproduces this causally: park the gate on slot 0 or 4 and
   near-cong falls to 0.04/0.02 while far-cong stays at 0.99. Park it on slot 1 or 3 and
   near-cong is 1.00 while far-cong only drops to 0.97/0.88. An attended-but-empty OUTER
   slot inverts the answer; an attended-but-empty INNER slot mostly does not.

   Stage 1 never showed a display in which the attended slot was empty — the target slot
   always carries signal — so what the weights do there is untrained extrapolation.

**4. Why the gate goes off centre: congruent trials teach it to defocus** (T16). The LU
   objective (weighted MSE to +-1) parked on each slot:

       near-cong    argmin = UNIFORM  (0.418, vs 0.858 at centre)
       far-cong     argmin = UNIFORM  (0.215, vs 1.890 at centre; slot 0 also beats centre)
       near-incong  argmin = centre   (0.186)
       far-incong   argmin = centre   (0.187)

   Incongruent trials teach the gate correctly. Congruent trials — half the list — push it
   away, and the sequential table shows the consequence: after a far-congruent trial the
   gate sits on an outer slot 25.7% of the time and near-cong minus far-cong is -0.107;
   after a far-INcongruent trial it sits there 10.7% of the time and the gap is -0.012.

**5. Why congruent trials push it away: the model overshoots** (T17). Stage 1 runs a FIXED
   oracle gate of softmax(one-hot) = peak 0.405, at which a Stage-1-style display outputs
   1.048 against its +-1 target — calibrated. Stage-2 Z is unconstrained and is SHARPER
   than 0.405 on 75.9% of trials; at peak 0.65 the output is ~2.1 and at 0.97 it is ~3.3.
   The LU descends MSE to +-1, so on a congruent trial — where every gate already gives the
   right sign — the only thing left to reduce is the output magnitude, and the update does
   that by turning the gate down. The controller is being taught to defocus by exactly the
   trials on which control does not matter.

**6. The RT effect is not slowing, it is non-responses.** RT among trials that actually
   crossed threshold is 0.319 (near-cong) vs 0.323 (far-cong) — indistinguishable. What
   differs is p(decided): 0.923 vs 0.969. Undecided trials are parked at rt =
   arrows_duration by the RT convention, and that is the whole +0.21 RT gap.

**7. It replicates: 6/6 seeds** (T13). near-cong - far-cong accuracy = -0.074 +- 0.011
   (SEM), negative in every seed; the interaction is -0.059 +- 0.021, i.e. near flankers
   hurt CONGRUENT trials more than incongruent ones — the reverse of the human prediction.

**8. It is specific to low stimulus noise** (T15, mean of seeds 0,1,2):

       arrow_noise_std   off-centre   dist_cong   dist_incong   RT gap   RT gap|decided
             0.4           0.336       -0.067       -0.047      +0.258      -0.005
             0.7           0.199       -0.030       -0.113      +0.225      -0.066
             1.0           0.174       -0.011       -0.137      +0.111      -0.058
             1.3           0.210       +0.008       -0.159      -0.009      -0.112

   As noise rises the gate stays on centre, the congruent-cell artefact melts away and the
   genuine flanker effect (near hurts more when incongruent) grows monotonically. Note
   RT|decided is negative at EVERY level: near-congruent trials that respond are never
   slower. run_flanker.py currently runs at 0.4 — the lowest variant in
   flanker_sweep_config.VARIANTS, labelled there "near-clean target". The sweep's own
   default variant is noise10.

WHAT TO DO ABOUT IT
    - Read the near/far contrast at noise >= 1.0, or state that at 0.4 the congruent cell
      is contaminated. The distance effect on INCONGRUENT trials is clean at every level.
    - Report RT next to dec_*, or use the decided-only RT, whenever a distance claim rests
      on RT. The raw RT gap here is entirely a non-response gap.
    - The design confound is real regardless of noise: "near" and "far" do not just move
      the flankers, they change which slots are empty. Filling the unused slots with
      arrow-level noise instead of bg noise (T11, "noise-filled") halves the gap,
      -0.079 -> -0.043, without touching anything else.
    - The overshoot is worth fixing on its own: nothing constrains the inferred gate to the
      sharpness the weights were trained at. Options are a softmax_temp that matches, a
      cap/penalty on Z magnitude, or training Stage 1 with the gate sharpness jittered.

ROUND 2 — WHY THE INVERSION HAPPENS, AND WHICH FIXES WORK
═══════════════════════════════════════════════════════════════════════════════

**A hypothesis that was WRONG.** The first guess was that attention on an empty outer
position produces a near-cancellation — the visible arrows' read-out weights summing to
about zero, so the sign is decided by residuals. T20 measures the output and refutes it.
Seed 42, p_corr [1, .75, .52, .25, .1], gate parked at raw Z = 1.5 x one-hot, 3000 matched
trials; the training target for |output| is 1.0 and the response threshold is 0.5:

    near-congruent display        accuracy   signed out   sd    p(responded)
      gate on 0  (EMPTY here)       0.005      -0.685    0.13      0.98
      gate on 4  (EMPTY here)       0.005      -1.096    0.15      1.00
    far-congruent display
      gate on 1  (EMPTY here)       0.760      +0.079    0.33      0.65
      gate on 3  (EMPTY here)       0.370      -0.377    0.32      0.69

The two cases are the other way round from the prediction. An empty OUTER position gives a
large, confident, inverted response (sd 0.13 over 3000 trials). An empty INNER position is
the near-cancellation case: output near zero, high variance, a third of trials never
responding. Both are failures; only the first costs accuracy, which is why the accuracy
effect falls on near displays and not on far ones.

**What does explain it: the read-out profile is positive at distance 1 and negative
beyond, and the two ends of the array have different numbers of distance-1 neighbours.**
T18 measures S[k, j] = d(output)/d(arrow at j) with attention parked at k, in output units
per input unit. Collapsed by |k - j|, seed 42:

    distance          0       1       2       3       4
    S  (anti tail)  +5.39   +0.72   -1.09   -3.09   -4.55
    S  (chance tail)+5.42   +0.14   -1.46   -1.99   -1.34

Now count neighbours. Position 0 or 4 has exactly ONE of the other four positions at
distance 1; position 1 or 3 has TWO. Summing S over the arrows a display actually shows:

    near display (arrows at 1,2,3), attention on empty 0: distances 1,2,3 -> +1.43 -2.08 -3.13 = -3.78
    near display,                   attention on empty 4: distances 3,2,1 -> -3.50 -1.56 +0.11 = -4.95
    far  display (arrows at 0,2,4), attention on empty 1: distances 1,1,3 -> +1.36 +0.69 -3.25 = -1.20
    far  display,                   attention on empty 3: distances 3,1,1 -> -2.47 +0.26 +1.10 = -1.11

One positive term against two negative ones at the edges; two positive against one in the
middle. That is the asymmetry, and it is combinatorial — a property of a 5-position line,
not of this seed. The sums predict the sign of the measured output in 3 of the 4 cells;
the miss is the near-zero cell, where a linearisation is least reliable.

**The anti-correlated tail was a contributor, not the cause.** p_corr_by_distance[d] below
0.5 means a companion at distance d predicted the OPPOSITE direction during Stage 1, so
the model has reason to learn a negative read-out weight there. Removing it (0.25, 0.10 ->
0.51, 0.50) cut |S| at distance 3 by 36% and at distance 4 by 71%. But S stays clearly
negative at distances 2-4 even when the training correlation is exactly chance, so
something else also produces suppression of non-attended positions — most likely that a
gated architecture implements "ignore position j" subtractively rather than by a zero
weight. That part is measured, not explained.

**Fix comparison.** Three Stage-1 seeds each, full retrain per cell, Stage 2 = 5000 trials.
`cong_eff` is congruent minus incongruent accuracy; `RTcong|dec` is the same contrast on RT
in timesteps, among trials that responded; `off-centre` is the fraction of trials whose
inherited gate peaks away from the target.

    arrow_noise_std = 0.4
      setting            acc   cong_eff  RTcong|dec  undec  off-centre  dist_cong  dist_incong  interaction
      p_corr anti_tail  0.869    0.162       -        -        0.336      -0.067      -0.047      -0.020
      p_corr chance     0.871    0.195     0.361    0.128      0.297      -0.028      -0.069      +0.041
      p_corr graded     0.894    0.170       -        -        0.224      -0.024      -0.053      +0.030
      softmax temp 0.5  0.985    0.025     0.256    0.033      0.011       0.000      -0.025      +0.025
      gate jitter .5-3  0.980    0.041     0.626    0.042      0.000       0.000      -0.047      +0.047
      display comp2     0.819    0.152     0.239    0.107      0.380      +0.010      -0.108      +0.118

    arrow_noise_std = 1.0
      p_corr chance     0.837    0.227     0.428    0.085      0.087      +0.013      -0.138      +0.151
      display comp2     0.833    0.219     0.430    0.092      0.105      +0.023      -0.184      +0.207
      gate jitter .5-3  0.838    0.269     0.568    0.065      0.001      +0.010      -0.213      +0.222
      gate jitter .5-5  0.823    0.314     0.633    0.086      0.000      +0.007      -0.248      +0.255

Reading it:

  - The p_corr edit already in configs.py ([1, .75, .52, .51, .5]) removes about 60% of the
    artefact at noise 0.4 and flips the interaction to the human-predicted sign. At 6 seeds
    the residual is dist_cong -0.033, still negative in 5/6 seeds.
  - Sharpening the trained gate (softmax_temp) or jittering it both drive the off-centre
    rate to ~0 and the artefact to exactly 0. At noise 0.4 they look like they destroy the
    congruency effect, but that is a CEILING artefact — accuracy is 0.98. The RT congruency
    effect among responded trials is the largest in the table for gate jitter (0.626
    timesteps) and the undecided rate falls from 12.8% to 4.2%.
  - At noise 1.0 gate jitter is the best setting tested on every criterion at once:
    off-centre 0.000, dist_cong +0.007, dist_incong -0.248, interaction +0.255, congruency
    effect 0.314 in accuracy and 0.633 timesteps in RT.
  - `comp2` (Stage 1 with two companions instead of one) removes the artefact WITHOUT
    stabilising the gate — its off-centre rate is the highest in the table. It fixes the
    consequence rather than the cause, and the two are therefore complementary.
  - `comp2sym` (two companions always at +-d, i.e. the exact Stage-2 layout) is WORSE than
    the baseline (dist_cong -0.092, interaction -0.063). Matching the test layout is not
    what helps; variety in which positions are occupied is.
  - Six-seed check on the two most promising: p_corr chance dist_cong -0.033, comp2 -0.025,
    comp2 interaction +0.049 vs +0.009. Three-seed numbers overstate comp2; use six.

RECOMMENDATION
  1. Keep the p_corr edit. Nothing below 0.5 unless an anti-predictive companion is
     intended; 0.5 is the value for "uninformative".
  2. Run the distance analyses at arrow_noise_std >= 1.0. At 0.4 the congruent cell is
     contaminated in 5-6 seeds out of 6; at 1.0 the pattern is already human-signed.
  3. Add the Stage-1 gate jitter (JitteredOracleRNN here). It is the only change that
     removes the cause — the gate stops leaving the target — and at noise 1.0 it also
     produces the largest congruency and distance effects of anything tested. It is a
     four-line change to the oracle path and does not touch Stage 2.
  4. Optionally add comp2 on top; untested in combination.


ALSO NOTED (unrelated to the above, but real)
    - run_flanker.py:243 calls `update_config(config)` a second time where it means
      `update_config(test_config)`, so Stage 1 trains at bg_noise_std = 0 while Stage 2
      tests at the class default 0.1. Not the cause here (T11 "bg-zero" changes the gap by
      0.005) but the two stages should match.
    - flanker_analyses.extract_trials computes centre_evidence from obs[:, search_from:],
      i.e. samples 1..4, but predict_first_frame means the model saw samples 0..3 in those
      timesteps. The evidence measure is off by one timestep against what drove the trial.
"""

import argparse
import copy
import os
import pickle

import numpy as np
import torch
import torch.nn as nn

import plot_style
plot_style.set_plot_style()
from plot_style import FLANKER_COLORS as COL, FigSize

from configs import FlankerTaskConfig, FlankerRandomTrialsConfig
from models import RNN_with_latent
from train_and_infer_functions import train_model
from flanker_analyses import (
    extract_trials, report_extraction, sync_gating, mirror_to_model, reset_Z_uniform,
    _interpolated_rt, _activate_latent, _focus_index,
)

SCRATCH = '/tmp/claude-140832107/-oscar-home-ahummos-code-NeuraGEM-essential/59f304d5-582e-460b-9a1c-ba4275bc8d91/scratchpad'
os.makedirs(SCRATCH, exist_ok=True)

CELL_NAMES = {0: 'near-cong', 1: 'near-incong', 2: 'far-cong', 3: 'far-incong'}
CELL_ORDER = [0, 2, 1, 3]           # cong pair, then incong; near before far
NEAR_SLOTS, FAR_SLOTS, CENTRE = [1, 3], [0, 4], 2


# ══════════════════════════════════════════════════════════════════════════════
# Setup — Stage 1 model (cached), Stage 2 session
# ══════════════════════════════════════════════════════════════════════════════

def build_configs(noise=0.4, bg_noise=0.0, seed=42, n_test=5000, match_bg=True,
                  p_corr=None, softmax_temp=None, display=None):
    """Stage-1 and Stage-2 configs, mirroring run_flanker.py.

    `p_corr` and `softmax_temp` are the two knobs the fix tests vary. Both are properties
    of Stage 1 that Stage 2 has to inherit unchanged — p_corr because it is what the
    weights learned from, softmax_temp because it defines the gate the weights were fitted
    under and `mirror_to_model` does not patch it.

    NOTE `match_bg`: run_flanker.py calls `update_config(config)` a second time where it
    means `update_config(test_config)`, so Stage 1 trains with bg_noise_std = 0 while
    Stage 2 tests with the class default 0.1. `match_bg=True` fixes that here; set it
    False to reproduce the script exactly.
    """
    cfg = FlankerTaskConfig(experiment_to_run='default')
    cfg.run_name = 'flanker_diag_pretrain'
    cfg.env_seed = seed
    cfg.arrow_noise_std = noise
    cfg.bg_noise_std = bg_noise

    tcfg = FlankerRandomTrialsConfig(experiment_to_run='default')
    tcfg.run_name = 'flanker_diag_test'
    tcfg.env_seed = seed
    tcfg.set_n_trials(n_test)
    tcfg.no_of_steps_in_latent_space = 1
    tcfg.arrow_noise_std = noise
    tcfg.bg_noise_std = bg_noise if match_bg else tcfg.bg_noise_std

    for c in (cfg, tcfg):
        if p_corr is not None:
            c.p_corr_by_distance = list(p_corr)
        if softmax_temp is not None:
            c.softmax_temp = float(softmax_temp)
    if display is not None:
        name, n_comp, sym = DISPLAY_VARIANTS[display]
        cfg.dataset_name = name           # Stage 1 only; Stage 2 keeps flanker_random
        cfg.n_companions = n_comp
        cfg.companion_symmetric = sym
        cfg.update_export_path()
    return cfg, tcfg


def get_pretrained(cfg, retrain=False):
    """Train Stage 1, or reload a cache that was built with the same stimulus settings."""
    tag = ''.join(f'{v:g}-' for v in cfg.p_corr_by_distance).rstrip('-')
    path = os.path.join(cfg.export_folder, 'models',
                        f'diag_flanker_seed{cfg.env_seed}'
                        f'_n{cfg.arrow_noise_std:g}_bg{cfg.bg_noise_std:g}'
                        f'_t{cfg.n_pretrain_trials}'
                        f'_p{tag}_st{float(getattr(cfg, "softmax_temp", 1.0)):g}'
                        f'_c{int(getattr(cfg, "n_companions", 1))}'
                        f'{"s" if getattr(cfg, "companion_symmetric", False) else ""}.pt')
    if os.path.exists(path) and not retrain:
        model = torch.load(path, weights_only=False)
        print(f'Stage 1 | loaded cache {path}')
        return model, None
    logger, model, cfg, _ = train_model(cfg, seed=cfg.env_seed, save_models=False,
                                        load_models=False, run_test_phase=False)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model, path)
    print(f'Stage 1 | trained and cached -> {path}')
    return model, logger


def run_stage2(model, cfg, tcfg, z_reset_scale=0.2):
    """One Stage-2 session, exactly as run_flanker.py does it."""
    sync_gating(tcfg, cfg)
    mirror_to_model(model, tcfg)
    reset_Z_uniform(model, scale=z_reset_scale, seed=tcfg.env_seed)
    logger, model, tcfg, _ = train_model(tcfg, seed=tcfg.env_seed, save_models=False,
                                         load_models=False, pretrained_model=model,
                                         run_test_phase=False)
    return logger


# ══════════════════════════════════════════════════════════════════════════════
# Probe harness — run the frozen model on stimuli we construct, with a Z we choose
# ══════════════════════════════════════════════════════════════════════════════

def make_stimuli(rng, n, flanker_slots, flanker_sign, cfg,
                 flanker_amp=1.0, centre_amp=1.0, centre_noise=None, true_dir=None,
                 n_slots=5):
    """(obs, true_dir). flanker_sign: +1 congruent, -1 incongruent, 0 = no flankers.

    `centre_noise` (n, ad) lets two conditions share identical centre-slot samples, so a
    near-vs-far contrast is paired on the only slot that carries the correct answer.
    """
    ad = cfg.arrows_duration
    if true_dir is None:
        true_dir = rng.choice([-1.0, 1.0], size=n)
    obs = rng.normal(0.0, cfg.bg_noise_std, size=(n, ad, n_slots)).astype(np.float64)
    if centre_noise is None:
        centre_noise = rng.normal(0.0, cfg.arrow_noise_std, size=(n, ad))
    obs[:, :, CENTRE] = true_dir[:, None] * centre_amp * cfg.signal_strength + centre_noise
    for s in flanker_slots:
        obs[:, :, s] = (flanker_sign * true_dir[:, None] * flanker_amp * cfg.signal_strength
                        + rng.normal(0.0, cfg.arrow_noise_std, size=(n, ad)))
    return obs, true_dir


Z_POOL = None      # set in main(): the session's per-trial raw Z, for 'sampled' probes


def resolve_z(z_spec, n, seed=0):
    """Turn a Z specification into an (n, Z_dim) array of RAW Z values.

    'sampled' draws one real trial's raw Z per probe trial, so the probe inherits the
    session's actual Z *distribution* rather than a summary of it. That matters: raw Z
    has an sd of ~1-1.8 across trials and softmax is nonlinear, so softmax(mean Z) is far
    more focused than the typical trial's gate.
    """
    if isinstance(z_spec, str):
        if z_spec == 'sampled':
            rng = np.random.default_rng(seed)
            return Z_POOL[rng.integers(0, len(Z_POOL), n)]
        raise ValueError(z_spec)
    return z_spec


def forward_probe(model, obs, true_dir, z_raw, cfg, chunk=2000, z_seed=0):
    """Frozen forward pass over constructed stimuli with a Z we specify.

    obs    (n, ad, n_slots) — the display, exactly as the dataset would emit it
    z_raw  (Z,) or (n, Z)   — RAW Z (pre-softmax), broadcast across the trial's timesteps
    returns output_traj (n, ad) — the decision variable, dim 5 of the output
    """
    m = copy.deepcopy(model).to(cfg.device)
    m.eval()
    n, ad, n_slots = obs.shape
    z_raw = np.asarray(resolve_z(z_raw, n, z_seed), dtype=np.float32)
    if z_raw.ndim == 1:
        z_raw = np.tile(z_raw, (n, 1))

    outs = []
    with torch.no_grad():
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            b = hi - lo
            x = np.zeros((b, ad, cfg.input_size), dtype=np.float32)
            x[:, :, :n_slots] = obs[lo:hi]
            x[:, :, -1] = true_dir[lo:hi, None]
            x = torch.tensor(x, device=cfg.device)
            mask = torch.tensor(cfg.input_feed_mask, dtype=x.dtype, device=x.device)
            x = x * mask
            # predict_first_frame: t=0 is a zero frame, so at t the model has seen t samples
            x = torch.cat((torch.zeros_like(x[:, :1, :]), x[:, :-1, :]), dim=1)

            Z = torch.tensor(np.repeat(z_raw[lo:hi, None, :], ad, axis=1), device=cfg.device)
            m.set_Z(Z)
            o, _ = m(x, what_latent='self')
            outs.append(torch.stack(o, dim=1)[:, :, -1].cpu().numpy())
    return np.concatenate(outs, axis=0)


def summarise(output_traj, true_dir, cfg, rt_threshold=0.5):
    """The same measures run_flanker.py reports, from a raw output trajectory."""
    ad = cfg.arrows_duration
    search_from = int(getattr(cfg, 'response_start_timestep', 1))
    rt_interp, rt_int, decided, cross_idx = _interpolated_rt(output_traj, rt_threshold, search_from)
    rows = np.arange(len(true_dir))
    correct_ts = (output_traj * true_dir[:, None] > 0).astype(float)
    return dict(
        acc=float(correct_ts[rows, cross_idx].mean()),          # correct_at_decision
        acc_ts=correct_ts.mean(axis=0),                          # P(target) per timestep
        rt=float(np.mean(rt_interp)),
        rt_dec=float(np.mean(rt_interp[decided])) if decided.any() else np.nan,
        dec=float(decided.mean()),
        out=output_traj.mean(axis=0),
        sgn_out=float(np.mean(output_traj[:, -1] * true_dir)),
        n=len(true_dir),
    )


def table(rows, cols, title):
    print(f'\n{title}')
    head = f'  {"":<22}' + ''.join(f'{c:>10}' for c in cols)
    print(head)
    print('  ' + '-' * (len(head) - 2))
    for name, vals in rows:
        cells = ''.join(f'{v:>10.3f}' if isinstance(v, (int, float, np.floating))
                        else f'{v:>10}' for v in vals)
        print(f'  {name:<22}{cells}')


# ══════════════════════════════════════════════════════════════════════════════
# T0 — reproduce the anomaly from the real Stage-2 session
# ══════════════════════════════════════════════════════════════════════════════

def t0_reproduce(trials, tcfg):
    print('\n' + '=' * 78)
    print('T0  Reproduce — the four cells of the real Stage-2 session')
    print('=' * 78)
    tt = trials['trial_type']
    ad = trials['ad']
    rows_acc, rows_ts = [], []
    for k in CELL_ORDER:
        m = tt == k
        rows_acc.append((CELL_NAMES[k], [
            m.sum(),
            trials['correct_at_decision'][m].mean(),
            trials['correct'][m][:, -1].mean(),
            trials['rt_interp'][m].mean(),
            trials['rt_interp'][m & trials['decided']].mean(),
            trials['decided'][m].mean(),
            np.abs(trials['output_traj'][m]).max(axis=1).mean(),
        ]))
        rows_ts.append((CELL_NAMES[k], trials['correct'][m].mean(axis=0)))
    table(rows_acc, ['n', 'acc@dec', 'acc_final', 'RT', 'RT|dec', 'p(dec)', 'max|out|'],
          'Behaviour by cell')
    table(rows_ts, [f't={t}' for t in range(ad)], 'P(target) by timestep')
    td1 = np.asarray(trials['true_dir'])
    td1 = td1[:, 0] if td1.ndim == 2 else td1
    signed = trials['output_traj'] * td1[:, None]
    table([(CELL_NAMES[k], signed[tt == k].mean(axis=0)) for k in CELL_ORDER],
          [f't={t}' for t in range(ad)], 'Signed output (toward the TRUE direction)')
    table([(CELL_NAMES[k], np.abs(trials['output_traj'][tt == k]).mean(axis=0))
           for k in CELL_ORDER],
          [f't={t}' for t in range(ad)], '|output| — how hard the model commits')
    print('\n  2x2 decomposition of acc@dec (additivity check):')
    a = {k: trials['correct_at_decision'][tt == k].mean() for k in range(4)}
    print(f'    congruency effect   near {a[0] - a[1]:+.4f}   far {a[2] - a[3]:+.4f}')
    print(f'    distance effect     cong {a[0] - a[2]:+.4f}   incong {a[1] - a[3]:+.4f}')
    print(f'    interaction (cong-incong of near-far): {(a[0]-a[2]) - (a[1]-a[3]):+.4f}')

    near_c, far_c = tt == 0, tt == 2
    print(f"\n  near-cong - far-cong :  acc {trials['correct_at_decision'][near_c].mean() - trials['correct_at_decision'][far_c].mean():+.4f}"
          f"   RT {trials['rt_interp'][near_c].mean() - trials['rt_interp'][far_c].mean():+.4f}")
    near_i, far_i = tt == 1, tt == 3
    print(f"  near-inc  - far-inc  :  acc {trials['correct_at_decision'][near_i].mean() - trials['correct_at_decision'][far_i].mean():+.4f}"
          f"   RT {trials['rt_interp'][near_i].mean() - trials['rt_interp'][far_i].mean():+.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# T1 — the transfer function: output vs flanker amplitude, Z frozen
# ══════════════════════════════════════════════════════════════════════════════

def t1_transfer(model, tcfg, z_probe, z_label, n=4000, seed=7, noiseless=False):
    """Sweep signed flanker amplitude from strongly-incongruent to strongly-congruent.

    Positive amp = flankers point WITH the target. If the model integrates evidence, the
    curve rises monotonically. If it turns over, the model is being driven past the total
    input magnitude Stage 1 ever produced (at most two active slots) and the extra
    congruent evidence starts costing accuracy — which is exactly the shape needed to make
    near-cong (two heavily-weighted agreeing flankers) worse than far-cong.
    """
    print('\n' + '=' * 78)
    print(f'T1  Flanker-amplitude transfer function   [Z = {z_label}, '
          f'{"no stimulus noise" if noiseless else "with stimulus noise"}]')
    print('=' * 78)
    cfg = copy.deepcopy(tcfg)
    if noiseless:
        cfg.arrow_noise_std = 0.0
        cfg.bg_noise_std = 0.0
    amps = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    for slots, name in [(NEAR_SLOTS, 'near (1,3)'), (FAR_SLOTS, 'far (0,4)')]:
        rows = []
        for a in amps:
            rng = np.random.default_rng(seed)      # same centre noise at every amplitude
            obs, td = make_stimuli(rng, n, slots, np.sign(a) if a else 0.0, cfg,
                                   flanker_amp=abs(a))
            s = summarise(forward_probe(model, obs, td, z_probe, cfg), td, cfg)
            rows.append((f'amp {a:+.1f}', [s['acc'], s['rt'], s['dec'], s['sgn_out'],
                                           *s['acc_ts']]))
        table(rows, ['acc', 'RT', 'p(dec)', 'out_end'] + [f'P(t={t})' for t in range(cfg.arrows_duration)],
              f'{name} flankers — signed amplitude ladder')


# ══════════════════════════════════════════════════════════════════════════════
# T2 — number of congruent flankers (Stage 1 only ever showed ONE companion)
# ══════════════════════════════════════════════════════════════════════════════

def t2_slot_count(model, tcfg, z_probe, z_label, n=4000, seed=11):
    print('\n' + '=' * 78)
    print(f'T2  How many flankers?  [Z = {z_label}]')
    print('=' * 78)
    cfg = copy.deepcopy(tcfg)
    combos = [([], 0.0, 'target alone'),
              ([1], +1, '1 near cong'), ([1, 3], +1, '2 near cong (= Stage 2)'),
              ([0], +1, '1 far cong'),  ([0, 4], +1, '2 far cong  (= Stage 2)'),
              ([1], -1, '1 near incong'), ([1, 3], -1, '2 near incong'),
              ([0], -1, '1 far incong'),  ([0, 4], -1, '2 far incong'),
              ([1, 2 + 1], +1, '')][:9]
    rows = []
    for slots, sign, name in combos:
        rng = np.random.default_rng(seed)
        obs, td = make_stimuli(rng, n, slots, sign, cfg)
        s = summarise(forward_probe(model, obs, td, z_probe, cfg), td, cfg)
        rows.append((name, [s['acc'], s['rt'], s['dec'], s['sgn_out'], *s['acc_ts']]))
    table(rows, ['acc', 'RT', 'p(dec)', 'out_end'] + [f'P(t={t})' for t in range(cfg.arrows_duration)],
          'Stage-1 (1 companion) vs Stage-2 (2 flankers) displays')


# ══════════════════════════════════════════════════════════════════════════════
# T3 — near vs far on IDENTICAL centre samples (paired)
# ══════════════════════════════════════════════════════════════════════════════

def t3_matched_centre(model, tcfg, z_probe, z_label, n=8000, seed=13):
    print('\n' + '=' * 78)
    print(f'T3  Near vs far with the centre slot held identical  [Z = {z_label}]')
    print('=' * 78)
    cfg = copy.deepcopy(tcfg)
    rng = np.random.default_rng(seed)
    td = rng.choice([-1.0, 1.0], size=n)
    centre_noise = rng.normal(0.0, cfg.arrow_noise_std, size=(n, cfg.arrows_duration))
    rows = []
    keep = {}
    for slots, sign, name in [(NEAR_SLOTS, +1, 'near-cong'), (FAR_SLOTS, +1, 'far-cong'),
                              (NEAR_SLOTS, -1, 'near-incong'), (FAR_SLOTS, -1, 'far-incong')]:
        r = np.random.default_rng(seed + 1)         # same flanker noise stream too
        obs, _ = make_stimuli(r, n, slots, sign, cfg, true_dir=td, centre_noise=centre_noise)
        o = forward_probe(model, obs, td, z_probe, cfg)
        keep[name] = o
        s = summarise(o, td, cfg)
        rows.append((name, [s['acc'], s['rt'], s['dec'], s['sgn_out'], *s['acc_ts']]))
    table(rows, ['acc', 'RT', 'p(dec)', 'out_end'] + [f'P(t={t})' for t in range(cfg.arrows_duration)],
          'paired near/far, identical centre evidence')
    for a, b in [('near-cong', 'far-cong'), ('near-incong', 'far-incong')]:
        sa = summarise(keep[a], td, cfg); sb = summarise(keep[b], td, cfg)
        print(f'  {a} - {b}:  acc {sa["acc"] - sb["acc"]:+.4f}   RT {sa["rt"] - sb["rt"]:+.4f}')
    return keep, td


# ══════════════════════════════════════════════════════════════════════════════
# T5 — where does each condition push Z?
# ══════════════════════════════════════════════════════════════════════════════

def t5_z_drivers(trials, tcfg):
    print('\n' + '=' * 78)
    print('T5  Where each condition pushes Z (per-slot delta_z, and dL/dZ)')
    print('=' * 78)
    tt = trials['trial_type']
    rows_dz, rows_g, rows_z = [], [], []
    for k in CELL_ORDER:
        m = (tt == k) & np.isfinite(trials['delta_z']).all(axis=1)
        rows_dz.append((CELL_NAMES[k], list(trials['delta_z'][m].mean(axis=0)) +
                        [trials['delta_focus'][m].mean()]))
        rows_z.append((CELL_NAMES[k], list(trials['z_act'][tt == k].mean(axis=0)) +
                       [trials['focus'][tt == k].mean()]))
        if trials['z_grad'] is not None:
            rows_g.append((CELL_NAMES[k], list(trials['z_grad'][tt == k].mean(axis=0))))
    slots = [f'slot{d}' for d in range(trials['z_act'].shape[1])]
    table(rows_z, slots + ['focus'], 'z_act (post-update level, activated gate)')
    table(rows_dz, slots + ['dfocus'], 'delta_z — the update the trial produced')
    if rows_g:
        table(rows_g, slots, 'aggregated dL/dZ (raw gradient; the update is -lr * this)')
    print(f"\n  session mean focus {trials['focus'].mean():.4f}   "
          f"mean z_act {np.round(trials['z_act'].mean(axis=0), 4)}")
    print(f"  raw Z mean {np.round(trials['z_raw'].mean(axis=0), 3)}  "
          f"sd across trials {np.round(trials['z_raw'].std(axis=0), 3)}")


# ══════════════════════════════════════════════════════════════════════════════
# T6 — is the within-trial decline a drift or a sign reversal?
# ══════════════════════════════════════════════════════════════════════════════

def t6_within_trial(trials, tcfg):
    print('\n' + '=' * 78)
    print('T6  Within-trial: does the decision variable turn around after t=1?')
    print('=' * 78)
    tt = trials['trial_type']
    td1 = np.asarray(trials['true_dir'])
    td1 = td1[:, 0] if td1.ndim == 2 else td1
    o = trials['output_traj'] * td1[:, None]   # signed toward the truth
    rows = []
    for k in CELL_ORDER:
        m = tt == k
        right_at_1 = o[m, 1] > 0
        flips = (o[m][right_at_1][:, -1] < 0).mean()
        recov = (o[m][~right_at_1][:, -1] > 0).mean()
        rows.append((CELL_NAMES[k], [o[m, 1].mean(), o[m, -1].mean(),
                                     o[m].max(axis=1).mean(), right_at_1.mean(),
                                     flips, recov,
                                     np.abs(o[m]).max(axis=1).mean()]))
    table(rows, ['out t=1', 'out t=end', 'peak', 'P(right@1)', 'P(flip)', 'P(recover)', 'peak|out|'],
          'Sign reversals within the trial')
    print('\n  P(flip)    = right at t=1 and wrong at the end')
    print('  P(recover) = wrong at t=1 and right at the end')


# ══════════════════════════════════════════════════════════════════════════════
# T7 — how far outside the Stage-1 input distribution is each Stage-2 cell?
# ══════════════════════════════════════════════════════════════════════════════

def t7_distribution(model, cfg, tcfg, logger1=None, n=20000, seed=17):
    print('\n' + '=' * 78)
    print('T7  Input drive: Stage-1 training displays vs Stage-2 cells')
    print('=' * 78)
    W = model.input_layer.weight.detach().cpu().numpy()      # (hidden, input_size)
    w_slot = W[:, :cfg.n_slots]

    def drive(obs):
        """||W_in x|| and the projection onto the readout-relevant direction."""
        x = obs.reshape(-1, obs.shape[-1])
        h = x @ w_slot.T
        return np.linalg.norm(h, axis=1).mean(), np.abs(x).sum(axis=1).mean()

    rng = np.random.default_rng(seed)
    rows = []
    # Stage-1: target + ONE companion, companion agrees with p_corr_by_distance[d]
    for d, slot in [(1, CENTRE - 1), (2, 0)]:
        p = cfg.p_corr_by_distance[d]
        agree = rng.random(n) < p
        obs, td = make_stimuli(rng, n, [slot], +1, cfg)
        obs[~agree, :, slot] *= -1
        nrm, l1 = drive(obs)
        rows.append((f'S1  target+companion d={d}', [p, l1, nrm]))
    for slots, sign, name in [(NEAR_SLOTS, +1, 'S2  near-cong'), (FAR_SLOTS, +1, 'S2  far-cong'),
                              (NEAR_SLOTS, -1, 'S2  near-incong'), (FAR_SLOTS, -1, 'S2  far-incong')]:
        obs, td = make_stimuli(rng, n, slots, sign, tcfg)
        nrm, l1 = drive(obs)
        rows.append((name, [np.nan, l1, nrm]))
    table(rows, ['p_corr', 'mean L1(x)', 'mean ||W x||'], 'Input magnitude by display type')
    print('\n  Stage 1 never shows more than two active slots; every Stage-2 display shows three.')


# ══════════════════════════════════════════════════════════════════════════════
# T4 — is it the weights or the inferred Z?
# ══════════════════════════════════════════════════════════════════════════════

def t4_replay(model, trials, tcfg):
    """Re-run the session's own stimuli under different Z, weights untouched.

    First a validation: replaying with the Z each trial actually inherited (the previous
    trial's logged raw Z, which is post-LU and therefore exactly what this trial used)
    must reproduce the logged outputs. If it does, every later substitution is trustworthy.
    """
    print('\n' + '=' * 78)
    print('T4  Replay the same stimuli under a different Z')
    print('=' * 78)
    td = np.asarray(trials['true_dir'])
    td = td[:, 0] if td.ndim == 2 else td
    obs, tt = trials['obs'], trials['trial_type']
    z_in_raw = np.roll(trials['z_raw'], 1, axis=0)
    z_in_raw[0] = trials['z_raw'][0]

    variants = [
        ('inherited (replay)', z_in_raw),
        ('session-mean Z',     np.tile(trials['z_raw'].mean(axis=0), (len(td), 1))),
        ('oracle centre 1-hot', np.tile(np.eye(5)[CENTRE], (len(td), 1))),
        ('uniform Z',          np.zeros((len(td), 5))),
        ('sharp centre (x5)',  np.tile(np.eye(5)[CENTRE] * 5.0, (len(td), 1))),
        ('shuffled inherited',  z_in_raw[np.random.default_rng(0).permutation(len(td))]),
    ]
    for label, Z in variants:
        o = forward_probe(model, obs, td, Z, tcfg)
        rows = []
        for k in CELL_ORDER:
            m = tt == k
            s = summarise(o[m], td[m], tcfg)
            rows.append((CELL_NAMES[k], [s['acc'], s['rt'], s['dec'], s['sgn_out'], *s['acc_ts']]))
        table(rows, ['acc', 'RT', 'p(dec)', 'out_end'] +
              [f'P(t={t})' for t in range(tcfg.arrows_duration)], f'Z = {label}')
        a = {CELL_NAMES[k]: summarise(o[tt == k], td[tt == k], tcfg) for k in CELL_ORDER}
        print(f"  near-cong - far-cong:  acc {a['near-cong']['acc'] - a['far-cong']['acc']:+.4f}"
              f"   RT {a['near-cong']['rt'] - a['far-cong']['rt']:+.4f}")
        if label == 'inherited (replay)':
            r = np.corrcoef(o.ravel(), trials['output_traj'].ravel())[0, 1]
            err = np.abs(o - trials['output_traj']).mean()
            print(f'  VALIDATION vs logged outputs: r = {r:.4f}, mean |diff| = {err:.4f}')


# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--retrain', action='store_true')
    ap.add_argument('--noise', type=float, default=0.4)
    ap.add_argument('--bg', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n-test', type=int, default=5000)
    ap.add_argument('--tests', type=str, default='0,1,2,3,4,5,6,7,8,9,A,B,C,D,E,F,G')
    ap.add_argument('--seeds', type=str, default='')
    ap.add_argument('--lr-sweep', action='store_true')
    ap.add_argument('--jitter-fix', action='store_true')
    ap.add_argument('--jitter-display', type=str, default='')
    ap.add_argument('--fix-sweep', type=str, default='',
                    help='comma-separated names from PCORR_PROFILES and/or SOFTMAX_TEMPS')
    ap.add_argument('--figure', action='store_true')
    ap.add_argument('--noise-sweep', type=str, default='',
                    help='comma-separated arrow_noise_std values; needs --seeds')
    ap.add_argument('--match-bg', type=int, default=0,
                    help='0 = reproduce run_flanker.py exactly (test bg noise 0.1); '
                         '1 = make Stage-2 bg noise match Stage 1')
    args = ap.parse_args()
    want = set(args.tests.split(','))

    cfg, tcfg = build_configs(noise=args.noise, bg_noise=args.bg, seed=args.seed,
                              n_test=args.n_test, match_bg=bool(args.match_bg))
    print(f'arrow_noise_std  train {cfg.arrow_noise_std}  test {tcfg.arrow_noise_std}')
    print(f'bg_noise_std     train {cfg.bg_noise_std}  test {tcfg.bg_noise_std}')
    print(f'p_corr_by_distance {cfg.p_corr_by_distance}   temporal weights '
          f'{[round(w, 3) for w in cfg.temporal_loss_weights]}')

    if args.jitter_fix:
        t21_jitter_fix([int(x) for x in (args.seeds or str(args.seed)).split(',')],
                       args.noise, args.bg, args.n_test, bool(args.match_bg),
                       display=(args.jitter_display or None), retrain=args.retrain)
        return
    if args.fix_sweep:
        names = args.fix_sweep.split(',')
        t19_fix_sweep([int(x) for x in (args.seeds or str(args.seed)).split(',')],
                      args.noise, args.bg, args.n_test, bool(args.match_bg),
                      profiles=[n for n in names if n in PCORR_PROFILES],
                      temps=[n for n in names if n in SOFTMAX_TEMPS],
                      displays=[n for n in names if n in DISPLAY_VARIANTS],
                      retrain=args.retrain)
        return
    if args.noise_sweep:
        t15_noise_sweep([int(x) for x in (args.seeds or str(args.seed)).split(',')],
                        [float(x) for x in args.noise_sweep.split(',')], args.bg,
                        args.n_test, bool(args.match_bg), retrain=args.retrain)
        return
    if args.seeds:
        t13_seeds([int(x) for x in args.seeds.split(',')], args.noise, args.bg,
                  args.n_test, bool(args.match_bg), retrain=args.retrain)
        return
    if args.lr_sweep:
        t14_gate_stability(args.seed, args.noise, args.bg, args.n_test,
                           bool(args.match_bg), retrain=args.retrain)
        return

    model, _ = get_pretrained(cfg, retrain=args.retrain)

    ptag = '-'.join(f'{v:g}' for v in cfg.p_corr_by_distance)
    cache = os.path.join(SCRATCH, f'stage2_n{args.noise:g}_bg{args.bg:g}_s{args.seed}'
                                  f'_t{args.n_test}_mb{args.match_bg}'
                                  f'_p{ptag}_st{float(cfg.softmax_temp):g}.pkl')
    if os.path.exists(cache) and not args.retrain:
        trials = pickle.load(open(cache, 'rb'))
        print(f'Stage 2 | loaded cached trials {cache}')
    else:
        logger_t = run_stage2(model, cfg, tcfg)
        trials = extract_trials(logger_t, tcfg, rt_threshold=0.5)
        pickle.dump(trials, open(cache, 'wb'))
    report_extraction(trials, label='Stage 2 | ')

    global Z_POOL
    Z_POOL = trials['z_raw']
    z_sess = trials['z_raw'].mean(axis=0)
    z_probes = [('sampled', 'sampled per-trial Z (the real distribution)'),
                (z_sess,    'softmax(mean raw Z) — a focused gate')]

    if '0' in want: t0_reproduce(trials, tcfg)
    if '6' in want: t6_within_trial(trials, tcfg)
    if '5' in want: t5_z_drivers(trials, tcfg)
    if '4' in want: t4_replay(model, trials, tcfg)
    for zp, zl in z_probes:
        if '3' in want: t3_matched_centre(model, tcfg, zp, zl)
        if '2' in want: t2_slot_count(model, tcfg, zp, zl)
        if '9' in want: t9_signal_vs_noise(model, tcfg, zp, zl)
        if '8' in want: t8_sensitivity(model, tcfg, zp, zl)
        if '1' in want:
            t1_transfer(model, tcfg, zp, zl, noiseless=True)
            t1_transfer(model, tcfg, zp, zl, noiseless=False)
    if '7' in want: t7_distribution(model, cfg, tcfg)
    if 'D' in want: t16_loss_landscape(model, tcfg)
    if 'E' in want: t17_gate_range(model, cfg, tcfg, trials)
    if 'F' in want: t18_readout_profile(model, cfg, tcfg)
    if 'G' in want: t20_empty_slot_readout(model, tcfg)
    if args.figure: make_figure(model, trials, tcfg)
    if 'A' in want: t10_z_landscape(model, tcfg)
    if 'B' in want:
        for zp, zl in z_probes:
            t11_fill_empty(model, tcfg, zp, zl)
    if 'C' in want: t12_empirical_z(trials)




# ══════════════════════════════════════════════════════════════════════════════
# T8 — effective read-weight on each slot, measured by finite differences
# ══════════════════════════════════════════════════════════════════════════════

def t8_sensitivity(model, tcfg, z_probe, z_label, delta=0.25, n=2000, seed=19):
    """d(signed output)/d(slot value) at several operating points.

    A linear evidence integrator has one number per slot regardless of context. If the
    near-slot sensitivity collapses once the near slots are driven, the model is gain
    limited, and adding congruent near flankers buys drive it cannot use while still
    paying their noise.
    """
    print('\n' + '=' * 78)
    print(f'T8  Per-slot sensitivity d(out)/d(x_slot)   [Z = {z_label}, delta={delta}]')
    print('=' * 78)
    cfg = copy.deepcopy(tcfg)
    cfg.arrow_noise_std, cfg.bg_noise_std = 0.0, 0.0
    ad, ns = cfg.arrows_duration, cfg.n_slots

    def base_obs(flanker_slots, sign):
        rng = np.random.default_rng(seed)
        return make_stimuli(rng, n, flanker_slots, sign, cfg)

    rows = []
    for slots, sign, name in [([], 0.0, 'target alone'),
                              (NEAR_SLOTS, +1, 'near-cong display'),
                              (FAR_SLOTS,  +1, 'far-cong display'),
                              (NEAR_SLOTS, -1, 'near-incong display'),
                              (FAR_SLOTS,  -1, 'far-incong display')]:
        obs, td = base_obs(slots, sign)
        sens = []
        for s in range(ns):
            up, dn = obs.copy(), obs.copy()
            up[:, :, s] += delta
            dn[:, :, s] -= delta
            o_up = forward_probe(model, up, td, z_probe, cfg)[:, -1] * td
            o_dn = forward_probe(model, dn, td, z_probe, cfg)[:, -1] * td
            sens.append(float((o_up - o_dn).mean() / (2 * delta)))
        o0 = forward_probe(model, obs, td, z_probe, cfg)[:, -1] * td
        rows.append((name, sens + [float(o0.mean())]))
    table(rows, [f'slot{d}' for d in range(ns)] + ['out_end'],
          'Sensitivity of the final signed output to each slot (noiseless displays)')


# ══════════════════════════════════════════════════════════════════════════════
# T9 — is the near-flanker cost their SIGNAL or their NOISE?
# ══════════════════════════════════════════════════════════════════════════════

def t9_signal_vs_noise(model, tcfg, z_probe, z_label, n=8000, seed=23):
    """2x2: near slots carry signal (yes/no) x carry noise (yes/no), against target alone.

    If the cost is direction independent it must come from magnitude, not from evidence.
    Turning the near slots' *noise* off while keeping their congruent signal separates
    'their noise is amplified by a high-gain channel' from 'their drive saturates the net'.
    """
    print('\n' + '=' * 78)
    print(f'T9  Near-flanker cost: signal or noise?   [Z = {z_label}]')
    print('=' * 78)
    cfg = copy.deepcopy(tcfg)
    ad, ns = cfg.arrows_duration, cfg.n_slots
    rng0 = np.random.default_rng(seed)
    td = rng0.choice([-1.0, 1.0], size=n)
    centre_noise = rng0.normal(0.0, cfg.arrow_noise_std, size=(n, ad))
    flank_noise = rng0.normal(0.0, cfg.arrow_noise_std, size=(n, ad, ns))

    def build(slots, sign, with_signal, with_noise):
        obs = np.random.default_rng(seed + 1).normal(0.0, cfg.bg_noise_std, size=(n, ad, ns))
        obs[:, :, CENTRE] = td[:, None] * cfg.signal_strength + centre_noise
        for s in slots:
            obs[:, :, s] = 0.0
            if with_signal:
                obs[:, :, s] += sign * td[:, None] * cfg.signal_strength
            if with_noise:
                obs[:, :, s] += flank_noise[:, :, s]
        return obs

    rows = []
    specs = [([], 0, False, False, 'target alone (slots 1,3,0,4 ~ 0)')]
    for slots, tag in [(NEAR_SLOTS, 'near'), (FAR_SLOTS, 'far')]:
        specs += [
            (slots, +1, False, True,  f'{tag}: noise only'),
            (slots, +1, True,  False, f'{tag}: cong signal, no noise'),
            (slots, +1, True,  True,  f'{tag}: cong signal + noise'),
            (slots, -1, True,  False, f'{tag}: incong signal, no noise'),
            (slots, -1, True,  True,  f'{tag}: incong signal + noise'),
        ]
    for slots, sign, sig, noi, name in specs:
        obs = build(slots, sign, sig, noi)
        s = summarise(forward_probe(model, obs, td, z_probe, cfg), td, cfg)
        rows.append((name, [s['acc'], s['rt'], s['dec'], s['sgn_out'], *s['acc_ts']]))
    table(rows, ['acc', 'RT', 'p(dec)', 'out_end'] +
          [f'P(t={t})' for t in range(ad)], 'Factorial: flanker signal x flanker noise')




# ══════════════════════════════════════════════════════════════════════════════
# T10 — the Z landscape: what each condition does when the gate points at slot k
# ══════════════════════════════════════════════════════════════════════════════

def t10_z_landscape(model, tcfg, n=3000, seed=29, sharpness=(1.0, 2.0, 4.0)):
    """Accuracy in each cell with the gate parked on each slot in turn.

    The empirical version of this (binning real trials by argmax(z_in)) is confounded:
    which slot Z sits on depends on the previous trial's type. Here Z is imposed, so the
    grid is causal. Read it as: what does a misdirected gate cost, and does it cost the
    same on near and on far displays?
    """
    print('\n' + '=' * 78)
    print('T10  Z landscape — gate parked on slot k, all four cells')
    print('=' * 78)
    cfg = copy.deepcopy(tcfg)
    rng0 = np.random.default_rng(seed)
    td = rng0.choice([-1.0, 1.0], size=n)
    centre_noise = rng0.normal(0.0, cfg.arrow_noise_std, size=(n, cfg.arrows_duration))
    displays = {}
    for slots, sign, name in [(NEAR_SLOTS, +1, 'near-cong'), (FAR_SLOTS, +1, 'far-cong'),
                              (NEAR_SLOTS, -1, 'near-incong'), (FAR_SLOTS, -1, 'far-incong')]:
        displays[name] = make_stimuli(np.random.default_rng(seed + 1), n, slots, sign, cfg,
                                      true_dir=td, centre_noise=centre_noise)[0]
    for s in sharpness:
        rows = []
        for k in range(cfg.n_slots):
            z = np.eye(cfg.n_slots)[k] * s
            gate = np.exp(z) / np.exp(z).sum()
            accs = [summarise(forward_probe(model, displays[nm], td, z, cfg), td, cfg)['acc']
                    for nm in ('near-cong', 'far-cong', 'near-incong', 'far-incong')]
            rows.append((f'Z on slot{k}  (gate {gate[k]:.2f})',
                         accs + [accs[0] - accs[1], accs[2] - accs[3]]))
        z = np.zeros(cfg.n_slots)
        accs = [summarise(forward_probe(model, displays[nm], td, z, cfg), td, cfg)['acc']
                for nm in ('near-cong', 'far-cong', 'near-incong', 'far-incong')]
        rows.append(('uniform Z', accs + [accs[0] - accs[1], accs[2] - accs[3]]))
        table(rows, ['near-cong', 'far-cong', 'near-inc', 'far-inc', 'nc-fc', 'ni-fi'],
              f'accuracy, raw Z = {s} x one-hot')
    print('\n  Slots 0 and 4 are EMPTY on near displays; slots 1 and 3 are EMPTY on far')
    print('  displays. A gate parked on an empty slot is the thing to look at.')


# ══════════════════════════════════════════════════════════════════════════════
# T11 — is the near/far difference really about which slots are EMPTY?
# ══════════════════════════════════════════════════════════════════════════════

def t11_fill_empty(model, tcfg, z_probe, z_label, n=8000, seed=31):
    """Refill the unused flanker slots and see whether the distance effect survives.

    In this task 'near' and 'far' do not just move the flankers, they change which slots
    are empty: a near display leaves slots 0 and 4 blank, a far display leaves 1 and 3
    blank. If the gate is ever misdirected, an empty attended slot yields whatever the
    weights do with silence — and near and far displays offer different silences.

    Variants
      standard      what the dataset generates now: unused slots get bg_noise_std
      noise-filled  unused slots get FULL arrow noise, zero signal ('neutral arrows'),
                    so every display occupies all five slots and only the signal moves
      bg-zero       unused slots exactly zero (what Stage 1 training actually showed)
      all-active    both pairs carry signal; the *other* pair is neutral-signed (0), i.e.
                    the closest thing to a display matched across distance
    """
    print('\n' + '=' * 78)
    print(f'T11  Refill the empty slots   [Z = {z_label}]')
    print('=' * 78)
    cfg = copy.deepcopy(tcfg)
    ad, ns = cfg.arrows_duration, cfg.n_slots
    rng0 = np.random.default_rng(seed)
    td = rng0.choice([-1.0, 1.0], size=n)
    cn = rng0.normal(0.0, cfg.arrow_noise_std, size=(n, ad))
    fill_noise = rng0.normal(0.0, cfg.arrow_noise_std, size=(n, ad, ns))
    bg = rng0.normal(0.0, 1.0, size=(n, ad, ns))

    def build(slots, sign, mode):
        obs = np.zeros((n, ad, ns))
        if mode == 'standard':
            obs += bg * cfg.bg_noise_std
        elif mode == 'noise-filled':
            obs += fill_noise                      # every slot carries an arrow-level sample
        obs[:, :, CENTRE] = td[:, None] * cfg.signal_strength + cn
        for s in slots:
            obs[:, :, s] = (sign * td[:, None] * cfg.signal_strength
                            + fill_noise[:, :, s])
        return obs

    for mode in ('standard', 'bg-zero', 'noise-filled'):
        rows, accs = [], {}
        for slots, sign, name in [(NEAR_SLOTS, +1, 'near-cong'), (FAR_SLOTS, +1, 'far-cong'),
                                  (NEAR_SLOTS, -1, 'near-incong'), (FAR_SLOTS, -1, 'far-incong')]:
            s = summarise(forward_probe(model, build(slots, sign, mode), td, z_probe, cfg),
                          td, cfg)
            accs[name] = s
            rows.append((name, [s['acc'], s['rt'], s['dec'], s['sgn_out'], *s['acc_ts']]))
        table(rows, ['acc', 'RT', 'p(dec)', 'out_end'] + [f'P(t={t})' for t in range(ad)],
              f'unused slots: {mode}')
        print(f"  distance effect   cong {accs['near-cong']['acc'] - accs['far-cong']['acc']:+.4f}"
              f"   incong {accs['near-incong']['acc'] - accs['far-incong']['acc']:+.4f}"
              f"   |  RT cong {accs['near-cong']['rt'] - accs['far-cong']['rt']:+.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# T12 — the empirical Z x cell table from the real session (selection-confounded,
#       but it is what the session actually did)
# ══════════════════════════════════════════════════════════════════════════════

def t12_empirical_z(trials):
    print('\n' + '=' * 78)
    print('T12  Real session: accuracy by the gate the trial inherited')
    print('=' * 78)
    z_in, tt, acc = trials['z_in'], trials['trial_type'], trials['correct_at_decision']
    ok = np.isfinite(z_in).all(axis=1)
    am = np.full(len(tt), -1)
    am[ok] = z_in[ok].argmax(axis=1)
    print(f'  argmax(z_in) lands on slot: '
          f'{np.round(np.bincount(am[ok], minlength=5) / ok.sum(), 3)}   '
          f'(off-centre on {1 - (am[ok] == CENTRE).mean():.1%} of trials)')
    rows = []
    for s in range(5):
        vals = []
        for k in CELL_ORDER:
            m = ok & (am == s) & (tt == k)
            vals.append(acc[m].mean() if m.sum() else np.nan)
        vals.append(vals[0] - vals[1])
        rows.append((f'argmax slot{s}  (n={int((ok & (am == s)).sum())})', vals))
    table(rows, [CELL_NAMES[k] for k in CELL_ORDER] + ['nc-fc'],
          'accuracy | inherited gate peaks at slot s')
    q = np.quantile(trials['focus_in'][ok], np.linspace(0, 1, 6))
    rows = []
    for i in range(5):
        b = ok & (trials['focus_in'] >= q[i]) & (
            trials['focus_in'] <= q[i + 1] if i == 4 else trials['focus_in'] < q[i + 1])
        vals = [acc[b & (tt == k)].mean() for k in CELL_ORDER]
        vals.append(vals[0] - vals[1])
        rows.append((f'Q{i + 1} focus_in [{q[i]:+.2f},{q[i + 1]:+.2f}]', vals))
    table(rows, [CELL_NAMES[k] for k in CELL_ORDER] + ['nc-fc'],
          'accuracy | inherited focus quintile')




# ══════════════════════════════════════════════════════════════════════════════
# T13 / T14 — does it hold across seeds, and does gate stability control it?
# ══════════════════════════════════════════════════════════════════════════════

def cell_summary(trials):
    """The four-cell numbers plus the two quantities the mechanism predicts."""
    tt, acc = trials['trial_type'], trials['correct_at_decision']
    z_in = trials['z_in']
    ok = np.isfinite(z_in).all(axis=1)
    off = 1.0 - (z_in[ok].argmax(axis=1) == CENTRE).mean()
    a = {k: acc[tt == k].mean() for k in range(4)}
    r = {k: trials['rt_interp'][tt == k].mean() for k in range(4)}
    d = {k: trials['decided'][tt == k].mean() for k in range(4)}
    rt_ce = 0.5 * (r[1] + r[3]) - 0.5 * (r[0] + r[2])     # incong minus cong, timesteps
    dd = {k: trials['decided'][tt == k] for k in range(4)}
    rt_ce_dec = (0.5 * (trials['rt_interp'][(tt == 1) & trials['decided']].mean()
                        + trials['rt_interp'][(tt == 3) & trials['decided']].mean())
                 - 0.5 * (trials['rt_interp'][(tt == 0) & trials['decided']].mean()
                          + trials['rt_interp'][(tt == 2) & trials['decided']].mean()))
    return dict(off_centre=off, acc_all=acc.mean(),
                cong_effect=0.5 * (a[0] + a[2]) - 0.5 * (a[1] + a[3]),
                rt_cong_effect=rt_ce, rt_cong_effect_dec=rt_ce_dec,
                undecided=1.0 - trials['decided'].mean(),
                acc_nc=a[0], acc_fc=a[2], acc_ni=a[1], acc_fi=a[3],
                dist_cong=a[0] - a[2], dist_incong=a[1] - a[3],
                interaction=(a[0] - a[2]) - (a[1] - a[3]),
                rt_dist_cong=r[0] - r[2], dec_nc=d[0], dec_fc=d[2],
                focus_in=np.nanmean(trials['focus_in']))


def t13_seeds(seeds, noise, bg, n_test, match_bg, retrain=False):
    print('\n' + '=' * 78)
    print(f'T13  Across seeds  (arrow_noise_std={noise}, n_test={n_test})')
    print('=' * 78)
    rows = []
    for sd in seeds:
        cfg, tcfg = build_configs(noise=noise, bg_noise=bg, seed=sd, n_test=n_test,
                                  match_bg=match_bg)
        model, _ = get_pretrained(cfg, retrain=retrain)
        logger_t = run_stage2(model, cfg, tcfg)
        tr = extract_trials(logger_t, tcfg, rt_threshold=0.5)
        s = cell_summary(tr)
        rows.append((f'seed {sd}', [s['off_centre'], s['acc_nc'], s['acc_fc'],
                                    s['dist_cong'], s['dist_incong'], s['interaction'],
                                    s['rt_dist_cong']]))
        print(f'  seed {sd}: off-centre {s["off_centre"]:.3f}  '
              f'near-cong {s["acc_nc"]:.3f}  far-cong {s["acc_fc"]:.3f}  '
              f'dist_cong {s["dist_cong"]:+.3f}')
    vals = np.array([r[1] for r in rows])
    rows.append(('MEAN', list(vals.mean(axis=0))))
    rows.append(('SEM', list(vals.std(axis=0, ddof=1) / np.sqrt(len(vals)))))
    table(rows, ['off-centre', 'acc n-cong', 'acc f-cong', 'dist_cong', 'dist_incong',
                 'interaction', 'RT dist_c'],
          'Per-seed distance effect and how often the gate sits off centre')
    dc = vals[:, 3]
    print(f'\n  dist_cong (near-cong - far-cong accuracy): '
          f'{dc.mean():+.4f} +- {dc.std(ddof=1) / np.sqrt(len(dc)):.4f}   '
          f'negative in {(dc < 0).sum()}/{len(dc)} seeds')
    inter = vals[:, 5]
    print(f'  interaction: {inter.mean():+.4f} +- {inter.std(ddof=1) / np.sqrt(len(inter)):.4f}'
          f'   (a real flanker-distance effect would make this clearly non-zero)')
    off = vals[:, 0]
    if len(off) > 2:
        print(f'  corr(off-centre rate, dist_cong) across seeds = '
              f'{np.corrcoef(off, dc)[0, 1]:+.3f}')
    return rows


def t14_gate_stability(seed, noise, bg, n_test, match_bg, lrs=(300., 100., 30., 10., 3.),
                       retrain=False):
    """Turn the latent learning rate down and watch the effect track gate stability.

    If the distance effect is produced by the gate wandering off centre, then anything
    that keeps the gate on the target should shrink it — without touching the stimulus,
    the weights, or the flankers.
    """
    print('\n' + '=' * 78)
    print(f'T14  Latent learning rate -> gate stability -> distance effect  (seed {seed})')
    print('=' * 78)
    cfg, _ = build_configs(noise=noise, bg_noise=bg, seed=seed, n_test=n_test, match_bg=match_bg)
    model0, _ = get_pretrained(cfg, retrain=retrain)
    rows = []
    for lr in lrs:
        _, tcfg = build_configs(noise=noise, bg_noise=bg, seed=seed, n_test=n_test,
                                match_bg=match_bg)
        tcfg.Z_lr = lr
        model = copy.deepcopy(model0)
        logger_t = run_stage2(model, cfg, tcfg)
        s = cell_summary(extract_trials(logger_t, tcfg, rt_threshold=0.5))
        rows.append((f'Z_lr {lr:g}', [s['off_centre'], s['focus_in'], s['acc_nc'], s['acc_fc'],
                                      s['dist_cong'], s['dist_incong'],
                                      s['acc_ni'], s['acc_fi']]))
        print(f'  Z_lr {lr:<6g} off-centre {s["off_centre"]:.3f}  dist_cong {s["dist_cong"]:+.4f}')
    table(rows, ['off-centre', 'focus_in', 'acc n-c', 'acc f-c', 'dist_cong', 'dist_incong',
                 'acc n-i', 'acc f-i'],
          'Gate stability vs the distance effect')




def t15_noise_sweep(seeds, noises, bg, n_test, match_bg, retrain=False):
    """The accuracy gap shrinks with stimulus noise but the RT gap does not — why?

    Under the gate-wandering account the two are carried by different things: accuracy by
    how often the gate is parked on a slot the near display leaves empty, RT (as measured)
    almost entirely by the trials that never cross threshold, since RT among decided trials
    is nearly identical across distance.
    """
    print('\n' + '=' * 78)
    print(f'T15  Distance effect against stimulus noise   seeds={list(seeds)}')
    print('=' * 78)
    rows = []
    for nz in noises:
        per = []
        for sd in seeds:
            cfg, tcfg = build_configs(noise=nz, bg_noise=bg, seed=sd, n_test=n_test,
                                      match_bg=match_bg)
            model, _ = get_pretrained(cfg, retrain=retrain)
            tr = extract_trials(run_stage2(model, cfg, tcfg), tcfg, rt_threshold=0.5)
            s = cell_summary(tr)
            tt = tr['trial_type']
            dec = tr['decided']
            rt_dec_gap = (tr['rt_interp'][(tt == 0) & dec].mean()
                          - tr['rt_interp'][(tt == 2) & dec].mean())
            per.append([s['off_centre'], s['dist_cong'], s['dist_incong'],
                        s['rt_dist_cong'], rt_dec_gap, s['dec_nc'] - s['dec_fc'],
                        s['acc_nc'], s['acc_fc']])
        v = np.array(per).mean(axis=0)
        rows.append((f'noise {nz:g}', list(v)))
        print(f'  noise {nz:<5g} off-centre {v[0]:.3f}  dist_cong {v[1]:+.4f}  '
              f'RT gap {v[3]:+.3f} (decided only {v[4]:+.3f})')
    table(rows, ['off-centre', 'dist_cong', 'dist_incong', 'RT gap', 'RT gap|dec',
                 'p(dec) gap', 'acc n-c', 'acc f-c'],
          'Distance effect vs arrow_noise_std (mean over seeds)')




# ══════════════════════════════════════════════════════════════════════════════
# Figure — symptom, mechanism, mediation
# ══════════════════════════════════════════════════════════════════════════════

def make_figure(model, trials, tcfg, path=None, n=3000, seed=29):
    """Three panels: what it looks like, where it comes from, and the causal grid."""
    import matplotlib.pyplot as plt
    tt = trials['trial_type']
    style = {0: ('near-cong', COL['near_cong'], '-'), 2: ('far-cong', COL['far_cong'], '--'),
             1: ('near-incong', COL['near_incong'], '-'), 3: ('far-incong', COL['far_incong'], '--')}

    fig, axes = plt.subplots(1, 3, figsize=(FigSize.large[0] * 3, FigSize.large[1]))

    ax = axes[0]
    for k in CELL_ORDER:
        lbl, c, ls = style[k]
        ax.plot(trials['correct'][tt == k].mean(axis=0), color=c, linestyle=ls, label=lbl)
    ax.axhline(0.5, color='k', lw=0.5, ls=':', alpha=0.4)
    ax.axvspan(-0.5, tcfg.response_start_timestep - 0.5, alpha=0.08, color='k')
    ax.set_xticks(range(trials['ad']))
    ax.set_xlabel('Timestep within trial'); ax.set_ylabel('P(target)')
    ax.set_title('Symptom: near-cong below far-cong', fontsize=6)
    ax.legend(fontsize=4.5)

    # Panel 2 — the same accuracy, split by where the inherited gate was pointing
    ax = axes[1]
    z_in = trials['z_in']
    ok = np.isfinite(z_in).all(axis=1)
    am = np.where(ok, z_in.argmax(axis=1), -1)
    w = 0.2
    for j, k in enumerate(CELL_ORDER):
        lbl, c, _ = style[k]
        vals = [trials['correct_at_decision'][ok & (am == s) & (tt == k)].mean()
                for s in range(5)]
        ax.bar(np.arange(5) + (j - 1.5) * w, vals, width=w, color=c, label=lbl)
    ax.axhline(0.5, color='k', lw=0.5, ls=':', alpha=0.4)
    ax.set_xticks(range(5))
    ax.set_xticklabels(['0\nfar L', '1\nnear L', '2\ntarget', '3\nnear R', '4\nfar R'], fontsize=4.5)
    ax.set_xlabel('Slot the inherited gate points at')
    ax.set_ylabel('Accuracy')
    ax.set_title('Mechanism: gate on an empty slot', fontsize=6)

    # Panel 3 — causal grid: gate imposed, not inherited
    ax = axes[2]
    cfg = copy.deepcopy(tcfg)
    rng0 = np.random.default_rng(seed)
    td = rng0.choice([-1.0, 1.0], size=n)
    cn = rng0.normal(0.0, cfg.arrow_noise_std, size=(n, cfg.arrows_duration))
    disp = {k: make_stimuli(np.random.default_rng(seed + 1), n,
                            NEAR_SLOTS if k in (0, 1) else FAR_SLOTS,
                            +1 if k in (0, 2) else -1, cfg,
                            true_dir=td, centre_noise=cn)[0] for k in range(4)}
    for j, k in enumerate(CELL_ORDER):
        lbl, c, _ = style[k]
        vals = [summarise(forward_probe(model, disp[k], td, np.eye(5)[s] * 1.5, cfg),
                          td, cfg)['acc'] for s in range(5)]
        ax.bar(np.arange(5) + (j - 1.5) * w, vals, width=w, color=c)
    ax.axhline(0.5, color='k', lw=0.5, ls=':', alpha=0.4)
    ax.set_xticks(range(5))
    ax.set_xticklabels(['0\nfar L', '1\nnear L', '2\ntarget', '3\nnear R', '4\nfar R'], fontsize=4.5)
    ax.set_xlabel('Slot the gate is PARKED on')
    ax.set_title('Causal: same, with Z imposed', fontsize=6)

    fig.tight_layout()
    path = path or (tcfg.export_path + 'diag_near_cong_mechanism.pdf')
    fig.savefig(path, bbox_inches='tight')
    print(f'\nExported: {path}')
    return fig




# ══════════════════════════════════════════════════════════════════════════════
# T16 — the LU objective as a function of where the gate points
# ══════════════════════════════════════════════════════════════════════════════

def t16_loss_landscape(model, tcfg, n=3000, seed=37, sharp=1.5):
    """The exact quantity the latent update descends, per condition, per parked slot.

    The LU minimises THIS trial's masked prediction error. If some condition's minimum
    sits on a slot that carries no arrow, the update is being pulled there on purpose,
    and the next trial inherits a gate parked on an empty slot.
    """
    print('\n' + '=' * 78)
    print(f'T16  LU objective vs where the gate points  (raw Z = {sharp} x one-hot)')
    print('=' * 78)
    cfg = copy.deepcopy(tcfg)
    ad, ns = cfg.arrows_duration, cfg.n_slots
    rng0 = np.random.default_rng(seed)
    td = rng0.choice([-1.0, 1.0], size=n)
    cn = rng0.normal(0.0, cfg.arrow_noise_std, size=(n, ad))
    tw = np.asarray(cfg.temporal_loss_weights, dtype=float)

    rows = []
    for slots, sign, name in [(NEAR_SLOTS, +1, 'near-cong'), (FAR_SLOTS, +1, 'far-cong'),
                              (NEAR_SLOTS, -1, 'near-incong'), (FAR_SLOTS, -1, 'far-incong')]:
        obs, _ = make_stimuli(np.random.default_rng(seed + 1), n, slots, sign, cfg,
                              true_dir=td, centre_noise=cn)
        losses = []
        for s in list(range(ns)) + [None]:
            z = np.zeros(ns) if s is None else np.eye(ns)[s] * sharp
            o = forward_probe(model, obs, td, z, cfg)
            # the LU objective: MSE against the true direction on dim 5, temporally weighted
            losses.append(float((((o - td[:, None]) ** 2) * tw[None, :]).sum(axis=1).mean()))
        best = int(np.argmin(losses))
        rows.append((name, losses + [best if best < ns else -1]))
    table(rows, [f'Z=slot{d}' for d in range(ns)] + ['uniform', 'argmin'],
          'weighted prediction loss (lower = where the LU pushes Z)')
    print('\n  Slots 0/4 carry no arrow on near displays; slots 1/3 carry none on far ones.')
    print('  An argmin on one of those is the update choosing a state that only works')
    print('  because "attend an empty slot" was never trained in Stage 1.')




# ══════════════════════════════════════════════════════════════════════════════
# T17 — the gate the weights were trained with vs the gate Stage 2 actually runs
# ══════════════════════════════════════════════════════════════════════════════

def t17_gate_range(model, cfg, tcfg, trials, n=4000, seed=41):
    """Stage 1's oracle Z is softmax(one-hot), a FIXED and rather soft gate. Stage 2's
    self-inferred Z is unconstrained and mostly runs sharper than that.

    That matters because the model is trained to output +-1. Driven through a sharper gate
    than it ever saw, it overshoots — and the LU, which descends MSE against +-1, can lower
    the loss simply by turning the gate down. On a congruent trial every gate gives the
    right sign, so shrinking the output is the *only* thing left to optimise, and the
    update spends the trial defocusing.
    """
    print('\n' + '=' * 78)
    print('T17  Gate sharpness: what Stage 1 trained vs what Stage 2 runs')
    print('=' * 78)
    ns = cfg.n_slots
    temp = float(getattr(cfg, 'softmax_temp', 1.0))
    train_gate = np.exp(np.eye(ns)[CENTRE] / temp)
    train_gate = train_gate / train_gate.sum()
    print(f'  Stage 1 oracle Z = softmax(one-hot / {temp:g}) -> peak {train_gate[CENTRE]:.3f}, '
          f'others {train_gate[0]:.3f}  (fixed, every trial)')
    peak = trials['z_act'].max(axis=1)
    pct = np.percentile(peak, [5, 25, 50, 75, 95])
    print(f'  Stage 2 self-inferred gate peak percentiles '
          f'5/25/50/75/95: {np.round(pct, 3)}')
    print(f'  fraction of Stage-2 trials sharper than the trained gate: '
          f'{(peak > train_gate[CENTRE]).mean():.1%}')

    cfgp = copy.deepcopy(tcfg)
    tw = np.asarray(cfgp.temporal_loss_weights, dtype=float)
    rng0 = np.random.default_rng(seed)
    td = rng0.choice([-1.0, 1.0], size=n)
    cn = rng0.normal(0.0, cfgp.arrow_noise_std, size=(n, cfgp.arrows_duration))
    displays = {
        'S1-style: target + 1 near': make_stimuli(np.random.default_rng(seed + 1), n, [1], +1,
                                                  cfgp, true_dir=td, centre_noise=cn)[0],
        'near-cong': make_stimuli(np.random.default_rng(seed + 1), n, NEAR_SLOTS, +1, cfgp,
                                  true_dir=td, centre_noise=cn)[0],
        'far-cong':  make_stimuli(np.random.default_rng(seed + 1), n, FAR_SLOTS, +1, cfgp,
                                  true_dir=td, centre_noise=cn)[0],
        'near-incong': make_stimuli(np.random.default_rng(seed + 1), n, NEAR_SLOTS, -1, cfgp,
                                    true_dir=td, centre_noise=cn)[0],
    }
    rows = []
    for s in (0.0, 1.0, 2.0, 3.0, 5.0):
        z = np.eye(ns)[CENTRE] * s
        g = np.exp(z) / np.exp(z).sum()
        vals = []
        for name, obs in displays.items():
            o = forward_probe(model, obs, td, z, cfgp)
            mse = float((((o - td[:, None]) ** 2) * tw[None, :]).sum(axis=1).mean())
            vals += [float((o[:, -1] * td).mean()), mse]
        rows.append((f'gate peak {g[CENTRE]:.2f} (Z={s:g})', vals))
    cols = []
    for name in displays:
        cols += [f'{name[:9]} out', f'{name[:9]} mse']
    table(rows, cols, 'output magnitude and LU loss vs gate sharpness (target is +-1)')
    print('\n  The output the weights were trained to produce is 1.0. Anywhere the "out"')
    print('  column exceeds it, sharpening the gate INCREASES the LU loss, so the update')
    print('  pushes the gate back toward uniform even though the answer is already right.')




# ══════════════════════════════════════════════════════════════════════════════
# T18 — the learned read-out profile: sensitivity to position j when attending k
# ══════════════════════════════════════════════════════════════════════════════

def readout_matrix(model, cfg, delta=0.3, n=1500, seed=53, sharp=1.5, baseline='blank'):
    """S[k, j] = d(final output) / d(arrow value at position j), with attention on k.

    Units: output units per input unit. `baseline='blank'` perturbs an empty display, so S
    is the linearised read-out; `baseline='centre'` perturbs a display that already carries
    the target arrow, so S includes whatever the operating point does to it.
    """
    ns, ad = cfg.n_slots, cfg.arrows_duration
    c = copy.deepcopy(cfg)
    c.arrow_noise_std, c.bg_noise_std = 0.0, 0.0
    rng = np.random.default_rng(seed)
    td = rng.choice([-1.0, 1.0], size=n)
    base = np.zeros((n, ad, ns))
    if baseline == 'centre':
        base[:, :, CENTRE] = td[:, None] * c.signal_strength
    S = np.zeros((ns, ns))
    for k in range(ns):
        z = np.eye(ns)[k] * sharp
        for j in range(ns):
            up, dn = base.copy(), base.copy()
            up[:, :, j] += delta
            dn[:, :, j] -= delta
            o_up = forward_probe(model, up, td, z, c)[:, -1]
            o_dn = forward_probe(model, dn, td, z, c)[:, -1]
            S[k, j] = float((o_up - o_dn).mean() / (2 * delta))
    return S


def t18_readout_profile(model, cfg, tcfg, sharp=1.5):
    """Test the standing hypothesis for the sign inversion.

    p_corr_by_distance[d] is the probability, during Stage 1, that a companion arrow d
    positions from the attended one agreed with it. Values BELOW 0.5 are anti-correlations,
    so the model has reason to learn a NEGATIVE read-out weight at those distances. The
    prediction: S[k, j] should track (p_corr[|k-j|] - 0.5) in sign, and the read-out with
    attention on an empty position is then the sum of the visible arrows' weights — which
    can be negative, and can nearly cancel.
    """
    print('\n' + '=' * 78)
    print(f'T18  Learned read-out profile   (attention parked, raw Z = {sharp} x one-hot)')
    print('=' * 78)
    pc = np.asarray(cfg.p_corr_by_distance, dtype=float)
    print(f'  p_corr_by_distance = {list(pc)}')
    print(f'  as evidence weight (p_corr - 0.5): {np.round(pc - 0.5, 3)}  '
          f'-> negative at distance {[d for d in range(len(pc)) if pc[d] < 0.5]}')

    for baseline in ('blank', 'centre'):
        S = readout_matrix(model, tcfg, sharp=sharp, baseline=baseline)
        table([(f'attend {k}', list(S[k])) for k in range(S.shape[0])],
              [f'arrow at {j}' for j in range(S.shape[1])],
              f'S[k,j] = d(output)/d(arrow_j), baseline = {baseline} display')
        # collapse by distance
        ns = S.shape[0]
        by_d = {d: [] for d in range(ns)}
        for k in range(ns):
            for j in range(ns):
                by_d[abs(k - j)].append(S[k, j])
        table([('mean S by distance', [np.mean(by_d[d]) for d in range(ns)]),
               ('p_corr - 0.5',       list(pc - 0.5))],
              [f'd={d}' for d in range(ns)],
              'read-out weight vs the Stage-1 companion correlation it was trained on')

        # what the read-out predicts when attention sits on an EMPTY position
        rows = []
        for slots, sign, name in [(NEAR_SLOTS, +1, 'near-cong'), (FAR_SLOTS, +1, 'far-cong'),
                                  (NEAR_SLOTS, -1, 'near-incong'), (FAR_SLOTS, -1, 'far-incong')]:
            active = {CENTRE: +1.0}
            for sl in slots:
                active[sl] = float(sign)
            pred = [sum(S[k, j] * v for j, v in active.items()) for k in range(ns)]
            rows.append((name, pred))
        table(rows, [f'attend {k}' for k in range(ns)],
              'linear prediction of the signed output (positive = correct answer)')
        print('  Compare the sign of each entry with the accuracy grid in T10: a negative')
        print('  entry predicts a below-chance cell, a near-zero one predicts an unstable cell.')




# ══════════════════════════════════════════════════════════════════════════════
# T19 — do the Stage-1 companion correlations control the artefact?
# ══════════════════════════════════════════════════════════════════════════════

#: Named p_corr_by_distance profiles. Index d = probability that a companion d positions
#: from the attended one agrees with it, during Stage 1. 0.5 is uninformative; BELOW 0.5
#: is anti-informative, i.e. that companion predicts the opposite direction.
PCORR_PROFILES = {
    'anti_tail':  [1.0, 0.75, 0.52, 0.25, 0.10],   # the value before this session's edit
    'chance_tail': [1.0, 0.75, 0.52, 0.51, 0.50],  # the current value in configs.py
    'flat_tail':  [1.0, 0.75, 0.50, 0.50, 0.50],   # exactly chance beyond distance 1
    'graded':     [1.0, 0.75, 0.62, 0.55, 0.50],   # monotone decay to chance, never below
}

#: Stage-1 oracle gate peak = softmax(one-hot / softmax_temp)[target]. temp 1 -> 0.405.
SOFTMAX_TEMPS = {'temp1.0': 1.0, 'temp0.5': 0.5, 'temp0.35': 0.35}


def _run_one(seed, noise, bg, n_test, match_bg, p_corr, softmax_temp, display=None,
             retrain=False):
    cfg, tcfg = build_configs(noise=noise, bg_noise=bg, seed=seed, n_test=n_test,
                              match_bg=match_bg, p_corr=p_corr, softmax_temp=softmax_temp,
                              display=display)
    model, _ = get_pretrained(cfg, retrain=retrain)
    tr = extract_trials(run_stage2(model, cfg, tcfg), tcfg, rt_threshold=0.5)
    s = cell_summary(tr)
    tt, dec = tr['trial_type'], tr['decided']
    s['rt_dist_cong_dec'] = (tr['rt_interp'][(tt == 0) & dec].mean()
                             - tr['rt_interp'][(tt == 2) & dec].mean())
    s['gate_peak'] = float(np.median(tr['z_act'].max(axis=1)))
    return s, model, cfg, tcfg


CONDITION_COLS = ['acc_all', 'cong_eff', 'RTcong_eff', 'RTcong|dec', 'undecided',
                  'off-centre', 'gate peak', 'dist_cong', 'dist_incong', 'interaction',
                  'RT gap', 'RT gap|dec', 'p(dec) gap']


def _condition_row(s):
    return [s['acc_all'], s['cong_effect'], s['rt_cong_effect'], s['rt_cong_effect_dec'],
            s['undecided'], s['off_centre'], s['gate_peak'],
            s['dist_cong'], s['dist_incong'], s['interaction'],
            s['rt_dist_cong'], s['rt_dist_cong_dec'], s['dec_nc'] - s['dec_fc']]


def t19_fix_sweep(seeds, noise, bg, n_test, match_bg, profiles=None, temps=None,
                  displays=None, retrain=False):
    """Retrain Stage 1 under each candidate setting and re-measure the whole fingerprint.

    A fix has to do two things: shrink the near-congruent artefact (dist_cong toward 0 or
    positive, interaction toward 0 or positive) AND leave the congruency effect intact.
    A setting that flattens everything is not a fix, it is a broken task.
    """
    print('\n' + '=' * 78)
    print(f'T19  Candidate fixes   seeds={list(seeds)}  arrow_noise_std={noise}')
    print('=' * 78)
    jobs = []
    for name in (profiles or []):
        jobs.append((f'p_corr {name}', PCORR_PROFILES[name], 1.0))
    for name in (temps or []):
        jobs.append((f'softmax {name}', None, SOFTMAX_TEMPS[name]))
    for name in (displays or []):
        jobs.append((f'display {name}', None, None, name))
    rows, per_seed = [], {}
    for job in jobs:
        label, pc, tmp = job[0], job[1], job[2]
        disp = job[3] if len(job) > 3 else None
        vals = []
        for sd in seeds:
            s, *_ = _run_one(sd, noise, bg, n_test, match_bg, pc, tmp, display=disp,
                             retrain=retrain)
            vals.append(_condition_row(s))
        v = np.array(vals)
        per_seed[label] = v
        rows.append((label, list(v.mean(axis=0))))
        dc = v[:, CONDITION_COLS.index('dist_cong')]
        print(f'  {label:<22} dist_cong {dc.mean():+.4f} '
              f'(neg in {(dc < 0).sum()}/{len(dc)})  '
              f'cong_eff {v[:, 1].mean():.3f}  RTcong_eff {v[:, 2].mean():+.3f}  '
              f'off-centre {v[:, 5].mean():.3f}')
    table(rows, CONDITION_COLS, f'mean over {len(list(seeds))} seeds')
    print('\n  dist_cong    near-congruent minus far-congruent accuracy (the artefact; want ~0)')
    print('  interaction  distance effect on congruent minus on incongruent (human sign: +)')
    print('  cong_eff     congruent minus incongruent accuracy (must stay large)')
    return per_seed




# ══════════════════════════════════════════════════════════════════════════════
# T20 — magnitude, not just sign, when attention sits on an empty position
# ══════════════════════════════════════════════════════════════════════════════

def t20_empty_slot_readout(model, tcfg, n=3000, seed=59, sharp=1.5, label=''):
    """Accuracy AND signed output when the gate is parked on each position.

    The cancellation account predicts a SMALL output (the visible arrows' read-out weights
    nearly cancel, so the sign is decided by residuals). A systematic-misread account
    predicts a LARGE output of the wrong sign. These are distinguishable, and only the
    second would be reproducible across positions and seeds.

    Output units: the model is trained to emit +-1, and the threshold for a response is
    |output| > 0.5.
    """
    print('\n' + '=' * 78)
    print(f'T20  What the model emits with the gate parked off centre {label}')
    print('=' * 78)
    cfg = copy.deepcopy(tcfg)
    rng0 = np.random.default_rng(seed)
    td = rng0.choice([-1.0, 1.0], size=n)
    cn = rng0.normal(0.0, cfg.arrow_noise_std, size=(n, cfg.arrows_duration))
    disp = {name: make_stimuli(np.random.default_rng(seed + 1), n, slots, sign, cfg,
                               true_dir=td, centre_noise=cn)[0]
            for slots, sign, name in [(NEAR_SLOTS, +1, 'near-cong'), (FAR_SLOTS, +1, 'far-cong'),
                                      (NEAR_SLOTS, -1, 'near-incong'), (FAR_SLOTS, -1, 'far-incong')]}
    for name in ('near-cong', 'far-cong'):
        rows = []
        empty = FAR_SLOTS if name.startswith('near') else NEAR_SLOTS
        for k in range(cfg.n_slots):
            o = forward_probe(model, disp[name], td, np.eye(cfg.n_slots)[k] * sharp, cfg)
            s = summarise(o, td, cfg)
            signed = o[:, -1] * td
            rows.append((f'gate on {k}' + ('  <- EMPTY here' if k in empty else ''),
                         [s['acc'], float(signed.mean()), float(np.abs(o[:, -1]).mean()),
                          float(signed.std()), s['dec']]))
        table(rows, ['accuracy', 'signed out', 'mean |out|', 'sd signed', 'p(dec)'],
              f'{name} display  (training target for |output| is 1.0)')




# ══════════════════════════════════════════════════════════════════════════════
# A third candidate fix: make the Stage-2 display geometry in-distribution
# ══════════════════════════════════════════════════════════════════════════════

from datasets import BaseTaskDataset, DATASET_REGISTRY


class FlankerMultiCompanionDataset(BaseTaskDataset):
    """Stage-1 pretraining with `config.n_companions` companions instead of one.

    The stock Stage-1 display has exactly two arrows (target + one companion); every
    Stage-2 display has three. The model therefore meets a three-arrow display, and the
    total input drive that comes with it, for the first time at test. With two companions
    drawn symmetrically about the target the training geometry matches the test geometry,
    which is the only way to test whether that mismatch matters without also changing the
    correlation structure.

    `config.companion_symmetric` draws the pair at +-d from the target when possible, i.e.
    the actual Stage-2 layout; otherwise companions are independent draws.
    """

    def generate_sequences(self):
        cfg = self.config
        n_comp = int(getattr(cfg, 'n_companions', 1))
        symmetric = bool(getattr(cfg, 'companion_symmetric', False))
        data_seq, context_ids_seq, hlcid_seq = [], [], []

        for block_idx, blk_size in enumerate(self.block_sizes):
            target_slot = block_idx % cfg.n_slots
            n_trials = blk_size // cfg.arrows_duration
            for _ in range(n_trials):
                true_direction = self.rng.choice([-1.0, 1.0])
                others = [s for s in range(cfg.n_slots) if s != target_slot]
                if symmetric and n_comp == 2:
                    pairs = [d for d in range(1, cfg.n_slots)
                             if target_slot - d >= 0 and target_slot + d < cfg.n_slots]
                    if pairs:
                        d = int(self.rng.choice(pairs))
                        comps = [target_slot - d, target_slot + d]
                    else:
                        comps = list(self.rng.choice(others, size=n_comp, replace=False))
                else:
                    comps = list(self.rng.choice(others, size=min(n_comp, len(others)),
                                                 replace=False))
                dirs = {}
                for c in comps:
                    p = cfg.p_corr_by_distance[abs(int(c) - target_slot)]
                    dirs[int(c)] = (true_direction if self.rng.random() < p
                                    else -true_direction)
                congruent = float(all(v == true_direction for v in dirs.values()))
                for _ in range(cfg.arrows_duration):
                    obs = self.rng.normal(0.0, cfg.bg_noise_std, cfg.n_slots).astype(np.float32)
                    obs[target_slot] = float(true_direction * cfg.signal_strength
                                             + self.rng.normal(0, cfg.arrow_noise_std))
                    for c, dr in dirs.items():
                        obs[c] = float(dr * cfg.signal_strength
                                       + self.rng.normal(0, cfg.arrow_noise_std))
                    data_seq.append(np.append(obs, np.float32(true_direction)))
                    context_ids_seq.append(float(target_slot))
                    hlcid_seq.append(congruent)
        return data_seq, context_ids_seq, hlcid_seq


DATASET_REGISTRY['flanker_pretrain_multi'] = FlankerMultiCompanionDataset

#: Stage-1 display variants: (dataset_name, n_companions, companion_symmetric)
DISPLAY_VARIANTS = {
    'comp1':      ('flanker_pretrain', 1, False),          # the stock Stage-1 display
    'comp2':      ('flanker_pretrain_multi', 2, False),     # two companions, free positions
    'comp2sym':   ('flanker_pretrain_multi', 2, True),      # two companions at +-d: the Stage-2 layout
}




# ══════════════════════════════════════════════════════════════════════════════
# A fourth candidate fix: train the read-out across a RANGE of gate sharpness
# ══════════════════════════════════════════════════════════════════════════════

class JitteredOracleRNN(RNN_with_latent):
    """Stage-1 model whose oracle gate sharpness is redrawn every trial.

    Why this exists. Stage 1 supplies the same oracle gate on every trial —
    softmax(one-hot / softmax_temp), peak 0.405 at the default temperature — so the
    read-out is calibrated to emit +-1 at that one gate and nowhere else. Stage 2 infers Z
    freely and runs sharper on ~76% of trials, where the output overshoots to 2-3 against a
    target of 1. The latent update descends squared error, so on a congruent trial (sign
    already correct at every gate) the only remaining gradient is 'shrink the output', and
    it shrinks it by defocusing.

    Lowering softmax_temp removes the overshoot but also removes the flanker influence
    entirely, because a sharp gate reads the centre and nothing else. Jittering the
    sharpness instead keeps the mean gate where it was and teaches the read-out to emit
    +-1 across the range Stage 2 actually visits, which removes the incentive to defocus
    without removing the congruency effect.

    The scale is redrawn once per forward pass, i.e. once per trial, and is shared across
    the trial's timesteps. It applies only to the oracle path (`what_latent='context_ids'`),
    so Stage 2, which uses `what_latent='self'`, is untouched.
    """

    oracle_scale_range = (0.5, 3.0)

    def __init__(self, config):
        super().__init__(config)
        self._oracle_scale = 1.0
        self.jitter_oracle = True

    def forward(self, input, taskID=None, what_latent='self'):
        if self.jitter_oracle and what_latent == 'context_ids':
            lo, hi = self.oracle_scale_range
            self._oracle_scale = float(np.random.uniform(lo, hi))
        else:
            self._oracle_scale = 1.0
        return super().forward(input, taskID, what_latent)

    def _encode_context_ids(self, ids):
        if self.oracle_context_encoding != 'one_hot':
            return super()._encode_context_ids(ids)
        slots = self._context_ids_to_slots(ids)
        latent = torch.zeros(*ids.shape[:-1], self.Z_dim, device=self.device)
        latent.scatter_(dim=-1, index=slots, value=float(self._oracle_scale))
        return self.latent_activation_function(latent)


def get_pretrained_jitter(cfg, scale_range=(0.5, 3.0), retrain=False):
    """Stage 1 with a jittered oracle gate. Cached separately from the stock models."""
    tag = ''.join(f'{v:g}-' for v in cfg.p_corr_by_distance).rstrip('-')
    path = os.path.join(cfg.export_folder, 'models',
                        f'diag_flankerJIT_seed{cfg.env_seed}_n{cfg.arrow_noise_std:g}'
                        f'_bg{cfg.bg_noise_std:g}_t{cfg.n_pretrain_trials}_p{tag}'
                        f'_r{scale_range[0]:g}-{scale_range[1]:g}'
                        f'_c{int(getattr(cfg, "n_companions", 1))}'
                        f'{"s" if getattr(cfg, "companion_symmetric", False) else ""}.pt')
    if os.path.exists(path) and not retrain:
        model = torch.load(path, weights_only=False)
        model.jitter_oracle = False          # the jitter is a training-time device only
        print(f'Stage 1 (jittered oracle) | loaded cache {path}')
        return model
    model = JitteredOracleRNN(cfg).to(cfg.device)
    model.oracle_scale_range = scale_range
    _, model, cfg, _ = train_model(cfg, seed=cfg.env_seed, save_models=False,
                                   load_models=False, pretrained_model=model,
                                   run_test_phase=False)
    model.jitter_oracle = False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model, path)
    print(f'Stage 1 (jittered oracle) | trained and cached -> {path}')
    return model


def t21_jitter_fix(seeds, noise, bg, n_test, match_bg, ranges=((0.5, 3.0), (0.5, 5.0)),
                   display=None, retrain=False):
    print('\n' + '=' * 78)
    print(f'T21  Jittered oracle gate sharpness in Stage 1   seeds={list(seeds)}')
    print('=' * 78)
    rows = []
    for rng_ in ranges:
        vals = []
        for sd in seeds:
            cfg, tcfg = build_configs(noise=noise, bg_noise=bg, seed=sd, n_test=n_test,
                                      match_bg=match_bg, display=display)
            model = get_pretrained_jitter(cfg, scale_range=rng_, retrain=retrain)
            tr = extract_trials(run_stage2(model, cfg, tcfg), tcfg, rt_threshold=0.5)
            s = cell_summary(tr)
            tt, dec = tr['trial_type'], tr['decided']
            s['rt_dist_cong_dec'] = (tr['rt_interp'][(tt == 0) & dec].mean()
                                     - tr['rt_interp'][(tt == 2) & dec].mean())
            s['gate_peak'] = float(np.median(tr['z_act'].max(axis=1)))
            vals.append(_condition_row(s))
        v = np.array(vals)
        i = CONDITION_COLS.index('dist_cong')
        rows.append((f'jitter {rng_[0]:g}-{rng_[1]:g}'
                     + (f' + {display}' if display else ''), list(v.mean(axis=0))))
        print(f'  jitter {rng_[0]:g}-{rng_[1]:g}: dist_cong {v[:, i].mean():+.4f} '
              f'(neg in {(v[:, i] < 0).sum()}/{len(v)})  cong_eff {v[:, 1].mean():.3f}  '
              f'RTcong_eff {v[:, 2].mean():+.3f}  off-centre {v[:, 5].mean():.3f}')
    table(rows, CONDITION_COLS, f'mean over {len(list(seeds))} seeds')


if __name__ == '__main__':
    main()
