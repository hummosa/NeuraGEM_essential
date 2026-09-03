"""
flanker_analyses.py — Reusable trial extraction, selection, and plotting utilities
for the flanker task. Import in run_flanker.py or analysis notebooks.

Three conventions worth knowing before reading any Z result
───────────────────────────────────────────────────────────
1. **One Z per trial.** The LU step aggregates the latent gradient over the trial's
   timesteps and broadcasts a single update back to all of them
   (`models._apply_exponential_increase`), so Z carries no within-trial dynamics.
   `extract_trials` averages over the timestep axis and reports the residual spread
   (`z_within_trial_spread`) as a diagnostic — it should be small, and comes only
   from the random Z initialisation, not from learning.

2. **Z is logged after the LU step.** The value stored on trial t already contains
   trial t's own update. Analyses asking "what control state did this trial
   inherit?" must use `z_in` (the previous trial's Z), not the trial's own value.
   `delta_z` is the update itself.

3. **Z is logged raw; the gate is softmaxed.** The RNN applies
   `softmax(Z / softmax_temp)` across the 5 slots (`models._get_Z_slice`). Because
   the softmax is across slots, a rise in raw dim 2 means nothing if the other four
   rose too. `z_act` is the activated version and is what should be reported;
   `focus` = z_act[center] - mean(z_act[flankers]) is the scalar summary.
"""

import itertools
import sys

import numpy as np

from plot_style import FLANKER_COLORS, flanker_color, outcome_style


# ── Console output ────────────────────────────────────────────────────────────

_SYMBOL_FALLBACKS = {'→': '->', '×': 'x', '±': '+/-',
                     'Δ': 'd', '−': '-'}


def _console_safe(text):
    """Make a label printable on a legacy console.

    Figure labels use arrows and multiplication signs, which render fine in PDFs but
    raise UnicodeEncodeError when printed to a cp1252 terminal. Transliterate the
    symbols we actually use, then fall back to replacement for anything else.
    """
    text = str(text)
    for symbol, plain in _SYMBOL_FALLBACKS.items():
        text = text.replace(symbol, plain)
    encoding = getattr(sys.stdout, 'encoding', None) or 'utf-8'
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        text = text.encode(encoding, errors='replace').decode(encoding, errors='replace')
    return text


# ── Latent helpers ────────────────────────────────────────────────────────────

def _activate_latent(z, config):
    """Apply the model's latent activation to raw Z values. Mirrors
    RNN_with_latent.latent_activation_function so analyses see the gate the RNN saw."""
    activation = getattr(config, 'latent_activation', 'softmax')
    if activation in ('softmax', 'softmax_chunked'):
        temp = float(getattr(config, 'softmax_temp', 1.0)) or 1.0
        x = z / temp
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=-1, keepdims=True)
    if activation == 'sigmoid':
        return 1.0 / (1.0 + np.exp(-z))
    return z


def _focus_index(z_act, center):
    """z_act[center] - mean(z_act[flankers]). Scalar 'attention to target' summary."""
    flankers = [d for d in range(z_act.shape[-1]) if d != center]
    return z_act[..., center] - z_act[..., flankers].mean(axis=-1)


def _lag(arr, k):
    """Shift arr forward by k trials. The first k entries become NaN (no history)."""
    out = np.roll(np.asarray(arr, dtype=float), k)
    out[:k] = np.nan
    return out


# ── Reaction time ─────────────────────────────────────────────────────────────

def _interpolated_rt(output_traj, threshold, search_from):
    """
    Continuous threshold-crossing time; trials that never decide sit at the trial end.

    The decision variable is sampled once per timestep, so an integer RT can take only a
    handful of values. Linearly interpolate the crossing between the two bracketing
    samples:

        rt = (t-1) + (threshold - |o(t-1)|) / (|o(t)| - |o(t-1)|)

    A trial whose decision variable never reaches threshold inside the window did not
    respond, and there is no honest number for when it would have. It is given
    `rt = arrows_duration` — the end of the trial — so those trials pile up in one visible
    bin instead of being dropped or projected to an invented value.

    Two conventions were tried before this one and both distorted the RT effects:

    - **Dropping them (NaN).** Failing to cross is roughly four times more common on
      incongruent trials, so censoring them biased incongruent RT downward and shrank the
      congruency effect.
    - **Extrapolating to a cap of 10.** About half of those trials are not rising at all,
      so they landed on the cap and dominated every mean: the congruency effect on RT read
      1.54 timesteps that way against 0.33 on decided trials alone. RT was mostly reporting
      failure-to-decide.

    Piling them at the trial end keeps every trial in the analysis, keeps the axis inside
    the range the model actually ran, and leaves the failure rate visible as its own
    quantity: `decided` marks them per trial and `undecided_frac` reports the rate.

    Parameters
    ----------
    output_traj : (n_trials, ad) decision variable
    threshold   : |output| level counted as a decision
    search_from : first timestep eligible to be a response (config.response_start_timestep);
                  earlier timesteps carry zero loss weight so their output is unconstrained.

    Returns
    -------
    rt_interp : (n,) float — crossing time; arrows_duration if it never crossed
    rt_int    : (n,) float — integer crossing timestep; ad if never crossed
    decided   : (n,) bool  — crossed inside the observation window
    cross_idx : (n,) int   — crossing timestep, clamped to the last timestep if not decided
    """
    a = np.abs(output_traj)
    n, ad = a.shape
    rows = np.arange(n)

    crossed   = a[:, search_from:] > threshold
    decided   = crossed.any(axis=1)
    cross_idx = crossed.argmax(axis=1) + search_from     # first crossing at/after search_from

    prev_idx  = np.maximum(cross_idx - 1, 0)
    cur_val   = a[rows, cross_idx]
    prev_val  = np.where(cross_idx > 0, a[rows, prev_idx], 0.0)

    denom = cur_val - prev_val
    frac  = np.where(denom > 0, (threshold - prev_val) / np.where(denom > 0, denom, 1.0), 0.0)
    frac  = np.clip(frac, 0.0, 1.0)

    rt_interp = np.where(decided, (cross_idx - 1) + frac, float(ad))
    rt_int    = np.where(decided, cross_idx.astype(float), float(ad))
    cross_idx = np.where(decided, cross_idx, ad - 1)

    return rt_interp, rt_int, decided, cross_idx


# ── Trial extraction ──────────────────────────────────────────────────────────

def extract_trials(logger, config, rt_threshold=0.5, search_from=None):
    """
    Unpack a logger into per-trial aligned arrays.

    Parameters
    ----------
    logger       : Logger object from train_model
    config       : stage config (arrows_duration, input_size, output_size,
                   response_start_timestep, n_slots, softmax_temp)
    rt_threshold : |output| threshold for declaring a decision
    search_from  : first timestep eligible as a response.
                   Defaults to config.response_start_timestep — timesteps before it
                   carry zero loss weight, so a "crossing" there is unconstrained noise.

    Returns
    -------
    dict with keys:

    Behaviour
        correct              (n, ad)  1.0 if sign-correct at each timestep
        output_traj          (n, ad)  raw decision variable (last output dim)
        output_full          (n, ad, output_size)  every output dim; all but the last
                                      carry zero loss weight and are unconstrained
        signed_output        (n, ad)  output sign-normalised by the trial's final decision
        true_dir             (n,)     true direction ±1.0
        is_correct           (n,)     bool — correct at the final timestep
        response_side        (n,)     sign of output at the final timestep
        correct_at_decision  (n,)     bool — correct at the threshold-crossing timestep
        resp_at_decision     (n,)     ±1 — response emitted at the crossing timestep
        rt                   (n,)     integer crossing timestep; ad if never crossed
        rt_interp            (n,)     crossing time, interpolated inside the window;
                                      arrows_duration when the trial never decided
        decided              (n,)     bool — threshold was crossed inside the window
        cross_idx            (n,)     crossing timestep index (clamped when not decided)

    Stimulus evidence (what each slot actually delivered, not what it nominally showed)
        obs               (n, ad, n_slots)  per-slot observation, noise included
        centre_evidence   (n,)  response-window mean of obs[centre] x true_dir. Positive
                                means the centre slot really did point at the target this
                                trial; the noise is large enough that it often does not.
        flanker_evidence  (n,)  same for whichever flanker pair was active this trial

    Condition labels
        trial_type   (n,)  hlcids per trial; 0-3 = near-cong/near-incong/far-cong/far-incong
        context_id   (n,)  context_ids per trial
        trial_idx    (n,)  0..n-1, session position

    Latent
        z_raw        (n, Z)  raw Z, averaged over the trial's timesteps
        z_act        (n, Z)  activated Z — the gate the RNN applied
        z_in         (n, Z)  previous trial's z_act — the state this trial inherited
        delta_z      (n, Z)  z_act - z_in, the update produced by this trial
        focus        (n,)    z_act[center] - mean(z_act[flankers])
        focus_in     (n,)    same for z_in
        delta_focus  (n,)    focus - focus_in
        z_traj       (n, ad, Z)  raw per-timestep Z (kept for backward compatibility)
        z_within_trial_spread   float — mean SD of raw Z across timesteps within a trial

    Optional (None when the logger did not record them)
        pe      (n,)     pre-LU prediction error on the response dim, response window mean
        z_grad  (n, Z)   aggregated dL/dZ that drove this trial's update

    Scalars
        rt_threshold, ad, n_trials, center_slot
    """
    ad = config.arrows_duration
    if search_from is None:
        search_from = int(getattr(config, 'response_start_timestep', 0))

    ii  = np.concatenate(logger.inputs,            axis=0).reshape(-1, config.input_size)
    oi  = np.concatenate(logger.predicted_outputs, axis=0).reshape(-1, config.output_size)
    tt  = np.concatenate(logger.hlcids,            axis=0).reshape(-1)
    cid = np.concatenate(logger.context_ids,       axis=0).reshape(-1)

    z_raw_all = np.concatenate(logger.latent_values, axis=0)
    z_flat    = z_raw_all.reshape(-1, z_raw_all.shape[-1])

    n_ts     = (len(ii) // ad) * ad
    n_trials = n_ts // ad
    rows     = np.arange(n_trials)

    true_dir    = ii[:n_ts, -1].reshape(n_trials, ad)
    output_traj = oi[:n_ts, -1].reshape(n_trials, ad)
    # Every output dim, not just the response one. Dims 0..n_slots-1 carry zero loss
    # weight (config.output_loss_mask), so they are unconstrained rather than meaningless
    # -- plot_trial shows them faintly, as the only honest way to read "what else the
    # read-out was doing" while it produced the decision variable.
    output_full = oi[:n_ts].reshape(n_trials, ad, -1)
    correct     = (true_dir * output_traj > 0).astype(float)
    z_traj      = z_flat[:n_ts].reshape(n_trials, ad, -1)

    rt_interp, rt_int, decided, cross_idx = _interpolated_rt(
        output_traj, rt_threshold, search_from
    )

    # Sign-normalize output by the model's own final decision (sign of output at t=-1).
    # Positive = accumulating toward the eventual decision; negative = against it.
    final_decision_sign = np.sign(output_traj[:, -1:])
    signed_output       = output_traj * final_decision_sign

    # ── Stimulus evidence actually delivered, signed by the true direction ───────
    # arrow_noise_std is 1.3 against a signal of 1.0, so on a sizeable minority of trials
    # the target slot's own samples point the wrong way. Analyses of *why* a trial went
    # wrong need that, not just the nominal condition label.
    n_slots  = int(getattr(config, 'n_slots', 5))
    obs      = ii[:n_ts, :n_slots].reshape(n_trials, ad, n_slots)
    centre_i = n_slots // 2
    near_i   = [centre_i - 1, centre_i + 1]
    far_i    = [0, n_slots - 1]
    signed   = obs[:, search_from:, :] * true_dir[:, search_from:, None]
    centre_evidence = signed[:, :, centre_i].mean(axis=1)
    near_evidence   = signed[:, :, near_i].mean(axis=(1, 2))
    far_evidence    = signed[:, :, far_i].mean(axis=(1, 2))
    # hlcids 0/1 = near cong/incong, 2/3 = far — so even trial types are the near ones.
    is_near_trial    = np.isin(tt[::ad][:n_trials], (0, 1))
    flanker_evidence = np.where(is_near_trial, near_evidence, far_evidence)

    # ── Latent: one value per trial, activated, and aligned to what it inherited ──
    z_within_trial_spread = float(z_traj.std(axis=1).mean())
    z_raw   = z_traj.mean(axis=1)                     # (n, Z)
    z_act   = _activate_latent(z_raw, config)
    center  = int(getattr(config, 'n_slots', z_act.shape[-1])) // 2

    z_in                = np.roll(z_act, 1, axis=0)
    z_in[0]             = np.nan
    delta_z             = z_act - z_in
    focus               = _focus_index(z_act, center)
    focus_in            = _focus_index(z_in,  center)
    delta_focus         = focus - focus_in

    # ── Optional logger channels (plumbing for the regression analyses) ──────────
    pe     = _optional_channel(logger.training_losses_before_latent_optimization,
                               n_ts, ad, config.output_size)
    if pe is not None:
        pe = pe[:, search_from:, -1].mean(axis=1)     # response-window mean on the response dim

    z_grad = _optional_channel(logger.gradients_corrections, n_ts, ad, z_act.shape[-1])
    if z_grad is not None:
        z_grad = z_grad.mean(axis=1)                  # identical across timesteps by construction

    return dict(
        # behaviour
        correct             = correct,
        output_traj         = output_traj,
        output_full         = output_full,
        signed_output       = signed_output,
        true_dir            = true_dir[:, 0],
        is_correct          = correct[:, -1].astype(bool),
        response_side       = np.sign(output_traj[:, -1]),
        correct_at_decision = correct[rows, cross_idx].astype(bool),
        resp_at_decision    = np.sign(output_traj[rows, cross_idx]),
        rt                  = rt_int,
        rt_interp           = rt_interp,
        decided             = decided,
        cross_idx           = cross_idx,
        # stimulus evidence
        obs              = obs,
        centre_evidence  = centre_evidence,
        flanker_evidence = flanker_evidence,
        # condition labels
        trial_type = tt[:n_ts].reshape(n_trials, ad)[:, 0],
        context_id = cid[:n_ts].reshape(n_trials, ad)[:, 0],
        trial_idx  = np.arange(n_trials),
        # latent
        z_raw       = z_raw,
        z_act       = z_act,
        z_in        = z_in,
        delta_z     = delta_z,
        focus       = focus,
        focus_in    = focus_in,
        delta_focus = delta_focus,
        z_traj      = z_traj,
        z_within_trial_spread = z_within_trial_spread,
        # optional
        pe     = pe,
        z_grad = z_grad,
        # scalars
        rt_threshold = rt_threshold,
        search_from  = search_from,
        ad           = ad,
        # Timesteps the target's onset was delayed. Carried here so every downstream
        # figure can mark onset without re-reading a config, and defaulted through
        # getattr so a result pickled before the knob existed still extracts.
        target_delay = int(getattr(config, 'target_delay', 0) or 0),
        n_trials     = n_trials,
        center_slot  = center,
    )


def _optional_channel(entries, n_ts, ad, dim):
    """Reshape an optional logger channel to (n_trials, ad, dim), or None if unusable."""
    if not entries:
        return None
    try:
        flat = np.concatenate(entries, axis=0).reshape(-1, dim)
    except (ValueError, TypeError):
        return None
    if len(flat) < n_ts:
        return None
    return flat[:n_ts].reshape(n_ts // ad, ad, dim)


def report_extraction(trials, label=''):
    """Print the diagnostics that should be eyeballed before trusting any figure."""
    n = trials['n_trials']
    undecided = int((~trials['decided']).sum())
    print(f'{label}trials={n}  undecided={undecided} ({100*undecided/max(n,1):.1f}%)  '
          f'accuracy={trials["correct_at_decision"].mean():.3f}  '
          f'Z within-trial spread={trials["z_within_trial_spread"]:.4f}')
    rt = trials['rt_interp']
    print(f'{label}RT (all trials): mean={rt.mean():.2f}  '
          f'min={rt.min():.2f}  max={rt.max():.2f}  (undecided sit at {trials["ad"]})')
    if trials['decided'].any():
        rt_d = rt[trials['decided']]
        print(f'{label}RT (crossed in window): mean={rt_d.mean():.2f}  '
              f'min={rt_d.min():.2f}  max={rt_d.max():.2f}')


# ── Trial selection ───────────────────────────────────────────────────────────

def select_trials(trials, trial_type=None, is_correct=None, response_side=None):
    """
    Return a boolean mask (n_trials,) for the requested trial subset.

    trial_type   : int, float, or list — match against trials['trial_type'] (hlcids)
    is_correct   : True = correct only, False = errors only (final-timestep definition)
    response_side: +1.0 rightward, -1.0 leftward
    """
    mask = np.ones(trials['n_trials'], dtype=bool)
    if trial_type is not None:
        mask &= np.isin(trials['trial_type'], np.atleast_1d(trial_type))
    if is_correct is not None:
        mask &= (trials['is_correct'] == is_correct)
    if response_side is not None:
        mask &= (trials['response_side'] == response_side)
    return mask


def decode_trial_types(trials, coding='distance'):
    """
    Decode hlcids into congruency and distance booleans.

    coding='distance'   : 0=near-cong, 1=near-incong, 2=far-cong, 3=far-incong
                          (the random-trials stage, and the archived Stage 3)
    coding='congruency' : 1.0=congruent, 0.0=incongruent (legacy Stage 1/2 coding)

    Returns dict(is_cong, is_near); is_near is None under the legacy coding.
    """
    tt = trials['trial_type']
    if coding == 'congruency':
        return dict(is_cong=(tt == 1.0), is_near=None)
    return dict(is_cong=np.isin(tt, [0, 2]), is_near=np.isin(tt, [0, 1]))


def lagged_factors(trials, n_back=2, coding='distance'):
    """
    Current and lagged trial factors, for history analyses and (later) regression.

    Boolean factors are stored as float 0.0/1.0 with NaN in the undefined leading
    slots, so `f['cong_1'] == 1` is safe — NaN never compares True.

    Returns dict with:
        cong, near, correct, rt, side           — current trial
        cong_k, near_k, correct_k, rt_k, side_k — for k in 1..n_back
        target      — the target's direction (+/-1); `side` is the response emitted,
                      so the two come apart exactly on error trials
        flanker     — the flankers' direction (+/-1) = target x congruency sign
        target_k    — lagged target direction, k = 1..n_back
        resp_rep    — this trial's response equals the previous trial's
        target_rep  — this trial's target direction equals the previous trial's
        cong_rep    — this trial's congruency equals the previous trial's
        alternated  — the previous response differed from the one before it
        valid       — trials with a full n_back history
    """
    d    = decode_trial_types(trials, coding=coding)
    cong = d['is_cong'].astype(float)
    near = d['is_near'].astype(float) if d['is_near'] is not None else np.full(trials['n_trials'], np.nan)
    corr = trials['correct_at_decision'].astype(float)
    rt   = trials['rt_interp']
    side = trials['resp_at_decision'].astype(float)
    targ = np.asarray(trials['true_dir'], dtype=float)
    # Congruent flankers point with the target, incongruent ones against it.
    flank = targ * np.where(cong == 1, 1.0, -1.0)

    f = dict(cong=cong, near=near, correct=corr, rt=rt, side=side,
             target=targ, flanker=flank)
    for k in range(1, n_back + 1):
        f[f'cong_{k}']    = _lag(cong, k)
        f[f'near_{k}']    = _lag(near, k)
        f[f'correct_{k}'] = _lag(corr, k)
        f[f'rt_{k}']      = _lag(rt,   k)
        f[f'side_{k}']    = _lag(side, k)
        f[f'target_{k}']  = _lag(targ, k)

    f['resp_rep'] = np.where(np.isnan(f['side_1']), np.nan,
                             (side == f['side_1']).astype(float))
    f['target_rep'] = np.where(np.isnan(f['target_1']), np.nan,
                               (targ == f['target_1']).astype(float))
    f['cong_rep'] = np.where(np.isnan(f['cong_1']), np.nan,
                             (cong == f['cong_1']).astype(float))
    if n_back >= 2:
        f['alternated'] = np.where(np.isnan(f['side_2']), np.nan,
                                   (f['side_1'] != f['side_2']).astype(float))
    else:
        f['alternated'] = np.full(trials['n_trials'], np.nan)

    valid = np.ones(trials['n_trials'], dtype=bool)
    valid[:n_back] = False
    f['valid'] = valid
    return f


def build_history_groups(trials, n_back=2, current_mask=None, congruent_types=(0, 2)):
    """
    Group trials by the congruency of the n_back preceding trials.

    For each trial passing `current_mask`, the history key is a tuple of booleans
    (True = congruent) ordered oldest-to-most-recent: (t-n_back, ..., t-1).
    All 2**n_back keys are returned, including the ones the old analysis skipped —
    (True, False) is CI→ and (False, True) is IC→, and they are not the same thing.

    Parameters
    ----------
    congruent_types : trial_type values treated as congruent.
        Default (0, 2) = near-cong and far-cong.
        For the legacy Stage 1/2 coding where trial_type IS the congruency flag, use (1,).

    Returns
    -------
    dict: history_key (tuple of bool, length n_back) → boolean mask (n_trials,)
    """
    is_cong = np.isin(trials['trial_type'], congruent_types)
    n       = trials['n_trials']
    if current_mask is None:
        current_mask = np.ones(n, dtype=bool)

    groups = {k: [] for k in itertools.product([True, False], repeat=n_back)}
    for t in range(n_back, n):
        if not current_mask[t]:
            continue
        key = tuple(bool(is_cong[t - n_back + i]) for i in range(n_back))
        groups[key].append(t)

    bool_groups = {}
    for k, idx_list in groups.items():
        m = np.zeros(n, dtype=bool)
        if idx_list:
            m[np.array(idx_list)] = True
        bool_groups[k] = m
    return bool_groups


def print_cell_counts(specs, label=''):
    """Print n for each (mask, label, color) spec. Thin cells should be visible, not silent."""
    if label:
        print(_console_safe(label))
    for spec in specs:
        mask, name = spec[0], spec[1]
        print(_console_safe(f'  {name}: n={int(mask.sum())}'))


# ── Per-trial scalar measures ─────────────────────────────────────────────────

def trial_measure(trials, measure):
    """
    Resolve a measure specification to (values, ylabel) with one value per trial.

    measure options
        'accuracy'      — correct at the threshold-crossing timestep
        'accuracy_final'— correct at the final timestep
        'rt'            — integer crossing timestep (ad when censored)
        'rt_interp'     — interpolated crossing time (NaN when censored)
        'focus'         — z_act[center] - mean(z_act[flankers]) for this trial
        'focus_in'      — same for the state inherited from the previous trial
        'delta_focus'   — focus - focus_in, i.e. this trial's update
        'pe'            — pre-LU prediction error
        ('z', d)        — activated Z, dimension d
        ('z_in', d)     — inherited activated Z, dimension d
        ('delta_z', d)  — update to activated Z, dimension d
        ('z_raw', d)    — raw (pre-softmax) Z, dimension d
        int d           — legacy alias for ('z', d)
        ('z_start', d) / ('z_end', d)
                        — legacy aliases. Z has no within-trial dynamics, so both
                          resolve to ('z', d); prefer ('z_in', d) if what you want
                          is the state the trial inherited.
    """
    if isinstance(measure, (int, np.integer)):
        measure = ('z', int(measure))
    if isinstance(measure, tuple) and measure[0] in ('z_start', 'z_end'):
        measure = ('z', measure[1])

    if measure == 'accuracy':
        return trials['correct_at_decision'].astype(float), 'Accuracy at decision'
    if measure == 'accuracy_final':
        return trials['correct'][:, -1], 'Accuracy at final timestep'
    if measure == 'rt':
        return trials['rt'], 'RT (timesteps)'
    if measure == 'rt_interp':
        return trials['rt_interp'], 'RT (timesteps, interpolated)'
    if measure == 'focus':
        return trials['focus'], 'Z focus on target'
    if measure == 'focus_in':
        return trials['focus_in'], 'Z focus inherited (start of trial)'
    if measure == 'delta_focus':
        return trials['delta_focus'], 'Δ Z focus (this trial\'s update)'
    if measure == 'pe':
        if trials['pe'] is None:
            raise ValueError("Prediction error was not logged for this run.")
        return trials['pe'], 'Prediction error (pre-LU)'

    if isinstance(measure, tuple) and len(measure) == 2:
        key, d = measure
        labels = {'z':       f'Z[{d}] (activated)',
                  'z_in':    f'Z[{d}] inherited',
                  'delta_z': f'Δ Z[{d}]',
                  'z_raw':   f'Z[{d}] (raw)'}
        if key in labels:
            return trials['z_act' if key == 'z' else key][:, d], labels[key]

    raise ValueError(f'Unknown measure: {measure!r}')


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _timestep_ticks(ad):
    """Tick positions for a within-trial axis, thinned so a panel-sized figure stays legible.

    A 10-timestep trial puts ten labels on an axis a couple of inches wide, which overprints
    at the figure sizes in plot_style.FigSize. Every second tick keeps the axis readable and
    still lands on the last timestep for an even `ad`.
    """
    step = 1 if ad <= 6 else 2
    return list(range(0, ad, step))


def target_onset_timestep(config):
    """First timestep at which the target arrow can influence the output, or None.

    With predict_first_frame=True the model at timestep t has been fed frames 0..t-1, so a
    target that first appears in frame `target_delay` reaches an output at t = delay + 1.
    Returns None when there is no delay, so callers can skip drawing a marker.
    """
    delay = int(getattr(config, 'target_delay', 0) or 0)
    return delay + 1 if delay > 0 else None


def mark_target_onset(ax, config, label=False):
    """Draw a vertical line where the delayed target first reaches the output.

    Purely a reading aid: no measure is referenced to onset. RT is measured from trial
    start, and the loss weights are unchanged by the delay, so this line marks when the
    evidence arrives — not when the model is allowed to respond.
    """
    t_on = target_onset_timestep(config)
    if t_on is None:
        return
    ax.axvline(t_on, color='k', linewidth=0.8, linestyle='-.', alpha=0.45, zorder=0,
               label='target onset' if label else None)


def _style_trial_ax(ax, ad, response_start, config=None):
    ax.axvspan(-0.5, response_start - 0.5, alpha=0.08, color='k')
    if config is not None:
        mark_target_onset(ax, config)
    ax.set_xticks(_timestep_ticks(ad))
    ax.set_xlabel('Timestep within trial')
    ax.legend(fontsize=5)


def plot_accuracy_by_timestep(ax, trials, specs, config, linestyles=None):
    """
    Plot P(target) per timestep for multiple trial groups.

    P(target) is the fraction of trials whose decision variable currently has the sign of
    the target — `trials['correct']`, evaluated at every timestep. It is **not** accuracy
    in the sense the bar panels use: there is no threshold and no commitment, so a trial
    that crossed threshold three timesteps ago still contributes its current sign, and one
    that never crossed contributes at every timestep. The decision variable keeps
    integrating after the response is emitted and often flips back toward the target
    (~4% of decided congruent trials, ~14% of decided incongruent ones), which this curve
    counts and the emitted response does not.

    The curve's last point therefore equals `measure='accuracy_final'`, not
    `measure='accuracy'` (`correct_at_decision`); expect the two to differ by a few points,
    and by more on incongruent trials.

    specs : list of (mask, label, color) — mask is boolean (n_trials,)
    """
    correct = trials['correct']
    ad      = trials['ad']
    if linestyles is None:
        linestyles = ['-'] * len(specs)
    for (mask, label, color), ls in zip(specs, linestyles):
        if mask.any():
            ax.plot(correct[mask].mean(axis=0), color=color, linestyle=ls,
                    label=f'{label} (n={mask.sum()})')
    ax.axhline(0.5, color='k', linewidth=0.5, linestyle=':', alpha=0.4)
    _style_trial_ax(ax, ad, config.response_start_timestep, config)
    ax.set_ylabel('P(target)')
    ax.set_ylim(0.3, 1.05)


def plot_rt(ax, trials, specs, config, interpolate=False, fit_gaussian=False,
            linestyles=None, bin_width=0.25):
    """
    RT for each spec, with the trials that never decided kept as their own `und.` category.

    interpolate=False (default)
        Empirical PMF over the integer crossing timestep, markers joined by lines, plus a
        final `und.` point carrying the trials that never crossed. Nothing is smoothed or
        interpolated: each point is the proportion of that condition's trials that crossed
        at exactly that timestep. With only four eligible timesteps this is the honest
        resolution, and the undecided pile reads as the separate outcome it is.

    interpolate=True
        Density of the interpolated crossing time over the trials that *did* decide,
        scaled so its area is the decided proportion, plus the same `und.` marker at the
        right. Sub-timestep structure becomes visible at the cost of implying more
        precision than four samples per trial support.

    Either way the undecided trials are shown rather than folded into the last real bin —
    that pile is a condition effect in its own right (roughly 5% of congruent trials
    against 16% of incongruent), and hiding it inside the RT axis makes a non-response
    look like a slow response.

    specs : list of (mask, label, color) — mask is boolean (n_trials,)
    fit_gaussian : overlay a Gaussian fit on the decided part (PMF mode only)
    """
    ad, rt_threshold = trials['ad'], trials['rt_threshold']
    decided_all = trials['decided']
    if linestyles is None:
        linestyles = ['-'] * len(specs)

    # The undecided marker sits one slot right of the axis proper, in both modes.
    und_x = ad + (0.5 if interpolate else 0.0)

    for (mask, label, color), ls in zip(specs, linestyles):
        if not mask.any():
            continue
        n_total = int(mask.sum())
        dec_m   = decided_all[mask]
        p_und   = float((~dec_m).mean())

        if interpolate:
            vals = trials['rt_interp'][mask][dec_m]
            lo   = max(int(trials['search_from']) - 1, 0)
            bins = np.arange(lo, float(ad) + bin_width, bin_width)
            ctrs = 0.5 * (bins[:-1] + bins[1:])
            if len(vals):
                dens, _ = np.histogram(vals, bins=bins, density=True)
                ax.plot(ctrs, dens * (1.0 - p_und), color=color, linestyle=ls,
                        linewidth=0.9, label=f'{label} (n={n_total})')
            ax.plot([und_x], [p_und], marker='o', markersize=4, color=color, zorder=3)
            ax.set_ylabel(f'Density  [threshold = {rt_threshold}]')
        else:
            rt_m = trials['rt'][mask]
            pmf  = np.array([(rt_m == t).sum() / n_total for t in range(ad)] + [p_und])
            ax.plot(list(range(ad)) + [und_x], pmf, marker='o', markersize=4, color=color,
                    linestyle=ls, linewidth=0.8, label=f'{label} (n={n_total})', zorder=3)
            if fit_gaussian and dec_m.sum() >= 3:
                try:
                    from scipy.stats import norm
                    mu, sigma = norm.fit(rt_m[dec_m])
                    x_fine = np.linspace(-0.5, ad - 0.5, 300)
                    ax.plot(x_fine, norm.pdf(x_fine, mu, sigma) * (1.0 - p_und), color=color,
                            linestyle='--', linewidth=1.2, alpha=0.7, zorder=2)
                except Exception:
                    pass
            ax.set_ylabel(f'P(RT = t)  [threshold = {rt_threshold}]')

    ticks = _timestep_ticks(ad)
    ax.set_xticks(ticks + [und_x])
    ax.set_xticklabels([str(t) for t in ticks] + ['und.'], fontsize=6)
    ax.set_xlabel('Timestep within trial')
    ax.axvspan(-0.5, config.response_start_timestep - 0.5, alpha=0.08, color='k')
    mark_target_onset(ax, config)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=5)


def plot_rt_distribution(ax, trials, specs, config, fit_gaussian=True, linestyles=None,
                         undecided='extra_bin'):
    """Deprecated name for `plot_rt(..., interpolate=False)`; `undecided` is ignored."""
    return plot_rt(ax, trials, specs, config, interpolate=False,
                   fit_gaussian=fit_gaussian, linestyles=linestyles)


def plot_rt_continuous(ax, trials, specs, config, bin_width=0.25, linestyles=None):
    """Deprecated name for `plot_rt(..., interpolate=True)`."""
    return plot_rt(ax, trials, specs, config, interpolate=True,
                   linestyles=linestyles, bin_width=bin_width)


def plot_z_by_timestep(ax, trials, specs, z_dim, config, linestyles=None):
    """
    Plot mean raw Z for dimension `z_dim` across timesteps within a trial.

    Kept for the archived blocked-stage script. Z has no within-trial dynamics
    (one aggregated update is broadcast to every timestep), so this line is flat by
    construction — use plot_scalar_bars with ('z_in', d) / ('delta_z', d) instead.

    specs : list of (mask, label, color) — mask is boolean (n_trials,)
    """
    z_traj = trials['z_traj']  # (n_trials, ad, Z_dim)
    ad     = trials['ad']
    if linestyles is None:
        linestyles = ['-'] * len(specs)
    for (mask, label, color), ls in zip(specs, linestyles):
        if mask.any():
            ax.plot(z_traj[mask, :, z_dim].mean(axis=0), color=color,
                    linestyle=ls, label=label)
    _style_trial_ax(ax, ad, config.response_start_timestep)
    ax.set_ylabel(f'Z dim {z_dim}')


def plot_scalar_bars(ax, trials, specs, measure, group_spacing=None, baseline=None,
                     hollow=None):
    """
    Bar chart with mean ± SEM for a scalar-per-trial measure.

    NaN-safe: censored RTs and undefined history entries are dropped per group, and
    the surviving n is what appears in the tick label.

    Parameters
    ----------
    specs   : list of (mask, label, color) — mask is boolean (n_trials,)
    measure : see trial_measure() for the full list
    group_spacing : list of int, optional
                    Extra gap (in bar-widths) inserted before the bar at each listed index.
    baseline : float, optional — horizontal reference line (e.g. 0.5 for accuracy)
    hollow  : list of bool, optional — one per spec. True draws the bar as an outline
              rather than a filled block. This is the house convention for an *error*
              cell (`plot_style.outcome_style`): hue stays on congruency and shade on
              distance, so outcome rides on fill and costs no colour. The bar-chart
              counterpart of the dashed line used for errors in the time-course panels.

    Returns
    -------
    x_positions : array — x positions used, for further annotation.
    """
    values_all, ylabel = trial_measure(trials, measure)

    hollow_all = [False] * len(specs) if hollow is None else list(hollow)

    means, sems, labels, colors, is_hollow, valid_idx = [], [], [], [], [], []
    for i, spec in enumerate(specs):
        mask, label, color = spec[0], spec[1], spec[2]
        v = values_all[mask]
        v = v[~np.isnan(v)]
        if len(v) == 0:
            continue
        means.append(v.mean())
        sems.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
        labels.append(f'{label}\n(n={len(v)})')
        colors.append(color)
        is_hollow.append(bool(hollow_all[i]))
        valid_idx.append(i)

    if not means:
        return np.array([])

    # Build x positions with optional extra gaps between groups
    gaps = set(group_spacing or [])
    x, pos = [], 0.0
    for k, orig_i in enumerate(valid_idx):
        if k > 0 and orig_i in gaps:
            pos += 0.7   # extra gap
        x.append(pos)
        pos += 1.0
    x = np.array(x)

    for xi, mean, color, hol in zip(x, means, colors, is_hollow):
        style = outcome_style(not hol, kind='bar', color=color)
        if not hol:
            style['alpha'] = 0.75
        ax.bar(xi, mean, width=0.6, zorder=2, **style)
    ax.errorbar(x, means, yerr=sems, fmt='none', color='k',
                capsize=3, linewidth=1, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5, rotation=30, ha='right')
    ax.set_ylabel(ylabel)

    # Zoom to the data rather than a fixed floor, so small differences stay visible
    # without hiding bars that fall below a hard-coded bound.
    means_a, sems_a = np.array(means), np.array(sems)
    lo = (means_a - 2 * sems_a).min()
    hi = (means_a + 2 * sems_a).max()
    pad = 0.15 * (hi - lo) if hi > lo else (abs(hi) * 0.1 + 1e-6)
    if baseline is not None:
        ax.axhline(baseline, color='k', linewidth=0.5, linestyle=':', alpha=0.4)
        lo, hi = min(lo, baseline), max(hi, baseline)
    ax.set_ylim(lo - pad, hi + pad)

    ax.grid(axis='y', linewidth=0.4, alpha=0.4, zorder=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return x


# ── Single-trial visualisation ────────────────────────────────────────────────
#
# What one trial looked like: the five slots' noisy observations, the gate the model
# applied to them, and the decision variable that came out.
#
# This used to require running under a debugger and evaluating a script against the
# locals of `predictive_learning`. It does not: `_log_batch` logs `inputs`, the raw
# UNMASKED batch tensor, not the masked/shifted `model_inputs` the RNN is fed, so the
# logger already holds both the arrow observations (dims 0..n_slots-1) and the hidden
# true direction (dim -1). `predicted_outputs` holds every output dim. Everything below
# is read from `extract_trials(logger, config)` — no live model, no debugger.
#
# Two alignment facts the panels depend on:
#
#   * `predict_first_frame=True` means the RNN at timestep t is fed the stimulus from
#     t-1 (t=0 gets a zero frame). The stimulus rows are drawn at the timestep the
#     stimulus was *presented*; the output row at the timestep it was emitted. So the
#     output at t responds to the arrows drawn at t-1 and earlier, never at t.
#   * `update_latent_before_weights=False` means the logged outputs were produced under
#     the Z the trial INHERITED, and the logged Z already contains this trial's own
#     update. The gate column therefore shows `z_in`, not `z_act` — see convention 2 in
#     this module's docstring.

_SLOT_LABELS = ('Far 1', 'Near 1', 'Center', 'Near 2', 'Far 2')

_TRIAL_TYPE_NAMES = {0: 'near-congruent',  1: 'near-incongruent',
                     2: 'far-congruent',   3: 'far-incongruent'}


def _slot_label(slot, n_slots):
    """Spatial name for a slot index, for any odd n_slots."""
    if n_slots == len(_SLOT_LABELS):
        return _SLOT_LABELS[slot]
    centre = n_slots // 2
    if slot == centre:
        return 'Center'
    return f'{"Far" if abs(slot - centre) > 1 else "Near"} {1 if slot < centre else 2}'


def trial_slot_roles(trials, config, trial):
    """
    Which slots carried an arrow on `trial`, and what each one was.

    Roles are a property of the dataset, so they are read from the labels rather than
    guessed, with one exception. In the random-trials stage the target is always centre
    and `trial_type` (hlcids) names the flanker pair. In the pretraining stage
    `context_id` is the target slot, but the companion slot is drawn per trial and never
    logged — it is recovered as the non-target slot whose observations carry a signal.
    That recovery picks the non-target slot with the largest |mean| over the trial's
    timesteps, and it is the only guess in the figure. It is exact whenever
    `bg_noise_std = 0` (what run_flanker.py and the sweep both run), since an empty slot
    is then identically zero. At the class default `bg_noise_std = 0.1` it is right on
    99.5% of trials at `arrow_noise_std = 0.9` and 98.1% at 1.3 (200k simulated trials) —
    it fails only when a companion's own noise happens to cancel its signal, which is a
    trial worth looking at twice anyway.

    Returns dict(target, flankers, empty, congruent, distance, name).
    """
    obs      = trials['obs'][trial]                    # (ad, n_slots)
    n_slots  = obs.shape[-1]
    tt       = float(trials['trial_type'][trial])
    # Trial types 0-3 (near/far x cong/incong) identify the random-trials stage; the
    # pretraining stage puts the congruency flag (0/1) here instead.
    random_stage = (getattr(config, 'dataset_name', '') == 'flanker_random'
                    or tt in (2.0, 3.0))

    if random_stage:
        target    = n_slots // 2
        near      = tt in (0.0, 1.0)
        flankers  = (target - 1, target + 1) if near else (0, n_slots - 1)
        congruent = tt in (0.0, 2.0)
        name      = _TRIAL_TYPE_NAMES.get(int(tt), f'type {tt:g}')
    else:
        target    = int(round(float(trials['context_id'][trial])))
        strength  = np.abs(obs.mean(axis=0))
        others    = [s for s in range(n_slots) if s != target]
        flankers  = (max(others, key=lambda s: strength[s]),)
        congruent = bool(tt == 1.0)
        name      = 'congruent' if congruent else 'incongruent'

    empty    = tuple(s for s in range(n_slots) if s != target and s not in flankers)
    distance = min(abs(s - target) for s in flankers)
    return dict(target=target, flankers=tuple(flankers), empty=empty,
                congruent=congruent, distance=distance, name=name)


def _bare(ax, keep_bottom=False):
    """Strip a stacked-trace row down to its data."""
    ax.set_yticks([])
    ax.set_xticks([])
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_visible(keep_bottom)


def plot_trial(trials, config, trial=0, show_gate=True, show_loss_weights=True,
               show_slot_channels=False, row_height=0.30, width=2.9):
    """
    One trial, top to bottom: each slot's observations, the true direction, the speed
    pressure, and the decision variable.

    Reading it
    ----------
    Each slot row plots that slot's observed value at every timestep — the quantity the
    model actually received, noise included. Up is rightward, down is leftward, and the
    dashed line is the noiseless level that slot's own arrow was drawn from (the true
    direction on the target, the flankers' direction beside it). A trace that spends the
    trial on the wrong side of zero is a slot that misled the model, which is what
    separates a bad-luck error from a control failure and is the quantity
    `centre_evidence` summarises across the session. Colour is role, not identity: the
    target is black, the flankers take the trial's congruency hue and distance shade from
    the house palette, and slots holding no arrow are grey (they still carry
    `bg_noise_std`, which is why they are drawn at all rather than omitted).

    The gate column is the softmaxed attention weight the trial INHERITED (`z_in`), one
    bar per slot, against a dotted line at the uniform value 1/n_slots. That is the gate
    under which these outputs were produced; the trial's own update lands on the next
    trial. It is blank on trial 0, which inherited nothing.

    The output row is the model's read-out on the response dimension. The other five
    output dims are hidden by default (`show_slot_channels=True` draws them faintly).
    The model is a next-frame predictor with `output_size == input_size`, but
    `output_loss_mask` zeroes every dim except the direction, so those five channels are
    never trained: across a session their SD is about 0.1 against observations with SD
    0.9, and their correlation with the slot they nominally predict is weak and
    inconsistent in sign. They are drift, and showing them by default only invites the
    question of what they are.

    Dashed rules mark +/- rt_threshold, the shaded band is the pre-response
    window (zero loss weight), and the marker sits at the interpolated crossing; the
    header says so when the trial never crossed. Note that `rt_interp` is
    `(cross_idx - 1) + frac`, so a crossing bracketed by t=0 and t=1 is reported below 1
    and its marker lands inside the shaded band. That is the convention, not a bug —
    the crossing is still counted only at or after `response_start_timestep`.

    Parameters
    ----------
    trials  : dict from extract_trials
    config  : the stage config those trials came from
    trial   : trial index within the session
    show_slot_channels : draw the five untrained output dims alongside the response one
    row_height, width : inches; a slot row is one row_height, the output row is taller.
                        Both go through FigSize.custom, so FigSize.dev() still applies.

    Returns the figure.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from plot_style import FigSize

    if not 0 <= trial < trials['n_trials']:
        raise IndexError(f'trial {trial} out of range (session has {trials["n_trials"]})')

    obs      = trials['obs'][trial]                      # (ad, n_slots)
    ad       = trials['ad']
    n_slots  = obs.shape[-1]
    t_axis   = np.arange(ad)
    roles    = trial_slot_roles(trials, config, trial)
    signal   = float(getattr(config, 'signal_strength', 1.0))
    resp_start = int(trials['search_from'])

    out_full = (trials['output_full'][trial] if trials.get('output_full') is not None
                else trials['output_traj'][trial][:, None])
    n_out    = out_full.shape[-1]

    tw = list(getattr(config, 'temporal_loss_weights', []) or []) if show_loss_weights else []
    gate = trials['z_in'][trial] if show_gate else None
    have_gate = gate is not None and not np.all(np.isnan(gate))

    # ── layout ────────────────────────────────────────────────────────────────
    extra = [0.55] + ([0.55] if tw else []) + [1.7]      # true direction | speed | output
    heights = [1.0] * n_slots + extra
    fig_h = row_height * sum(heights) + 0.65
    fig = plt.figure(figsize=FigSize.custom(width, fig_h))
    gs = gridspec.GridSpec(len(heights), 2, height_ratios=heights,
                           width_ratios=[1.0, 0.17 if have_gate else 0.001],
                           hspace=0.25, wspace=0.06)

    gt_dir = float(trials['true_dir'][trial])
    flank_color = flanker_color(roles['congruent'], near=(roles['distance'] == 1))
    role_color = {roles['target']: '#000000'}
    role_color.update({s: flank_color for s in roles['flankers']})
    role_color.update({s: '#a8a8a8' for s in roles['empty']})

    # ── slot rows ─────────────────────────────────────────────────────────────
    lim = max(1.05 * np.abs(obs).max(), signal * 1.4)
    slot_axes = []
    for s in range(n_slots):
        ax = fig.add_subplot(gs[s, 0])
        active = s not in roles['empty']
        ax.axhline(0, color='k', linewidth=0.4, linestyle=':', alpha=0.5)
        if active:
            # The noiseless level this slot's arrow was drawn from: the true direction on
            # the target, and the flankers' own direction beside it. A trace that spends
            # the trial on the wrong side of zero is a slot that misled the model, which
            # is the single most useful thing to be able to see on one trial.
            nominal = signal * (gt_dir if s == roles['target']
                                else gt_dir * (1.0 if roles['congruent'] else -1.0))
            ax.axhline(nominal, color=role_color[s], linewidth=0.5, linestyle='--',
                       alpha=0.35)
        # A slot holding no arrow is drawn, not omitted: at bg_noise_std = 0 its trace is
        # identically zero, so it has to sit ON TOP of the dotted zero reference (zorder)
        # or it reads as a missing line rather than a flat one.
        ax.plot(t_axis, obs[:, s], color=role_color[s],
                linewidth=1.4 if active else 1.0, alpha=1.0 if active else 0.95,
                solid_capstyle='round', zorder=3 if active else 2.5)
        role = ('target' if s == roles['target']
                else 'flanker' if s in roles['flankers'] else 'empty')
        label = _slot_label(s, n_slots) + (f'\n{role}' if role else '')
        ax.set_ylabel(label, rotation=0, ha='right', va='center', labelpad=4,
                      color=role_color[s] if active else '#909090')
        ax.set_xlim(-0.5, ad - 0.5)
        ax.set_ylim(-lim, lim)
        _bare(ax)
        slot_axes.append(ax)

    # ── inherited gate, one bar per slot ──────────────────────────────────────
    if have_gate:
        gmax = max(float(np.nanmax(gate)) * 1.15, 1.0 / n_slots * 1.6)
        for s, ax_slot in enumerate(slot_axes):
            axg = fig.add_subplot(gs[s, 1])
            axg.barh([0], [gate[s]], height=1.2, color=role_color[s],
                     alpha=0.85 if s not in roles['empty'] else 0.5, linewidth=0)
            axg.axvline(1.0 / n_slots, color='k', linewidth=0.4, linestyle=':', alpha=0.5)
            axg.set_xlim(0, gmax)
            axg.set_ylim(-0.8, 0.8)
            _bare(axg)
            if s == 0:
                axg.set_title('gate', pad=2)

    # ── true direction ────────────────────────────────────────────────────────
    row = n_slots
    ax_gt = fig.add_subplot(gs[row, 0])
    ax_gt.axhline(0, color='k', linewidth=0.4, linestyle=':', alpha=0.5)
    ax_gt.hlines(gt_dir, -0.5, ad - 0.5, colors='k', linewidth=3, zorder=3)
    ax_gt.set_ylabel('target\ndirection', rotation=0, ha='right', va='center', labelpad=4)
    ax_gt.set_xlim(-0.5, ad - 0.5)
    ax_gt.set_ylim(-1.6, 1.6)
    _bare(ax_gt)

    # ── speed pressure ────────────────────────────────────────────────────────
    if tw:
        row += 1
        ax_tw = fig.add_subplot(gs[row, 0])
        w = (list(tw) + [0.0] * ad)[:ad]
        ax_tw.bar(t_axis, w, color='#555555', alpha=0.55, width=0.6)
        ax_tw.set_ylabel('loss\nweight', rotation=0, ha='right', va='center', labelpad=4)
        ax_tw.set_xlim(-0.5, ad - 0.5)
        _bare(ax_tw)

    # ── decision variable ─────────────────────────────────────────────────────
    row += 1
    ax_out = fig.add_subplot(gs[row, 0])
    ax_out.axvspan(-0.5, resp_start - 0.5, color='k', alpha=0.08, linewidth=0)
    mark_target_onset(ax_out, config)
    if show_slot_channels:
        for d in range(n_out - 1):
            ax_out.plot(t_axis, out_full[:, d], color='#bbbbbb', linewidth=0.5, alpha=0.7,
                        zorder=1, label='slot channels (no loss)' if d == 0 else None)
        # loc='best' rather than a fixed corner: the response trace ends top-right on a
        # rightward decision and bottom-right on a leftward one, so no corner is safe.
        ax_out.legend(frameon=False, handlelength=1.0, borderpad=0.2, labelspacing=0.25,
                      loc='best')
    ax_out.plot(t_axis, out_full[:, -1], color='#1f4e79', linewidth=1.6,
                solid_capstyle='round', zorder=3)
    thr = float(trials['rt_threshold'])
    ax_out.axhline(0, color='k', linewidth=0.4, linestyle=':', alpha=0.5)
    for level in (thr, -thr):
        ax_out.axhline(level, color='k', linewidth=0.5, linestyle='--', alpha=0.35)
    if bool(trials['decided'][trial]):
        rt = float(trials['rt_interp'][trial])
        ax_out.axvline(rt, color='k', linewidth=0.6, alpha=0.55)
        ax_out.plot([rt], [np.sign(trials['resp_at_decision'][trial]) * thr],
                    marker='o', markersize=3.5, color='#1f4e79', zorder=4)
    ax_out.set_xlim(-0.5, ad - 0.5)
    ax_out.set_xticks(t_axis)
    ax_out.set_xlabel('Timestep within trial')
    ax_out.set_ylabel('decision\nvariable', rotation=0, ha='right', va='center', labelpad=4)
    ax_out.spines['top'].set_visible(False)
    ax_out.spines['right'].set_visible(False)

    # ── header: the trial's identity, which no axis can carry ─────────────────
    if bool(trials['decided'][trial]):
        outcome = 'correct' if bool(trials['correct_at_decision'][trial]) else 'ERROR'
        verdict = f'{outcome}, RT {float(trials["rt_interp"][trial]):.2f}'
    else:
        verdict = 'no response'
    slot_axes[0].set_title(
        f'trial {trial} · {roles["name"]}\n'
        f'target {"right" if gt_dir > 0 else "left"} · {verdict}',
        loc='left', pad=3)

    fig.align_ylabels(slot_axes + [ax_gt, ax_out])
    return fig


def export_trial_figure(trials, config, trial=0, path=None, **kwargs):
    """
    `plot_trial` written straight to a PDF.

    `path` defaults to <config.export_path>/flanker_trial_<trial>.pdf. Returns the path.
    """
    import os
    import matplotlib.pyplot as plt

    fig = plot_trial(trials, config, trial=trial, **kwargs)
    if path is None:
        os.makedirs(config.export_path, exist_ok=True)
        path = f'{config.export_path}flanker_trial_{trial}.pdf'
    fig.savefig(path, bbox_inches='tight')
    print(f'Exported: {path}')
    plt.close(fig)
    return path


def example_trial_indices(trials, config, one_per_condition=True, seed=0):
    """
    A representative trial index per condition, for figures that want a small gallery.

    Returns [(label, index), ...] in FLANKER_CELLS order for the random-trials stage,
    congruent-then-incongruent for the pretraining stage. Trials that decided are
    preferred, so the RT marker has something to point at; a condition with no decided
    trial falls back to any trial of that condition, and one with no trials is dropped.
    """
    rng = np.random.default_rng(seed)
    tt  = trials['trial_type']
    types = ([(0, 'near-cong'), (2, 'far-cong'), (1, 'near-incong'), (3, 'far-incong')]
             if getattr(config, 'dataset_name', '') == 'flanker_random'
             else [(1, 'congruent'), (0, 'incongruent')])
    if not one_per_condition:
        types = types[:1]

    picks = []
    for code, label in types:
        idx = np.flatnonzero(tt == code)
        if not len(idx):
            continue
        decided = idx[trials['decided'][idx]]
        pool = decided if len(decided) else idx
        picks.append((label, int(rng.choice(pool))))
    return picks


def plot_correlation_structure(config=None, alphas=(0.5, 1.0, 1.5, 2.0)):
    """
    Stage-1 companion correlation against slot distance: the config's own profile
    alongside the power-law family `p(d) = 1 / (1 + d)**alpha` it was chosen from.

    This is the knob that teaches the model near != far — the gap between index 1 and 2
    sets the ceiling on any distance effect, and the profile must stay above 0.5
    everywhere or the model learns to ignore the companion entirely. Nothing in it is
    per-trial, so it takes a config rather than a trials dict.
    """
    import matplotlib.pyplot as plt
    from plot_style import FigSize

    n = len(getattr(config, 'p_corr_by_distance', [0] * 5)) if config else 5
    d = np.arange(n)

    fig, ax = plt.subplots(figsize=FigSize.wide)
    # shades = plt.cm.Greys(np.linspace(0.35, 0.75, len(alphas)))
    # for alpha, color in zip(alphas, shades):
    #     ax.plot(d, 1.0 / (1.0 + d) ** alpha, color=color, linewidth=0.8,
    #             label=f'α={alpha}')
    if config is not None:
        ax.plot(d, np.asarray(config.p_corr_by_distance, dtype=float),
                color=FLANKER_COLORS['near_incong'], marker='o', markersize=3,
                linewidth=1.4, label='config', zorder=3)
    ax.axhline(0.5, color='k', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.text(n - 1, 0.52, 'floor', ha='right', va='bottom', alpha=0.6)
    ax.set_xlabel('Slot distance from target')
    ax.set_ylabel('P(companion congruent)')
    ax.set_xticks(d)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, handlelength=1.0, borderpad=0.2, labelspacing=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig


# ── Config / model helpers ────────────────────────────────────────────────────

def sync_gating(stage_config, from_config):
    """
    Copy runtime gating flags from from_config to stage_config.
    Stage configs are freshly constructed with class defaults, so a runtime choice
    (e.g. gating='post') would otherwise be silently reverted.
    """
    for attr in ('pre_gating', 'post_gating', 'use_add_gating', 'use_mul_gating'):
        setattr(stage_config, attr, getattr(from_config, attr))


def mirror_to_model(model, stage_config, attrs=('Z_lr', 'Z_decay', 'no_of_steps_in_latent_space')):
    """
    Mirror stage_config attributes onto model.config, and onto the live Z optimizer.

    Two separate things have to happen, and only the first is obvious:

    1. model.config is a direct reference to the config the model was built with
       (not a copy). `no_of_steps_in_latent_space` is re-read from it every batch,
       so setting it here is enough.

    2. `Z_lr` and `Z_decay` are NOT re-read per batch — they are baked into the Adam
       optimizer when it is constructed (models.RNN_with_latent.__init__ →
       _build_Z_optimizer), and `set_Z` only rebuilds that optimizer if Z's *shape*
       changes. Across flanker stages the shape never changes, so assigning
       config.Z_lr alone silently has no effect and the LU keeps stepping at the
       learning rate the model was born with. Push the values into the existing
       param groups too.
    """
    for attr in attrs:
        if hasattr(stage_config, attr):
            setattr(model.config, attr, getattr(stage_config, attr))

    optimizer = getattr(model, 'Z_optimizer', None)
    if optimizer is not None:
        for group in optimizer.param_groups:
            if hasattr(stage_config, 'Z_lr'):
                group['lr'] = float(stage_config.Z_lr)
            if hasattr(stage_config, 'Z_decay') and 'weight_decay' in group:
                group['weight_decay'] = float(stage_config.Z_decay or 0.0)


def reset_Z_uniform(model, scale=0.2, seed=None):
    """
    Re-seed Z before a new stage, identically across the trial's timesteps.

    `torch.randn_like(model.Z) * scale` gives every timestep a *different* random
    start. Because LU broadcasts one aggregated update to all timesteps, that
    difference is never corrected — it persists as frozen noise on the gate, and
    makes the within-trial Z axis look like dynamics when it is initialisation.
    Seeding one vector and sharing it across timesteps keeps Z genuinely shared,
    as models.adjust_Z_grads assumes.

    Also rebuilds the Z optimizer, which (a) clears Adam moments carried over from
    the previous stage's LU and (b) picks up the current config.Z_lr — see
    mirror_to_model for why that does not happen on its own. Call mirror_to_model
    first so the rebuild reads the intended learning rate.
    """
    import torch
    gen = None
    if seed is not None:
        gen = torch.Generator(device='cpu').manual_seed(int(seed))
    b, t, d = model.Z.shape
    z0 = torch.randn(b, 1, d, generator=gen) * scale
    model.set_Z(z0.expand(b, t, d).contiguous().to(model.Z.device))
    model._rebuild_Z_optimizer()
