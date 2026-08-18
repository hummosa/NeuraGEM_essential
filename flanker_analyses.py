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
    Continuous threshold-crossing time.

    The decision variable is sampled once per timestep, so an integer RT can take
    only a handful of values. Linearly interpolate the crossing between the two
    bracketing samples:

        rt = (t-1) + (threshold - |o(t-1)|) / (|o(t)| - |o(t-1)|)

    Parameters
    ----------
    output_traj : (n_trials, ad) decision variable
    threshold   : |output| level counted as a decision
    search_from : first timestep eligible to be a response (config.response_start_timestep);
                  earlier timesteps carry zero loss weight so their output is unconstrained.

    Returns
    -------
    rt_interp : (n,) float — NaN where the threshold was never crossed (censored)
    rt_int    : (n,) float — integer crossing timestep; ad if never crossed
    decided   : (n,) bool
    cross_idx : (n,) int — crossing timestep, clamped to the last timestep if censored
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

    rt_interp = np.where(decided, (cross_idx - 1) + frac, np.nan)
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
        signed_output        (n, ad)  output sign-normalised by the trial's final decision
        true_dir             (n,)     true direction ±1.0
        is_correct           (n,)     bool — correct at the final timestep
        response_side        (n,)     sign of output at the final timestep
        correct_at_decision  (n,)     bool — correct at the threshold-crossing timestep
        resp_at_decision     (n,)     ±1 — response emitted at the crossing timestep
        rt                   (n,)     integer crossing timestep; ad if never crossed
        rt_interp            (n,)     interpolated crossing time; NaN if censored
        decided              (n,)     bool — threshold was crossed inside the window
        cross_idx            (n,)     crossing timestep index (clamped for censored trials)

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
    correct     = (true_dir * output_traj > 0).astype(float)
    z_traj      = z_flat[:n_ts].reshape(n_trials, ad, -1)

    rt_interp, rt_int, decided, cross_idx = _interpolated_rt(
        output_traj, rt_threshold, search_from
    )

    # Sign-normalize output by the model's own final decision (sign of output at t=-1).
    # Positive = accumulating toward the eventual decision; negative = against it.
    final_decision_sign = np.sign(output_traj[:, -1:])
    signed_output       = output_traj * final_decision_sign

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
    censored = int((~trials['decided']).sum())
    print(f'{label}trials={n}  censored RT={censored} ({100*censored/max(n,1):.1f}%)  '
          f'accuracy={trials["correct_at_decision"].mean():.3f}  '
          f'Z within-trial spread={trials["z_within_trial_spread"]:.4f}')
    if trials['decided'].any():
        rt = trials['rt_interp'][trials['decided']]
        print(f'{label}RT (decided): mean={rt.mean():.2f}  '
              f'min={rt.min():.2f}  max={rt.max():.2f}')


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
        resp_rep    — this trial's response equals the previous trial's
        alternated  — the previous response differed from the one before it
        valid       — trials with a full n_back history
    """
    d    = decode_trial_types(trials, coding=coding)
    cong = d['is_cong'].astype(float)
    near = d['is_near'].astype(float) if d['is_near'] is not None else np.full(trials['n_trials'], np.nan)
    corr = trials['correct_at_decision'].astype(float)
    rt   = trials['rt_interp']
    side = trials['resp_at_decision'].astype(float)

    f = dict(cong=cong, near=near, correct=corr, rt=rt, side=side)
    for k in range(1, n_back + 1):
        f[f'cong_{k}']    = _lag(cong, k)
        f[f'near_{k}']    = _lag(near, k)
        f[f'correct_{k}'] = _lag(corr, k)
        f[f'rt_{k}']      = _lag(rt,   k)
        f[f'side_{k}']    = _lag(side, k)

    f['resp_rep'] = np.where(np.isnan(f['side_1']), np.nan,
                             (side == f['side_1']).astype(float))
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

def _style_trial_ax(ax, ad, response_start):
    ax.axvspan(-0.5, response_start - 0.5, alpha=0.08, color='k')
    ax.set_xticks(range(ad))
    ax.set_xlabel('Timestep within trial')
    ax.legend(fontsize=5)


def plot_accuracy_by_timestep(ax, trials, specs, config, linestyles=None):
    """
    Plot mean accuracy per timestep for multiple trial groups.

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
    _style_trial_ax(ax, ad, config.response_start_timestep)
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.3, 1.05)


def plot_rt_distribution(ax, trials, specs, config, fit_gaussian=True, linestyles=None,
                         undecided='extra_bin'):
    """
    Plot RT as an empirical probability mass function over integer timesteps
    (markers connected by lines) plus an optional Gaussian fit (dashed overlay).

    PMF sums to 1.0; undecided trials are always included.
    See plot_rt_continuous for the interpolated-RT version.

    Parameters
    ----------
    undecided : 'extra_bin'  — add a t=ad bin to the right for undecided trials (default)
                'last_bin'   — stack undecided trials onto the last existing bin (t=ad-1)

    specs : list of (mask, label, color) — mask is boolean (n_trials,)
    linestyles : applied to the empirical PMF line; Gaussian always plotted dashed.
    """
    rt           = trials['rt']
    ad           = trials['ad']
    rt_threshold = trials['rt_threshold']
    if linestyles is None:
        linestyles = ['-'] * len(specs)

    use_extra = (undecided == 'extra_bin')
    x_bins = list(range(ad + 1)) if use_extra else list(range(ad))

    for (mask, label, color), ls in zip(specs, linestyles):
        if not mask.any():
            continue
        n_total   = int(mask.sum())
        rt_m      = rt[mask]
        decided   = rt_m < ad
        n_decided = int(decided.sum())

        if use_extra:
            # decided bins 0..ad-1, undecided bin = ad
            pmf = np.array([(rt_m == t).sum() / n_total for t in range(ad + 1)])
        else:
            # decided bins 0..ad-2, undecided stacked into bin ad-1
            pmf = np.array([(rt_m == t).sum() / n_total for t in range(ad - 1)]
                           + [(rt_m >= ad - 1).sum() / n_total])

        ax.plot(x_bins, pmf, marker='o', markersize=4, color=color,
                linestyle=ls, linewidth=0.8, label=f'{label} (n={n_total})', zorder=3)

        if fit_gaussian and n_decided >= 3:
            try:
                from scipy.stats import norm
                p_decided = n_decided / n_total
                mu, sigma = norm.fit(rt_m[decided])
                x_fine = np.linspace(-0.5, ad - 0.5, 300)
                y_fine = norm.pdf(x_fine, mu, sigma) * p_decided
                ax.plot(x_fine, y_fine, color=color, linestyle='--',
                        linewidth=1.2, alpha=0.7, zorder=2)
            except Exception:
                pass

    # x-axis: decided bins always labelled by timestep; extra bin labelled 'und.'
    if use_extra:
        tick_labels = [str(t) for t in range(ad)] + ['und.']
        ax.set_xticks(x_bins)
        ax.set_xticklabels(tick_labels, fontsize=6)
        ax.set_xlabel('Timestep within trial')
        ax.axvspan(-0.5, config.response_start_timestep - 0.5, alpha=0.08, color='k')
        ax.legend(fontsize=5)
    else:
        _style_trial_ax(ax, ad, config.response_start_timestep)
    ax.set_ylabel(f'P(RT = t)  [threshold = {rt_threshold}]')
    ax.set_ylim(bottom=0)


def plot_rt_continuous(ax, trials, specs, config, bin_width=0.25, linestyles=None):
    """
    Plot the distribution of interpolated RTs — the continuous counterpart to
    plot_rt_distribution. Censored (never-crossed) trials cannot be placed on the
    time axis, so they are excluded here and their proportion is put in the legend.

    specs : list of (mask, label, color) — mask is boolean (n_trials,)
    """
    rt   = trials['rt_interp']
    ad   = trials['ad']
    lo   = max(int(trials['search_from']) - 1, 0)
    bins = np.arange(lo, ad + bin_width, bin_width)
    ctrs = 0.5 * (bins[:-1] + bins[1:])
    if linestyles is None:
        linestyles = ['-'] * len(specs)

    for (mask, label, color), ls in zip(specs, linestyles):
        if not mask.any():
            continue
        n_total = int(mask.sum())
        vals    = rt[mask]
        vals    = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        censored = 100.0 * (n_total - len(vals)) / n_total
        dens, _  = np.histogram(vals, bins=bins, density=True)
        ax.plot(ctrs, dens, color=color, linestyle=ls, linewidth=0.9,
                label=f'{label} (n={n_total}, {censored:.0f}% cens.)')

    ax.axvspan(lo - 0.5, config.response_start_timestep - 0.5, alpha=0.08, color='k')
    ax.set_xlabel('RT (timesteps, interpolated)')
    ax.set_ylabel(f'Density  [threshold = {trials["rt_threshold"]}]')
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=5)


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


def plot_scalar_bars(ax, trials, specs, measure, group_spacing=None, baseline=None):
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

    Returns
    -------
    x_positions : array — x positions used, for further annotation.
    """
    values_all, ylabel = trial_measure(trials, measure)

    means, sems, labels, colors, valid_idx = [], [], [], [], []
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

    ax.bar(x, means, color=colors, alpha=0.75, width=0.6, zorder=2)
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
