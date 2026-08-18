"""
rotation_decoding_analysis.py — Decode the rotation angle from Z vs. hidden activity.

Question: during the generalization phase (Phase 3a, novel rotations), to what degree is
the rotation angle recoverable from the latent Z compared to from RNN hidden activity,
for each model in a comparison?

Design decisions baked in here:

  Cross-validation : held-out *angles*.  The decoder is fit on a subset of the Phase-3a
                     angles and tested on angles it has never seen, so a good score means
                     the representation carries a continuous/metric code for rotation,
                     not just a lookup table of trained contexts.
  Dimensionality   : Z_dim is 2-3 while hidden_size is 64, but a linear decoder onto the
                     2-D target [cos θ, sin θ] has a rank-2 readout no matter how many
                     inputs it gets, so the extra width buys no readout power — only more
                     freedom to overfit, which the held-out-angle CV and the null already
                     control (a null at chance means the 64 dims are not cheating).
                     Reducing the hidden state to Z_dim is therefore a *diagnostic of how
                     distributed the code is*, not a fairness control, and it is not in the
                     default bars.  Use `decode_component_sweep` for that question:
                     unsupervised PCA ranks directions by variance and can miss a
                     low-variance but decodable coordinate, so 'H_pls' (supervised, fit
                     in-fold) is the honest fixed-width reduction and 'H_pca' the
                     variance-ordered one.
  Chance level     : held-out folds contain only a handful of angles, so the analytic 90°
                     is not the right reference.  Each representation gets its own null
                     from permuting angle labels *across state-blocks* and re-running the
                     identical CV.
  Timesteps        : all frames by default (`frames='all'`).  Note that at outcome frames
                     the attack (x, y) is in the input, so part of any hidden-state
                     advantage is read off the current stimulus rather than held context.
                     Pass frames='cue' to measure held context only.

Every public function accepts either a single logger or a list of loggers (see `as_runs`),
fits decoders per run, and aggregates scores across runs.  Timesteps are never pooled
across seeds — different networks have different hidden bases.

Entry points:
    decode_rotation_across_models(results)   → per-model results from a run script's dict
    plot_decoding_comparison(...)            → MAE per representation, grouped by model
    plot_decoded_vs_true(...)                → out-of-fold decoded angle vs true angle
    component_sweep_across_models(results)   → error vs #components, per model
    plot_component_sweep(...)                → how distributed the rotation code is
    summarize_decoding(...)                  → printed table

Run `python rotation_decoding_analysis.py` for the synthetic self-test.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plot_style import Color_scheme, FigSize, get_model_color
from rotating_targets_analysis import (
    _mean_sem,
    _nearest_rotation_deg,
    flatten_logger,
    get_block_switches,
    get_phase2_last_third,
    get_phase3a_range,
)

# Representations reported by default. Dimensionality-matched variants exist (see
# ALL_REPS) but are diagnostics, not fairness controls — see the module docstring.
REPS = ('Z', 'H')

# Every representation `decode_rotation` understands.
ALL_REPS = ('Z', 'H', 'H_pls', 'H_pca')

_REP_LABELS = {
    'Z':     'latent Z',
    'H':     'hidden (full)',
    'H_pls': 'hidden (PLS→Z_dim)',
    'H_pca': 'hidden (PCA→Z_dim)',
}

_ALPHAS = np.logspace(-3, 5, 17)


# ──────────────────────────────────────────────────────────────────────────────
# Logger extraction
# ──────────────────────────────────────────────────────────────────────────────

def flatten_hidden_states(logger, config):
    """
    Return the logged RNN hidden states as a flat (T, hidden_size) array, or None.

    `models.RNN_with_latent.forward` returns only the *final* (h, c) of the sequence, and
    `_log_batch` logs it without stride slicing, so each logger entry is (batch, hidden).
    That final h is the state whose readout produced the logged prediction for the same
    batch, so the returned array aligns 1:1 with `flatten_logger`'s arrays.

    Entries shaped (batch, seq, hidden) are also accepted — the last `stride` steps are
    taken, matching how inputs/outputs/Z are logged — in case hidden-state logging is
    ever changed to record the full sequence.

    Raises ValueError if the result cannot be aligned with the other logged arrays, which
    is what happens under `log_initial_burn_in_timesteps=True` (the burn-in branch of
    `_log_batch` logs seq_len input entries but only one hidden state).
    """
    if not getattr(logger, 'hidden_states', None):
        return None

    stride = config.stride
    chunks = []
    for entry in logger.hidden_states:
        entry = np.asarray(entry)
        if entry.ndim == 3:          # (batch, seq, hidden)
            entry = entry[:, -stride:, :]
        elif entry.ndim == 2:        # (batch, hidden) — the current logging behaviour
            entry = entry[:, np.newaxis, :]
        else:
            raise ValueError(
                f'Unexpected hidden-state entry with shape {entry.shape}; '
                'expected (batch, hidden) or (batch, seq, hidden).'
            )
        chunks.append(entry)

    hh = np.concatenate(chunks, axis=0)
    hh = hh.reshape(-1, hh.shape[-1])

    n_expected = len(np.concatenate(logger.inputs, axis=0).reshape(-1, config.input_size))
    if len(hh) != n_expected:
        raise ValueError(
            f'Hidden states ({len(hh)} timesteps) do not align with inputs '
            f'({n_expected} timesteps). This happens with '
            'log_initial_burn_in_timesteps=True, which logs seq_len input entries but '
            'only one hidden state for the first batch. Re-run with it disabled.'
        )
    return hh


def verify_hidden_state_alignment(logger, config, model, phase='phase3a',
                                  n_check=200, atol=1e-4):
    """
    Confirm that logged hidden states are the states that produced the logged predictions.

    Pushes each logged h through the model's output layer and compares against the logged
    prediction. The whole decoding pipeline rests on this alignment, so it is worth
    checking rather than assuming.

    Only valid over a phase in which the weights were frozen — during Phases 1-2 the
    output layer is updated after every batch, so `model.output_layer` at the end of
    training is not the readout that produced an early logged prediction. Phase 3a runs
    with `test_no_of_steps_in_weight_space=0`, so the check is exact there, and Phase 3a
    is the phase this analysis decodes from anyway.

    Returns (ok: bool, max_abs_diff: float).
    """
    import torch

    hh = flatten_hidden_states(logger, config)
    if hh is None:
        raise ValueError('No hidden states logged; set config.log_hidden_states = True.')

    t_start, t_end, _ = _phase_range_and_rotations(logger, config, phase)
    if t_start is None:
        raise ValueError(f"Phase '{phase}' not found in logger.")

    _, oi, _, _ = flatten_logger(logger, config)
    t_end = min(t_end, len(hh))
    idx = np.linspace(t_start, t_end - 1, min(n_check, t_end - t_start)).astype(int)

    with torch.no_grad():
        h = torch.as_tensor(hh[idx], dtype=torch.float32, device=config.device)
        pred = model.output_layer(h).cpu().numpy()

    max_diff = float(np.abs(pred - oi[idx]).max())
    return bool(max_diff < atol), max_diff


def _phase_range_and_rotations(logger, config, phase):
    """Return (t_start, t_end, rotation_set_degrees) for a named phase."""
    if phase == 'phase3a':
        t_start, t_end = get_phase3a_range(logger)
        if t_start is None:
            return None, None, None
        rotations = list(config.test_rotations) or list(config.train_rotations)
    elif phase == 'phase2':
        t_start, t_end = get_phase2_last_third(logger)
        rotations = list(config.train_rotations)
    elif phase == 'all':
        t_start = 0
        t_end = len(np.concatenate(logger.inputs, axis=0).reshape(-1, config.input_size))
        rotations = sorted(set(list(config.train_rotations)
                               + list(config.test_rotations or [])))
    else:
        raise ValueError(f"Unknown phase '{phase}'; expected 'phase3a', 'phase2' or 'all'.")
    return int(t_start), int(t_end), rotations


def extract_decode_samples(logger, config, phase='phase3a', frames='all', z_lag=0):
    """
    Collect aligned (Z, hidden, rotation) samples over one phase of one run.

    Parameters
    ----------
    phase  : 'phase3a' (novel rotations) | 'phase2' (last third, trained rotations) | 'all'
    frames : 'all' | 'cue' | 'outcome'
        Which timesteps to sample.  At outcome frames the attack (x, y) is present in the
        input, so the hidden state can read rotation from the current stimulus; 'cue'
        restricts to timesteps where rotation must come from held context.
    z_lag  : int
        Z is sampled at index t + z_lag.  `latent_values[t]` is Z *after* the LU step of
        batch t, while `hidden_states[t]` comes from the weight-update forward pass, which
        ran with the Z carried over from batch t-1.  z_lag=0 reads both at the logged
        timestep; z_lag=-1 pairs each hidden state with the Z that actually drove it.

    Returns a dict of aligned arrays (or None if the phase is absent from the logger):
        Z (N, Z_dim), H (N, hidden_size), theta_rad (N,), angle_deg (N,),
        block_idx (N,), frame_type (N,) [0=cue, 1=outcome],
        miniblock_since_switch (N,), t_index (N,)
    """
    t_start, t_end, rotations = _phase_range_and_rotations(logger, config, phase)
    if t_start is None:
        return None

    ii, _, ll, li = flatten_logger(logger, config)
    hh = flatten_hidden_states(logger, config)
    if li is None:
        raise ValueError('No latent values logged.')
    if hh is None:
        raise ValueError('No hidden states logged; set config.log_hidden_states = True.')

    nc = config.n_colors
    t_end = min(t_end, len(ii))

    # Block boundaries within the phase: the phase start plus every llcid change.
    boundaries = [t_start] + get_block_switches(ll, t_start, t_end)
    boundaries = sorted(set(boundaries))
    mb_len = nc * 2  # timesteps per mini-block (cue + outcome per color)

    Z, H, theta, angles, blocks, ftypes, mbs, tix = [], [], [], [], [], [], [], []

    for bi, b_start in enumerate(boundaries):
        b_end = boundaries[bi + 1] if bi + 1 < len(boundaries) else t_end
        for t in range(b_start, b_end):
            t_z = t + z_lag
            if not (0 <= t_z < len(li)):
                continue
            deg = _nearest_rotation_deg(ll[t], rotations)
            if deg is None:
                continue
            is_cue = ii[t, :nc].sum() > 0.5
            if frames == 'cue' and not is_cue:
                continue
            if frames == 'outcome' and is_cue:
                continue

            Z.append(li[t_z])
            H.append(hh[t])
            theta.append(float(ll[t]))
            angles.append(float(deg))
            blocks.append(bi)
            ftypes.append(0 if is_cue else 1)
            mbs.append((t - b_start) // mb_len)
            tix.append(t)

    if not Z:
        return None

    return dict(
        Z=np.asarray(Z, dtype=float),
        H=np.asarray(H, dtype=float),
        theta_rad=np.asarray(theta, dtype=float),
        angle_deg=np.asarray(angles, dtype=float),
        block_idx=np.asarray(blocks, dtype=int),
        frame_type=np.asarray(ftypes, dtype=int),
        miniblock_since_switch=np.asarray(mbs, dtype=int),
        t_index=np.asarray(tix, dtype=int),
        phase=phase,
        frames=frames,
        z_lag=z_lag,
    )


# ──────────────────────────────────────────────────────────────────────────────
# One-or-many normalizer
# ──────────────────────────────────────────────────────────────────────────────

def as_runs(loggers, configs=None):
    """
    Normalize a single logger, a list of loggers, or a list of (logger, config) pairs
    into a list of (logger, config) tuples.

    `configs` may be a single config (applied to every logger) or a list, one per logger.
    When omitted, each logger's own `.config` is used — `train_model` sets it.

    This is the hook that lets every analysis function below take one run today and a
    list of seeds later without any call-site change.
    """
    if loggers is None:
        return []

    # A bare logger (has the attributes a Logger has) rather than a sequence of them.
    if hasattr(loggers, 'inputs'):
        loggers = [loggers]
    loggers = list(loggers)

    # A list of (logger, config) pairs.
    if loggers and isinstance(loggers[0], (tuple, list)) and len(loggers[0]) == 2:
        return [(lg, cf) for lg, cf in loggers]

    if configs is None:
        cfgs = []
        for lg in loggers:
            cf = getattr(lg, 'config', None)
            if cf is None:
                raise ValueError(
                    'No config given and logger has no .config attribute. '
                    'Pass configs= explicitly.'
                )
            cfgs.append(cf)
    elif isinstance(configs, (list, tuple)):
        if len(configs) != len(loggers):
            raise ValueError(f'Got {len(loggers)} loggers but {len(configs)} configs.')
        cfgs = list(configs)
    else:
        cfgs = [configs] * len(loggers)

    return list(zip(loggers, cfgs))


# ──────────────────────────────────────────────────────────────────────────────
# Circular metrics
# ──────────────────────────────────────────────────────────────────────────────

def _angular_error_deg(theta_pred, theta_true):
    """Absolute angular error in degrees, wrapped to [0, 180]."""
    d = np.arctan2(np.sin(theta_pred - theta_true), np.cos(theta_pred - theta_true))
    return np.abs(np.degrees(d))


def _circular_r2(theta_pred, theta_true):
    """
    1 − (residual circular dispersion) / (total circular dispersion about the mean angle).

    0 means no better than always predicting the mean direction; 1 means exact.
    """
    ss_res = np.mean(1.0 - np.cos(theta_pred - theta_true))
    mean_dir = np.arctan2(np.mean(np.sin(theta_true)), np.mean(np.cos(theta_true)))
    ss_tot = np.mean(1.0 - np.cos(theta_true - mean_dir))
    return float(1.0 - ss_res / (ss_tot + 1e-12))


# ──────────────────────────────────────────────────────────────────────────────
# Decoder
# ──────────────────────────────────────────────────────────────────────────────

def _transform(X_train, X_test, rep, n_components):
    """
    Standardize (and for 'H_pca', reduce) with every transform fit on the training fold
    only.  Fitting PCA on the full dataset would leak held-out-angle structure into the
    projection, which is exactly what this analysis is trying to measure.
    """
    scaler = StandardScaler().fit(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)
    if rep == 'H_pca':
        k = int(min(n_components, Xtr.shape[1], Xtr.shape[0]))
        pca = PCA(n_components=k).fit(Xtr)
        Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
    return Xtr, Xte


def _fit_predict_angle(X_train, y_train, X_test, alpha):
    """Ridge onto (cos θ, sin θ), then recover the angle with atan2."""
    model = Ridge(alpha=alpha).fit(X_train, y_train)
    p = model.predict(X_test)
    return np.arctan2(p[:, 1], p[:, 0])


def _fit_predict_angle_pls(X_train, y_train, X_test, n_components):
    """
    PLS onto (cos θ, sin θ) — a supervised fixed-width reduction.

    Unlike PCA, PLS picks the directions of X most predictive of the target, so it does
    not miss a low-variance but decodable coordinate. Its own regularization is the
    component count, so there is no separate ridge penalty to select.
    """
    k = int(min(n_components, X_train.shape[1]))
    model = PLSRegression(n_components=k).fit(X_train, y_train)
    p = np.asarray(model.predict(X_test))
    return np.arctan2(p[:, 1], p[:, 0])


def _select_alpha(X, theta, groups, rep, n_components, alphas, n_inner=3):
    """
    Choose the ridge penalty by an inner group-CV over the *training* angles.

    Grouping the inner split by angle too matters: samples within a state-block are
    heavily autocorrelated, so a random or leave-one-out inner split would leak and pick
    an alpha that is far too small — penalising the 64-dim hidden state most.
    """
    uniq = np.unique(groups)
    n_splits = int(min(n_inner, len(uniq)))
    if n_splits < 2:
        return float(np.median(alphas))

    y = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    errs = np.zeros(len(alphas))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        Xtr, Xte = _transform(X[tr], X[te], rep, n_components)
        for ai, a in enumerate(alphas):
            pred = _fit_predict_angle(Xtr, y[tr], Xte, a)
            errs[ai] += _angular_error_deg(pred, theta[te]).mean()
    return float(alphas[int(np.argmin(errs))])


def _cv_decode(X, theta, groups, rep, n_components, n_folds, alphas, fold_alphas=None):
    """
    Held-out-angle cross-validation.  Returns (theta_pred, alphas_used).

    `groups` are angle labels, so every fold tests angles absent from its training set.
    When `fold_alphas` is given it is used instead of running the inner search — the null
    permutations reuse the alpha selected on the real data for the matching fold, which
    keeps the null paired with the real fit and keeps the runtime sane.
    """
    uniq = np.unique(groups)
    n_splits = int(min(n_folds, len(uniq)))
    if n_splits < 2:
        raise ValueError(
            f'Need at least 2 distinct rotation angles to hold angles out; got {len(uniq)}.'
        )

    y = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    theta_pred = np.full(len(theta), np.nan)
    used = []

    for fi, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        # Held-out angles must be genuinely unseen — assert rather than trust.
        assert not (set(groups[tr]) & set(groups[te])), 'angle leaked across a CV fold'

        if rep == 'H_pls':
            # PLS is both the reduction and the regressor; the component count is its
            # regularization, so there is no alpha to select.
            Xtr, Xte = _transform(X[tr], X[te], 'H', n_components)
            theta_pred[te] = _fit_predict_angle_pls(Xtr, y[tr], Xte, n_components)
            used.append(np.nan)
            continue

        alpha = (fold_alphas[fi] if fold_alphas is not None
                 else _select_alpha(X[tr], theta[tr], groups[tr], rep, n_components, alphas))
        Xtr, Xte = _transform(X[tr], X[te], rep, n_components)
        theta_pred[te] = _fit_predict_angle(Xtr, y[tr], Xte, alpha)
        used.append(alpha)

    return theta_pred, used


def _permute_angles_across_blocks(angle_deg, block_idx, rng):
    """
    Reassign each state-block another block's angle.

    Permuting at the block level (rather than shuffling samples) destroys the
    representation↔angle mapping while preserving the within-block temporal structure
    that makes this data autocorrelated in the first place.
    """
    blocks = np.unique(block_idx)
    block_angle = np.array([angle_deg[block_idx == b][0] for b in blocks])
    permuted = rng.permutation(block_angle)
    lookup = dict(zip(blocks, permuted))
    return np.array([lookup[b] for b in block_idx])


def decode_rotation(samples, rep='Z', n_folds=6, n_shuffles=20, seed=0,
                    n_components=None, alphas=_ALPHAS, max_miniblocks=None):
    """
    Decode the rotation angle from one representation of one run, holding angles out.

    Parameters
    ----------
    samples : dict from `extract_decode_samples`
    rep     : one of ALL_REPS — 'Z', 'H', or a fixed-width reduction of H
              ('H_pls' supervised, 'H_pca' variance-ordered)
    n_folds : number of held-out-angle folds (capped at the number of distinct angles)
    n_shuffles : block-permuted null repetitions.  0 skips the null.
    n_components : component target for 'H_pls'/'H_pca'; defaults to the run's Z_dim.

    Returns a dict with mae_deg, median_ae_deg, circ_r2, the null summary, the out-of-fold
    theta_pred, and MAE binned by mini-block since the last block switch.
    """
    if rep not in ALL_REPS:
        raise ValueError(f"Unknown representation '{rep}'; expected one of {ALL_REPS}.")

    X = samples['Z'] if rep == 'Z' else samples['H']
    theta = samples['theta_rad']
    angle_deg = samples['angle_deg']
    block_idx = samples['block_idx']
    n_components = samples['Z'].shape[1] if n_components is None else n_components

    theta_pred, fold_alphas = _cv_decode(
        X, theta, angle_deg, rep, n_components, n_folds, alphas
    )
    err = _angular_error_deg(theta_pred, theta)

    # Null: same splits, same alphas, angles shuffled across state-blocks.
    rng = np.random.default_rng(seed)
    null_mae, null_r2 = [], []
    for _ in range(n_shuffles):
        perm_deg = _permute_angles_across_blocks(angle_deg, block_idx, rng)
        perm_theta = np.radians(perm_deg)
        p_pred, _ = _cv_decode(X, perm_theta, perm_deg, rep, n_components,
                               n_folds, alphas, fold_alphas=fold_alphas)
        null_mae.append(_angular_error_deg(p_pred, perm_theta).mean())
        null_r2.append(_circular_r2(p_pred, perm_theta))

    # MAE as a function of mini-blocks since the block switch.
    mb = samples['miniblock_since_switch']
    n_mb = int(max_miniblocks if max_miniblocks is not None else mb.max() + 1)
    mae_by_mb = np.full(n_mb, np.nan)
    for m in range(n_mb):
        sel = mb == m
        if sel.any():
            mae_by_mb[m] = err[sel].mean()

    return dict(
        rep=rep,
        mae_deg=float(err.mean()),
        median_ae_deg=float(np.median(err)),
        circ_r2=_circular_r2(theta_pred, theta),
        null_mae_deg=float(np.mean(null_mae)) if null_mae else np.nan,
        null_mae_deg_std=float(np.std(null_mae)) if null_mae else np.nan,
        null_circ_r2=float(np.mean(null_r2)) if null_r2 else np.nan,
        mae_by_miniblock=mae_by_mb,
        theta_pred=theta_pred,
        theta_true=theta,
        alphas=fold_alphas,
        n_samples=int(len(theta)),
        # Dimensionality the readout actually sees — for the reduced reps that is the
        # component target, which is the whole point of them.
        n_features=int(min(n_components, X.shape[1])
                       if rep in ('H_pca', 'H_pls') else X.shape[1]),
        n_features_raw=int(X.shape[1]),
        n_angles=int(len(np.unique(angle_deg))),
        n_folds=int(min(n_folds, len(np.unique(angle_deg)))),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Component sweep — how distributed is the rotation code?
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_SWEEP_KS = (1, 2, 3, 5, 10, 20, 40, 64)


def _nanmean_or_nan(vals):
    """Mean ignoring NaNs, returning NaN for an all-NaN input without warning."""
    vals = np.asarray(vals, dtype=float)
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan


def decode_component_sweep(samples, ks=DEFAULT_SWEEP_KS, method='pls', n_folds=6,
                           alphas=_ALPHAS, tol=0.1):
    """
    Decoding error from the top-k components of the hidden state, for a range of k.

    This is the question the fixed-width reduction was really asking: *how many
    dimensions of hidden activity do you need to recover the rotation?* Z answers it in
    Z_dim by construction, so a hidden state that needs far more is holding the same
    information in a far more distributed form.

    method : 'pls' — supervised, components chosen for predicting rotation (recommended)
             'pca' — unsupervised, components ordered by variance.  Worth plotting
                     alongside: a large pls/pca gap means the rotation code lives in
                     low-variance directions, which is why a PCA-matched bar understates
                     the hidden state so badly.

    Reductions are fit inside each fold. Returns dict with ks, errors, and the reference
    error decoding from Z.
    """
    rep = 'H_pls' if method == 'pls' else 'H_pca'
    H, theta, ang = samples['H'], samples['theta_rad'], samples['angle_deg']
    ks = [int(k) for k in ks if k <= H.shape[1]]

    errors = []
    for k in ks:
        pred, _ = _cv_decode(H, theta, ang, rep, k, n_folds, alphas)
        errors.append(float(_angular_error_deg(pred, theta).mean()))

    z_pred, _ = _cv_decode(samples['Z'], theta, ang, 'Z', samples['Z'].shape[1],
                           n_folds, alphas)
    z_err = float(_angular_error_deg(z_pred, theta).mean())

    errors = np.array(errors)
    # Matching Z within a tolerance rather than strictly: the two ceilings are often
    # within a degree of each other, and a strict test makes the statistic undefined on a
    # gap far smaller than the seed-to-seed spread.
    thresh = z_err * (1.0 + tol)
    reached = [k for k, e in zip(ks, errors) if e <= thresh]
    reached_strict = [k for k, e in zip(ks, errors) if e <= z_err]

    return dict(
        ks=np.array(ks),
        errors=errors,
        z_error=z_err,
        z_dim=int(samples['Z'].shape[1]),
        method=method,
        tol=tol,
        # Smallest k at which the hidden state comes within `tol` of Z. NaN when it never
        # does within the swept range — itself a result, not a failure.
        components_to_match_z=float(reached[0]) if reached else np.nan,
        components_to_match_z_strict=float(reached_strict[0]) if reached_strict else np.nan,
    )


def component_sweep_for_runs(loggers, configs=None, phase='phase3a', frames='all',
                             ks=DEFAULT_SWEEP_KS, method='pls', z_lag=0, **kw):
    """Component sweep over one or many runs, aggregating curves across seeds."""
    runs = as_runs(loggers, configs)
    per_seed = []
    for lg, cf in runs:
        samples = extract_decode_samples(lg, cf, phase=phase, frames=frames, z_lag=z_lag)
        if samples is not None:
            per_seed.append(decode_component_sweep(samples, ks=ks, method=method, **kw))
    if not per_seed:
        return None

    curves = np.array([s['errors'] for s in per_seed])
    mean, sem = _agg(curves)
    return dict(
        per_seed=per_seed,
        ks=per_seed[0]['ks'],
        errors_mean=mean,
        errors_sem=sem,
        z_error_mean=float(np.mean([s['z_error'] for s in per_seed])),
        z_dim=per_seed[0]['z_dim'],
        components_to_match_z=_nanmean_or_nan([s['components_to_match_z']
                                               for s in per_seed]),
        tol=per_seed[0]['tol'],
        method=method,
        n_seeds=len(per_seed),
    )


def component_sweep_across_models(results, verbose=True, **kw):
    """Component sweep for every model in a run script's `results` dict."""
    out = {}
    for label, entry in results.items():
        loggers, configs = _runs_from_result_entry(entry)
        res = component_sweep_for_runs(loggers, configs, **kw)
        if res is None:
            continue
        out[label] = res
        if verbose:
            n = res['components_to_match_z']
            n_txt = f'{n:.0f}' if np.isfinite(n) else f'>{res["ks"][-1]}'
            print(f'  {label}: hidden needs {n_txt} components to match Z '
                  f'({res["z_dim"]} dims, {res["z_error_mean"]:.1f}°)')
    return out


def plot_component_sweep(sweep_results, sweep_results_pca=None, figsize=None):
    """
    Decoding error against the number of hidden components retained.

    The horizontal line is Z at its native dimensionality. Where the hidden curve crosses
    it is how many dimensions of neural activity carry what Z holds in Z_dim — the
    distributedness of the rotation code.

    Pass `sweep_results_pca` (the same sweep with method='pca') to overlay the
    variance-ordered curve; a large gap means the code lives in low-variance directions.
    """
    labels = list(sweep_results.keys())
    if not labels:
        raise ValueError('No models to plot.')

    # One panel, one line per model — not one panel per model. Model identity is carried
    # by colour, which keeps the figure the same size no matter how many conditions run.
    # Extra width over `wide` is for the legend, which sits outside the axes: with model
    # names this long an in-axes legend would cover the curves.
    fig, ax = plt.subplots(1, 1, figsize=figsize or FigSize.custom(2.9, 1.4))

    for label in labels:
        res = sweep_results[label]
        c = get_model_color(label)
        ks, m, e = res['ks'], res['errors_mean'], res['errors_sem']
        ax.plot(ks, m, '-o', ms=2, color=c, label=label)
        if res['n_seeds'] > 1:
            ax.fill_between(ks, m - e, m + e, color=c, alpha=0.2, lw=0)
        if sweep_results_pca and label in sweep_results_pca:
            p = sweep_results_pca[label]
            ax.plot(p['ks'], p['errors_mean'], '--', lw=0.7, color=c, alpha=0.7)
        # Z's own error at its native width — where the hidden curve drops below this
        # marker is how many neural dimensions match the latent.
        ax.plot([res['z_dim']], [res['z_error_mean']], '*', ms=6, color=c,
                mec='white', mew=0.4)

    ax.axhline(90, color='0.7', ls='--', lw=0.6)
    ax.set_xscale('log')
    ks_ref = list(sweep_results[labels[0]]['ks'])
    ax.set_xticks(ks_ref)
    # Tick at every k but label alternate ones (always keeping the last): eight labels on
    # a log axis this narrow overlap into an unreadable smear.
    show = set(ks_ref[::2]) | {ks_ref[-1]}
    ax.set_xticklabels([str(k) if k in show else '' for k in ks_ref])
    ax.minorticks_off()
    ax.set_xlabel('Hidden components retained')
    ax.set_ylabel('Decoding error (deg)')

    style = [Line2D([], [], color='0.35', lw=0.9, label='hidden, PLS'),
             Line2D([], [], ls='none', marker='*', ms=6, color='0.35', label='latent Z')]
    if sweep_results_pca:
        style.insert(1, Line2D([], [], color='0.35', lw=0.7, ls='--', label='hidden, PCA'))
    handles = [Line2D([], [], color=get_model_color(l), lw=0.9, label=l) for l in labels]
    ax.legend(handles=handles + style, frameon=False, loc='upper left',
              bbox_to_anchor=(1.02, 1.0), handlelength=1.2, borderpad=0.2,
              labelspacing=0.25, handletextpad=0.4)
    plt.tight_layout()
    return fig


def _label_width(labels, minimum=16):
    """Column width that fits the longest model label — names vary a lot across runs."""
    return max(minimum, max((len(str(l)) for l in labels), default=0) + 2)


def summarize_component_sweep(sweep_results):
    """Print how many hidden components each model needs to match its own Z."""
    first = next(iter(sweep_results.values()))
    print('\n── How distributed is the rotation code? (held-out angles) ──')
    w = _label_width(sweep_results)
    print(f'{"Model":<{w}}{"Z dims":>8}{"Z err°":>9}{"components to match Z":>24}')
    for label, res in sweep_results.items():
        n = res['components_to_match_z']
        n_txt = f'{n:.0f}' if np.isfinite(n) else f'>{res["ks"][-1]}'
        print(f'{label:<{w}}{res["z_dim"]:>8}{res["z_error_mean"]:>9.1f}{n_txt:>24}')
    print(f'(hidden components ({first["method"].upper()}) needed to come within '
          f'{first["tol"]*100:.0f}% of Z\'s error)')


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation over runs (seeds) and models
# ──────────────────────────────────────────────────────────────────────────────

_SCALARS = ('mae_deg', 'median_ae_deg', 'circ_r2', 'null_mae_deg', 'null_circ_r2')


def _agg(arr):
    """
    `_mean_sem` over axis 0, returning NaN for all-NaN columns without warning.

    All-NaN columns are the normal case for the null metrics when n_shuffles=0, which is
    how the lag diagnostic runs — not an error worth a RuntimeWarning on every call.
    """
    arr = np.atleast_2d(np.asarray(arr, dtype=float))
    mean = np.full(arr.shape[1], np.nan)
    sem = np.full(arr.shape[1], np.nan)
    usable = ~np.all(np.isnan(arr), axis=0)
    if usable.any():
        mean[usable], sem[usable] = _mean_sem(arr[:, usable])
    return mean, sem


def decode_rotation_for_runs(loggers, configs=None, phase='phase3a', frames='all',
                             reps=REPS, z_lag=0, **decode_kw):
    """
    Run the decoding analysis over one or many runs (seeds) of the same model.

    Decoders are fit per run and only the *scores* are aggregated — pooling timesteps
    across seeds would mix incompatible hidden bases.

    Returns
    -------
    dict with:
      per_seed : list of {rep: decode_rotation(...) result}
      mean/sem : {rep: {scalar_metric: value}} aggregated across seeds
      mae_by_miniblock_mean/sem : {rep: (n_miniblocks,)}
      n_seeds, phase, frames, z_lag
    """
    runs = as_runs(loggers, configs)
    if not runs:
        return None

    max_mb = max(int(getattr(cf, 'n_miniblocks_per_state_block', 0)) for _, cf in runs) or None

    per_seed = []
    for lg, cf in runs:
        samples = extract_decode_samples(lg, cf, phase=phase, frames=frames, z_lag=z_lag)
        if samples is None:
            continue
        per_seed.append({
            rep: decode_rotation(samples, rep=rep, max_miniblocks=max_mb, **decode_kw)
            for rep in reps
        })

    if not per_seed:
        return None

    mean, sem, mb_mean, mb_sem = {}, {}, {}, {}
    for rep in reps:
        vals = np.array([[s[rep][k] for k in _SCALARS] for s in per_seed])
        m, e = _agg(vals)
        mean[rep] = dict(zip(_SCALARS, m))
        sem[rep] = dict(zip(_SCALARS, e))

        curves = np.array([s[rep]['mae_by_miniblock'] for s in per_seed])
        mb_mean[rep], mb_sem[rep] = _agg(curves)

    return dict(
        per_seed=per_seed,
        mean=mean,
        sem=sem,
        mae_by_miniblock_mean=mb_mean,
        mae_by_miniblock_sem=mb_sem,
        reps=tuple(reps),
        n_seeds=len(per_seed),
        phase=phase,
        frames=frames,
        z_lag=z_lag,
    )


def _runs_from_result_entry(entry):
    """
    Pull (loggers, configs) out of one entry of a run script's `results` dict.

    Accepts the current single-seed shape (`entry['logger']`, `entry['cfg']`) and the
    multi-seed shape (`entry['loggers']`, `entry['cfgs']`), so moving to many seeds is a
    one-line change in the run script rather than an edit here.
    """
    if 'loggers' in entry:
        return entry['loggers'], entry.get('cfgs', entry.get('cfg'))
    return entry['logger'], entry.get('cfg')


def decode_rotation_across_models(results, phase='phase3a', frames='all', reps=REPS,
                                  z_lag=0, verbose=True, **decode_kw):
    """
    Apply the decoding analysis to every model in a run script's `results` dict.

    `results` maps model label → dict holding either a single logger/cfg or lists of them.
    Returns {label: decode_rotation_for_runs(...) result}, skipping models with no data
    in the requested phase.
    """
    out = {}
    for label, entry in results.items():
        loggers, configs = _runs_from_result_entry(entry)
        res = decode_rotation_for_runs(loggers, configs, phase=phase, frames=frames,
                                       reps=reps, z_lag=z_lag, **decode_kw)
        if res is None:
            if verbose:
                print(f'  {label}: no {phase} data — skipped')
            continue
        out[label] = res
        if verbose:
            line = '  '.join(
                f'{_REP_LABELS[r]} {res["mean"][r]["mae_deg"]:5.1f}°' for r in reps
            )
            print(f'  {label}: {line}   (n_seeds={res["n_seeds"]})')
    return out


def compare_z_lags(results, lags=(0, -1), reps=REPS, phase='phase3a',
                   frames='all', n_shuffles=0, **decode_kw):
    """
    Re-run the decoding at several Z lags to quantify the Z/hidden timing asymmetry.

    `latent_values[t]` is Z after the latent update of batch t, while `hidden_states[t]`
    comes from the weight-update forward pass, which ran with the Z carried over from
    batch t-1.  Z is therefore one latent-update step ahead of h at the same index.
    lag 0 reads both at the logged timestep; lag -1 pairs each hidden state with the Z
    that actually drove it.  The null is off by default here since this is a diagnostic.

    Returns {lag: {label: result}}.
    """
    return {
        lag: decode_rotation_across_models(results, phase=phase, frames=frames, reps=reps,
                                           z_lag=lag, verbose=False,
                                           n_shuffles=n_shuffles, **decode_kw)
        for lag in lags
    }


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────

# Figures here encode *model* as colour (via plot_style.MODEL_COLORS, so a model keeps its
# colour across every figure in the project) and *representation* as fill opacity. That
# leaves the legend needing only two greyscale entries instead of one per model-rep pair,
# which is what keeps these panels small enough to drop into a multi-panel figure.
_REP_ALPHA = {'Z': 1.0, 'H': 0.45, 'H_pls': 0.70, 'H_pca': 0.25}


def _rep_alpha(rep):
    return _REP_ALPHA.get(rep, 0.6)


def _rep_style_handles(reps):
    """Greyscale proxy handles so the legend explains opacity once, not per model."""
    return [Patch(facecolor='0.35', alpha=_rep_alpha(r), label=_REP_LABELS[r])
            for r in reps]


def plot_decoding_comparison(decode_results, config=None, reps=None, figsize=None):
    """
    Mean absolute angular error per representation, grouped by model.

    Lower is better.  Each bar carries its own block-permuted null as a grey marker;
    a bar at or above its null carries no decodable rotation information.  Error bars are
    the SEM across seeds and are omitted for a single run.
    """
    labels = list(decode_results.keys())
    if not labels:
        raise ValueError('No models to plot.')
    reps = reps or decode_results[labels[0]]['reps']
    n, r = len(labels), len(reps)

    # One panel regardless of model count; width grows only with the number of bars.
    fig, ax = plt.subplots(
        1, 1, figsize=figsize or FigSize.custom(max(1.4, 0.24 * n * r + 0.55), 1.4))

    x = np.arange(n)
    w = 0.85 / r
    bar_colors = [get_model_color(l) for l in labels]
    n_seeds = decode_results[labels[0]]['n_seeds']

    all_nulls = []
    for ri, rep in enumerate(reps):
        off = (ri - (r - 1) / 2) * w
        vals = [decode_results[l]['mean'][rep]['mae_deg'] for l in labels]
        errs = [decode_results[l]['sem'][rep]['mae_deg'] for l in labels]
        ax.bar(x + off, vals, width=w, color=bar_colors, alpha=_rep_alpha(rep),
               yerr=errs if n_seeds > 1 else None, capsize=1.2, error_kw=dict(lw=0.5))
        all_nulls += [decode_results[l]['mean'][rep]['null_mae_deg'] for l in labels]

    # One grey band rather than a null marker per bar: with n models × r reps those were a
    # dozen near-identical ticks all saying the same thing, and pinning the axis to 110
    # left a 40-90° gap carrying no data. The band is the empirically correct chance
    # reference (see the module docstring), so the analytic 90° line is dropped with it.
    all_nulls = [v for v in all_nulls if np.isfinite(v)]
    if all_nulls:
        ax.axhspan(min(all_nulls), max(all_nulls), color='0.85', lw=0, zorder=0)
        ax.set_ylim(0, max(all_nulls) * 1.06)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    # Short label deliberately: at this panel height the axes area is ~0.55in, and a
    # 19-character ylabel centred on it overruns the figure and gets clipped.
    ax.set_ylabel('Error (deg)')
    ax.set_xlim(-0.6, n - 0.4)

    handles = _rep_style_handles(reps) + [Patch(facecolor='0.85', label='shuffled null')]
    ax.legend(handles=handles, frameon=False, loc='lower left',
              bbox_to_anchor=(0.0, 1.0, 1.0, 0.1), mode='expand',
              ncol=len(handles), handlelength=0.9, borderpad=0.15,
              labelspacing=0.2, handletextpad=0.4, columnspacing=0.8)
    plt.tight_layout()
    return fig


def plot_decoded_vs_true(decode_results, reps=None, seed_index=0, figsize=None):
    """
    Out-of-fold decoded angle against the true angle, one panel per model.

    Every point comes from a fold in which its angle was held out of training, so points
    on the diagonal mean the representation supports interpolation to unseen rotations.
    """
    labels = list(decode_results.keys())
    reps = reps or decode_results[labels[0]]['reps']
    n = len(labels)

    # Scatters genuinely need one panel per model. Panels are narrower than `small` and
    # the axes are shared, so only the leftmost carries tick labels — with `aspect='equal'`
    # the data area is square and the leftover width would otherwise become dead space.
    fig, axes = plt.subplots(
        1, n, figsize=figsize or FigSize.custom(1.05 * n + 0.5, 1.35),
        squeeze=False, sharex=True, sharey=True)

    # Draw the grey hidden-state cloud first so the model-coloured Z points sit on top of
    # it rather than under it.
    draw_order = [r for r in reps if r != 'Z'] + [r for r in reps if r == 'Z']

    for ci, label in enumerate(labels):
        ax = axes[0, ci]
        per_seed = decode_results[label]['per_seed'][seed_index]
        for rep in draw_order:
            r = per_seed[rep]
            # Z in the model's colour, other representations in grey, so each panel reads
            # as "this model's latent vs its neural activity".
            c = get_model_color(label) if rep == 'Z' else '0.55'
            ax.plot(np.degrees(r['theta_true']) % 360,
                    np.degrees(r['theta_pred']) % 360,
                    '.', ms=0.8, alpha=0.2, color=c, rasterized=True)
        ax.plot([0, 360], [0, 360], color='0.6', ls='--', lw=0.6)
        # Model name as a title, not an xlabel: this is the grid case the style guide
        # allows titles for, and it frees the x-axis for the quantity. A supxlabel would
        # collide with the per-panel labels at this height.
        ax.set_title(label)
        ax.set_xticks([0, 180, 360])
        ax.set_yticks([0, 180, 360])
        ax.set_aspect('equal', adjustable='box')

    # Shared axes: label only the leftmost panel.
    axes[0, 0].set_ylabel('Decoded rotation (deg)')
    axes[0, 0].set_xlabel('True rotation (deg)')

    # Colour-neutral swatches: Z is drawn in each model's own colour, so a single coloured
    # key would misrepresent every panel but the first.
    handles = [Line2D([], [], ls='none', marker='o', ms=2.5, color='0.15',
                      label='latent Z (model colour)'),
               Line2D([], [], ls='none', marker='o', ms=2.5, color='0.55',
                      label='hidden (grey)')]
    fig.legend(handles=handles, frameon=False, loc='lower center',
               bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=0.9, borderpad=0.15,
               labelspacing=0.2, handletextpad=0.4, columnspacing=1.0)
    fig.tight_layout()
    return fig



def summarize_decoding(decode_results, reps=None):
    """Print the decoding table: model × representation → error, null, circular R²."""
    labels = list(decode_results.keys())
    if not labels:
        print('No decoding results.')
        return
    reps = reps or decode_results[labels[0]]['reps']
    first = decode_results[labels[0]]

    print(f'\n── Rotation decoding — phase={first["phase"]}, frames={first["frames"]}, '
          f'z_lag={first["z_lag"]}, held-out angles ──')
    w = _label_width(decode_results)
    print(f'{"Model":<{w}}{"Representation":<22}{"MAE°":>8}{"null°":>8}'
          f'{"circR²":>9}{"dims":>7}{"n":>8}')
    for label in labels:
        res = decode_results[label]
        for rep in reps:
            m = res['mean'][rep]
            s = res['sem'][rep]
            n_feat = res['per_seed'][0][rep]['n_features']
            n_samp = res['per_seed'][0][rep]['n_samples']
            mae = (f'{m["mae_deg"]:.1f}±{s["mae_deg"]:.1f}' if res['n_seeds'] > 1
                   else f'{m["mae_deg"]:.1f}')
            print(f'{label:<{w}}{_REP_LABELS[rep]:<22}{mae:>8}'
                  f'{m["null_mae_deg"]:>8.1f}{m["circ_r2"]:>9.3f}'
                  f'{n_feat:>7}{n_samp:>8}')
    print(f'(chance ≈ 90°; n_seeds={first["n_seeds"]})')


def print_lag_table(lag_results, reps=REPS):
    """Print MAE at each Z lag — quantifies the Z/hidden timing asymmetry."""
    lags = list(lag_results.keys())
    labels = list(next(iter(lag_results.values())).keys())
    print('\n── Z-lag diagnostic (MAE°, held-out angles) ──')
    w = _label_width(labels)
    print(f'{"Model":<{w}}{"Representation":<22}' + ''.join(f'{f"lag {l}":>10}' for l in lags))
    for label in labels:
        for rep in reps:
            cells = ''.join(
                f'{lag_results[l][label]["mean"][rep]["mae_deg"]:>10.1f}' for l in lags
            )
            print(f'{label:<{w}}{_REP_LABELS[rep]:<22}{cells}')
    print('lag 0: Z and h both read at the logged timestep.  '
          'lag -1: h paired with the Z that drove it.')


# ──────────────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────────────

def self_test(verbose=True):
    """
    Synthetic checks on the decoder itself, independent of any trained model.

    1. A representation that *is* the rotation ([cos θ, sin θ] + noise) must decode to
       near-zero error, including on angles held out of training.
    2. A representation independent of rotation must land at the null (~90°).
    3. The block-permuted null must sit near chance with near-zero circular R².
    """
    rng = np.random.default_rng(0)
    angles = np.arange(0, 360, 15).astype(float)
    per_block = 120

    angle_deg = np.repeat(angles, per_block)
    theta = np.radians(angle_deg)
    block_idx = np.repeat(np.arange(len(angles)), per_block)

    informative = np.stack([np.cos(theta), np.sin(theta)], axis=1) \
        + 0.05 * rng.standard_normal((len(theta), 2))
    noise = rng.standard_normal((len(theta), 64))

    def _samples(Z, H):
        return dict(Z=Z, H=H, theta_rad=theta, angle_deg=angle_deg, block_idx=block_idx,
                    miniblock_since_switch=np.zeros(len(theta), dtype=int))

    good = decode_rotation(_samples(informative, noise), rep='Z', n_shuffles=5, seed=0)
    bad = decode_rotation(_samples(noise[:, :2], noise), rep='Z', n_shuffles=5, seed=0)

    checks = [
        ('informative representation decodes held-out angles', good['mae_deg'] < 10),
        ('uninformative representation is at chance', bad['mae_deg'] > 60),
        ('null is near chance', 70 < good['null_mae_deg'] < 110),
        ('null circular R² is near zero', abs(good['null_circ_r2']) < 0.25),
        ('informative circular R² is high', good['circ_r2'] > 0.9),
    ]
    if verbose:
        print('── rotation_decoding_analysis self-test ──')
        for name, ok in checks:
            print(f'  [{"ok" if ok else "FAIL"}] {name}')
        print(f'  informative: MAE={good["mae_deg"]:.2f}°  circR²={good["circ_r2"]:.3f}  '
              f'null={good["null_mae_deg"]:.1f}°')
        print(f'  noise:       MAE={bad["mae_deg"]:.2f}°   circR²={bad["circ_r2"]:.3f}')
    return all(ok for _, ok in checks)


if __name__ == '__main__':
    raise SystemExit(0 if self_test() else 1)
