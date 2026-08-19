"""
rotation_geometry_analysis.py — What kind of code does the model use for rotation?

Decoding (see rotation_decoding_analysis.py) answers *how much* rotation information a
representation carries. This module asks a different question: *what shape* is that code?

The method is representational similarity analysis (RSA), chosen because it transfers
directly to neural data — everything below operates on a matrix of pairwise distances
between the states for each rotation, which you can compute from Z, from RNN hidden
activity, or from voxels/spikes with no change to the comparison step.

Procedure
---------
1. Average the representation within each rotation angle → one pattern per angle.
2. Build the empirical RDM: pairwise distance between those patterns.
3. Build a candidate RDM for each hypothesis about the code (below).
4. Rank-correlate empirical against each candidate. Highest wins.
5. Compare against a split-half noise ceiling, so a low correlation can be read as
   "wrong hypothesis" rather than "noisy data".

Candidate codes
---------------
Reported by default (CANDIDATES):
  circular      State turns smoothly with angle and wraps: 10° and 350° are neighbours.
                (head-direction-like)
  blend         State for a novel angle is a weighted mixture of the trained-context
                states — like mixing paint. Smooth like `circular`, but its geometry is
                inherited from the trained set rather than from the circle, so it has no
                wraparound and squashes angles far from any trained context.
  nearest       State reports only which trained context is closest. Flat within a
                trained angle's basin, sharp jump at the boundary. (categorical perception)

Available via `candidates=ALL_CANDIDATES` (OPTIONAL_CANDIDATES):
  periodic      Codes the target *constellation* rather than the colour→location mapping.
                With n_colors targets evenly spaced, rotating by 360/n_colors maps the set
                of target positions onto itself, so such a code cannot resolve θ from
                θ + 360/n_colors. Note this period is set by n_colors, NOT by which
                rotations were trained. This is the analysis's negative control — it lands
                at ~0 for a model that tracks the colour→location mapping, which is what
                demonstrates the method can return "no". Worth running once per new task
                configuration rather than on every figure.
  seamed        Smooth in angle but cut somewhere, like a number line whose ends are far
                apart — what the 'normalized' oracle encoding builds. Correlates ~0.72 with
                `circular` by construction, so its bar largely tracks circular's; reach for
                it when wrap-vs-no-wrap is the actual question.

The oracles do NOT calibrate this in Phase 3a, despite the temptation to treat them as
known-circular / known-categorical references. `reconfigure_for_prediction` sets
`what_latent_to_use='self'` for every condition, so in the test phase no oracle uses its
designed encoding — all conditions self-infer Z and differ only in their weights. Measured:
'Oracle Z (one-hot)' returns a near-perfect *circular* fit there. With no calibration
available, lean on the noise ceiling as the upper reference and on `periodic` (run once,
via ALL_CANDIDATES) as the lower one.

Caveat on reading a circular result: θ is a circular variable and Z is found by smooth
gradient descent on a loss that is smooth in θ, so a closed manifold is close to inevitable
regardless of what the model "chose" to encode. The informative question is whether the
ring is metrically *uniform* (equal arc per degree) or warped toward the trained angles —
the latter is the categorical-perception signature.

Entry points:
    geometry_across_models(results)     → RSA fits per model
    plot_candidate_rdms(...)            → what each hypothesis predicts (start here)
    plot_rdm_fits(...)                  → which hypothesis wins, vs the noise ceiling
    plot_geometry_embedding(...)        → 2-D map of the code: ring? triangle? clusters?

Run `python rotation_geometry_analysis.py` for the synthetic self-test.
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from plot_style import FigSize, get_model_color
from rotation_decoding_analysis import (
    _agg,
    as_runs,
    extract_decode_samples,
)

#: Hypotheses reported by default, in report order.
CANDIDATES = ('circular', 'blend', 'nearest')

#: Available but off by default — pass `candidates=ALL_CANDIDATES` to include them.
#: `periodic` is the negative control: it lands at ~0 whenever the model tracks the
#: colour→location mapping rather than the bare target constellation, so it is worth
#: running once per new task configuration to confirm the analysis can return "no".
#: `seamed` correlates ~0.72 with `circular` by construction, so its bar largely tracks
#: circular's and adds little on its own; reach for it when wrap-vs-no-wrap is the actual
#: question (e.g. checking a 1-D latent for a discontinuity).
OPTIONAL_CANDIDATES = ('periodic', 'seamed')

ALL_CANDIDATES = CANDIDATES + OPTIONAL_CANDIDATES

_CANDIDATE_LABELS = {
    'circular':  'circular (wraps)',
    'blend':     'blend of trained',
    'nearest':   'nearest trained',
    'periodic':  'constellation (360/n_colors)',
    'seamed':    'seamed (no wrap)',
}


# ──────────────────────────────────────────────────────────────────────────────
# Angle geometry
# ──────────────────────────────────────────────────────────────────────────────

def circular_distance_deg(a, b, period=360.0):
    """Shortest distance between two angles in degrees, wrapped to [0, period/2]."""
    d = np.abs(np.asarray(a, float) - np.asarray(b, float)) % period
    return np.minimum(d, period - d)


def _blend_weights(angles, train_rotations, width_deg=60.0):
    """
    Soft assignment of each angle to the trained contexts — the 'mixing paint' code.

    Weights fall off with circular distance to each trained angle (a von Mises-style
    kernel) and sum to 1, so each angle is a convex mixture of trained contexts.
    `width_deg` sets how sharply: small → approaches `nearest`, large → approaches uniform.
    """
    kappa = (180.0 / max(width_deg, 1e-6)) ** 2 / 2.0
    d = circular_distance_deg(np.asarray(angles)[:, None], np.asarray(train_rotations)[None, :])
    logits = kappa * np.cos(np.radians(d))
    w = np.exp(logits - logits.max(axis=1, keepdims=True))
    return w / w.sum(axis=1, keepdims=True)


def candidate_rdms(angles, train_rotations, n_colors=5, blend_width_deg=60.0,
                   seam_deg=0.0):
    """
    Predicted dissimilarity matrix for each hypothesis, over the given angles.

    Only the *rank order* of entries matters — the fits below are rank correlations — so
    the arbitrary scale of each matrix is irrelevant.
    """
    angles = np.asarray(angles, float)
    A, B = angles[:, None], angles[None, :]
    out = {}

    # Smooth and wrapping.
    out['circular'] = circular_distance_deg(A, B)

    # Mixture over trained contexts: distance between mixing weights.
    w = _blend_weights(angles, train_rotations, blend_width_deg)
    out['blend'] = squareform(pdist(w, metric='euclidean'))

    # Categorical: same trained basin or not.
    nearest = np.argmin(
        circular_distance_deg(angles[:, None], np.asarray(train_rotations)[None, :]), axis=1)
    out['nearest'] = (nearest[:, None] != nearest[None, :]).astype(float)

    # Constellation symmetry — period set by n_colors, not by the trained rotations.
    period = 360.0 / max(int(n_colors), 1)
    out['periodic'] = circular_distance_deg(A, B, period=period)

    # Smooth but cut at `seam_deg`: unwrapped distance along the cut line.
    lin = (angles - seam_deg) % 360.0
    out['seamed'] = np.abs(lin[:, None] - lin[None, :])

    return out


def candidate_confusability(angles, train_rotations, n_colors=5, candidates=CANDIDATES,
                            blend_width_deg=60.0):
    """
    How similar the hypotheses are to *each other*, given this angle design.

    The candidates are not orthogonal — with trained rotations spread around the circle,
    'circular' and 'blend' can correlate above 0.7, so a winning margin smaller than that
    means little. Check this before reading a fit table, and note it is a design lever:
    which rotations you train on changes how separable the hypotheses are, so a set that
    lowers these correlations makes the experiment sharper.

    Returns (matrix, candidate_names) of pairwise Spearman correlations.
    """
    cand = candidate_rdms(angles, train_rotations, n_colors, blend_width_deg)
    names = list(candidates)
    M = np.eye(len(names))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            M[i, j] = M[j, i] = rdm_fit(cand[names[i]], cand[names[j]])
    return M, names


def print_confusability(angles, train_rotations, n_colors=5, candidates=CANDIDATES):
    """Print the hypothesis-vs-hypothesis correlation matrix for this angle design."""
    M, names = candidate_confusability(angles, train_rotations, n_colors, candidates)
    w = max(len(n) for n in names) + 2
    print(f'\n── Hypothesis confusability (trained {list(train_rotations)}, '
          f'n_colors={n_colors}) ──')
    print(' ' * w + ''.join(f'{n[:9]:>11}' for n in names))
    for i, n in enumerate(names):
        print(f'{n:<{w}}' + ''.join(f'{M[i, j]:>11.2f}' for j in range(len(names))))
    off = M[np.triu_indices(len(names), k=1)]
    print(f'Max off-diagonal correlation = {np.max(off):.2f}. A winning fit needs to beat '
          'its rivals by more than this to mean much.')


# ──────────────────────────────────────────────────────────────────────────────
# Empirical RDMs
# ──────────────────────────────────────────────────────────────────────────────

def mean_pattern_by_angle(samples, key='Z', mask=None):
    """Average the representation within each rotation → (n_angles, dim) and the angles."""
    X = samples[key]
    ang = samples['angle_deg']
    if mask is not None:
        X, ang = X[mask], ang[mask]
    angles = np.unique(ang)
    patterns = np.stack([X[ang == a].mean(axis=0) for a in angles])
    return patterns, angles


def build_rdm(patterns, metric='euclidean'):
    """
    Pairwise dissimilarity between per-angle patterns.

    Features are z-scored first so that a few high-variance units cannot dominate — which
    matters when comparing a 3-dim Z against a 64-dim hidden state.
    """
    P = np.asarray(patterns, float)
    sd = P.std(axis=0, keepdims=True)
    P = (P - P.mean(axis=0, keepdims=True)) / np.where(sd < 1e-12, 1.0, sd)
    return squareform(pdist(P, metric=metric))


def _upper(rdm):
    """Upper triangle (excluding the diagonal) as a vector."""
    iu = np.triu_indices(rdm.shape[0], k=1)
    return np.asarray(rdm)[iu]


def classical_mds(rdm, n_components=2):
    """
    Classical (Torgerson) MDS — coordinates whose distances approximate the RDM.

    Done directly rather than via sklearn.manifold.MDS: this is deterministic (no random
    restarts, so the picture does not change between runs) and free of the iterative
    solver's version-to-version default changes.
    """
    D = np.asarray(rdm, float)
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    vals, vecs = np.linalg.eigh(B)
    order = np.argsort(vals)[::-1][:n_components]
    return vecs[:, order] * np.sqrt(np.clip(vals[order], 0, None))


def rdm_fit(empirical, candidate):
    """Spearman correlation between two RDMs — rank-based, so scale is irrelevant."""
    r, _ = spearmanr(_upper(empirical), _upper(candidate))
    return float(r) if np.isfinite(r) else np.nan


def _split_halves(samples):
    """
    Two disjoint halves of the samples for a split-half reliability estimate.

    Split by state-block where an angle has more than one, otherwise by mini-block parity.
    Splitting at random would let temporally adjacent (highly correlated) timesteps land on
    both sides and inflate the ceiling.
    """
    ang, blocks = samples['angle_deg'], samples['block_idx']
    mb = samples['miniblock_since_switch']
    first = np.zeros(len(ang), dtype=bool)
    for a in np.unique(ang):
        sel = np.flatnonzero(ang == a)
        b = blocks[sel]
        uniq = np.unique(b)
        if len(uniq) >= 2:
            keep = np.isin(b, uniq[::2])
        else:
            keep = (mb[sel] % 2) == 0
        first[sel[keep]] = True
    return first, ~first


def noise_ceiling(samples, key='Z', metric='euclidean'):
    """
    Split-half reliability of the empirical RDM — the most any hypothesis could score.

    A candidate correlating well below this is genuinely the wrong shape; a candidate near
    it explains everything the data can support.
    """
    h1, h2 = _split_halves(samples)
    if h1.sum() < 2 or h2.sum() < 2:
        return np.nan
    p1, a1 = mean_pattern_by_angle(samples, key, mask=h1)
    p2, a2 = mean_pattern_by_angle(samples, key, mask=h2)
    common = np.intersect1d(a1, a2)
    if len(common) < 4:
        return np.nan
    r1 = build_rdm(p1[np.isin(a1, common)], metric)
    r2 = build_rdm(p2[np.isin(a2, common)], metric)
    # Spearman-Brown: the ceiling for the full dataset, not for one half of it.
    r = rdm_fit(r1, r2)
    return float(2 * r / (1 + r)) if np.isfinite(r) and r > -1 else np.nan


# ──────────────────────────────────────────────────────────────────────────────
# Per-run / per-model analysis
# ──────────────────────────────────────────────────────────────────────────────

def analyze_rotation_geometry(samples, config, key='Z', metric='euclidean',
                              candidates=CANDIDATES, blend_width_deg=60.0):
    """RSA for one representation of one run. Returns fits, RDMs and the noise ceiling."""
    patterns, angles = mean_pattern_by_angle(samples, key)
    emp = build_rdm(patterns, metric)
    cand = candidate_rdms(angles, config.train_rotations,
                          n_colors=getattr(config, 'n_colors', 5),
                          blend_width_deg=blend_width_deg)
    return dict(
        key=key,
        angles=angles,
        patterns=patterns,
        rdm=emp,
        candidate_rdms={c: cand[c] for c in candidates},
        fits={c: rdm_fit(emp, cand[c]) for c in candidates},
        noise_ceiling=noise_ceiling(samples, key, metric),
        train_rotations=list(config.train_rotations),
        n_angles=len(angles),
    )


def geometry_for_runs(loggers, configs=None, keys=('Z', 'H'), phase='phase3a',
                      frames='all', candidates=CANDIDATES, **kw):
    """
    RSA over one or many runs (seeds) of the same model.

    As with the decoding analysis, patterns are never pooled across seeds — RDMs are built
    per run and only the *fits* are averaged.
    """
    runs = as_runs(loggers, configs)
    per_seed = []
    for lg, cf in runs:
        samples = extract_decode_samples(lg, cf, phase=phase, frames=frames)
        if samples is None:
            continue
        per_seed.append({k: analyze_rotation_geometry(samples, cf, key=k,
                                                      candidates=candidates, **kw)
                         for k in keys})
    if not per_seed:
        return None

    mean, sem, ceil = {}, {}, {}
    for k in keys:
        vals = np.array([[s[k]['fits'][c] for c in candidates] for s in per_seed])
        m, e = _agg(vals)
        mean[k] = dict(zip(candidates, m))
        sem[k] = dict(zip(candidates, e))
        ceil[k] = float(np.nanmean([s[k]['noise_ceiling'] for s in per_seed]))

    return dict(per_seed=per_seed, mean=mean, sem=sem, noise_ceiling=ceil,
                keys=tuple(keys), candidates=tuple(candidates),
                n_seeds=len(per_seed), phase=phase, frames=frames)


def geometry_across_models(results, verbose=True, **kw):
    """RSA for every model in a run script's `results` dict."""
    from rotation_decoding_analysis import _runs_from_result_entry

    out = {}
    for label, entry in results.items():
        loggers, configs = _runs_from_result_entry(entry)
        res = geometry_for_runs(loggers, configs, **kw)
        if res is None:
            continue
        out[label] = res
        if verbose:
            k = res['keys'][0]
            best = max(res['candidates'], key=lambda c: (res['mean'][k][c]
                                                         if np.isfinite(res['mean'][k][c])
                                                         else -np.inf))
            print(f'  {label}: {k} best fit = {_CANDIDATE_LABELS[best]} '
                  f'(rho={res["mean"][k][best]:.2f}, ceiling={res["noise_ceiling"][k]:.2f})')
    return out


def summarize_geometry(geometry_results, key='Z'):
    """Print the RSA fit table: model × hypothesis → Spearman rho."""
    labels = list(geometry_results.keys())
    if not labels:
        print('No geometry results.')
        return
    cands = geometry_results[labels[0]]['candidates']
    w = max(16, max(len(l) for l in labels) + 2)

    print(f'\n── Rotation code geometry (RSA, {key}) ──')
    print(f'{"Model":<{w}}' + ''.join(f'{c[:11]:>13}' for c in cands) + f'{"ceiling":>10}')
    for label in labels:
        res = geometry_results[label]
        row = ''.join(f'{res["mean"][key][c]:>13.2f}' for c in cands)
        print(f'{label:<{w}}{row}{res["noise_ceiling"][key]:>10.2f}')
    print('Spearman rho between the empirical RDM and each hypothesis; '
          'ceiling = split-half reliability.')


# ──────────────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────────────

def plot_candidate_rdms(angles, train_rotations, n_colors=5, candidates=CANDIDATES,
                        blend_width_deg=60.0, figsize=None):
    """
    What each hypothesis *predicts*, before looking at any data.

    Read this first: each panel is a matrix over angle pairs, dark = similar states, light
    = different. The characteristic looks are a smooth wrapping gradient (circular), blocky
    plateaus (nearest), a repeating fine grid (periodic), and a single bright corner-to-
    corner ramp with no wraparound (seamed).
    """
    cand = candidate_rdms(angles, train_rotations, n_colors, blend_width_deg)
    n = len(candidates)
    fig, axes = plt.subplots(1, n, figsize=figsize or FigSize.custom(0.85 * n + 0.35, 1.15),
                             squeeze=False)
    for i, c in enumerate(candidates):
        ax = axes[0, i]
        ax.imshow(cand[c], cmap='viridis', origin='lower')
        ax.set_title(_CANDIDATE_LABELS[c].split(' (')[0])
        ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_ylabel('rotation')
    fig.tight_layout()
    return fig


def plot_rdm_fits(geometry_results, key='Z', figsize=None):
    """
    How well each hypothesis explains the measured code, per model.

    The dashed line is the split-half noise ceiling: a bar near it explains everything the
    data supports, and a bar far below it is genuinely the wrong shape.
    """
    labels = list(geometry_results.keys())
    if not labels:
        raise ValueError('No models to plot.')
    cands = geometry_results[labels[0]]['candidates']
    n, c = len(labels), len(cands)

    fig, ax = plt.subplots(
        1, 1, figsize=figsize or FigSize.custom(max(1.6, 0.20 * n * c + 0.6), 1.25))
    x = np.arange(n)
    w = 0.85 / c
    shades = np.linspace(1.0, 0.35, c)

    for ci, cand in enumerate(cands):
        off = (ci - (c - 1) / 2) * w
        vals = [geometry_results[l]['mean'][key][cand] for l in labels]
        errs = [geometry_results[l]['sem'][key][cand] for l in labels]
        n_seeds = geometry_results[labels[0]]['n_seeds']
        ax.bar(x + off, vals, width=w, alpha=shades[ci],
               color=[get_model_color(l) for l in labels],
               yerr=errs if n_seeds > 1 else None, capsize=1.2, error_kw=dict(lw=0.5))

    for xi, l in enumerate(labels):
        nc_val = geometry_results[l]['noise_ceiling'][key]
        if np.isfinite(nc_val):
            ax.plot([xi - 0.45, xi + 0.45], [nc_val, nc_val], ls='--', lw=0.7, color='0.3')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel('RDM fit (rho)')
    ax.set_xlim(-0.6, n - 0.4)
    ax.axhline(0, color='0.7', lw=0.5)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [Patch(facecolor='0.35', alpha=shades[i],
                     label=_CANDIDATE_LABELS[c_].split(' (')[0])
               for i, c_ in enumerate(cands)]
    handles.append(Line2D([], [], ls='--', lw=0.7, color='0.3', label='noise ceiling'))
    ax.legend(handles=handles, frameon=False, loc='lower left',
              bbox_to_anchor=(0.0, 1.0, 1.0, 0.1), mode='expand', ncol=3,
              handlelength=0.9, borderpad=0.15, labelspacing=0.2,
              handletextpad=0.4, columnspacing=0.8)
    fig.tight_layout()
    return fig


def plot_geometry_embedding(geometry_results, key='Z', seed_index=0, figsize=None):
    """
    A 2-D map of the code — the picture that makes the hypotheses concrete.

    Each point is one rotation, placed so that distances match the empirical RDM, and
    coloured by angle on a circular colormap. A ring that runs smoothly through the colour
    wheel is a circular code; a triangle with points bunched at its corners is a blend over
    trained contexts; tight separate clumps are a categorical code. Trained rotations are
    ringed in black.
    """
    labels = list(geometry_results.keys())
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=figsize or FigSize.custom(1.05 * n + 0.4, 1.3),
                             squeeze=False)
    cmap = plt.get_cmap('hsv')

    for ci, label in enumerate(labels):
        ax = axes[0, ci]
        res = geometry_results[label]['per_seed'][seed_index][key]
        rdm, angles = res['rdm'], res['angles']
        xy = classical_mds(rdm, n_components=2)
        ax.plot(xy[:, 0], xy[:, 1], '-', lw=0.5, color='0.75', zorder=1)
        ax.scatter(xy[:, 0], xy[:, 1], s=9, c=[cmap(a / 360.0) for a in angles], zorder=2)
        trained = [i for i, a in enumerate(angles)
                   if circular_distance_deg(a, np.array(res['train_rotations'])).min() < 1e-3]
        if trained:
            ax.scatter(xy[trained, 0], xy[trained, 1], s=26, facecolors='none',
                       edgecolors='k', linewidths=0.6, zorder=3)
        ax.set_title(label)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect('equal', adjustable='datalim')
    axes[0, 0].set_ylabel(f'MDS of {key}')
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────────────

def self_test(verbose=True):
    """
    Build representations with a known code and check RSA recovers the right hypothesis.

    If a synthetic ring does not score highest on 'circular', or a synthetic 3-slot code on
    'nearest', the comparison is broken and nothing downstream can be trusted.
    """
    rng = np.random.default_rng(0)
    train = [0.0, 120.0, 210.0]
    angles = np.arange(0, 360, 15.0)
    per_angle = 60

    ang = np.repeat(angles, per_angle)
    th = np.radians(ang)
    blocks = np.repeat(np.arange(len(angles)), per_angle)
    mb = np.tile(np.arange(per_angle), len(angles))

    def _s(X):
        return dict(Z=X, H=X, angle_deg=ang, block_idx=blocks,
                    miniblock_since_switch=mb, theta_rad=th)

    class _Cfg:
        train_rotations = train
        n_colors = 5

    ring = np.stack([np.cos(th), np.sin(th)], 1) + 0.03 * rng.standard_normal((len(th), 2))
    nearest_idx = np.argmin(circular_distance_deg(ang[:, None], np.array(train)[None, :]), 1)
    onehot = np.eye(3)[nearest_idx] + 0.03 * rng.standard_normal((len(th), 3))
    blend = _blend_weights(ang, train, 60.0) + 0.03 * rng.standard_normal((len(th), 3))

    results = {}
    for name, X in [('circular', ring), ('nearest', onehot), ('blend', blend)]:
        # Exercise the optional candidates too, so they cannot rot while off by default.
        r = analyze_rotation_geometry(_s(X), _Cfg(), key='Z', candidates=ALL_CANDIDATES)
        best = max(r['fits'], key=lambda c: r['fits'][c])
        results[name] = (best, r['fits'], r['noise_ceiling'])

    checks = [(f'synthetic {k} code recovered as {k!r}', v[0] == k)
              for k, v in results.items()]
    checks.append(('noise ceiling is high for clean data',
                   all(v[2] > 0.8 for v in results.values())))

    if verbose:
        print('── rotation_geometry_analysis self-test ──')
        for name, ok in checks:
            print(f'  [{"ok" if ok else "FAIL"}] {name}')
        for k, (best, fits, nc) in results.items():
            fit_txt = '  '.join(f'{c}={fits[c]:.2f}' for c in ALL_CANDIDATES)
            print(f'  {k:<9} best={best:<9} ceiling={nc:.2f}   {fit_txt}')
    return all(ok for _, ok in checks)


if __name__ == '__main__':
    raise SystemExit(0 if self_test() else 1)
