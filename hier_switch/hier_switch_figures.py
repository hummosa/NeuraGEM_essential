"""
hier_switch_figures.py — the paper panels, built once and drawn by whoever asks.

Each panel is a `spec_*` builder that takes **a list of per-seed reports** (what
hier_switch_group.session_report writes) and returns a spec dict; `draw(ax, spec)` renders
it. A single session is a list of one, so the group figure and the single-session figure
are literally the same panel with a different number of replicates — the flanker lesson,
and the reason nothing here takes a "which script am I" flag.

    .venv/bin/python hier_switch/hier_switch_figures.py                  # the group figures
    .venv/bin/python hier_switch/hier_switch_figures.py <session dir>    # one session

Conventions (docs/figure_style.md): sizes only through `FigSize`, model colours through
`plot_style.get_model_color`, one panel per question with one line per model — never a
panel per model — and the seed is the unit of analysis, so every mean carries its seeds as
dots and its SEM as a whisker or band.
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import plot_style
from plot_style import FigSize, get_model_color, outcome_color, outcome_line

from hier_switch_group import EXPORTS, collect, session_report

plot_style.set_plot_style()

# Conditions get a line style, models a colour (never a panel each).
STYLES = {'softmax_rc_none': ('-', 'NeuraGEM'), 'softmax_rc_low': ('-', 'low conflict'),
          'softmax_rc_high': ('--', 'high conflict'),
          'rnn_rc_none': ('-', 'RNN'), 'rnn_rc_low': ('-', 'RNN, low'),
          'rnn_rc_high': ('--', 'RNN, high'),
          'sigmoid_zlr30000': (':', 'sigmoid')}
COL_OBS = plot_style.get_model_color('bayesian')      # the ideal observer
COL_NG = get_model_color('neuragem')
COL_RNN = get_model_color('rnn')
CONFLICT = [0.0, 0.125, 0.286, 0.5, 0.8]
#: Error and correct carry a hue and a marker of their own (plot_style.outcome_line), not
#: only a dash — a dashed line is unreadable in a paper-sized legend.
OUTCOMES = ((True, 'correct'), (False, 'error'))
#: Hue is the model, shade is the early-reversal conflict: lighter = low, full = high.
LOW_SHADE = 0.45


def shade(color, f=LOW_SHADE):
    """`color` mixed f of the way to white: the lighter member of a condition pair."""
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def split_style(label, split):
    """Colour and line style for one arm of a low/high early-conflict pair."""
    col = get_model_color(label)
    return (shade(col), '-') if split == 'low' else (col, '--')


# ── Reading reports ───────────────────────────────────────────────────────────

def _get(rep, path, default=np.nan):
    """rep['a']['b'][2] as _get(rep, 'a.b.2')."""
    cur = rep
    for part in path.split('.'):
        try:
            cur = cur[int(part)] if part.lstrip('-').isdigit() else cur[part]
        except (KeyError, IndexError, TypeError):
            return default
    return default if cur is None else cur


def stack(reports, path, default=np.nan):
    """(n_seeds, ...) array of one field across seeds, NaN where a seed lacks it."""
    vals = [np.atleast_1d(np.asarray(_get(r, path, default), dtype=float)) for r in reports]
    n = max(len(v) for v in vals)
    out = np.full((len(vals), n), np.nan)
    for i, v in enumerate(vals):
        out[i, :len(v)] = v
    return out[:, 0] if n == 1 else out


# ── Primitives ────────────────────────────────────────────────────────────────

def band(ax, x, arr, label=None, color='k', ls='-'):
    """Mean ± SEM across seeds, arr = (n_seeds, n_points)."""
    arr = np.asarray(arr, dtype=float)
    n = np.sum(np.isfinite(arr), axis=0)
    mu = np.nanmean(arr, axis=0)
    se = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    ax.plot(x, mu, color=color, linestyle=ls, linewidth=1.0, label=label)
    if arr.shape[0] > 1:
        ax.fill_between(x, mu - se, mu + se, color=color, alpha=0.18, linewidth=0)
    return mu


def series(ax, x, arr, label=None, color='k', ls='-', marker='o', dots=True, **kw):
    """Mean ± SEM across seeds at each level, with one faint line per seed.

    Extra kwargs (markerfacecolor, markeredgecolor, …) pass through, so an outcome series
    can be styled with `plot_style.outcome_line(correct)` in one call.
    """
    # outcome_line() hands over colour, marker and line style in one dict, so accept those
    # names as overrides rather than colliding with the positional defaults.
    ls = kw.pop('linestyle', ls)
    marker = kw.pop('marker', marker)
    color = kw.pop('color', color)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    mu = np.nanmean(arr, axis=0)
    n = np.sum(np.isfinite(arr), axis=0)
    se = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    if dots and arr.shape[0] > 1:
        for row in arr:
            ax.plot(x, row, color=color, alpha=0.18, linewidth=0.4, zorder=1)
    kw.setdefault('linewidth', 1.0)
    kw.setdefault('markersize', 2.5)
    ax.errorbar(x, mu, yerr=se, marker=marker, color=color, linestyle=ls,
                capsize=1.5, label=label, zorder=3, **kw)


def bars(ax, groups, ylabel=None, baseline=None, connect=True, rotation=0, gap_after=(),
         gap=0.6, clusters=None):
    """groups = [(values_per_seed, label, color)]; bars are means, dots are seeds.

    `gap_after` holds indices after which an extra gap is inserted, so a panel can be
    blocked into clusters (criterion, condition pair) instead of one long row of bars.
    `clusters` is [(label, first index, last index)] and puts one tick label centred under
    each cluster instead of one per bar — the fix for pair labels colliding.
    """
    x = np.arange(len(groups), dtype=float)
    for i in gap_after:
        x[i + 1:] += gap
    vals = [np.asarray(v, dtype=float).ravel() for v, _, _ in groups]
    mu = [np.nanmean(v) for v in vals]
    se = [np.nanstd(v, ddof=1) / np.sqrt(max(np.sum(np.isfinite(v)), 1))
          if np.sum(np.isfinite(v)) > 1 else 0.0 for v in vals]
    for xi, m, s, (_, _, c) in zip(x, mu, se, groups):
        ax.bar(xi, m, width=0.6, color=c, alpha=0.75, zorder=2)
        ax.errorbar(xi, m, yerr=s, color='k', capsize=2, linewidth=0.8, zorder=3)
    n = min(len(v) for v in vals)
    # One dot per seed. The joining line is the within-seed contrast, so it only means
    # something with more than one seed; with a single session it would imply a trend
    # across unrelated bars.
    for i in range(n):
        y = [v[i] for v in vals]
        ax.plot(x, y, color='0.25', alpha=0.45, linewidth=0.4, marker='o', markersize=1.8,
                zorder=4, linestyle='-' if (connect and n > 1) else 'none')
    if baseline is not None:
        ax.axhline(baseline, color='k', linewidth=0.5, alpha=0.4)
    if clusters:
        ax.set_xticks([x[a:b + 1].mean() for _, a, b in clusters])
        ax.set_xticklabels([l for l, _, _ in clusters], rotation=rotation,
                           ha='right' if rotation else 'center')
    else:
        ax.set_xticks(x)
        ax.set_xticklabels([l for _, l, _ in groups], rotation=rotation,
                           ha='right' if rotation else 'center')
    if ylabel:
        ax.set_ylabel(ylabel)


def legend(ax, **kw):
    kw.setdefault('frameon', False)
    # Long enough that a dashed handle actually shows its dashes: the short 1.2 default made
    # solid and dashed series indistinguishable in the legend.
    kw.setdefault('handlelength', 2.4)
    kw.setdefault('borderpad', 0.2)
    kw.setdefault('labelspacing', 0.25)
    ax.legend(**kw)


# ── Panels ────────────────────────────────────────────────────────────────────

def spec_psychometric(groups):
    """P1: accuracy against cue conflict, steady state, with the observer's ceiling.

    groups: {label: reports}. The ideal observer's per-level cue accuracy is taken from the
    first group's reports — it is a property of the task, not of the model.
    """
    def panel(ax):
        for label, reps in groups.items():
            series(ax, CONFLICT, stack(reps, 'behaviour.psychometric.acc'),
                   label=label, color=get_model_color(label))
        first = next(iter(groups.values()))
        series(ax, CONFLICT, stack(first, 'observer.p_c'), label='ideal observer',
               color=COL_OBS, ls='--', marker='s', dots=False)
        ax.axhline(0.5, color='k', linewidth=0.5, alpha=0.3)
        ax.set_xlabel('Cue conflict')
        ax.set_ylabel('Accuracy (steady state)')
        ax.set_ylim(0.4, 1.02)
        legend(ax, loc='lower left')
    return panel


def spec_reversal(groups, key='acc', ylabel='Accuracy'):
    """P2/P3: a measure from 5 trials before a reversal to 15 after, low vs high early
    conflict. `key` is any curve behaviour() returns: acc, z_side, z_evidence, rt,
    undecided, abs_dec."""
    def panel(ax):
        for label, reps in groups.items():
            k = np.asarray(_get(reps[0], 'behaviour.reversal.all.k'), dtype=float)
            for split in ('low', 'high'):
                arr = stack(reps, f'behaviour.reversal.{split}.{key}')
                if np.isfinite(arr).any():
                    col, ls = split_style(label, split)
                    band(ax, k, arr, label=f'{split} early conflict', color=col, ls=ls)
        ax.axvline(0.5, color='k', linewidth=0.5, alpha=0.4)
        if key == 'acc':
            ax.axhline(0.5, color='k', linewidth=0.5, alpha=0.3)
        ax.set_xlabel('Trials from reversal')
        ax.set_ylabel(ylabel)
        legend(ax, loc='lower right')
    return panel


#: The three switch criteria, in the order the figures show them.
CRITERIA = (('switch', 'behaviour'), ('z_switch', 'Z side'), ('dec_switch', 'decided'))


def _switch_arms(groups, criteria=None):
    """[(values_per_seed, criterion label, split, colour)] for the model and the observer.

    `criteria` selects a subset of CRITERIA — the story figure shows two (the paper's
    behavioural one on decided trials, and the latent one), the supplement keeps all three.
    """
    arms = []
    for label, reps in groups.items():
        for crit, name in (criteria or CRITERIA):
            for split in ('low', 'high'):
                arms.append((stack(reps, f'behaviour.switch.{split}.{crit}'), name, split,
                             split_style(label, split)[0]))
    first = next(iter(groups.values()))
    for split in ('low', 'high'):
        # merge_splits carries the observer's latency from each forced session; an unforced
        # session only has the pooled one.
        arr = stack(first, f'observer.switch_{split}')
        arms.append((arr if np.isfinite(arr).any() else stack(first, 'observer.switch'),
                     'observer', split, shade(COL_OBS) if split == 'low' else COL_OBS))
    return arms


def _split_legend(ax, loc='upper left'):
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor='0.75', label='low early conflict'),
                       Patch(facecolor='0.35', label='high early conflict')],
              frameon=False, handlelength=1.0, borderpad=0.2, labelspacing=0.25, loc=loc)


def spec_switch(groups, criteria=None):
    """How long the network stays on the old rule, by the conflict of the first five trials.

    Four clusters — the paper's behavioural criterion, the latent one (Z back on the true
    context), the behavioural one restricted to decided trials, and the ideal observer on the
    same trials — each a low/high pair. Shade is the early conflict, hue is the model.
    """
    def panel(ax):
        arms = _switch_arms(groups, criteria)
        g = [(v, '', col) for v, _, _, col in arms]
        # One tick label per low/high pair, centred under it.
        clusters = [(arms[i][1], i, i + 1) for i in range(0, len(arms), 2)]
        bars(ax, g, ylabel='Trials to switch', connect=False, clusters=clusters,
             gap_after=list(range(1, len(g) - 1, 2)))
        ax.tick_params(axis='x', length=0)
        _split_legend(ax)
    return panel


def spec_switch_cost(groups, criteria=None):
    """The same thing as one number per criterion: how many extra trials a high-conflict
    start costs, paired within seed. Positive = slower after an ambiguous start."""
    def panel(ax):
        arms = _switch_arms(groups, criteria)
        g = []
        for i in range(0, len(arms), 2):
            low, name, _, _ = arms[i]
            high = arms[i + 1][0]
            col = COL_OBS if name == 'observer' else COL_NG
            g.append((np.asarray(high) - np.asarray(low), name, col))
        bars(ax, g, ylabel='Extra trials to switch\nafter a high-conflict start', baseline=0.0,
             connect=False)
        ax.tick_params(axis='x', length=0)
    return panel


def spec_z_belief(groups):
    """B1: Z on the context axis around a reversal, against the observer's belief."""
    def panel(ax):
        for label, reps in groups.items():
            k = np.asarray(_get(reps[0], 'latent.belief.k'), dtype=float)
            band(ax, k, stack(reps, 'latent.belief.z_curve'), label=f'{label}: Z',
                 color=get_model_color(label))
            band(ax, k, stack(reps, 'latent.belief.obs_curve'), label='ideal observer',
                 color=COL_OBS, ls='--')
        ax.axvline(0.5, color='k', linewidth=0.5, alpha=0.4)
        ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
        ax.set_xlabel('Trials from reversal')
        ax.set_ylabel('Evidence for the true context\n(±1 = a prototype)')
        legend(ax, loc='lower right')
    return panel


def spec_z_uncertainty(groups):
    """B2/P3: Z uncertainty after a reversal, low vs high early conflict (Fig 1k,l)."""
    def panel(ax):
        for label, reps in groups.items():
            k = np.asarray(_get(reps[0], 'latent.uncertainty.all.k'), dtype=float)
            for split in ('low', 'high'):
                arr = stack(reps, f'latent.uncertainty.{split}.curve')
                if np.isfinite(arr).any():
                    col, ls = split_style(label, split)
                    band(ax, k, arr, label=f'{split} early conflict', color=col, ls=ls)
        ax.axvline(0.5, color='k', linewidth=0.5, alpha=0.4)
        ax.set_xlabel('Trials from reversal')
        ax.set_ylabel('Z uncertainty\n(1 = the middle of the axis)')
        legend(ax, loc='upper right')
    return panel


def spec_z_update(reports, state='stale'):
    """B5: how far one trial moves Z toward the other context, by conflict and outcome.

    `s` is the fraction of the distance between the two context prototypes that this
    trial's error part of the update covered, signed toward the context Z was *not*
    holding. Stale = Z held the wrong context (mostly just after a reversal): the paper's
    credit-assignment split.
    """
    st = {'aligned': 0, 'stale': 1}[state]

    def panel(ax):
        for correct, name in OUTCOMES:
            err = 0 if correct else 1
            arr = np.full((len(reports), 5), np.nan)
            for i, r in enumerate(reports):
                for c in range(5):
                    arr[i, c] = _get(r, f'z_updates.{st}__{err}__{c}.s')
            series(ax, CONFLICT, arr, label=f'{state}, {name}', **outcome_line(correct))
        ax.axhline(0, color='k', linewidth=0.5, alpha=0.4)
        ax.set_xlabel('Cue conflict')
        ax.set_ylabel('Move to the other context\n(× prototype distance)')
        legend(ax, loc='upper right')
    return panel


def spec_tipped(reports):
    """B5: how often an error tips Z across the midpoint, by conflict."""
    def panel(ax):
        for st, name, ls, mk in ((1, 'Z stale', '-', 'o'), (0, 'Z aligned', ':', 's')):
            arr = np.full((len(reports), 5), np.nan)
            for i, r in enumerate(reports):
                for c in range(5):
                    arr[i, c] = _get(r, f'z_updates.{st}__1__{c}.tipped')
            series(ax, CONFLICT, arr, label=name, color=COL_NG, ls=ls, marker=mk)
        ax.set_xlabel('Cue conflict')
        ax.set_ylabel('P(the error tips Z across)')
        ax.set_ylim(-0.02, 1.02)
        legend(ax, loc='upper right')
    return panel


def spec_normative(reports):
    """B5: the model's update against the ideal observer's log-odds update, per cell."""
    def panel(ax):
        for correct, name in OUTCOMES:
            xs, ys = [], []
            for r in reports:
                for c in r.get('normative', {}).get('cells', []):
                    if bool(c['err']) != correct:
                        xs.append(c['dlogit'])
                        ys.append(c['s'])
            st = outcome_line(correct)
            ax.scatter(xs, ys, s=6, marker=st['marker'], color=st['color'],
                       linewidth=0.6, label=name)
        r_all = stack(reports, 'normative.r')
        ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
        ax.axvline(0, color='k', linewidth=0.5, alpha=0.3)
        ax.set_xlabel("Ideal observer's log-odds update\ntoward the context Z did not hold")
        ax.set_ylabel('Model: move toward it')
        ax.text(0.03, 0.95, f'r = {np.nanmean(r_all):.2f}', transform=ax.transAxes,
                va='top', ha='left')
        legend(ax, loc='lower right')
    return panel


def spec_eps_cw(reports):
    """B4: the pooled |dL/dZ| by conflict — the ACC-like conflict-weighted error."""
    def panel(ax):
        for correct, name in OUTCOMES:
            key = 'g_contrast_by_conf_correct' if correct else 'g_contrast_by_conf'
            series(ax, CONFLICT, stack(reports, f'latent.eps_cw.{key}'), label=name,
                   **outcome_line(correct))
        ax.set_xlabel('Cue conflict')
        ax.set_ylabel('|dL/dZ| along the context axis')
        legend(ax, loc='upper right')
    return panel


def spec_gain(by_condition, softmax_key='softmax_rc_none', sigmoid_key='sigmoid_zlr30000'):
    """The gain axis: what one trial does to how far *open* the gate is, not to which
    context it selects.

    Only a gate without the softmax has this direction at all — the softmax's two units sum
    to one, so their mean cannot move. The sigmoid-at-test sessions are what make it visible.
    """
    def panel(ax):
        sig = by_condition.get(sigmoid_key, [])
        sm = by_condition.get(softmax_key, [])
        for correct, name in OUTCOMES:
            err = 0 if correct else 1
            key = lambda r, c: _get(r, f'z_updates_gain.{err}__{c}.dgain')
            if sig:
                series(ax, CONFLICT, np.array([[key(r, c) for c in range(5)] for r in sig]),
                       label=f'sigmoid, {name}', **outcome_line(correct))
            if sm:
                series(ax, CONFLICT, np.array([[key(r, c) for c in range(5)] for r in sm]),
                       label=f'softmax, {name}', dots=False,
                       **dict(outcome_line(correct, filled=False), linestyle=':', linewidth=0.8))
        ax.axhline(0, color='k', linewidth=0.5, alpha=0.4)
        ax.set_xlabel('Cue conflict')
        ax.set_ylabel('Δ gain this trial\n(mean of the two Z units)')
        legend(ax, loc='lower left')
    return panel


def spec_gain_ladder(by_condition, keys=('sigmoid_zlr30000', 'sigmoid_zlr30000_wd1e-05',
                                         'sigmoid_zlr30000_wd3e-05'),
                     labels=('0.09', '0.3', '0.9')):
    """The leak in one number per setting: how much further down an error pushes the gain
    than a correct trial does, at the most ambiguous cue, for each weight-decay strength."""
    def panel(ax):
        g = []
        for key, lab in zip(keys, labels):
            reps = by_condition.get(key, [])
            if not reps:
                continue
            d = (stack(reps, 'z_updates_gain.1__4.dgain')
                 - stack(reps, 'z_updates_gain.0__4.dgain'))
            g.append((d, lab, outcome_color(False)))
        sm = by_condition.get('softmax_rc_none', [])
        if sm:
            g.append((stack(sm, 'z_updates_gain.1__4.dgain')
                      - stack(sm, 'z_updates_gain.0__4.dgain'), 'softmax', COL_NG))
        bars(ax, g, ylabel='Δ gain, error − correct', baseline=0.0, connect=False,
             gap_after=[len(g) - 2] if sm else ())
        ax.set_xlabel('Sigmoid gate: Z_lr × Z_decay')
    return panel


def spec_gain_cost(by_condition, keys=('softmax_rc_none', 'sigmoid_zlr30000',
                                       'sigmoid_zlr30000_wd1e-05', 'sigmoid_zlr30000_wd3e-05'),
                   labels=('softmax', '0.09', '0.3', '0.9')):
    """What the leak costs: steady-state accuracy of the same weights under each gate."""
    def panel(ax):
        g = [(stack(by_condition[k], 'behaviour.acc_steady'), lab,
              COL_NG if k.startswith('softmax') else outcome_color(False))
             for k, lab in zip(keys, labels) if by_condition.get(k)]
        bars(ax, g, ylabel='Accuracy (steady state)', baseline=0.5, connect=True,
             gap_after=[0])
        ax.set_ylim(0.4, 1.0)
        ax.set_xlabel('Gate (sigmoid: Z_lr × Z_decay)')
    return panel


def spec_rt(groups):
    """P6: RT against conflict, and the undecided rate next to it."""
    def panel(ax):
        for label, reps in groups.items():
            series(ax, CONFLICT, stack(reps, 'behaviour.psychometric.rt'), label=label,
                   color=get_model_color(label))
        ax.set_xlabel('Cue conflict')
        ax.set_ylabel('RT (timesteps)')
        legend(ax, loc='upper left')
    return panel


def spec_rt_reversal(groups):
    """P6: RT around a reversal — fast perseveration, then a peak where Z is uncertain."""
    def panel(ax):
        for label, reps in groups.items():
            arr = stack(reps, 'behaviour.rt.by_since')
            series(ax, np.arange(1, arr.shape[1] + 1), arr, label=label,
                   color=get_model_color(label), dots=False)
        ax.set_xlabel('Trials since reversal')
        ax.set_ylabel('RT (timesteps)')
        legend(ax, loc='upper right')
    return panel


def spec_decoding(reports, which='rule'):
    """C1/P5: cross-validated decoding of cue or rule from the hidden state, per timestep,
    in the steady state and over the first 5 trials after a reversal."""
    def panel(ax):
        for cond, ls, name in (('steady_aligned', '-', 'steady state'),
                               ('early', '--', 'first 5 after a reversal')):
            arr = stack(reports, f'hidden.decoding.{cond}.{which}.acc')
            band(ax, np.arange(arr.shape[1]), arr, label=name, color=COL_NG, ls=ls)
        ax.axhline(0.5, color='k', linewidth=0.5, alpha=0.3)
        ax.axvspan(1, 16, color='0.85', alpha=0.35, linewidth=0, zorder=0)
        ax.set_xlabel('Timestep (grey: the cue)')
        ax.set_ylabel(f'{which.capitalize()} decoding accuracy')
        ax.set_ylim(0.4, 1.02)
        legend(ax, loc='lower right')
    return panel


def spec_integration(reports):
    """The paper's integration index and cue velocity, by condition."""
    def panel(ax):
        g = [(stack(reports, f'hidden.integration.{c}.index'), n, COL_NG)
             for c, n in (('steady_aligned', 'steady'), ('early', 'first 5'),
                          ('stale', 'Z stale'), ('conf0', 'conflict 0'), ('conf4', 'conflict .8'))]
        bars(ax, g, ylabel='Integration index\n(late / early cue period)', rotation=45)
    return panel


def spec_unit_classes(groups):
    """C2: the paper's three PFC classes, as fractions of the tuned hidden units."""
    def panel(ax):
        from matplotlib.patches import Patch
        g = []
        for label, reps in groups.items():
            for cls in ('CueS', 'CueL', 'Rule'):
                g.append((stack(reps, f'hidden.unit_classes.frac_of_tuned.{cls}'),
                          cls, get_model_color(label)))
        bars(ax, g, ylabel='Fraction of tuned units', connect=False)
        ax.legend(handles=[Patch(facecolor=get_model_color(l), alpha=0.75, label=l)
                           for l in groups], frameon=False, handlelength=1.0,
                  borderpad=0.2, labelspacing=0.25, loc='upper left')
    return panel


def spec_clamp_grid(cells, measure='index', ylabel='Integration index'):
    """E2: one line per clamped gain level, the measure against the clamped contrast.

    `cells` pools every seed's grid, so each (gain, contrast) cell is a mean over seeds with
    its SEM and one faint dot per seed — the same convention as every other panel.
    """
    def value(c):
        return (_get(c, f'integration.{measure}') if measure in ('index', 'cue_velocity')
                else _get(c, measure))

    def panel(ax):
        ms = sorted({c['m'] for c in cells})
        ds = sorted({c['d'] for c in cells})
        cmap = plt.cm.viridis
        for i, m in enumerate(ms):
            col = cmap(0.15 + 0.7 * i / max(len(ms) - 1, 1))
            per = [[value(c) for c in cells if c['m'] == m and c['d'] == d] for d in ds]
            n = max(len(v) for v in per)
            arr = np.full((n, len(ds)), np.nan)
            for j, v in enumerate(per):
                arr[:len(v), j] = v
            series(ax, ds, arr, label=f'gain {m:+.1f}' if len(ms) > 1 else None, color=col,
                   dots=False)
            if arr.shape[0] > 1:
                for j, d in enumerate(ds):
                    ax.plot([d] * arr.shape[0], arr[:, j], '.', color=col, markersize=1.6,
                            alpha=0.45, zorder=1)
        ax.set_xlabel('Clamped contrast (z₀ − z₁)/2')
        ax.set_ylabel(ylabel)
        if len(ms) > 1:
            legend(ax, loc='best')
    return panel


def spec_clamp_gain(cells, d=1.0, measure='cue_velocity', ylabel='Cue velocity'):
    """The clamp grid's simple cut: contrast held at one committed value, gain varied.

    One line, x = the clamped gain (the mean of the two Z units), seeds as dots.
    """
    def value(c):
        return (_get(c, f'integration.{measure}') if measure in ('index', 'cue_velocity')
                else _get(c, measure))

    def panel(ax):
        rows = [c for c in cells if abs(c['d'] - d) < 1e-9]
        ms = sorted({c['m'] for c in rows})
        per = [[value(c) for c in rows if c['m'] == m] for m in ms]
        n = max(len(v) for v in per) if per else 0
        arr = np.full((n, len(ms)), np.nan)
        for j, v in enumerate(per):
            arr[:len(v), j] = v
        series(ax, ms, arr, color=COL_NG)
        for j, mm in enumerate(ms):
            ax.plot([mm] * arr.shape[0], arr[:, j], '.', color=COL_NG, markersize=1.8,
                    alpha=0.45, zorder=1)
        # The fixed contrast belongs in the caption, not on a paper-width axis label.
        ax.set_xlabel('Clamped gain (z₀ + z₁)/2')
        ax.set_ylabel(ylabel)
    return panel


# ── Figures ───────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
# The story figure's panels
# ══════════════════════════════════════════════════════════════════════════════

def nolegend(p):
    """The same panel without its legend.

    Twenty panels at paper size cannot each carry a key. The story figure keeps a legend
    only where it says something a reader cannot get from the caption, and drops the rest —
    shade and dash mean the same thing in every panel (light solid = low early conflict,
    dark dashed = high), so repeating that key twenty times costs space and buys nothing.
    """
    def panel(ax):
        p(ax)
        lg = ax.get_legend()
        if lg is not None:
            lg.remove()
    return panel


def rotate_xticks(p, deg=30):
    """The same panel with its x tick labels rotated — for a narrow panel whose cluster
    labels would otherwise run into each other."""
    def panel(ax):
        p(ax)
        for t in ax.get_xticklabels():
            t.set_rotation(deg)
            t.set_ha('right')
    return panel


def relegend(p, **kw):
    """The same panel with its legend redrawn — a smaller font, or somewhere else."""
    def panel(ax):
        p(ax)
        h, l = ax.get_legend_handles_labels()
        lg = ax.get_legend()
        if lg is not None:
            h = h or lg.legend_handles
            l = l or [t.get_text() for t in lg.get_texts()]
            lg.remove()
        if h:
            kw.setdefault('frameon', False)
            kw.setdefault('handlelength', 1.4)
            kw.setdefault('borderpad', 0.15)
            kw.setdefault('labelspacing', 0.2)
            kw.setdefault('fontsize', 'small')
            ax.legend(h, l, **kw)
    return panel


#: The signals a task variable can be decoded from, in the order every panel shows them.
#: `hidden_pc2` is the dimension-matched control for `hidden_t16` and is drawn hollow, in
#: the same hue, because it is the same signal with 62 dimensions taken away.
SOURCES = (('hidden_t16', 'hidden state', 'hidden state', False),
           ('hidden_pc2', 'hidden state', 'hidden, 2 PCs', True),
           ('z_in', 'latent z', 'Z', False),
           ('step', 'z update', 'ΔZ', False),
           ('grad', 'z gradient', 'dL/dZ', False))
#: The subset the story figure's decoding panel shows — four bars per cluster is what fits.
#: ΔZ is dropped there because the whole of row 5 is about it; it stays in the variance
#: panel, in the json and in the single-session figure.
STORY_SOURCES = tuple(x for x in SOURCES if x[0] != 'step')
#: The variables each source is tested against, and how they are labelled on an axis.
MATRIX_VARS = (('cue', 'cue'), ('rule', 'rule'), ('context', 'context'),
               ('conflict', 'conflict'))
#: The full set, including the paper's two uncertainties, for the variance panel.
VARIANCE_VARS = MATRIX_VARS + (('outcome', 'outcome'),
                               ('rule_uncertainty', 'rule uncert.'),
                               ('cue_uncertainty', 'cue uncert.'))


def _source_color(key):
    return get_model_color(dict((k, c) for k, c, _, _ in SOURCES)[key])


def spec_psychometric_ctx(groups):
    """The paper's Fig 1e: accuracy against cue conflict, one curve per context.

    The two contexts are ordered relative to training — solid is the context the model saw
    last, dashed the other — because which one is labelled "0" is an accident of the data
    stream, while "the one the weights and Z were sitting in when training stopped" is a
    real asymmetry. The paper's claim is that the two coincide.
    """
    def panel(ax):
        main = next(iter(groups))          # the model whose two contexts are spelt out
        for label, reps in groups.items():
            col = get_model_color(label)
            for k, (name, ls) in enumerate((('last trained context', '-'),
                                            ('other context', '--'))):
                # The contrast this panel is about is within the main model, so only its two
                # curves are named; a baseline gets its own name once and no context split.
                key = name if label == main else (label if k == 0 else None)
                series(ax, CONFLICT, stack(reps, f'behaviour.psychometric.acc_ctx.{k}'),
                       label=key, color=col if k == 0 else shade(col), ls=ls,
                       dots=(label == main), marker='o' if k == 0 else 's')
        first = next(iter(groups.values()))
        series(ax, CONFLICT, stack(first, 'observer.p_c'), label='ideal observer',
               color=COL_OBS, ls=':', marker='', dots=False)
        ax.axhline(0.5, color='k', linewidth=0.5, alpha=0.3)
        ax.set_xlabel('Cue conflict')
        ax.set_ylabel('Accuracy (steady state)')
        ax.set_ylim(0.4, 1.02)
        legend(ax, loc='lower left')
    return panel


def spec_switch_vs_early_conflict(groups, criterion='dec_switch'):
    """The paper's Fig 1f on uncontrolled reversals: switch latency against how ambiguous
    the block's first five trials happened to be.

    The forced low/high sessions (spec_switch) are the controlled version and the stronger
    test. This one is the paper's own design: bin each seed's reversals by their own early
    conflict and see whether latency rises with it.
    """
    def panel(ax):
        for label, reps in groups.items():
            x = np.nanmean(stack(reps, 'behaviour.by_early_conf.conflict'), axis=0)
            series(ax, x, stack(reps, f'behaviour.by_early_conf.{criterion}'),
                   label=label, color=get_model_color(label))
        first = next(iter(groups.values()))
        x = np.nanmean(stack(first, 'behaviour.by_early_conf.conflict'), axis=0)
        obs = stack(first, 'behaviour.by_early_conf.observer')
        if np.isfinite(obs).any():
            series(ax, x, obs, label='ideal observer', color=COL_OBS, ls='--', marker='s',
                   dots=False)
        ax.set_xlabel('Cue conflict of the first 5 trials')
        ax.set_ylabel('Trials to switch')
        legend(ax, loc='upper left')
    return panel


def spec_decoding_matrix(reports, variables=MATRIX_VARS, sources=SOURCES):
    """What each signal carries: cross-validated decoding, one cluster per variable.

    Hue is the source (hidden state, Z, ΔZ, dL/dZ); the hollow bar is the hidden state cut
    down to two principal components, so a hidden-vs-Z difference cannot be explained by
    the hidden state simply having 32× the dimensions. The dashed line is the shuffled-
    label null pooled over every cell.
    """
    def panel(ax):
        groups, gaps, clusters, hollow_at = [], [], [], []
        i = 0
        for vkey, vlabel in variables:
            first = i
            for skey, _, slabel, hollow in sources:
                groups.append((stack(reports, f'hidden.encoding.decoding.{skey}.{vkey}'),
                               slabel, _source_color(skey)))
                if hollow:
                    hollow_at.append(i)
                i += 1
            clusters.append((vlabel, first, i - 1))
            gaps.append(i - 1)
        bars(ax, groups, ylabel='Decoding accuracy\n(balanced, cross-validated)',
             baseline=0.5, connect=False, clusters=clusters, gap_after=gaps[:-1],
             rotation=45)
        # Hollow the dimension-matched control, the way outcome is hollowed elsewhere.
        # bars() draws one patch per group, in order, so the indices line up.
        patches = [p for p in ax.patches]
        for j in hollow_at:
            if j < len(patches):
                patches[j].set_facecolor('none')
                patches[j].set_edgecolor(groups[j][2])
                patches[j].set_linewidth(0.8)
        ax.set_ylim(0.4, 1.02)
        ax.tick_params(axis='x', length=0)
        from matplotlib.patches import Patch
        # One horizontal row above the axes: the bars fill the panel, so an in-axes legend
        # would sit on top of them (docs/figure_style.md offers this as the alternative).
        ax.legend(handles=[Patch(facecolor='none' if h else _source_color(k),
                                 edgecolor=_source_color(k), label=l)
                           for k, _, l, h in sources],
                  frameon=False, handlelength=0.9, borderpad=0.2, labelspacing=0.2,
                  columnspacing=0.8, ncol=2, loc='lower left',
                  bbox_to_anchor=(0, 1.0, 1, 0.22), mode='expand', fontsize='small')
    return panel


def spec_encoding_variance(reports, sources=SOURCES, variables=VARIANCE_VARS):
    """How much of each signal's variance each variable uniquely explains.

    The paper's Fig 2k asked at the population level: one stacked bar per signal, one
    segment per variable, plus the variance that two variables share and the variance
    nothing explains. A signal that stacks into one tall segment is demixed; one that
    stacks into several is mixed.

    Segments are means over seeds. A stacked bar cannot carry a SEM or a dot per seed
    without becoming unreadable, so the spread lives in the group table instead: the row
    "Encoding: Z is demixed" is the same contrast with its per-seed values and sign count.
    """
    def panel(ax):
        cmap = plt.cm.tab20
        cols = {v: cmap(j / 20.0) for j, (v, _) in enumerate(variables)}
        x = np.arange(len(sources), dtype=float)
        for i, (skey, _, slabel, _) in enumerate(sources):
            bottom = 0.0
            for vkey, vlabel in variables:
                h = float(np.nanmean(stack(reports, f'hidden.encoding.variance.{skey}.unique.{vkey}')))
                h = max(h, 0.0)
                ax.bar(x[i], h, bottom=bottom, width=0.7, color=cols[vkey], linewidth=0,
                       label=vlabel if i == 0 else None, zorder=2)
                bottom += h
            for key, col in (('shared', '0.6'), ('residual', '0.88')):
                h = max(float(np.nanmean(stack(reports, f'hidden.encoding.variance.{skey}.{key}'))), 0.0)
                ax.bar(x[i], h, bottom=bottom, width=0.7, color=col, linewidth=0,
                       label=key if i == 0 else None, zorder=2)
                bottom += h
        ax.set_xticks(x)
        ax.set_xticklabels([l for _, _, l, _ in sources], rotation=45, ha='right')
        ax.tick_params(axis='x', length=0)
        ax.set_ylabel('Fraction of the signal’s variance')
        ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, handlelength=0.9, borderpad=0.2, labelspacing=0.18,
                  loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize='small')
    return panel


def spec_decoding_timecourse(reports, which=('cue', 'rule')):
    """Cue and rule decoding from the hidden state, per timestep, steady state against the
    first five trials after a reversal — the two codes and what a reversal does to each."""
    def panel(ax):
        for name, ls in ((which[0], '-'), (which[1], '--')):
            for cond, f, tag in (('steady_aligned', 1.0, 'steady'),
                                 ('early', LOW_SHADE, 'first 5 after reversal')):
                arr = stack(reports, f'hidden.decoding.{cond}.{name}.acc')
                col = COL_NG if f == 1.0 else shade(COL_NG)
                band(ax, np.arange(arr.shape[1]), arr, color=col, ls=ls,
                     label=f'{name}, {tag}')
        ax.axhline(0.5, color='k', linewidth=0.5, alpha=0.3)
        ax.axvspan(1, 16, color='0.85', alpha=0.35, linewidth=0, zorder=0)
        ax.set_xlabel('Timestep (grey: the cue)')
        ax.set_ylabel('Decoding accuracy')
        ax.set_ylim(0.4, 1.02)
        legend(ax, loc='lower right', fontsize='small')
    return panel


def spec_integration_reversal(groups, measure='index',
                              ylabel='Integration index', splits=('all',)):
    """The paper's Fig 3c: a population measure of the PFC regime, trial by trial around a
    reversal. `measure` is 'index' (late/early cue-period activity, high = rule-driven) or
    'cue_velocity' (the fastest rise along the cue axis, high = input-driven)."""
    def panel(ax):
        for label, reps in groups.items():
            col = get_model_color(label)
            k = np.asarray(_get(reps[0], 'hidden.integration_reversal.k'), dtype=float)
            arr = stack(reps, f'hidden.integration_reversal.{measure}')
            if not np.isfinite(arr).any():
                continue
            band(ax, k[:arr.shape[1]], arr, label=label, color=col)
        ax.axvline(0.5, color='k', linewidth=0.5, alpha=0.4)
        ax.set_xlabel('Trials from reversal')
        ax.set_ylabel(ylabel)
        if len(groups) > 1:
            legend(ax, loc='best', fontsize='small')
    return panel


def spec_trace(groups, key='z_evidence', ylabel='Z on the true context'):
    """One of the three latent signals around a reversal, one line per condition.

    `key` is a trace name from latent()['traces']: z_evidence (the persistent context code),
    step (the size of the trial's own latent update — the transient), grad (the raw error
    gradient, the signal that drives it), or gain.
    """
    def panel(ax):
        for label, reps in groups.items():
            k = np.asarray(_get(reps[0], f'latent.traces.{key}.k'), dtype=float)
            arr = stack(reps, f'latent.traces.{key}.mean')
            if not np.isfinite(arr).any():
                continue
            band(ax, k[:arr.shape[1]], arr, label=label, color=get_model_color(label))
        ax.axvline(0.5, color='k', linewidth=0.5, alpha=0.4)
        ax.set_xlabel('Trials from reversal')
        ax.set_ylabel(ylabel)
        if len(groups) > 1:
            legend(ax, loc='best', fontsize='small')
    return panel


#: The manipulations, as (condition prefix, its control's prefix, short label). Kept here
#: so the panels, the group table and the captions all name them the same way.
MANIPULATIONS = (('softmax_lu0', 'softmax', 'update off'),
                 ('softmax_lu3', 'softmax', 'update ×3'),
                 ('softmax_lu10', 'softmax', 'update ×10'),
                 ('softmax_blast', 'softmax', 'Z→(1,1)'),
                 ('sigmoid_blast', 'sigmoid', 'Z→(1,1), sigmoid'))


def _manip_pairs(by_condition, manipulations=MANIPULATIONS, criterion='dec_switch'):
    """[(label, split, perturbed per seed, control per seed)] for what is on disk."""
    out = []
    for name, base, label in manipulations:
        for split in ('low', 'high'):
            pert = by_condition.get(f'{name}_rc_{split}')
            ctrl = by_condition.get(f'{base}_rc_{split}')
            if not pert or not ctrl:
                continue
            key = f'behaviour.switch.all.{criterion}'
            out.append((label, split, stack(pert, key), stack(ctrl, key)))
    return out


def spec_manipulation_switch(by_condition, criterion='dec_switch'):
    """Trials to switch under each manipulation, beside its own unperturbed control.

    One cluster per manipulation × early-conflict level; within a cluster the control comes
    first and the perturbed session second, so the comparison the panel is about is the
    adjacent pair. Shade is the early conflict, as everywhere.
    """
    def panel(ax):
        pairs = _manip_pairs(by_condition, criterion=criterion)
        if not pairs:
            ax.axis('off')
            return
        groups, clusters, gaps = [], [], []
        for label, split, pert, ctrl in pairs:
            col = split_style('NeuraGEM', split)[0]
            i = len(groups)
            groups += [(ctrl, '', shade(col, 0.75)), (pert, '', col)]
            clusters.append((f'{label}\n{split}', i, i + 1))
            gaps.append(i + 1)
        bars(ax, groups, ylabel='Trials to switch', connect=False, clusters=clusters,
             gap_after=gaps[:-1], rotation=45)
        ax.tick_params(axis='x', length=0)
    return panel


def spec_manipulation_cost(by_condition, criterion='dec_switch'):
    """Each manipulation as one number: trials to switch minus its own control, within seed.

    Positive = slower to switch. Silencing the latent update is predicted to slow switching
    (the paper's ACC→MD result); driving the latent is predicted to speed it up.
    """
    def panel(ax):
        pairs = _manip_pairs(by_condition, criterion=criterion)
        if not pairs:
            ax.axis('off')
            return
        groups = [(np.asarray(pert) - np.asarray(ctrl), f'{label}\n{split}',
                   split_style('NeuraGEM', split)[0])
                  for label, split, pert, ctrl in pairs]
        bars(ax, groups, ylabel='Extra trials to switch\nvs the same session unperturbed',
             baseline=0.0, connect=False, rotation=45)
        ax.tick_params(axis='x', length=0)
    return panel


def figure(panels, path, ncol=None, panel=None, letters=False):
    """Draw a list of panel callables into one row/grid and save it.

    A `None` entry leaves a blank slot, so a multi-row figure can keep its columns aligned
    when one row is shorter. With `letters`, each occupied panel is lettered a, b, c … in
    reading order, which is what the captions refer to.
    """
    n = len(panels)
    ncol = ncol or min(n, 3)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=FigSize.grid(nrow, ncol, panel or FigSize.wide))
    axes = np.atleast_1d(axes).ravel()
    drawn = 0
    for ax, p in zip(axes, panels):
        if p is None:
            ax.axis('off')
            continue
        p(ax)
        if letters:
            ax.text(-0.28, 1.06, chr(ord('a') + drawn), transform=ax.transAxes,
                    fontweight='bold', va='bottom', ha='left')
        drawn += 1
    for ax in axes[n:]:
        ax.axis('off')
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Exported: {path}  {np.round(fig.get_size_inches(), 2)} in')


def group_figures(out_dir=None):
    data = collect()
    out_dir = out_dir or os.path.join(EXPORTS, 'group', 'figures')
    ng = [data[('NG', 'softmax_rc_none')][s] for s in sorted(data.get(('NG', 'softmax_rc_none'), {}))]
    rnn = [data[('RNN', 'rnn_rc_none')][s] for s in sorted(data.get(('RNN', 'rnn_rc_none'), {}))]
    low = [data[('NG', 'softmax_rc_low')][s] for s in sorted(data.get(('NG', 'softmax_rc_low'), {}))]
    high = [data[('NG', 'softmax_rc_high')][s] for s in sorted(data.get(('NG', 'softmax_rc_high'), {}))]
    if not ng:
        raise SystemExit('no NG reports on disk yet: run hier_switch_group.py task first')
    groups = {'NeuraGEM': ng}
    if rnn:
        groups['RNN'] = rnn
    # The rc_low / rc_high pair carries the split, so behaviour panels merge the two
    # sessions' splits into the one NeuraGEM entry.
    split = {'NeuraGEM': [merge_splits(a, b) for a, b in zip(low, high)]} if low and high else groups

    # Every NG condition as {condition: [report per seed]}, for the gain figure.
    by_condition = {cond: [d[s] for s in sorted(d)]
                    for (model, cond), d in data.items() if model == 'NG'}

    # Retired into the story figure: the reversal-aligned RT panel (story row 4) and the
    # whole of switching.pdf (story rows 1 and 4). What stays here is what the story does
    # not carry: accuracy and RT against conflict pooled over contexts.
    figure([spec_psychometric(groups), spec_rt(groups)],
           os.path.join(out_dir, 'behaviour.pdf'), ncol=2)
    # Four labelled clusters per panel need more width than the standard wide preset.
    figure([spec_switch(split), spec_switch_cost(split)],
           os.path.join(out_dir, 'switch_latency.pdf'), ncol=2,
           panel=FigSize.custom(2.6, 1.5))
    # The latent panels are NeuraGEM only: the RNN's Z never moves, so it has no belief,
    # no uncertainty and no gradient on Z to show.
    figure([spec_z_belief({'NeuraGEM': ng}), spec_z_uncertainty(split), spec_eps_cw(ng)],
           os.path.join(out_dir, 'latent.pdf'))
    figure([spec_gain(by_condition), spec_gain_ladder(by_condition), spec_gain_cost(by_condition)],
           os.path.join(out_dir, 'gain.pdf'))
    figure([spec_decoding(ng, 'cue'), spec_decoding(ng, 'rule'), spec_integration(ng),
            spec_unit_classes(groups)], os.path.join(out_dir, 'hidden.pdf'), ncol=2)
    for act in ('softmax', 'sigmoid'):
        cells = clamp_cells_on_disk(act)
        if cells:
            figure([spec_clamp_grid(cells, 'index'),
                    spec_clamp_grid(cells, 'cue_velocity', 'Cue velocity'),
                    spec_clamp_grid(cells, 'acc_match', 'Accuracy\n(gate matches context)')],
                   os.path.join(out_dir, f'clamp_{act}.pdf'))
    # The simple cut through the sigmoid grid: contrast fixed, gain varied.
    cells = clamp_cells_on_disk('sigmoid')
    if cells:
        figure([spec_clamp_gain(cells, 1.0, 'cue_velocity', 'Cue velocity'),
                spec_clamp_gain(cells, 1.0, 'index', 'Integration index'),
                spec_clamp_gain(cells, 1.0, 'acc_match', 'Accuracy\n(gate matches context)')],
               os.path.join(out_dir, 'clamp_sigmoid_gain.pdf'))
    story_figure(out_dir)


#: The two switch criteria the story figure shows. The raw behavioural one is contaminated
#: by hedging (an undecided trial's sign is a coin flip), so the story uses the decided-only
#: version of the paper's criterion and the latent one; all three stay in switch_latency.pdf.
STORY_CRITERIA = (('dec_switch', 'decided'), ('z_switch', 'Z side'))


def story_figure(out_dir=None):
    """The whole result as one figure: five rows of four panels, a–t.

    Deliberately larger than a single panel preset (docs/figure_style.md allows it for a
    figure that genuinely summarises a lot of data, and asks for a comment saying why): it
    is the paper's argument end to end, and each row is also written on its own so a row
    can be iterated without rebuilding the rest.

      1  behaviour, against the paper's Fig 1e and 1f
      2  what each signal encodes — the hidden state against Z, its update and its gradient
      3  what the gate does when it is held still: gain against contrast
      4  the same measures trial by trial around a reversal (the paper's Fig 3c cut)
      5  the three latent signals around a reversal: persistent, transient, and the error
    """
    data = collect()
    out_dir = out_dir or os.path.join(EXPORTS, 'group', 'figures')
    pick = lambda key: [data[key][s] for s in sorted(data.get(key, {}))]
    ng, rnn = pick(('NG', 'softmax_rc_none')), pick(('RNN', 'rnn_rc_none'))
    low, high = pick(('NG', 'softmax_rc_low')), pick(('NG', 'softmax_rc_high'))
    sig = pick(('NG', 'sigmoid_zlr30000'))
    if not ng:
        raise SystemExit('no NG reports on disk yet: run hier_switch_group.py task first')
    groups = {'NeuraGEM': ng}
    if rnn:
        groups['RNN'] = rnn
    split = ({'NeuraGEM': [merge_splits(a, b) for a, b in zip(low, high)]}
             if low and high else groups)
    cells = clamp_cells_on_disk('sigmoid')

    # A legend is kept only where it carries something the caption cannot: which context,
    # which signal, which variable, correct against error, the gain ladder, which model.
    # Shade and dash mean low against high early conflict in every panel that has them.
    rows = [
        ('story_1_behaviour', [
            relegend(spec_psychometric_ctx(groups), loc='lower left'),
            relegend(spec_switch_vs_early_conflict({'NeuraGEM': ng}), loc='upper left'),
            rotate_xticks(nolegend(spec_switch(split, STORY_CRITERIA))),
            nolegend(spec_reversal(split, 'acc', 'Accuracy'))]),
        ('story_2_encoding', [
            spec_decoding_matrix(ng, sources=STORY_SOURCES),
            nolegend(spec_decoding_timecourse(ng)),
            relegend(spec_eps_cw(ng), loc='upper right'),
            spec_encoding_variance(ng)]),
        ('story_3_gate', [
            nolegend(spec_clamp_grid(cells, 'acc_match', 'Accuracy\n(gate matches context)')),
            nolegend(spec_clamp_grid(cells, 'rt', 'RT (timesteps)')),
            nolegend(spec_clamp_grid(cells, 'index', 'Integration index')),
            relegend(spec_clamp_grid(cells, 'cue_velocity', 'Cue velocity'),
                     loc='center left', bbox_to_anchor=(1.02, 0.5))]),
        ('story_4_reversal', [
            nolegend(spec_reversal(split, 'undecided', 'Undecided rate')),
            relegend(spec_rt_reversal(groups), loc='upper right'),
            nolegend(spec_integration_reversal({'NeuraGEM': ng}, 'index', 'Integration index')),
            nolegend(spec_integration_reversal({'NeuraGEM': ng}, 'cue_velocity', 'Cue velocity'))]),
        ('story_5_latent', [
            relegend(spec_z_belief({'NeuraGEM': ng}), loc='lower right'),
            nolegend(spec_trace({'NeuraGEM': ng}, 'step', 'ΔZ toward the true context\n(|update|)')),
            nolegend(spec_trace({'NeuraGEM': ng}, 'grad', '|dL/dZ| on the context axis')),
            nolegend(spec_trace({'sigmoid at test': sig} if sig else {'NeuraGEM': ng}, 'gain',
                                'Z gain (mean of the units)'))]),
    ]
    # Row 6, the manipulations, appears only once those sessions exist (run_manipulations.sh);
    # until then the story figure is the five rows above. by_condition carries every NG
    # condition on disk, so the row builds itself as soon as the data lands.
    by_condition = {cond: [d[s] for s in sorted(d)]
                    for (model, cond), d in data.items() if model == 'NG'}
    if _manip_pairs(by_condition):
        mom = {f'momentum {mu}': pick(('NG', f'softmax_mom{mu}_rc_none')) for mu in ('0.5', '0.9')}
        mom = {k: v for k, v in mom.items() if v}
        traces = {'no momentum': ng, **mom}
        rows.append(('story_6_manipulations', [
            rotate_xticks(spec_manipulation_switch(by_condition)),
            rotate_xticks(spec_manipulation_cost(by_condition)),
            relegend(spec_trace(traces, 'step', 'ΔZ toward the true context\n(|update|)'),
                     loc='upper right'),
            nolegend(spec_trace(traces, 'grad', '|dL/dZ| on the context axis'))]))

    panel = FigSize.custom(1.7, 1.35)
    for name, panels in rows:
        figure(panels, os.path.join(out_dir, f'{name}.pdf'), ncol=4, panel=panel,
               letters=True)
    figure([p for _, ps in rows for p in ps], os.path.join(out_dir, 'story.pdf'),
           ncol=4, panel=panel, letters=True)


def merge_splits(low_rep, high_rep):
    """One report whose 'low' and 'high' splits come from the two forced-conflict sessions."""
    out = json.loads(json.dumps(low_rep))
    for field in ('reversal', 'switch'):
        out['behaviour'][field]['low'] = low_rep['behaviour'][field]['all']
        out['behaviour'][field]['high'] = high_rep['behaviour'][field]['all']
    if 'latent' in out:
        out['latent']['uncertainty']['low'] = low_rep['latent']['uncertainty']['all']
        out['latent']['uncertainty']['high'] = high_rep['latent']['uncertainty']['all']
    out['observer']['switch_low'] = low_rep['observer']['switch']
    out['observer']['switch_high'] = high_rep['observer']['switch']
    return out


def clamp_cells_on_disk(activation):
    """Every clamp cell of every seed, pooled (each cell carries its own m and d)."""
    root = os.path.join(EXPORTS, 'clamp')
    cells = []
    if not os.path.isdir(root):
        return cells
    for tag in sorted(os.listdir(root)):
        f = os.path.join(root, tag, f'clamp_grid_{activation}.json')
        if os.path.exists(f):
            with open(f) as fh:
                cells += json.load(fh)['cells']
    return cells


def session_figures(path, out_dir=None):
    """The same panels for one session — a list of one replicate."""
    f = os.path.join(path, 'results.json')
    rep = json.load(open(f)) if os.path.exists(f) else session_report(path)
    reps = [rep]
    out_dir = out_dir or os.path.join(path, 'figures')
    groups = {'NeuraGEM' if rep['model_type'] == 'NG' else 'RNN': reps}
    figure([spec_psychometric(groups), spec_rt(groups), spec_rt_reversal(groups)],
           os.path.join(out_dir, 'behaviour.pdf'))
    if 'latent' in rep:
        figure([spec_z_belief(groups), spec_z_uncertainty(groups), spec_eps_cw(reps)],
               os.path.join(out_dir, 'latent.pdf'))
        if rep['activation'] != 'softmax':
            figure([spec_gain({rep['condition']: reps}, softmax_key=None,
                              sigmoid_key=rep['condition'])],
                   os.path.join(out_dir, 'gain.pdf'), ncol=1)
    if 'hidden' in rep:
        figure([spec_decoding(reps, 'cue'), spec_decoding(reps, 'rule'),
                spec_integration(reps)], os.path.join(out_dir, 'hidden.pdf'))
        if 'encoding' in rep['hidden']:
            figure([spec_decoding_matrix(reps), spec_encoding_variance(reps)],
                   os.path.join(out_dir, 'encoding.pdf'), ncol=2,
                   panel=FigSize.custom(2.2, 1.5))


if __name__ == '__main__':
    args = [a for a in sys.argv[1:]]
    if args and args[0] == 'story':
        story_figure()
    elif args:
        for p in args:
            session_figures(p)
    else:
        group_figures()
