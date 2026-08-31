"""
flanker_metrics.py — every per-seed scalar measure computed from one test session.

One session in, one flat dict of named effects out. `session_effects()` is a thin
composition of five focused blocks, so a new measure goes in the block it belongs to
rather than at the bottom of a growing function:

    condition_masks     trial masks every block shares (congruency, distance, history)
    behaviour_effects   accuracy and RT: cell means, congruency, distance
    rt_outcome_effects  RT split by correct vs error, and how often no decision came
    history_effects     sequential congruency, lag contrasts, the repetition control
    post_error_effects  post-error slowing, accuracy, PERI
    control_effects     the latent: focus, attention profile, what drives the update

The statistical contract does not live here: each seed is a synthetic subject, so every
number below is *within* one session and is only meaningful once one-sample t-tested
across seeds (see flanker_sweep_analysis.summarize).

`SIGNATURES` at the bottom is the registry of human benchmark effects with the sign each
should take — the single source of truth for the scorecard figure and any pass/fail table.
"""

from __future__ import annotations

import numpy as np

from flanker_analyses import lagged_factors

# Slot layout: [far_left, near_left, centre, near_right, far_right]
NEAR_SLOTS, FAR_SLOTS, CENTRE_SLOT = [1, 3], [0, 4], 2

DISTANCES  = ('near', 'far')
CONGRUENCY = ('cong', 'incong')


# ── Shared helpers ────────────────────────────────────────────────────────────

def _mean(vals, mask):
    """NaN-safe mean over a boolean mask; NaN if the cell is empty."""
    v = vals[mask]
    v = v[~np.isnan(v)]
    return v.mean() if len(v) else np.nan


def _frac(mask, within):
    """Fraction of `within` trials that also satisfy `mask`; NaN if `within` is empty."""
    n = int(within.sum())
    return float((mask & within).sum()) / n if n else np.nan


def condition_masks(trials):
    """Every trial mask the effect blocks share. Returns a dict of boolean arrays."""
    f = lagged_factors(trials, n_back=2)
    m = {
        'cong':    f['cong'] == 1,   'incong':   f['cong'] == 0,
        'near':    f['near'] == 1,   'far':      f['near'] == 0,
        'valid':   f['valid'],
        'pc':      f['correct_1'] == 1,  'perr':     f['correct_1'] == 0,
        'p2c':     f['correct_2'] == 1,
        'p_cong':  f['cong_1'] == 1,     'p_incong': f['cong_1'] == 0,
        'p2_cong': f['cong_2'] == 1,     'p2_incong': f['cong_2'] == 0,
        'p_near':  f['near_1'] == 1,     'p_far':    f['near_1'] == 0,
        'rep':     f['resp_rep'] == 1,   'sw':       f['resp_rep'] == 0,
        'corr':    trials['correct_at_decision'].astype(bool),
        'decided': trials['decided'].astype(bool),
    }
    m['err'] = ~m['corr']
    return m


def _cells(m):
    """The four condition cells as (name, mask) pairs."""
    return [(f'{d}_{c}', m[d] & m[c]) for c in CONGRUENCY for d in DISTANCES]


# ── Block 1: accuracy and RT ──────────────────────────────────────────────────

def behaviour_effects(trials, m):
    """Cell means, the congruency effect, and the distance effect within congruency."""
    acc = trials['correct_at_decision'].astype(float)
    rt  = trials['rt_interp']
    e   = {'acc_overall': float(acc.mean())}

    for cn in CONGRUENCY:
        e[f'acc_{cn}'] = _mean(acc, m[cn])
        e[f'rt_{cn}']  = _mean(rt,  m[cn])
    for name, mask in _cells(m):
        e[f'acc_{name}'] = _mean(acc, mask)
        e[f'rt_{name}']  = _mean(rt,  mask)

    e['cong_effect_acc'] = e['acc_cong'] - e['acc_incong']
    e['cong_effect_rt']  = e['rt_incong'] - e['rt_cong']
    for d in DISTANCES:
        e[f'cong_effect_acc_{d}'] = e[f'acc_{d}_cong'] - e[f'acc_{d}_incong']
        e[f'cong_effect_rt_{d}']  = e[f'rt_{d}_incong'] - e[f'rt_{d}_cong']
    e['interaction_acc'] = e['cong_effect_acc_near'] - e['cong_effect_acc_far']
    e['interaction_rt']  = e['cong_effect_rt_near']  - e['cong_effect_rt_far']

    # Simple effects of distance *within* each congruency. These carry the directional
    # predictions — near flankers help when they agree with the target and hurt when they
    # conflict — which the interaction collapses into a single number.
    for cn in CONGRUENCY:
        e[f'dist_effect_acc_{cn}'] = e[f'acc_near_{cn}'] - e[f'acc_far_{cn}']
        e[f'dist_effect_rt_{cn}']  = e[f'rt_near_{cn}']  - e[f'rt_far_{cn}']
    return e


# ── Block 2: RT by outcome, and extrapolation bookkeeping ─────────────────────

def rt_outcome_effects(trials, m):
    """
    RT split by whether the response was correct, per cell.

    Errors on incongruent trials are flanker-driven, so they should be *fast*; that is a
    different signature from the overall RT cost and it can point the opposite way. Every
    RT cell here is paired with the fraction of its trials that actually crossed threshold
    inside the window (`dec_*`), because the rest sit at the trial end by convention — a
    cell with a low decided fraction is reporting failure-to-decide, not speed.
    """
    rt  = trials['rt_interp']
    dec = m['decided']
    e   = {'undecided_frac': float((~dec).mean())}

    for cn in CONGRUENCY:
        e[f'undecided_frac_{cn}'] = 1.0 - _frac(dec, m[cn])
    # Congruency effect on RT using only trials that crossed inside the window — the
    # sensitivity check against the extrapolation, not the headline number.
    e['cong_effect_rt_decided'] = _mean(rt, m['incong'] & dec) - _mean(rt, m['cong'] & dec)

    for name, mask in _cells(m):
        e[f'dec_{name}'] = _frac(dec, mask)
        for on, om in (('corr', m['corr']), ('err', m['err'])):
            e[f'rt_{name}_{on}']  = _mean(rt, mask & om)
            e[f'dec_{name}_{on}'] = _frac(dec, mask & om)
        # Positive = errors are faster than correct responses in this cell.
        e[f'fasterr_{name}'] = e[f'rt_{name}_corr'] - e[f'rt_{name}_err']

    # The contrast the distance prediction is about, computed separately for correct and
    # error responses: does flanker distance move RT the same way when the target wins?
    for cn in CONGRUENCY:
        for on in ('corr', 'err'):
            e[f'dist_effect_rt_{cn}_{on}'] = e[f'rt_near_{cn}_{on}'] - e[f'rt_far_{cn}_{on}']
    return e


# ── Block 3: trial history ────────────────────────────────────────────────────

def history_effects(trials, m):
    """Sequential congruency: the four history cells, lag contrasts, repetition control."""
    acc = trials['correct_at_decision'].astype(float)
    rt  = trials['rt_interp']
    e   = {}

    # Post-correct only: incongruent trials fail more often, so an unrestricted history
    # cell partly measures post-error adaptation rather than conflict adaptation.
    ok = m['valid'] & m['pc'] & m['p2c']
    for lbl, c2, c1 in [('CC', 'p2_cong', 'p_cong'), ('CI', 'p2_cong', 'p_incong'),
                        ('IC', 'p2_incong', 'p_cong'), ('II', 'p2_incong', 'p_incong')]:
        cell = ok & m[c2] & m[c1]
        for tgt in CONGRUENCY:
            suffix = 'I' if tgt == 'incong' else 'C'
            e[f'acc_{lbl}_to_{suffix}'] = _mean(acc, cell & m[tgt])
            e[f'rt_{lbl}_to_{suffix}']  = _mean(rt,  cell & m[tgt])

    # Which lag drives the history effect? Both contrasts on current-incongruent trials.
    inc = m['incong']
    e['lag1_contrast_acc'] = _mean(acc, ok & m['p_incong'] & inc)  - _mean(acc, ok & m['p_cong'] & inc)
    e['lag2_contrast_acc'] = _mean(acc, ok & m['p2_incong'] & inc) - _mean(acc, ok & m['p2_cong'] & inc)
    e['lag1_contrast_rt']  = _mean(rt,  ok & m['p_cong'] & inc)    - _mean(rt,  ok & m['p_incong'] & inc)
    e['lag2_contrast_rt']  = _mean(rt,  ok & m['p2_cong'] & inc)   - _mean(rt,  ok & m['p2_incong'] & inc)

    # Gratton interaction with and without the response-repetition control. A sequential
    # effect that survives only when repetitions are pooled in is feature-integration
    # priming, not control (Mayr, Awh & Laurey 2003).
    for rname, rmask in [('all', np.ones_like(m['rep'])), ('switch', m['sw']), ('repeat', m['rep'])]:
        base = m['valid'] & m['pc'] & rmask
        ce_rt_c = _mean(rt, base & m['p_cong'] & inc)     - _mean(rt, base & m['p_cong'] & m['cong'])
        ce_rt_i = _mean(rt, base & m['p_incong'] & inc)   - _mean(rt, base & m['p_incong'] & m['cong'])
        ce_ac_c = _mean(acc, base & m['p_cong'] & m['cong'])   - _mean(acc, base & m['p_cong'] & inc)
        ce_ac_i = _mean(acc, base & m['p_incong'] & m['cong']) - _mean(acc, base & m['p_incong'] & inc)
        e[f'sce_rt_{rname}']  = ce_rt_c - ce_rt_i
        e[f'sce_acc_{rname}'] = ce_ac_c - ce_ac_i
    return e


# ── Block 4: post-error adaptation ────────────────────────────────────────────

def post_error_effects(trials, m):
    """Post-error slowing and accuracy, with trial A restricted to incongruent trials."""
    acc, rt, foc_in = (trials['correct_at_decision'].astype(float),
                       trials['rt_interp'], trials['focus_in'])
    e = {}
    # Trial B split by congruency: a target-focused state helps incongruent B and hurts
    # congruent B, so pooling them averages an effect against its own opposite.
    for bn, bm in [('I', m['incong']), ('C', m['cong'])]:
        after_err  = m['valid'] & m['p_incong'] & m['perr'] & bm
        after_corr = m['valid'] & m['p_incong'] & m['pc']   & bm
        e[f'pia_B{bn}'] = _mean(acc, after_err) - _mean(acc, after_corr)
        e[f'pes_B{bn}'] = _mean(rt,  after_err) - _mean(rt,  after_corr)
        e[f'focus_in_diff_B{bn}'] = _mean(foc_in, after_err) - _mean(foc_in, after_corr)

    ce_err  = (_mean(rt, m['valid'] & m['p_incong'] & m['perr'] & m['incong'])
               - _mean(rt, m['valid'] & m['p_incong'] & m['perr'] & m['cong']))
    ce_corr = (_mean(rt, m['valid'] & m['p_incong'] & m['pc'] & m['incong'])
               - _mean(rt, m['valid'] & m['p_incong'] & m['pc'] & m['cong']))
    e['peri'] = ce_corr - ce_err          # post-error reduction of interference
    return e


# ── Block 5: the control state ────────────────────────────────────────────────

def control_effects(trials, m):
    """
    Focus, the attention profile over slots, and what produces the update.

    `focus_in` is the state a trial *inherited*, which is what drove its behaviour;
    reading a trial's own focus after conditioning on its outcome is circular. The
    attention profile is the softmaxed gate the RNN actually applied, averaged over
    trials — uniform would be 0.200 per slot.
    """
    foc_in, dfoc, z_act = trials['focus_in'], trials['delta_focus'], trials['z_act']
    e = {
        'focus_all':    float(np.nanmean(foc_in)),
        'focus_cong':   _mean(foc_in, m['cong']),
        'focus_incong': _mean(foc_in, m['incong']),
        'att_centre':   float(np.nanmean(z_act[:, CENTRE_SLOT])),
        'att_near':     float(np.nanmean(z_act[:, NEAR_SLOTS])),
        'att_far':      float(np.nanmean(z_act[:, FAR_SLOTS])),
    }
    e['att_near_minus_far'] = e['att_near'] - e['att_far']

    # Where the inherited gate actually points, and how sharp it runs. These are the
    # mechanism behind the spurious near-vs-far accuracy difference: a gate peaking off
    # the target lands on a slot the display may leave empty, and near and far displays
    # leave different slots empty, so they pay different prices for the same wandering.
    # `gate_peak` is comparable to the Stage-1 oracle's own peak — softmax(one-hot /
    # softmax_temp), 0.405 with five slots at temp 1 — which is the only sharpness the
    # weights were ever calibrated at unless config.oracle_gate_jitter is set.
    z_in = trials['z_in']
    ok   = np.isfinite(z_in).all(axis=1)
    e['gate_off_centre'] = (float(1.0 - (z_in[ok].argmax(axis=1) == CENTRE_SLOT).mean())
                            if ok.any() else np.nan)
    e['gate_peak'] = float(np.median(z_in[ok].max(axis=1))) if ok.any() else np.nan

    # The update is measured on the trial that produced it, not the one after.
    for d in DISTANCES:
        for on, om in (('err', ~m['corr']), ('corr', m['corr'])):
            e[f'dfocus_{d}_{on}'] = _mean(dfoc, m[d] & m['incong'] & om)
    e['dfocus_near_minus_far_err'] = e['dfocus_near_err'] - e['dfocus_far_err']

    # The same update across the *full* congruency x distance grid, and split by outcome.
    # The four keys above are incongruent-only and keep their old names; these carry the
    # congruency in the key, matching `acc_{cell}` and `rt_{cell}_{outcome}` elsewhere, so
    # a panel can ask what each kind of trial teaches Z rather than only what an
    # incongruent one does. Congruent trials are the half of the list that has nothing to
    # be misled by, so their update is the baseline the incongruent cells are read against.
    for name, mask in _cells(m):
        e[f'dfocus_{name}'] = _mean(dfoc, mask)
        for on, om in (('corr', m['corr']), ('err', m['err'])):
            e[f'dfocus_{name}_{on}'] = _mean(dfoc, mask & om)
    for cn in CONGRUENCY:
        e[f'dfocus_{cn}'] = _mean(dfoc, m[cn])
    e['dfocus_cong_effect'] = e['dfocus_incong'] - e['dfocus_cong']
    e['focus_in_after_near_err'] = _mean(foc_in, m['valid'] & m['p_incong'] & m['p_near'] & m['perr'])
    e['focus_in_after_far_err']  = _mean(foc_in, m['valid'] & m['p_incong'] & m['p_far']  & m['perr'])
    return e


# ── Block 6: why errors fail to recruit control ───────────────────────────────

def error_diagnosis_effects(trials, m):
    """
    Split incongruent errors by *what caused them*, and ask what each kind teaches Z.

    arrow_noise_std is 1.3 against a signal of 1.0, so the target slot's own samples
    point the wrong way on a sizeable minority of trials. Two very different things
    therefore produce an error:

      - the centre slot itself was misleading (`_noisy`), in which case attending the
        centre *less* genuinely lowers this trial's prediction error, or
      - the centre was fine and the flankers simply won (`_clean`), where attending the
        centre more is the fix.

    The latent update minimises this trial's prediction error, so it cannot tell the two
    apart the way an error monitor would. These keys quantify the consequence: the split
    is a median split on `centre_evidence` taken within incongruent trials, so 0.5 of
    them fall either side by construction and `frac_err_noisy` above 0.5 means poor
    centre evidence is over-represented among errors.
    """
    dfoc, foc_in = trials['delta_focus'], trials['focus_in']
    centre_ev, flank_ev = trials['centre_evidence'], trials['flanker_evidence']
    z_grad = trials.get('z_grad')
    inc = m['incong']

    e = {}
    for cn in CONGRUENCY:
        for on, om in (('corr', m['corr']), ('err', m['err'])):
            e[f'centre_ev_{cn}_{on}'] = _mean(centre_ev, m[cn] & om)
            e[f'flank_ev_{cn}_{on}']  = _mean(flank_ev,  m[cn] & om)

    median = np.nanmedian(centre_ev[inc]) if inc.any() else np.nan
    noisy, clean = centre_ev < median, centre_ev >= median
    groups = {'err_noisy':  inc & m['err']  & noisy, 'err_clean':  inc & m['err']  & clean,
              'corr_noisy': inc & m['corr'] & noisy, 'corr_clean': inc & m['corr'] & clean}
    for name, mask in groups.items():
        e[f'dfocus_{name}']   = _mean(dfoc, mask)
        e[f'focus_in_{name}'] = _mean(foc_in, mask)
        e[f'n_{name}']        = float(mask.sum())
        if z_grad is not None:
            e[f'zgrad_centre_{name}'] = _mean(z_grad[:, CENTRE_SLOT], mask)

    n_err = float((inc & m['err']).sum())
    e['frac_err_noisy'] = float(groups['err_noisy'].sum()) / n_err if n_err else np.nan
    # Positive = flanker-driven errors teach more control than noise-driven ones do.
    e['dfocus_err_clean_minus_noisy'] = e['dfocus_err_clean'] - e['dfocus_err_noisy']
    return e


# ── Block 7: the circularity behind the missing post-error improvement ────────

#: Trial positions relative to the event, for the event-locked analysis.
EVENT_LAGS = np.arange(-2, 3)


def event_locked(trials, lags=EVENT_LAGS, n_bins=8):
    """
    Focus and accuracy on the trials surrounding an incongruent error, vs a correct one.

    This is what explains why the model shows no post-error improvement even when errors
    *do* recruit control. `focus_in` is the state a trial inherited, so lag 0 is what the
    event trial itself started from and lag +1 is what the next trial inherited — the step
    from 0 to +1 therefore contains the event's own update.

    The control deficit **precedes** the error: a trial that is about to go wrong already
    inherited a lower focus. The error's own correction is real and larger than a correct
    trial's, but it starts from further back and does not catch up within one trial, so the
    next trial still begins below par. Conditioning on the outcome and then reading the
    state is circular for exactly this reason — a low state is what produced the error.

    Returns per-*event* arrays (n_events, n_lags) so the caller decides what the replicate
    is: sessions at group level, events within one session.
    """
    m = condition_masks(trials)
    n = trials['n_trials']
    idx = np.arange(n)
    foc = trials['focus_in']
    acc = trials['correct_at_decision'].astype(float)
    pad = int(np.max(np.abs(lags)))
    inside = (idx >= pad) & (idx < n - pad)

    out = {}
    for key, om in (('err', m['err']), ('corr', m['corr'])):
        events = idx[m['incong'] & om & inside]
        out[f'focus_{key}'] = np.array([foc[events + l] for l in lags]).T
        out[f'acc_{key}']   = np.array([acc[events + l] for l in lags]).T

    # The gap when the error happens, and the update each outcome produces.
    dfoc = trials['delta_focus']
    out['start_gap'] = (_mean(foc, m['incong'] & m['err'])
                        - _mean(foc, m['incong'] & m['corr']))
    out['upd_err']  = _mean(dfoc, m['incong'] & m['err'])
    out['upd_corr'] = _mean(dfoc, m['incong'] & m['corr'])

    # Why the average correction above is unreliable: an incongruent error is either the
    # centre slot's own evidence misleading the model (noise-driven — attending it *less*
    # genuinely lowered this trial's error) or clean centre evidence that the flankers
    # simply outweighed (flanker-driven — attending it *more* is the fix). Same median
    # split as error_diagnosis_effects, kept here so this figure carries the argument on
    # its own: the two kinds get very different updates, and the average blurs that.
    centre_ev = trials['centre_evidence']
    inc_all = m['incong']
    median = np.nanmedian(centre_ev[inc_all]) if inc_all.any() else np.nan
    noisy, clean = centre_ev < median, centre_ev >= median
    out['frac_err_noisy']   = _frac(noisy, inc_all & m['err'])
    out['frac_err_clean']   = _frac(clean, inc_all & m['err'])
    out['dfocus_err_noisy'] = _mean(dfoc, inc_all & m['err'] & noisy)
    out['dfocus_err_clean'] = _mean(dfoc, inc_all & m['err'] & clean)

    # How much accuracy a given inherited focus buys on incongruent trials — the exchange
    # rate that turns the control gap into the behavioural one. `curve_rt` is the same
    # bins read in RT, which is the other half of the price: control that is not there has
    # to be paid for either in accuracy or in time, and a model that pays in only one of
    # them is not doing what a human does. RT is `rt_interp`, so trials that never crossed
    # sit at the trial end by the house convention rather than being dropped.
    rt  = trials['rt_interp']
    inc = m['incong'] & ~np.isnan(foc)
    edges = np.nanquantile(foc[inc], np.linspace(0, 1, n_bins + 1))
    centres, means, rts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = inc & (foc >= lo) & (foc < hi)
        if sel.sum() > 20:
            centres.append(np.nanmean(foc[sel]))
            means.append(acc[sel].mean())
            rts.append(np.nanmean(rt[sel]))
    out['curve_x'], out['curve_y'] = np.array(centres), np.array(means)
    out['curve_rt'] = np.array(rts)
    out['lags'] = np.asarray(lags)
    return out


# ── Composition ───────────────────────────────────────────────────────────────

def session_effects(trials):
    """Every scalar effect for one session, as a flat dict."""
    m = condition_masks(trials)
    e = {}
    for block in (behaviour_effects, rt_outcome_effects, history_effects,
                  post_error_effects, control_effects, error_diagnosis_effects):
        e.update(block(trials, m))
    return e


# ── Benchmark registry ────────────────────────────────────────────────────────
#
# The human signatures the model is being judged against. `sign` is the direction a
# human dataset shows, so a scorecard can flip every effect to "positive = matches
# humans" and put them on one axis. Keep this list as the single source of truth —
# a figure that hard-codes its own list will drift from the tables.

SIGNATURES = [
    ('cong_effect_acc',        'Congruency effect (accuracy)',      +1, 'Eriksen & Eriksen 1974'),
    ('cong_effect_rt',         'Congruency effect (RT)',            +1, 'Eriksen & Eriksen 1974'),
    ('dist_effect_acc_incong', 'Distance: accuracy, incongruent',   -1, 'flanker proximity'),
    ('dist_effect_acc_cong',   'Distance: accuracy, congruent',     +1, 'flanker proximity'),
    ('dist_effect_rt_incong',  'Distance: RT, incongruent',         +1, 'flanker proximity'),
    ('pes_BI',                 'Post-error slowing (B incong)',     +1, 'Rabbitt 1966'),
    ('pia_BI',                 'Post-error accuracy (B incong)',    +1, 'post-error improvement'),
    ('peri',                   'PERI (interference drops)',         +1, 'Ridderinkhof 2002'),
    ('sce_acc_repeat',         'Gratton, response repeats',         +1, 'Gratton 1992'),
    ('sce_acc_switch',         'Gratton, response switches',        +1, 'Mayr, Awh & Laurey 2003'),
    ('lag2_contrast_acc',      'Lag-2 history contrast',            +1, 'model prediction'),
]
