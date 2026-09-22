"""
hier_switch_hooks.py — per-trial perturbations of the latent update (the paper's optogenetics).

Lam et al. manipulate the switch itself: they silence the ACC→MD terminals during the
feedback of the first four trials after a reversal (switching is delayed, Fig 4h), and they
drive MD during the feedback of the first five trials after a *high-conflict* reversal
(switching speeds up and PFC cue velocity rises, Fig 5d,g). Both act on a handful of trials
at a known position in the block, then stop.

A `TrialHook` is that: a window defined in trials-since-the-reversal, and one thing done to
the latent update inside it. It is **default-off** — `config.trial_hook` is absent on every
config that existed before this module, and `predictive_learning` skips it — so no earlier
run changes.

Three kinds, each a plain dict so it survives json and travels in a session's meta:

    dict(kind='lu_scale', k=0,   trials=[1, 4])   ACC→MD silencing (k=0) and, at k>1, an
                                                  "ACC stimulation" that the paper never
                                                  did: label it as ours, not theirs.
    dict(kind='z_set',   z=[1, 1], trials=[1, 1]) the MD-activation analogue: drive both
                                                  latent units, then let the gradient take
                                                  over from there.

**What k = 0 does and does not stop.** It multiplies the optimiser's learning rate, so Z
does not move; the gradient is still computed, still pooled and still logged. That is the
point — in the paper the ACC error signal survives the silencing of its output to MD, and
here the logged `grad` is exactly that surviving signal.

    .venv/bin/python hier_switch/hier_switch_hooks.py        # self-test on a trained model
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

KINDS = ('lu_scale', 'z_set')


class TrialHook:
    """One perturbation, applied on the trials of a window after each reversal.

    `predictive_learning` calls `pre()` before the latent update of every trial and
    `post()` after it. The hook keeps its own count of trials-since-the-reversal from the
    context it is handed, so it needs nothing from the dataset.

    It also records, per trial, what it did: `lu_scale` (the factor the learning rate was
    multiplied by) and `clamped` (Z was overwritten, so the trial's Δz is not an update).
    Both arrays travel into `session.npz`, and the analyses exclude the trials they mark
    from anything that reads a latent update as an update.
    """

    def __init__(self, spec):
        spec = dict(spec)
        kind = spec.get('kind')
        if kind not in KINDS:
            raise ValueError(f"trial_hook kind must be one of {KINDS}, got {kind!r}")
        lo, hi = spec.get('trials', [1, 5])
        self.spec, self.kind = spec, kind
        self.lo, self.hi = int(lo), int(hi)
        self.k = float(spec.get('k', 1.0))
        self.z = np.asarray(spec.get('z', [0.0, 0.0]), dtype=float)
        self._prev_ctx, self._since = None, 0
        self._active, self._base_lr = False, None
        self.lu_scale, self.clamped = [], []

    @classmethod
    def from_spec(cls, spec):
        """A hook, or None when there is nothing to do (the default)."""
        return None if not spec else cls(spec)

    # ── the two call sites ────────────────────────────────────────────────────

    def pre(self, model, config, context_ids=None):
        """Before the trial's latent update: open or close the window, and apply."""
        ctx = self._context(context_ids)
        self._since = 1 if (self._prev_ctx is None or ctx != self._prev_ctx) else self._since + 1
        self._prev_ctx = ctx
        self._active = self._in_window()
        g = model.Z_optimizer.param_groups[0]
        if self._base_lr is None:
            self._base_lr = float(g['lr'])
        scale = 1.0
        if self._active and self.kind == 'lu_scale':
            scale = self.k
            g['lr'] = self._base_lr * self.k
        self.lu_scale.append(scale)
        self.clamped.append(False)

    def post(self, model, config):
        """After the trial's latent update: restore, and overwrite Z if that is the kind."""
        g = model.Z_optimizer.param_groups[0]
        if self.kind == 'lu_scale':
            g['lr'] = self._base_lr
        if self._active and self.kind == 'z_set':
            import torch
            with torch.no_grad():
                model.Z.copy_(torch.as_tensor(
                    np.broadcast_to(self.z, model.Z.shape).copy(),
                    dtype=model.Z.dtype, device=model.Z.device))
            model.Z.requires_grad_(True)
            model.latent = model.Z
            self.clamped[-1] = True

    # ── bookkeeping ───────────────────────────────────────────────────────────

    def _in_window(self):
        # Restricting a hook to low- or high-conflict reversals is done by running it on the
        # matching forced session (reversal_conflict), where every reversal is that level,
        # rather than by filtering here.
        return self.lo <= self._since <= self.hi

    @staticmethod
    def _context(context_ids):
        if context_ids is None:
            return None
        try:
            return int(np.asarray(context_ids.detach().cpu()).ravel()[-1])
        except AttributeError:
            return int(np.asarray(context_ids).ravel()[-1])

    def arrays(self, n=None):
        """The per-trial record, trimmed or padded to `n` trials."""
        out = dict(lu_scale=np.asarray(self.lu_scale, dtype=np.float32),
                   clamped=np.asarray(self.clamped, dtype=bool))
        if n is None:
            return out
        for k, v in out.items():
            if len(v) >= n:
                out[k] = v[:n]
            else:
                pad = np.zeros(n - len(v), dtype=v.dtype)
                if k == 'lu_scale':
                    pad = pad + 1
                out[k] = np.concatenate([v, pad])
        return out


def _self_test():
    """Run each kind on a trained model and check it did exactly what it says.

    Needs a saved model; skips with a message if none is on disk.
    """
    import copy
    from hier_switch_analyses import extract_trials, recover_grad, session_arrays
    from hier_switch_train import load_model, run_test

    from hier_switch_group import NG_SEEDS, NG_TAG
    path = os.path.join(_ROOT, 'exports', 'hier_switch', f'tune_{NG_TAG}',
                        f'NG_s{NG_SEEDS[0]}', 'model.pt')
    if not os.path.exists(path):
        print(f'no model at {path}: skipping'); return
    model, cfg = load_model(path)
    N = 200

    kept = {}

    def go(**kw):
        m = copy.deepcopy(model)
        logger, _, tcfg = run_test(m, copy.deepcopy(cfg), run_name='scratch/hook_test',
                                   n_trials=N, **kw)
        tr = extract_trials(logger, tcfg)
        kept['logger'], kept['cfg'] = logger, tcfg
        hook = getattr(tcfg, 'trial_hook', None)
        return tr, (hook.arrays(tr['n']) if hook is not None else None)

    base, _ = go()

    # The recorder takes the gradient from the logger rather than from Δz, so that it
    # survives a zeroed update. Two things have to hold for that: the pooled gradient is
    # broadcast to every timestep of the trial (so row 0 is the whole story), and on an
    # unperturbed trial the logged route agrees with the Δz route.
    tcfg0 = kept['cfg']
    gc = np.concatenate(kept['logger'].gradients_corrections, axis=0)
    gc = gc.reshape(base['n'], tcfg0.trial_len, -1)
    spread = float(np.nanmax(np.abs(gc - gc[:, :1, :])))
    logged = session_arrays(kept['logger'], tcfg0)['grad_logged']
    rec = recover_grad(base['z'], base['z_in'], tcfg0.Z_lr, tcfg0.Z_decay)
    fin = np.isfinite(rec).all(axis=1)
    diff = float(np.abs((logged - tcfg0.Z_decay * base['z_in'])[fin] - rec[fin]).max())
    ok = spread < 1e-12 and diff < 1e-10
    print(f"{'OK ' if ok else 'FAIL'} logged gradient: identical across the trial's "
          f"{tcfg0.trial_len} rows (spread {spread:.1e}) and agrees with the Δz route "
          f"(max diff {diff:.1e})")
    same, _ = go(perturb=dict(kind='lu_scale', k=1.0, trials=[1, 4]))
    ok = np.allclose(base['z'], same['z'], rtol=0, atol=0)
    print(f"{'OK ' if ok else 'FAIL'} lu_scale k=1 is bit-identical to no hook "
          f"(max |Δz| {np.abs(base['z'] - same['z']).max():.2e})")

    off, arr = go(perturb=dict(kind='lu_scale', k=0.0, trials=[1, 4]))
    # The session's very first trial has no incoming Z, so it is dropped from every check.
    fin = np.isfinite(off['z_in']).all(axis=1)
    win = (arr['lu_scale'] == 0) & fin
    dz = np.abs(off['z'][win] - off['z_in'][win])
    moved = np.abs(off['z'][~win & fin] - off['z_in'][~win & fin])
    frozen = dz.max() < 1e-12 and moved.max() > 0
    print(f"{'OK ' if frozen else 'FAIL'} lu_scale k=0 freezes Z on {win.sum()} window "
          f"trials (max |Δz| there {dz.max():.2e}; outside the window {moved.max():.2e}, "
          f"so the rest of the session still runs)")

    zs, arr = go(perturb=dict(kind='z_set', z=[1.0, 1.0], trials=[1, 1]))
    hit = arr['clamped']
    at = bool(hit.any()) and np.allclose(zs['z'][hit], 1.0, atol=1e-6)
    # The trial after each clamp must start from (1, 1) and then be free to move again.
    nxt = np.zeros_like(hit)
    nxt[1:] = hit[:-1]
    off_it = np.abs(zs['z'][nxt] - zs['z_in'][nxt]).max()
    print(f"{'OK ' if at and off_it > 0 else 'FAIL'} z_set puts Z at (1, 1) on "
          f"{hit.sum()} trials, and the next trial moves off it (max |Δz| {off_it:.2e})")


if __name__ == '__main__':
    _self_test()
