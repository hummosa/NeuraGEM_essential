# labs/ — testing an idea without cluttering the main code

A **lab** is a small, closed-off space for testing one idea: an intervention, an
effect, a mechanism we're not sure about. It builds on the main code but adds
nothing to it. The lab writes down what was changed and what happened (ideally
with a figure). Later, if that intervention turns out to be needed, the record is
there, and so is the code to bring back.

## Why

The main code — `models.py`, `train_and_infer_functions.py`, `datasets.py`,
`configs.py`, the task folders, and their docs under `docs/` — should only hold
things we'll keep reading. A config switch added "to try something", a training
trick, or a probe of whether an effect survives a manipulation, has a reader for
a week. Put it in core and it stays forever: a branch in the training loop, a
config key nobody remembers, a paragraph in a task doc saying something that
stopped being true. Put it in a lab and it stays where it was tried, with its
result next to it.

`hier_switch/` is the model for keeping code self-contained. It is a whole task
in one folder: it puts the repo root on `sys.path` and imports `models`,
`train_and_infer_functions` and `datasets`. Core never imports it back. Its one
reach into the training loop is a seam that is off by default
(`config.trial_hook`, a few lines in `_latent_update_step`). A lab works the same
way, at a smaller scale: one question rather than one task. Its notes stay in its
own folder, not in `docs/`.

## Structure

One folder per **question**, starting with the date it began (so folders sort
by date; the rest of the name is what you search for):

    labs/
      README.md                       index: one line per lab
      2026-10-01_z-weight-decay/
        NOTE.md                       intervention · how to run · outcome · action on main
        *.py, *.sh                    the lab's code, flat (like hier_switch/)
        figures/                      the figures NOTE.md cites (small PDF/PNG, committed)
        papers/                       optional: references specific to this question

Heavy outputs (checkpoints, recorded sessions, sweeps) go to
`exports/labs/<lab-folder>/`, which is gitignored like the rest of `exports/`.
Only the figures and numbers that NOTE.md cites are committed, in `figures/`.

To start a lab: create the folder, copy the NOTE.md template below, and add a
line to `labs/README.md`.

## NOTE.md

```markdown
# <one-line question>

**Status:** open | closed
**Ran at:** <commit sha of the main code the results were produced against>
**Action on main:** none | <what, and where it landed: sha / PR>

## Intervention
What is changed relative to main, exactly: config keys and values, which
function is wrapped or replaced, which model/task/seeds. Someone reading
this in a year should be able to rebuild it without the lab's code.

## How to run
    .venv/bin/python labs/<lab>/<script>.py ...
Script → outputs, one line per run (local or the sbatch script used).

## Outcome
The numbers and the verdict, specific enough to believe without re-running.
![](figures/<main-figure>.png)
What it does *not* show, and the explanations not yet tested (label them as
hypotheses).

## Log
- 2026-10-01 — what happened this session.
```

The **Intervention** section is the part that has to stay true longest. Main
will move on and the lab's code may stop running. The description should still
be enough to rebuild the intervention.

## Rules

- **Importing the repo.** Lab scripts run from the repo root under its venv and
  put the root on the path the way `hier_switch/` does:

      _HERE = os.path.dirname(os.path.abspath(__file__))
      _ROOT = os.path.dirname(os.path.dirname(_HERE))   # labs/<lab>/ → repo root
      for _p in (_HERE, _ROOT):
          if _p not in sys.path:
              sys.path.insert(0, _p)

      .venv/bin/python labs/<lab>/<script>.py

  A lab may import a task folder (`hier_switch/`) the same way.
- **Imports go one way.** Labs import main. Main never imports a lab, and no
  lab imports another lab: copy what you need and note the source.
- **Prefer not to edit core.** Make the intervention from inside the lab first:
  subclass a config, wrap or replace a function, or subclass `Logger`. If the
  change really has to reach inside the training loop, add a seam to core that
  is **off by default**: a `getattr(config, '<name>', None)` check of a few
  lines, commented with the lab's path, so that no existing config changes
  behaviour (the `trial_hook` pattern). Make it its own small commit and record
  its sha in NOTE.md.
- **Write only into the lab.** Lab code writes to `labs/<lab>/figures/` and
  `exports/labs/<lab>/`. It never writes into `docs/`, a task folder, or
  another experiment's `exports/` directory. Anything that has to change an
  existing artifact works on a copy.
- **Figures.** Use `plot_style` and the `FigSize` presets
  (`docs/figure_style.md`), so a figure can be promoted without redrawing.
  Otherwise lab figures don't need to meet the paper's standards.
- **Cluster runs** follow the same rules as everywhere on Oscar: `sbatch` only,
  nothing heavy on a login node. The sbatch script lives in the lab folder,
  `cd`s to the repo root, and writes logs to `./slurm/` like
  `hier_switch/run_*.sh`. Show the script to the user before submitting.
- **Docs stay out of `docs/`.** While the lab is open, nothing about it goes
  into a task doc, `overview.md` or a handoff, except possibly a one-line
  pointer ("tried in `labs/<lab>/`, see NOTE.md").
- **Agents** (Claude Code) doing exploratory work in this repo put it in a
  lab, not in a new top-level script or a new section of a task doc.

## Closing a lab

At the end of a work session on a lab, and again when it is done:

1. **Update NOTE.md.** Put the numbers and verdict in *Outcome*, keep *How to
   run* current, add a dated line to *Log*, and set *Ran at* to the sha the
   results came from.
2. **Don't regenerate cited figures** after the Outcome is written, unless you
   update the Outcome along with them.
3. **Set the Action on main line**, then flip Status to `closed`: either
   `none` (a negative or null result — still worth keeping), or each action
   with where it landed. A lab with an action not yet done stays `open`.
4. **Update its line in `labs/README.md`** with the status and a one-line
   verdict.
5. **Leave the folder in place.** Closed labs are the record. They are not
   kept up to date with main, and they are not deleted.

## Promoting into main

When a lab's intervention is judged necessary:

- **Rewrite, don't move.** Rewrite the code to main's conventions, in the
  module where it belongs, with a config key whose default leaves existing
  runs unchanged. Don't `git mv` lab scripts into the root.
- **Document it where it now lives:** the task's doc under `docs/`, or
  `configs.md`. Link back to the lab's NOTE.md for the evidence.
- **Make it a small named commit.** Record its sha on the lab's *Action on
  main* line.
- **Close the loop.** Once the action is recorded, flip the lab's Status to
  `closed` and update its line in `labs/README.md` (as in *Closing a lab*).
