# lab/ — investigations that live beside the code, not in it

`lab/` sits at the repo root and is **ignored by git** (`.gitignore`). Nothing in it
reaches a collaborator through the repo; what they need arrives through an
*Action on main* (below). Git is not its backup — keep that in mind before deleting a
checkout.

## Why

The tracked repo holds things with a permanent reader: the core (`configs.py`,
`models.py`, `datasets.py`, `train_and_infer_functions.py`, `functions_and_utils.py`), the
`<task>_*.py` families and `hier_switch/`, the `run_*.sh` / `submit_job.sh` wrappers that
produce results we keep, and the writeups under `docs/`. Everything else — "does this
knob matter", "why do these two numbers disagree", one-off probes, parameter pokes, sanity
checks, paper comparisons — is an *investigation* with a reader for a week. It goes in a
lab so that an intervention that may never be adopted does not leave a config flag, a
sweep variant, a figure function and a paragraph of docs behind in the main code.

A lab **extends** the main code without editing it: it imports the repo, overrides what
it needs at runtime, and writes only into its own folders.

**What is not a lab.** A lab is for exploratory changes that would otherwise leave a
feature behind in main: changing the model or its training, or fundamentally changing an
analysis convention that many figures depend on. Adding a figure, a few per-seed measures,
or a re-read of results already on disk is not — put it straight into the task family (a
`fig_*` in the figure script, drawn by default, with its `spec_*` panel shared by the
workbench). Example: the post-conflict figure (`group_14_post_conflict`) and the
congruent-trial control axes (`group_13_control_axes_cong`) were briefly a lab and should
not have been.

## Structure

One folder per **question**, prefixed by start date (dates sort, the slug is the
retrieval key):

    lab/
      2026-10-01_oracle-gate-jitter/     the worked example
        NOTE.md     question · intervention · what was run · findings · action on main
        papers/     PDFs / references specific to this question (optional)
        code/       scripts and the sbatch wrapper; import the repo (see Rules)
        out/        small outputs: figures, CSV, text logs

Large artifacts — trained models, session pickles — do not go in `out/`. They go under
`exports/lab/<issue>/`, next to every other experiment's results (`exports/` is the shared
data symlink). NOTE.md says where.

To start a new question, copy the worked example's NOTE.md into a new dated folder, empty
`code/` and `out/`, and rewrite its sections.

## NOTE.md

Sections, in order:

- **Status** — `open` | `closed`.
- **Question** — one or two sentences.
- **Intervention** — exactly what is changed relative to main, and how (which config
  attribute, which runtime override), so it can be re-implemented without the lab code.
- **Prediction** — written before looking at results.
- **What was run** — one line per run: script → outputs.
- **Findings** — numbers and verdicts, specific enough that nothing needs re-running to be
  believed, with the figure that shows it (`out/*.png`). Report the way the project
  reports anything else: mean ± SEM across seeds and the number of seeds in the predicted
  direction.
- **Log** — one dated line per work session.
- **Action on main:** — the last line. `none`, or what should change and where it landed
  (commit sha).

## Closing a question

At the end of every work session on it, and again when it is done:

1. **NOTE.md is the audit trail.** Update *Findings*, add a *Log* line, keep *What was
   run* current.
2. **RECOMMENDATIONS.md is the deliverable**, written only when the lab produces actions
   beyond itself (a change to main, a doc correction, a figure for the paper). It must be
   readable standalone: a short summary of the evidence, then concrete "change X at
   location Y" items, then an explicit *not recommended* list. NOTE.md links to it from
   the *Action on main* line.
3. **Close the loop explicitly.** Flip *Status* to `closed` only when *Action on main* is
   final: `none`, or each action resolved with the commit it landed in. A lab with
   pending recommendations stays `open`.
4. **Summaries live in the lab folder**, not in chat scrollback or a shared doc. A result
   that must reach main goes in as a small named commit through the normal route —
   rewritten to the repo's conventions, with the task's `docs/*.md` updated — and NOTE.md
   records the sha.
5. Outputs a summary cites (`out/*.png`, `out/*.txt`) stay put. Do not regenerate them
   after the summary is written unless the summary is updated with them.

## Rules

- **Importing the repo.** Scripts put the repo on the path and run under its venv:

      REPO = Path(__file__).resolve().parents[3]
      sys.path.insert(0, str(REPO))
      os.chdir(REPO)          # repo paths (./exports/...) are relative to the root

      .venv/bin/python lab/<issue>/code/script.py      # from the repo root

- **Override, don't edit.** Change behaviour by setting config attributes or module
  globals at runtime (e.g. `flanker_sweep.RUN_NAME`, `flanker_sweep.VARIANTS`), never by
  editing a tracked file. If the intervention genuinely cannot be expressed without a
  code change, that is itself a finding — note it, and make the change on a branch.
- **Write only into the lab's own places**: `lab/<issue>/out/` and
  `exports/lab/<issue>/`. Never into another experiment's export folder, never into
  `docs/`. Lab scripts may *read* any existing results, e.g. a sweep's pickles as the
  baseline.
- **Conventions are relaxed, not abandoned.** Lab figures should still use `plot_style`
  and `FigSize` so a panel can be lifted into a paper figure, but quick diagnostics are
  fine. Lab code is never pasted into main as is; promote it by rewriting it into the
  relevant task family.
- **Cluster.** Same law as the repo: training and sweeps go to Slurm, never the login
  node. Put the sbatch wrapper in `code/`, show it before submitting, and get explicit
  approval for array jobs. Logs go to `./slurm/` as usual.
- Every NOTE.md ends with an explicit **Action on main:** line. It is the only path from
  lab to main.
- Agents (Claude Code) put exploratory work here rather than adding flags, variants or
  scripts to main.
