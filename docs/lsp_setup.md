# Python LSP for Claude Code

Set up 2026-08-24 on Oscar. Gives Claude Code's built-in LSP tool go-to-definition,
find-references, and type diagnostics after every edit, instead of grep-based navigation.

Three pieces: a language-server **binary**, the **plugin** that wires it into Claude Code,
and a **project config** so the server resolves `torch`/`numpy` from the right venv.

---

## 1. Language server binary

`basedpyright` (a Pyright fork) installed into the shared venv `/users/ahummos/venvs/neo`,
which this repo's `.venv` symlinks to:

```bash
.venv/bin/pip install basedpyright
```

Chosen over `pip install pyright` because its `nodejs-wheel-binaries` dependency bundles a
Node runtime in the wheel. There is no `node`/`npm` on Oscar's default PATH — only a
`node-js/22.16.0-5pf2` module — and stock `pyright` would otherwise download Node on first
run. This way nothing depends on a module being loaded when Claude Code starts.

The plugin (step 2) invokes the bare command `pyright-langserver`, so it must resolve on the
PATH of the Claude Code process. `~/.local/bin` is *not* on PATH here; `~/bin` is:

```bash
ln -sfn /users/ahummos/venvs/neo/bin/basedpyright-langserver ~/bin/pyright-langserver
```

The venv's console scripts carry an absolute shebang (`#!/users/ahummos/venvs/neo/bin/python3`),
so the symlink works without activating the venv.

> **Note:** step 1 modifies the shared `neo` venv, which other NeuraGEM checkouts also
> symlink to. It adds ~75 MB and no runtime imports — nothing in the training code
> imports basedpyright.

## 2. Plugin

```bash
claude plugin install pyright-lsp@claude-plugins-official
```

`pyright-lsp` is a wiring-only plugin from the official marketplace — it ships a README and a
`lspServers` entry (`pyright-langserver --stdio`, mapping `.py`/`.pyi` → `python`) and no
binary. Installed at **user scope**, so it applies to every project on this account; it lands
in `~/.claude/settings.json` under `enabledPlugins`. **Restart Claude Code** for the server to
start.

## 3. Project config

`pyrightconfig.json` at the repo root. Without `venvPath`/`venv` the server uses whichever
interpreter it finds first and reports every scientific import as missing:

```json
{
  "venvPath": "/users/ahummos/venvs",
  "venv": "neo",
  "pythonVersion": "3.11",
  "typeCheckingMode": "basic"
}
```

`typeCheckingMode` matters: basedpyright defaults to `recommended`, which turns on rules like
`reportAny` and `reportUnusedCallResult` and buries real errors in style noise on a research
codebase. `basic` keeps the checks that catch actual bugs.

Also set:

| Key | Value | Why |
| --- | --- | --- |
| `exclude` | `__pycache__`, `.venv`, `archive`, `exports` | Don't index dead code or run outputs |
| `reportMissingModuleSource` | `none` | Silences "stub found but source missing" for C-extension packages |
| `reportUnusedExpression` | `none` | Bare expressions are normal in analysis scripts |

---

## Verifying

```bash
.venv/bin/basedpyright --outputjson models.py
```

Expect `filesAnalyzed: 1`, a couple of seconds, and **zero** `reportMissingImports` — that's
the signal the venv wiring is right. On first run against `models.py` it reported 4 real
`Optional`-related errors (lines 394, 517, 518, 601) and nothing else, so the baseline noise
level is low.

If imports suddenly go missing, check that `neo` still exists and that
`~/bin/pyright-langserver` isn't a dangling symlink.
