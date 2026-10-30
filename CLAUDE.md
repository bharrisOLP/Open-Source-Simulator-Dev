# os-simulator-dev: GPU versions of open-source photonics simulators

This repository makes open-source, GPU-accelerated versions of the simulators that a photonics R&D team uses
(FDTD, mode solvers, EME, band solvers, ray and physical optics, charge, thermal). The first target hardware is an
AMD GPU (Radeon RX 7900 XT, gfx1100, ROCm). Remote: `https://github.com/bharrisOLP/Open-Source-Simulator-Dev`
(**public**).

`STATUS.md` holds the current tool, the queue and the decisions. Read it first. Update it at each step.

## The rules of the user (2026-10-30)

1. Make open-source versions of all the tools: Meep, meow, Tidy3D, EME, FDTD, MPB, ZEMAX; also charge and thermal.
2. Work on one tool at a time. Keep `STATUS.md` current: the tool in work, and the queue in order of complexity.
3. Do not ask the user questions. Do not ask for approvals. Keep going, and make new ideas, until the session ends.
4. Never delete anything on the Ubuntu test computer (the GPU computer), or on the Mac outside this repository. Full
   control inside this repository. Put each file that you want deleted outside the repository into
   `docs/deletion-requests.md`; the user reviews that list.
5. "Before you commit in this repo, always make sure the changes go through an adversarial review."
6. "You should never [show a confirmation dialog] in this repo." (Also from subagents.) The hook
   `tools/claude/allow_in_repo.py` (registered in `~/.claude/settings.json`, PermissionRequest, for Bash, Write and
   Edit) approves a minimal allow list: ONE simple command `rm`, `rmdir`, `mv`, `git rm` or `git mv` with plain
   relative paths inside the repository (no dot folders, no protected paths), and Write/Edit inside the
   repository. Any other call from the repository or the project scratchpad is DENIED with the reason (never a
   dialog). Rules for the main agent and for each subagent prompt:
   - Do not use WebFetch (the hook does not cover it; new domains open a dialog). WebSearch is allowed.
   - Do not use `git clean`, `git reset --hard`, `find -delete` or compound commands with `rm` or `mv`.
   - Run scratch git tests inside a `uv run` Python script with `tempfile.TemporaryDirectory`, or with pytest.

## Public repository: no confidential data

The repository is public. Do not add PDK data, stack values, process data, client names, client results,
measurement data, credentials, host names, IP addresses, private folder paths, or the work email of the user.
`tools/remote.sh` names the host and private folders, so it is a local file (git-ignored). Use synthetic
test structures with values from public literature. Git commits use the GitHub no-reply email
(`336213665+bharrisOLP@users.noreply.github.com`).

## Commits

ONE adversarial review per commit, never a second one (user, 2026-10-30: "you should never have more than 1
adversarial review"). Keep each commit small, so that one review covers it.

1. Stage the change. Run `bash tools/review.sh new`. It writes `reviews/<date>-<hash12>.md` for the staged diff.
2. Start ONE independent reviewer (a subagent that did not write the change), with a time limit (about 15
   minutes) and the tool limits of rule 6. It reads `bash tools/review.sh diff`, tries to break the change (wrong
   physics, wrong numbers, untested paths, security, private data), fills the record and sets `verdict: pass` or
   `verdict: fail`.
3. If the verdict is fail: fix each blocker and major finding, stage the fixes, run
   `bash tools/review.sh fixed <record>`, and write one line per finding under "Fixes by the author" (the fix with
   file:line, or why no change). Do not start a second review.
4. Stage the record and commit. `.githooks/pre-commit` runs `tools/hygiene.py` (no private data) and
   `tools/review.sh check` (a passed or fixed record for the exact staged diff). Do not use `--no-verify`. Run
   `git config core.hooksPath .githooks` once per clone.

## The GPU computer

- Use only `bash tools/remote.sh <command>` (status, usage, push, sh, setup-env). The file is local (not in git). The simulation gate (a hook)
  blocks a direct ssh, scp or rsync, and it reads each `remote.sh sh` heredoc.
- `remote.sh sh` runs builds and short development tests (minutes, small memory). A long benchmark or a validation
  run whose numbers go into a report runs as a job, with a log and a record.
- The simulation gate blocks some solver words in shell commands. Put such code into a file in the repository.
  Never reword a command only to pass the gate.
- Share the computer: other sessions run jobs there. Run `usage` before a large job. Use one GPU job at a time.
- Do not delete files on the computer. `push` deletes nothing there.
- **Run every test and every bench on the GPU computer, never on the Mac** (user, 2026-10-30: "All tests need to go
  to the ubuntu pc"). Also the tests of agents and reviewers. Push first, then run through the GPU lock:
  `bash tools/remote.sh sh <<'EOF'` / `bash tools/gpu_run.sh <folder> python -m pytest -q` / `EOF`.
  `tools/gpu_run.sh` takes the lock `.local/gpu.lock` (one gpu_run.sh process at a time in this repository; test
  runs at the same time failed and gave wrong timings) and uses `.venv-rocm` (`.venv-optics` for optics_gpu). The
  lock does not cover jobs of other sessions: run `usage` before a large run. On the Mac, only
  static checks: `tools/hygiene.py`, `tools/review.sh`, reading code.

## Code

- Python only through uv (`uv run`, `uv add`). JavaScript only through bun. Set `PYTHONDONTWRITEBYTECODE=1`.
  A new dependency must also go into the GPU environment (`tools/env/rocm_jax.sh` or `tools/env/optics_rocm.sh`).
- Each solver has a CPU reference path and tests that compare GPU against CPU and against an independent solver
  (the upstream tool, an analytic case, or a published number).
- Keep the license of each upstream project. A fork of a GPL project stays GPL.
- Write prose in ASD-STE100 Simplified Technical English.
