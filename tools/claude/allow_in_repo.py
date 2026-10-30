#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""PermissionRequest hook: approve the permission dialog of a simple file operation inside this repository.

User rules (2026-10-30): "You have full and absolute control of all the code in this repo and in this directory."
and "Why did you just ask me for a confirmation dialog? You should never do that in this repo".

MINIMAL ALLOW LIST (after two adversarial reviews, reviews/2026-10-30-*.md). The hook approves a call only when:
  - the working folder is the repository or a folder in it;
  - Bash: the command is ONE simple command (no operator, redirect, quote, expansion or cd), the program is rm,
    rmdir, mv, "git rm" or "git mv", the options come from a fixed set, and each other word is a plain relative
    path ([A-Za-z0-9_][A-Za-z0-9_./-]*) that resolves inside the repository;
  - Write/Edit: an absolute path inside the repository;
  - and no path: is the repository root, has a symlink in it (any component), is a PROTECTED path or inside one,
    or contains a PROTECTED path (an ancestor of it); Bash also may not touch NO_DELETE (the review records).
    The protected check ignores case: the disk ignores case.
Anything else from this project (working folder in the repository or in the project scratchpad) gets a
DENY with the reason: the user wants no dialogs here, and a deny runs nothing. Calls from other projects get no
decision: the normal permission flow continues there. The PreToolUse hooks (mounted disks, AWS,
simulation gate) run before this hook and still block. Each approval goes to .local/permission-hook.log.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(os.path.realpath(Path(__file__).resolve().parents[2]))
# the scratchpad folder of the Claude Code sessions of this project (subagents work there too)
SCRATCH = Path(os.path.realpath(f"/private/tmp/claude-{os.getuid()}")) / ("-" + str(REPO).strip("/").replace("/", "-"))
DENY_MESSAGE = ("os-simulator-dev rule: no permission dialogs in this project (user, 2026-10-30), so this call is "
                "denied instead of asked. Allowed without a dialog: ONE simple command rm, rmdir, mv, git rm or git mv "
                "with plain relative paths inside the repository (no dot folders, no protected paths), and "
                "Write/Edit inside the repository. For scratch work, use a uv Python script (tempfile cleans up "
                "itself) instead of rm/mv/git reset in a shell. See CLAUDE.md, rule 6.")
PROTECTED = [".git", ".githooks", ".claude", "tools/claude", "tools/review.sh", "tools/hygiene.py"]
NO_DELETE = PROTECTED + ["reviews"]     # reviewers write their records; nobody deletes them
FLAGS = {"rm": {"-r", "-f", "-rf", "-fr", "-R", "-Rf", "-v"}, "rmdir": {"-p"}, "mv": {"-f", "-n", "-v"},
         "git rm": {"-r", "-f", "-q", "--cached", "-rf"}, "git mv": {"-f", "-k", "-v"}}
WORD = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./-]*$")
SIMPLE = re.compile(r"^[A-Za-z0-9_./ -]+$")          # the whole command: no quotes, $, ~, ;, &, |, <, >, (, ), etc.


def _key(p: Path) -> str:
    return os.path.normcase(str(p)).lower()


def path_ok(p: Path, protected: list[str] = PROTECTED) -> bool:
    """p is absolute and normalized (no ..); it must be strictly inside the repo, symlink-free and not touch a
    protected path."""
    if p == REPO or not p.is_relative_to(REPO):
        return False
    rel = p.relative_to(REPO)
    cur = REPO
    for part in rel.parts:                       # no symlink in any component (also a broken one)
        cur = cur / part
        if cur.is_symlink():
            return False
    if os.path.realpath(p) != str(p):
        return False
    k = _key(rel)
    for q in protected:
        qk = q.lower()
        if k == qk or k.startswith(qk + "/") or qk.startswith(k + "/"):
            return False
    return True


def bash_ok(command: str, cwd: Path) -> bool:
    if not SIMPLE.match(command) or "\n" in command:
        return False
    words = command.split()
    if not words:
        return False
    prog = words[0]
    args = words[1:]
    if prog == "git":
        if len(words) < 2 or words[1] not in ("rm", "mv"):
            return False
        prog, args = "git " + words[1], words[2:]
    if prog not in FLAGS:
        return False
    paths = [a for a in args if not a.startswith("-")]
    if not paths or any(a not in FLAGS[prog] for a in args if a.startswith("-")):
        return False
    for a in paths:
        if not WORD.match(a) or ".." in Path(a).parts:
            return False
        if not path_ok(Path(os.path.normpath(cwd / a)), NO_DELETE):
            return False
    return True


def in_project(cwd: Path) -> bool:
    return cwd == REPO or cwd.is_relative_to(REPO) or cwd == SCRATCH or cwd.is_relative_to(SCRATCH)


def decide(data: dict) -> bool:
    cwd = Path(os.path.realpath(data.get("cwd") or os.getcwd()))
    if not (cwd == REPO or cwd.is_relative_to(REPO)):
        return False
    tool = data.get("tool_name", "")
    ti = data.get("tool_input") or {}
    if tool in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
        p = ti.get("file_path") or ti.get("notebook_path")
        if not p or not os.path.isabs(p) or ".." in Path(p).parts:
            return False
        return path_ok(Path(os.path.normpath(p)))
    if tool == "Bash":
        return isinstance(ti.get("command"), str) and bash_ok(ti["command"], cwd)
    return False


def main() -> int:
    try:
        data = json.load(sys.stdin)
        ok = decide(data)
        project = in_project(Path(os.path.realpath(data.get("cwd") or os.getcwd())))
    except Exception:  # noqa: BLE001 - on any doubt, no decision
        return 0
    if not ok:
        if project:      # never a dialog in this project: deny (nothing runs) with the reason
            print(json.dumps({"hookSpecificOutput": {"hookEventName": "PermissionRequest",
                                                     "decision": {"behavior": "deny", "message": DENY_MESSAGE}}}))
        return 0
    if not os.environ.get("ALLOW_IN_REPO_NOLOG"):
        try:
            log = REPO / ".local" / "permission-hook.log"
            log.parent.mkdir(exist_ok=True)
            ti = data.get("tool_input") or {}
            what = ti.get("command") or ti.get("file_path") or ""
            with log.open("a") as f:
                f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{data.get('tool_name')}\t{what!r}\n")
        except OSError:
            pass
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PermissionRequest",
                                             "decision": {"behavior": "allow"}}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
