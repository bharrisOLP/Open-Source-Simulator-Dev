"""Tests of the adversarial review gate (tools/review.sh, .githooks/pre-commit) in a temporary git repository."""
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def run(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(list(args), cwd=cwd, capture_output=True, text=True, check=check)


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    (tmp_path / "tools").mkdir()
    shutil.copy(REPO / "tools" / "review.sh", tmp_path / "tools" / "review.sh")
    shutil.copy(REPO / "tools" / "hygiene.py", tmp_path / "tools" / "hygiene.py")
    shutil.copytree(REPO / ".githooks", tmp_path / ".githooks")
    run(tmp_path, "git", "init", "-q", "-b", "main")
    run(tmp_path, "git", "config", "user.name", "t")
    run(tmp_path, "git", "config", "user.email", "t@example.com")
    run(tmp_path, "git", "config", "core.hooksPath", ".githooks")
    return tmp_path


def commit(repo: Path) -> subprocess.CompletedProcess:
    return run(repo, "git", "commit", "-q", "-m", "x", check=False)


def review(repo: Path, verdict: str = "pass", reviewer: str = "independent agent", extra: str = "") -> Path:
    rec = repo / run(repo, "bash", "tools/review.sh", "new").stdout.strip()
    text = rec.read_text().replace("verdict: TODO (pass or fail)", f"verdict: {verdict}")
    text = text.replace("reviewer: TODO (an independent agent or person, not the author of the change)",
                        f"reviewer: {reviewer}")
    rec.write_text(text + extra)
    run(repo, "git", "add", str(rec))
    return rec


def first_commit(repo: Path) -> None:
    (repo / "a.txt").write_text("a\n")
    run(repo, "git", "add", "-A")
    review(repo)
    assert commit(repo).returncode == 0


def test_commit_without_review_is_blocked(repo):
    (repo / "a.txt").write_text("a\n")
    run(repo, "git", "add", "-A")
    r = commit(repo)
    assert r.returncode != 0
    assert "no passed adversarial review" in r.stderr


def test_failed_review_is_blocked(repo):
    (repo / "a.txt").write_text("a\n")
    run(repo, "git", "add", "-A")
    review(repo, verdict="fail")
    assert commit(repo).returncode != 0


def test_verdict_in_the_body_does_not_count(repo):
    (repo / "a.txt").write_text("a\n")
    run(repo, "git", "add", "-A")
    review(repo, verdict="fail", extra="\nverdict: pass\n")
    assert commit(repo).returncode != 0


def test_reviewer_todo_is_blocked(repo):
    (repo / "a.txt").write_text("a\n")
    run(repo, "git", "add", "-A")
    review(repo, reviewer="TODO later")
    assert commit(repo).returncode != 0


def test_passed_review_commits_and_a_later_change_needs_a_new_review(repo):
    first_commit(repo)
    (repo / "a.txt").write_text("b\n")
    run(repo, "git", "add", "-A")
    assert commit(repo).returncode != 0


def test_unstaged_record_does_not_count(repo):
    (repo / "a.txt").write_text("a\n")
    run(repo, "git", "add", "a.txt", "tools", ".githooks")
    rec = repo / run(repo, "bash", "tools/review.sh", "new").stdout.strip()
    rec.write_text(rec.read_text().replace("verdict: TODO (pass or fail)", "verdict: pass")
                   .replace("reviewer: TODO (an independent agent or person, not the author of the change)",
                            "reviewer: x"))
    assert commit(repo).returncode != 0


def test_other_files_under_reviews_need_a_review(repo):
    first_commit(repo)
    (repo / "reviews" / "evil.py").write_text("print(1)\n")
    run(repo, "git", "add", "-A")
    assert commit(repo).returncode != 0


def test_textconv_does_not_hide_content(repo):
    (repo / ".gitattributes").write_text("*.py diff=hide\n")
    run(repo, "git", "config", "diff.hide.textconv", "sh -c 'echo same'")
    (repo / "x.py").write_text("one\n")
    run(repo, "git", "add", "-A")
    h1 = run(repo, "bash", "tools/review.sh", "hash").stdout
    (repo / "x.py").write_text("two\n")
    run(repo, "git", "add", "-A")
    h2 = run(repo, "bash", "tools/review.sh", "hash").stdout
    assert h1 != h2
    assert "two" in run(repo, "bash", "tools/review.sh", "diff").stdout


def test_a_staged_change_of_the_gate_cannot_approve_itself(repo):
    first_commit(repo)
    gate = repo / "tools" / "review.sh"
    gate.write_text(gate.read_text().replace('  check) cmd_check ;;', '  check) exit 0 ;;'))
    run(repo, "git", "add", "-A")
    assert commit(repo).returncode != 0


def test_a_first_commit_without_the_gate_is_blocked(repo):
    (repo / "a.txt").write_text("a\n")
    run(repo, "git", "add", "a.txt")                 # tools/ not staged: no gate in the index, no HEAD
    assert commit(repo).returncode != 0


def test_hygiene_blocks_private_data_also_in_records(repo):
    first_commit(repo)
    rec = repo / "reviews" / "2026-01-01-0123456789ab.md"
    rec.write_text("---\ndiff_sha256: " + "0" * 64 + "\n---\nhost " + "192." + "168.1.20\n")
    run(repo, "git", "add", str(rec))
    r = commit(repo)
    assert r.returncode != 0 and "hygiene" in r.stderr


def test_hygiene_blocks_private_data_in_reviewed_files(repo):
    first_commit(repo)
    (repo / "b.txt").write_text("see " + "/Us" + "ers/someone/notes\n")
    run(repo, "git", "add", "-A")
    review(repo)
    r = commit(repo)
    assert r.returncode != 0 and "hygiene" in r.stderr


def test_one_review_then_fixes_commits_without_a_second_review(repo):
    first_commit(repo)
    (repo / "b.txt").write_text("v1\n")
    run(repo, "git", "add", "-A")
    rec = review(repo, verdict="fail")                  # the one review finds a problem
    (repo / "b.txt").write_text("v2 fixed\n")           # the author fixes it
    run(repo, "git", "add", "b.txt")
    run(repo, "bash", "tools/review.sh", "fixed", str(rec.relative_to(repo)))
    run(repo, "git", "add", str(rec))
    assert commit(repo).returncode != 0, "the fix list still has TODO"
    rec.write_text(rec.read_text().replace("TODO: for each finding of this review: the fix (file:line) or why no change.",
                                           "- F1: b.txt:1 corrected."))
    run(repo, "git", "add", str(rec))
    assert commit(repo).returncode == 0


def test_fixed_record_must_match_the_fixed_diff(repo):
    first_commit(repo)
    (repo / "b.txt").write_text("v1\n")
    run(repo, "git", "add", "-A")
    rec = review(repo, verdict="fail")
    run(repo, "bash", "tools/review.sh", "fixed", str(rec.relative_to(repo)))
    rec.write_text(rec.read_text().replace("TODO: for each finding of this review: the fix (file:line) or why no change.",
                                           "- F1: fixed."))
    (repo / "b.txt").write_text("v3 changed after the fixed hash\n")
    run(repo, "git", "add", "-A")
    assert commit(repo).returncode != 0
