#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Public-repository hygiene check: no private data in the staged files (CLAUDE.md, "Public repository").

    uv run --script tools/hygiene.py            check the staged copy of each added or changed file (pre-commit)
    uv run --script tools/hygiene.py FILE...    check these files on disk

It blocks: private IPv4 addresses, e-mail addresses (except GitHub no-reply, noreply@anthropic.com, example.com
and the ssh remote syntax git@github.com), home-folder paths, the names of the private folders, hosts and organisation of the user. The words
come in parts below, so that this file does not match itself. Exit 1 lists each hit as path:line: rule.
"""
import re
import subprocess
import sys

J = "".join
RULES = [
    ("private IPv4 address", re.compile(r"\b(?:192\.168|10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b")),
    ("e-mail address",
     re.compile(r"(?<![\w.+-])(?!git@github\.com)[\w.+-]+@(?!users\.noreply\.github\.com|anthropic\.com|example\.com)[\w-]+\.[\w.-]+")),
    ("home-folder path", re.compile(r"/Users/[A-Za-z]|/home/[A-Za-z]|[A-Za-z]:\\\\Users")),
    ("private folder", re.compile(J(["~/(?:sim", "ulations|pd", "ks|sk", "ua|know", "ledge|data|co", "de)\\b"]))),
    ("private name", re.compile(J(["(?i)\\b(?:open", "light|ph18", "da|ori", "ole|bren", "dan|linux-", "sim|win-", "sim)\\b"]))),
]


def staged_files() -> list[str]:
    out = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMRT", "-z"],
                         capture_output=True, check=True).stdout
    return [p for p in out.decode().split("\0") if p]


def read(path: str, staged: bool) -> str:
    if staged:
        data = subprocess.run(["git", "show", f":{path}"], capture_output=True, check=True).stdout
    else:
        data = open(path, "rb").read()
    return data.decode("utf-8", errors="replace")


def scan(path: str, text: str) -> list[str]:
    hits = []
    for n, line in enumerate(text.splitlines(), 1):
        for name, rx in RULES:
            if rx.search(line):
                hits.append(f"{path}:{n}: {name}")
    return hits


def main(argv: list[str]) -> int:
    staged = not argv
    hits = []
    for path in (staged_files() if staged else argv):
        if path.endswith(("uv.lock", ".png", ".svg", ".gds")):
            continue                      # lock files and binaries: package versions look like addresses
        hits += scan(path, read(path, staged))
    if hits:
        print("BLOCKED by the hygiene check (public repository, CLAUDE.md):", *hits, sep="\n  ", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
