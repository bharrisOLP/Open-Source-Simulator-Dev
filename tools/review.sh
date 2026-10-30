#!/usr/bin/env bash
# Adversarial review gate of this repository (user rule, 2026-10-30: "Before you commit in this repo, always make
# sure the changes go through an adversarial review.").
#
#   bash tools/review.sh hash     the SHA-256 of the staged diff (review records are not part of it)
#   bash tools/review.sh diff     the staged diff that the reviewer reads (the same bytes as the hash)
#   bash tools/review.sh new      write the template reviews/<date>-<hash12>.md for the staged diff
#   bash tools/review.sh fixed R  after ONE failed review: bind record R to the fixed staged diff (fixed_sha256,
#                                 verdict: fixed) and add the "## Fixes by the author" section to fill
#   bash tools/review.sh check    the pre-commit check (.githooks/pre-commit runs the copy of HEAD)
#
# ONE adversarial review per commit (user, 2026-10-30: "you should never have more than 1 adversarial review").
# A review record is a file reviews/YYYY-MM-DD-<12 hex>.md. Only such files are outside the hashed diff; any other
# file under reviews/ is part of it. The check passes if a record that is staged in this commit has, in its front
# matter (between the first two "---" lines), a "reviewer:" that is not empty and not TODO, and either
#   - "diff_sha256: <hash of the staged diff>" and "verdict: pass"  (the review found nothing to fix), or
#   - "fixed_sha256: <hash of the staged diff>" and "verdict: fixed", and a "## Fixes by the author" section with
#     no TODO in it (the author fixed the findings of the one review and lists each fix; no second review).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

RECORD_RE='^reviews/[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9a-f]{12}\.md$'
RECORD_GLOB='reviews/[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]-[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f].md'

staged_diff() {
  git diff --cached --binary --full-index --no-color --no-ext-diff --no-textconv -- . ":(exclude,glob)$RECORD_GLOB"
}

diff_hash() {
  staged_diff | shasum -a 256 | cut -d' ' -f1
}

front_matter() {  # front_matter PATH: the lines between the first two "---" lines of the staged copy
  git show ":$1" | awk 'NR == 1 { if ($0 != "---") exit; next } $0 == "---" { exit } { print }'
}

fixes_section() {  # the "## Fixes by the author" section of the staged record (to the next "## " heading)
  git show ":$1" | awk '/^## Fixes by the author/ { on = 1; print; next } /^## / { on = 0 } on { print }'
}

record_ok() {  # record_ok PATH HASH: 0 if the staged record passes the diff HASH
  local fm fx
  fm=$(front_matter "$1")
  if grep -qxF "diff_sha256: $2" <<<"$fm" && grep -qx 'verdict: pass' <<<"$fm"; then
    :
  elif grep -qxF "fixed_sha256: $2" <<<"$fm" && grep -qx 'verdict: fixed' <<<"$fm"; then
    fx=$(fixes_section "$1")
    [[ $(wc -l <<<"$fx") -ge 2 ]] || return 1
    ! grep -q 'TODO' <<<"$fx" || return 1
  else
    return 1
  fi
  local rv
  rv=$(grep -E '^reviewer:' <<<"$fm" | head -n 1 | sed -E 's/^reviewer:[[:space:]]*//')
  [[ -n $rv && $rv != TODO* ]] || return 1
}

cmd_check() {
  local h staged others records f ok=""
  staged=$(git diff --cached --name-only --diff-filter=ACMRT)
  records=$(grep -E "$RECORD_RE" <<<"$staged" || true)
  others=$(git diff --cached --name-only -- . ":(exclude,glob)$RECORD_GLOB")
  if [[ -z $others ]]; then
    for f in $records; do
      front_matter "$f" | grep -qE '^diff_sha256: [0-9a-f]{64}$' \
        || { echo "BLOCKED: $f is not a review record (no diff_sha256 in its front matter)." >&2; exit 1; }
    done
    exit 0                       # only review records: nothing else to review
  fi
  h=$(diff_hash)
  for f in $records; do
    if record_ok "$f" "$h"; then ok=$f; break; fi
  done
  if [[ -z $ok ]]; then
    {
      echo "BLOCKED: no passed adversarial review for this staged diff (sha256 $h)."
      echo "Need a staged record with 'diff_sha256: $h' and 'verdict: pass', or (after the ONE review and the"
      echo "fixes) 'fixed_sha256: $h', 'verdict: fixed' and a filled 'Fixes by the author' section"
      echo "(bash tools/review.sh fixed <record>). Never a second review. See CLAUDE.md, 'Commits'."
    } >&2
    exit 1
  fi
  echo "review gate: $ok passes the staged diff $h"
}

cmd_fixed() {  # cmd_fixed RECORD: bind the record of the one review to the fixed staged diff
  local f=${1:-} h
  [[ -f $f && $f =~ $RECORD_RE ]] || { echo "usage: bash tools/review.sh fixed reviews/<record>.md" >&2; exit 2; }
  TMPF=$(mktemp)
  trap 'rm -f "${TMPF:-}"' EXIT
  h=$(diff_hash)
  awk -v h="$h" '
    NR == 1 && $0 == "---" { fm = 1; print; next }
    fm && /^fixed_sha256:/ { next }
    fm && /^verdict:/ { print "verdict: fixed"; next }
    fm && $0 == "---" { print "fixed_sha256: " h; fm = 0; print; next }
    { print }' "$f" >"$TMPF"
  cat "$TMPF" >"$f"
  if ! grep -q '^## Fixes by the author' "$f"; then
    printf '\n## Fixes by the author\nTODO: for each finding of this review: the fix (file:line) or why no change.\n' >>"$f"
  fi
  echo "$f: fixed_sha256 $h; fill the section 'Fixes by the author', then stage the record"
}

cmd_new() {
  local h f
  h=$(diff_hash)
  f="reviews/$(date +%Y-%m-%d)-${h:0:12}.md"
  if [[ -e $f ]]; then
    echo "$f"
    return
  fi
  mkdir -p reviews
  cat >"$f" <<EOF
---
diff_sha256: $h
date: $(date +%Y-%m-%dT%H:%M:%S%z)
reviewer: TODO (an independent agent or person, not the author of the change)
verdict: TODO (pass or fail)
---
# Adversarial review

## Scope
$(git diff --cached --stat -- . ":(exclude,glob)$RECORD_GLOB")

## Findings
TODO: each finding with file:line, severity (blocker, major, minor), and the action taken.

## What the reviewer tried to break
TODO
EOF
  echo "$f"
}

case "${1:-}" in
  hash) diff_hash ;;
  diff) staged_diff ;;
  new) cmd_new ;;
  fixed) cmd_fixed "${2:-}" ;;
  check) cmd_check ;;
  *) sed -n '2,20p' "$0"; exit 2 ;;
esac
