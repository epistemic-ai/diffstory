#!/usr/bin/env bash
# Explicitly publish this reviewed release to a NEW public organization repository.
# Never force-push, reuse existing history, or put credentials into repository files.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPOSITORY="epistemic-ai/diffstory"
for executable in git gh python3; do
  if ! command -v "$executable" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n' "$executable" >&2
    exit 1
  fi
done
if [[ -e "$ROOT/.git" ]]; then
  printf 'Refusing an existing Git history. Run from a fresh release directory.\n' >&2
  exit 1
fi
# Avoid inheriting a parent checkout, another Git directory, or an injected index.
if [[ -n "${GIT_DIR:-}" || -n "${GIT_WORK_TREE:-}" || -n "${GIT_INDEX_FILE:-}" ]]; then
  printf 'Unset GIT_DIR, GIT_WORK_TREE and GIT_INDEX_FILE before publishing.\n' >&2
  exit 1
fi
cd "$ROOT"
python3 scripts/check_release.py --manifest
python3 -m unittest discover -s tests -v
gh auth status --hostname github.com
# Success here means the name is already taken; a nonzero response does NOT prove
# it is available. Only the subsequent create request can establish that.
if gh repo view "$REPOSITORY" --json nameWithOwner >/dev/null 2>&1; then
  printf 'Refusing to modify existing repository %s.\n' "$REPOSITORY" >&2
  exit 1
fi
# Use configured identity or verified authenticated profile; never a guessed email.
AUTHOR_NAME="$(git config --get user.name || true)"
AUTHOR_EMAIL="$(git config --get user.email || true)"
if [[ -z "$AUTHOR_NAME" || -z "$AUTHOR_EMAIL" ]]; then
  PROFILE="$(gh api user --jq '[.name // .login, .id, .login] | @tsv')"
  IFS=$'\t' read -r PROFILE_NAME PROFILE_ID PROFILE_LOGIN <<< "$PROFILE"
  [[ -n "$PROFILE_LOGIN" && "$PROFILE_ID" =~ ^[0-9]+$ ]] || { printf 'Could not resolve commit identity.\n' >&2; exit 1; }
  AUTHOR_NAME="${AUTHOR_NAME:-$PROFILE_NAME}"
  AUTHOR_EMAIL="${AUTHOR_EMAIL:-${PROFILE_ID}+${PROFILE_LOGIN}@users.noreply.github.com}"
fi
git -c core.hooksPath=/dev/null init --initial-branch=main
git config --local core.hooksPath /dev/null
git config --local user.name "$AUTHOR_NAME"
git config --local user.email "$AUTHOR_EMAIL"
# Add precisely the sealed file set, never `git add .`.
python3 - <<'PY'
import json
import subprocess
from pathlib import Path
manifest = json.loads(Path('PUBLICATION.json').read_text())
paths = [item['path'] for item in manifest['files']] + ['PUBLICATION.json']
# The checker already validated every relative path; separate flags from pathspecs.
subprocess.run(['git', 'add', '--', *paths], check=True)
PY
python3 scripts/check_release.py --manifest
git -c commit.gpgSign=false commit -m 'Release Diffstory 0.3.0: literate code review by Epistemic AI'
printf 'Creating NEW PUBLIC repository %s and pushing main.\n' "$REPOSITORY"
if ! gh repo create "$REPOSITORY" --public --source="$ROOT" --remote=origin --push \
    --disable-wiki --homepage='https://www.epistemic.ai' \
    --description='Literate, evidence-bound code walkthroughs. Read a diff as a story — local-first, offline HTML, Python AST analysis.'; then
  printf 'Create/push did not finish. Inspect GitHub and this local checkout before retrying; no remote cleanup or force-push was attempted.\n' >&2
  exit 1
fi
LOCAL_SHA="$(git rev-parse HEAD)"
REMOTE_SHA="$(gh api "repos/$REPOSITORY/git/ref/heads/main" --jq '.object.sha')"
IS_PRIVATE="$(gh repo view "$REPOSITORY" --json isPrivate --jq '.isPrivate')"
if [[ "$LOCAL_SHA" != "$REMOTE_SHA" || "$IS_PRIVATE" != 'false' ]]; then
  printf 'Remote verification failed: inspect the repository before announcing publication.\n' >&2
  exit 1
fi
URL="$(gh repo view "$REPOSITORY" --json url --jq '.url')"
printf '\nPublished and verified: %s\nCommit: %s\n' "$URL" "$LOCAL_SHA"
printf 'CI was triggered by the push; its result is not verified by this helper.\n'
