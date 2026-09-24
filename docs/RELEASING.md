# Publishing and releasing

## Create the organization repository

Use an authenticated GitHub CLI account that can create public repositories under `epistemic-ai`. The public destination is `epistemic-ai/diffstory`.

From the unpacked release directory:

```bash
gh auth login
bash scripts/publish.sh
```

The script requires a fresh directory without `.git`, checks the publication manifest and files, runs the unit tests, initializes a clean `main` branch, commits only manifest-listed files, and calls `gh repo create --public --source ... --push`. It does not modify an existing repository, re-use private Git history, or force-push. Commit identity comes from your Git configuration or your authenticated GitHub profile.

Repository creation, pushes, and CI execution require your local GitHub authorization. Preparing this archive does not imply those remote actions have happened. The helper prints the verified repository URL and compares remote and local commit IDs after a successful push. If a create/push fails, preserve the output and inspect both local and remote state before retrying.

## Release a new version

1. Update the package version and changelog; keep the rendered footer in sync.
2. Run unit and browser tests, rebuild the synthetic demo, and run the release checker.
3. Build with `python -m build` and validate with `python -m twine check dist/*`.
4. Confirm GitHub CI passed on the exact commit, review the release file set, then create a tag and a GitHub release from that commit.

Do not publish a package to PyPI until ownership/availability of the `diffstory` name is confirmed. No PyPI publication workflow or credentials are included.

After repository creation, enable private vulnerability reporting and a protected-branch/ruleset policy appropriate to the organization. Those are repository-administration settings, not actions performed by the code in this archive.

## Manifest

`PUBLICATION.json` lists the exact files and SHA-256 hashes in this prepared release. `scripts/check_release.py --manifest` verifies it. Generated distribution files, untracked reports, virtual environments, and caches are not publication inputs. Future maintainer releases should regenerate their manifest with `python scripts/check_release.py --write-manifest` only after reviewing all included files.
