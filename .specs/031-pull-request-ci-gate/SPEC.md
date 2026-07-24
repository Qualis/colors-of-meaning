# Feature: Pull-request CI gate

## Overview

Run the full quality gate on every pull request **before** merge, and make it a required
status check. Today all three workflows (`builder`, `development`, `service`) trigger only
on `push: main`, `schedule`, and `workflow_dispatch` — so the 8-gate `tox` run happens
*after* a branch has already landed on `main`. A broken PR can merge and fail only
post-merge, which defeats much of the value of an otherwise excellent gate and undermines
the "always shippable" impression a capability showcase needs. This feature adds a
`pull_request`-triggered workflow that runs the same gate on a native runner installing
from the committed lock (feature 030), with least-privilege permissions, superseded-run
cancellation, SHA-pinned Actions, and no repository secrets (so it runs on fork PRs).
Depends on 030 so the gate installs pinned inputs.

## User Stories

- As a maintainer, I want a broken pull request to fail CI before it can merge, so `main`
  stays green and the demonstration is always in a shippable state.
- As a contributor (including from a fork), I want the gate to run on my PR without needing
  repository secrets, so first-time contributions are checked automatically.
- As an evaluator, I want a visible "PRs are gated" signal (a passing check and a badge), so
  the project reads as professionally maintained.
- As a maintainer, I want superseded CI runs cancelled when I push a new commit to a PR, so
  runner time is not wasted.

## Acceptance Criteria

- [ ] Given a pull request targeting `main`, when it is opened or updated, then
  `.github/workflows/pr-gate.yml` runs the full `tox` gate (flake8, black, bandit, semgrep,
  pip-audit, radon, xenon, mypy, and pytest at 100% coverage) plus `shellcheck` and
  `ansible-lint` (matching `bin/test`).
- [ ] Given the workflow, when it is inspected, then it declares `permissions: contents:
  read` (least privilege) and a `concurrency` group keyed to the PR ref with
  `cancel-in-progress: true`.
- [ ] Given the workflow, when it runs, then it references **no** repository secrets and
  therefore executes on pull requests from forks; all Actions are pinned to commit SHAs
  (matching the repo convention).
- [ ] Given the gate installs dependencies, when it sets up the environment, then it
  installs from the committed `requirements.lock` (feature 030), so the PR gate is
  reproducible.
- [ ] Given `README.MD`, when read, then it shows a build badge for the PR gate alongside
  the existing three, and the repository documents the one manual owner step — making the
  `pr-gate` check **required** in branch protection — which cannot be committed as a file.

## Hexagonal Layer Impact

Repository / CI configuration only; **no `src/` layer changes**, no architecture-suite
change. Files:

- `.github/workflows/pr-gate.yml` (new)
- `README.MD` (add the PR-gate badge; document the branch-protection step, or place that
  note in `CONTRIBUTING.rst`)

### Shared Layer

No code change.

## API Contracts

None.

## CLI Impact

None. The developer-facing change is that PRs are now gated; the gate command is the
existing `tox` (plus `shellcheck`/`ansible-lint`), unchanged.

## Dependency Injection

None.

## Observability

The workflow uploads `build/pytest.xml` as a CI artifact (`if: always()`), mirroring
`service.yml`, so a failed gate is inspectable. No application observability change.

## Open Questions

- **Runner: native `tox` vs the builder image.** Recommended: a native job (checkout →
  `setup-python` 3.11 → install `tox` + the lock → `tox`), which needs no secrets, runs on
  forks, and exercises the same `tox` the maintainers run — proving feature 030's lock. The
  alternative, reusing `./test.sh` (the `svanosselaer/colors-of-meaning-builder` image),
  gives exact parity with the post-merge gate but depends on that image being pullable
  without secrets; kept as a fallback if env-parity is preferred over fork-safety.
- **Caching.** `actions/setup-python` pip caching keyed on `requirements.lock` would speed
  the gate; optional, added if the run is slow (the ML stack is heavy).
- **Overlap with `service.yml`.** The post-merge `service.yml` build stays as-is (it also
  builds and pushes the image); this feature only adds the pre-merge check. De-duplicating
  the two is out of scope.
- **Branch protection is not a file.** Requiring the check is a GitHub repository setting the
  owner must toggle once; this feature documents it but cannot enforce it in-repo.
