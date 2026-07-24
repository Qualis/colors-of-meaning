# Plan: Pull-request CI gate

## Implementation Strategy

Add one `pull_request` workflow that runs the existing gate on a native runner installing
from feature 030's lock — no secrets, SHA-pinned, least-privilege, concurrency-cancelling.
Reuse the exact `tox` maintainers run (highest honesty and lowest maintenance) rather than a
bespoke step list. Keep the three existing workflows untouched. The only non-committable
piece — making the check *required* — is documented for the owner. Depends on feature 030
(the lock must exist for a reproducible install).

New file: `.github/workflows/pr-gate.yml`. Modified: `README.MD` (badge + branch-protection
note; or the note in `CONTRIBUTING.rst`).

## Layer Changes

No domain / application / infrastructure / interface / shared source changes. CI
configuration only.

- **`.github/workflows/pr-gate.yml` (new).**
  - `on: pull_request: { branches: [main] }` (keep existing `push:main` post-merge builds).
  - Top-level `permissions: { contents: read }`.
  - `concurrency: { group: pr-gate-${{ github.head_ref || github.ref }}, cancel-in-progress: true }`.
  - One `ubuntu-latest` job:
    1. `actions/checkout@<sha>` (same SHA the repo already pins).
    2. `actions/setup-python@<sha>` with `python-version: '3.11'` (optionally `cache: pip`,
       `cache-dependency-path: requirements.lock`).
    3. Install pinned tooling: `pip install -c requirements.lock tox` (and `ansible-lint`),
       plus system `shellcheck` (present on `ubuntu-latest`).
    4. `ansible-lint -p infrastructure/ansible/playbook-*.yml`.
    5. `shellcheck -x ./*.sh bin/*`.
    6. `tox` (the full 8-gate + 100%-coverage run; tox installs the project constrained by
       the lock per feature 030).
    7. `actions/upload-artifact@<sha>` for `build/pytest.xml`, `if: always()`.
  - No `secrets.*` references anywhere → runs on fork PRs.
- **`README.MD`.** Add the PR-gate `actions/workflows/pr-gate.yml/badge.svg` badge next to the
  existing three; add a short "Branch protection" line noting the owner must require the
  `pr-gate` check (Settings → Branches).

## Dependency Injection

None.

## Task List

1. [ ] ci: add `.github/workflows/pr-gate.yml` (pull_request→main; least-priv perms;
   concurrency cancel; SHA-pinned checkout/setup-python/upload-artifact; native
   `tox` + `shellcheck` + `ansible-lint`; lock-constrained install; no secrets).
2. [ ] docs: add the PR-gate badge to `README.MD`; document the required-check
   branch-protection step (README or `CONTRIBUTING.rst`).
3. [ ] verify: open a draft PR (or run `actionlint`) to confirm the workflow parses and the
   gate passes end-to-end on a real PR; confirm no secret is referenced.

## Testing Strategy

No `src/` change, so no unit tests and no coverage delta. Validation is operational:

- **Self-proof:** a draft PR shows the gate running and passing (the workflow gating itself
  is the acceptance evidence).
- **Static check:** `actionlint` (if available) parses the YAML; a manual review confirms
  least-privilege `permissions`, `concurrency`, SHA pins, and absence of `secrets.*`.
- **Fork-safety:** confirmed by the no-secrets property (a fork PR cannot access secrets, so
  a secret-free workflow is the requirement).
- The gate it runs is the existing `tox` — no new test infrastructure.

## Observability Plan

CI-level only: `build/pytest.xml` uploaded as an artifact on every run (`if: always()`) so a
failure is inspectable. No application observability change.

## Risks and Mitigations

- **Workflow doesn't pass on itself / YAML error.** → Validate via a draft PR and/or
  `actionlint` (task 3) before relying on it.
- **Builder image / secret coupling.** → The native-`tox`-from-lock job needs no secrets and
  runs on forks; the image-parity option is documented as a fallback only.
- **Heavy ML install makes the gate slow.** → Optional `setup-python` pip cache keyed on
  `requirements.lock`; acceptable even without caching for a PR gate.
- **Branch protection not enforced** (it is not a file). → Documented as the single manual
  owner action required to complete the feature; called out explicitly so it is not missed.
- **Depends on feature 030's lock.** → Sequence 030 before 031; if 031 lands first, fall back
  to `pip install tox` unpinned until the lock exists (noted, not preferred).

## Validation against the spec

Full gate on every PR → task 1; least-priv + concurrency → task 1; SHA-pinned + secret-free +
fork-safe → task 1; installs from the lock → task 1; badge + documented required-check →
task 2; end-to-end proof → task 3.
