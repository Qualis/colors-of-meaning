# Feature: Reproducible & auditable dependency supply chain

## Overview

Make every `tox` run resolve the **same** dependency versions on every machine and every
week, and turn the four currently-silent `pip-audit` suppressions into a managed, dated
decision. Today `setup.cfg` declares bare package names (no lower/upper bounds, no lock),
`tox` uses `tox-recreate` so each run re-resolves against the latest upstream, and
`semgrep scan --config auto` fetches its ruleset over the network at gate time — the exact
combination that has already broken the build when matplotlib / starlette / semgrep shipped
new releases. This feature adds a committed lockfile the gate installs from, pins semgrep
(tool and rules), and records each vulnerability suppression with a rationale and a
review-by date. No runtime dependency is added; `setup.cfg` keeps its abstract ranges so
the package stays installable as a library. First in the production-hardening sequence
(030 → 031 → 032 → 033); it makes the PR gate (031) reproducible from day one.

## User Stories

- As a maintainer, I want `tox` to install a pinned, committed set of versions so a green
  run today reproduces next week (the observed matplotlib / starlette / semgrep drift stops).
- As a reviewer, I want dependency and Actions updates to arrive as explicit, reviewable
  pull requests rather than as invisible drift on the next CI run.
- As a security-conscious evaluator, I want each suppressed `pip-audit` finding to carry a
  package, a rationale, and a review-by date, so suppressions are demonstrably managed.
- As a contributor, I want the security scan to run the same rules every time, so a scan
  that passes locally passes in CI.

## Acceptance Criteria

- [ ] Given a clean checkout, when `tox` runs, then it installs from a committed, fully
  pinned `requirements.lock` (project + `testing` extra) and the gate is green at 100%
  coverage; `setup.cfg` `install_requires` remains abstract (unpinned ranges).
- [ ] Given `requirements.lock` is recompiled from unchanged inputs, when compared to the
  committed file, then the resolution is identical (stable, no churn).
- [ ] Given the security gate, when it runs, then `semgrep` executes a **pinned** ruleset
  (not `--config auto`) and a **pinned** tool version (resolved through the lock, not the
  ambient builder image), and surfaces the same findings deterministically.
- [ ] Given the four current `pip-audit --ignore-vuln` IDs, when a reader opens
  `docs/security/audit-suppressions.md`, then each row states the ID, package, severity,
  why it is currently un-actionable, and a review-by date; any ID that no longer fires
  against the locked versions has been removed from both the doc and `tox.ini`.
- [ ] Given `.github/dependabot.yml`, when a dependency or a pinned Action SHA has an
  update, then Dependabot opens a PR (ecosystems: `pip` for the lock, `github-actions`),
  so pinning does not become staleness and the weekly run is the standing re-review trigger
  for the suppressions register.
- [ ] Given the new committed files, when `git status --ignored` is checked, then
  `requirements.lock` and `docs/security/audit-suppressions.md` are tracked (not swallowed
  by a `.gitignore` rule — the feature-028 gotcha).

## Hexagonal Layer Impact

This is repository / tooling engineering; **no `src/` layer changes** and therefore no
change to domain, application, infrastructure, or interface code, and no change to the
`pytest-archon` architecture suite. Files touched are build and CI configuration only:

- `requirements.lock` (new, committed)
- `tox.ini` (install against the lock; pin the semgrep config)
- `setup.cfg` (add `semgrep` to the `testing` extra so the lock pins it — a test-only
  dependency, not `install_requires`)
- `.github/dependabot.yml` (new)
- `docs/security/audit-suppressions.md` (new)
- `.gitignore` (verify the two new files are tracked)

### Shared Layer

No code change. The reproducibility contract is expressed in build config, not in
`shared/`.

## API Contracts

None.

## CLI Impact

None. No CLI command changes; the developer-facing change is that `tox` (and any
`pip install -c requirements.lock`) now resolves pinned versions.

## Dependency Injection

None. No Lagom registrations change.

## Observability

None added. (No metrics/tracing — consistent with the repo's "stdlib logger + uuid
correlation-id, don't invent infra" convention.)

## Open Questions

- **Lockfile format.** `requirements.lock` (pip-compile format via `uv pip compile`) is
  recommended over a full `uv.lock`: Dependabot understands it and it needs zero migration
  of the `setup.cfg` metadata. A `uv.lock` (metadata moved to `pyproject.toml [project]`) is
  a larger, separate modernization — deferred.
- **Semgrep hermeticity.** Pinning to explicit rule packs (`p/python`, `p/security-audit`,
  …) is deterministic but still fetches from the registry; fully hermetic means vendoring
  the rules into the repo. Default: pinned rule packs + pinned tool version now; vendoring
  noted as a stronger follow-up.
- **Hashes in the lock.** `--generate-hashes` adds supply-chain integrity but can complicate
  local installs of platform wheels (torch). Default: pin versions now; hashes deferred as
  an option.
- **The four suppressed IDs.** Their current details could not be fetched offline; the plan
  attempts to drop each against the freshly-locked versions and only keeps (with rationale)
  those that still fire.
