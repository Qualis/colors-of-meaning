# Plan: Reproducible & auditable dependency supply chain

## Implementation Strategy

Add a committed lock the gate installs from, pin the one remaining network-fetched gate
(semgrep), and document the security-suppression debt — all in build/CI config, with **no
`src/` change**. Land it first so 031's PR gate runs pinned inputs. Keep `setup.cfg`
`install_requires` abstract (a distributable library should declare ranges, not pins); the
lock is the concrete resolution layered on top for CI and developers. Validate by a green
`tox` and a no-op re-compile.

New files: `requirements.lock`, `.github/dependabot.yml`, `docs/security/audit-suppressions.md`.
Modified: `tox.ini`, `setup.cfg`, `.gitignore` (only if a rule would swallow the new files).

## Layer Changes

No domain / application / infrastructure / interface / shared source changes. Build and CI
configuration only:

- **`requirements.lock` (new).** Generate with `uv pip compile` from the project including
  the `testing` extra (`uv pip compile setup.py --extra testing -o requirements.lock`, uv
  reading `setup.cfg` through `setup.py`). Fully pinned; hashes deferred (Open Question).
- **`tox.ini`.** In `[testenv]`, constrain installs to the lock so the env resolves pinned
  versions while `extras = testing` + the editable install still provide the package:
  `deps = -c {toxinidir}/requirements.lock` (constraint, not `-r`, so setup.cfg stays the
  source of *what* is needed and the lock fixes *which version*). Replace
  `semgrep scan --config auto --error .` with an explicit pinned ruleset, e.g.
  `semgrep scan --config p/python --config p/security-audit --error .`. Keep every other gate
  line unchanged.
- **`setup.cfg`.** Add `semgrep` to the `[options.extras_require] testing` list so its
  version is captured in the lock (today it is ambient from the builder image). It is a
  test/gate tool only — **not** added to `install_requires`.
- **`.github/dependabot.yml` (new).** `version: 2`; updates for `pip` (targets
  `requirements.lock`) and `github-actions` (bumps the SHA pins in all workflows), weekly,
  with sensible PR limits and grouping.
- **`docs/security/audit-suppressions.md` (new).** A dated table — `ID | package | severity |
  why un-actionable now | review-by` — one row per current ignore (GHSA-4xh5-x5gv-qwph,
  CVE-2026-4539, PYSEC-2026-3447, PYSEC-2025-194). For each, attempt `pip-audit` without that
  ignore against the locked versions; drop any that no longer fire from both the doc and the
  `tox.ini` line.
- **`.gitignore`.** `docs/` is not currently ignored, so likely a no-op; verify with
  `git status --ignored` and add an explicit un-ignore only if a rule matches (feature-028
  reports/*-swallow gotcha).

## Dependency Injection

None.

## Task List

Each task is independently committable; no inside-out layer order applies (config only).

1. [ ] tooling: generate `requirements.lock` (uv pip compile, `testing` extra); confirm a
   second compile is byte-identical (stable resolution).
2. [ ] tooling: constrain `[testenv]` installs to the lock in `tox.ini`; run `tox`; confirm
   green at 100% coverage with pinned versions.
3. [ ] security: add `semgrep` to the `testing` extra (`setup.cfg`), re-lock, and pin the
   semgrep ruleset in `tox.ini`; re-run the scan; triage any newly-surfaced finding
   (extract-to-variable per the repo's semgrep gotcha — **no `# nosemgrep`**, the no-comments
   rule forbids it) and record any genuine suppression.
4. [ ] security: write `docs/security/audit-suppressions.md` (dated rows); attempt to drop
   each `--ignore-vuln` against the lock and prune those that no longer fire.
5. [ ] ci: add `.github/dependabot.yml` (`pip` + `github-actions`, weekly).
6. [ ] verify: `git status --ignored` shows the lock and the security doc tracked; full
   `tox` green; a repeat run resolves identical versions.

## Testing Strategy

No new `src/` code, so no new unit tests and no coverage delta. Validation is by the gate
itself and by reproducibility checks:

- **Reproducibility:** a second `uv pip compile` produces an identical `requirements.lock`;
  two `tox` runs install the same versions.
- **Gate parity:** `tox` stays green at 100% coverage with the constrained install; the
  pinned semgrep ruleset produces the same (or explicitly-triaged) findings.
- **Suppressions:** each retained `--ignore-vuln` still corresponds to a finding that fires
  without it; each removed one no longer fires.
- Verify with `tox`, never `pytest` directly (all 8 gates).

## Observability Plan

None.

## Risks and Mitigations

- **Constraining tox to the lock breaks resolution or an env.** → Land tasks 1–2 alone,
  `tox` green before anything builds on it; keep `setup.cfg` ranges abstract so the package
  still installs uninstrained elsewhere.
- **Pinned semgrep ruleset surfaces new findings** (different set than `auto`). → Triage in
  isolation (task 3); fix by extracting the flagged expression to a variable (the documented
  repo gotcha), not by a `# nosemgrep` comment; record any real suppression.
- **Adding semgrep to the testing extra bloats/instabilifies the lock.** → It is already a
  required gate tool; pinning it is strictly more reproducible than the ambient version.
- **`.gitignore` swallowing the new committed files** (feature-028 gotcha). → `git status
  --ignored` check in task 6; explicit un-ignore only if needed.
- **Dependabot noise.** → Weekly cadence, grouped updates, PR limits; the point is that
  updates become explicit reviewable PRs, which also drives the suppressions re-review.
- **torch/platform wheels in the lock.** → Pin versions without hashes initially (hashes are
  an Open Question) to avoid platform-wheel friction on developer machines.

## Validation against the spec

Pinned committed lock + tox-constrained install → tasks 1–2; stable re-resolution → tasks 1
and 6; pinned semgrep tool+ruleset → task 3; dated suppressions register with prune → task 4;
Dependabot for pip + actions → task 5; new files tracked (no gitignore swallow) → task 6.
