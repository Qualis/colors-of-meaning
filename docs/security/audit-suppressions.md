# Dependency vulnerability suppressions

`tox` runs `pip-audit` as a quality gate. A small number of advisories are suppressed with
`--ignore-vuln` in `tox.ini`. Each active suppression is recorded here with a rationale and
a review-by date so the decision is auditable and does not rot silently. Suppressions are
re-evaluated on every Dependabot dependency PR and at the latest by the review-by date.

The audited closure is the one resolved into `uv.lock` and installed by `uv sync --locked`
(feature 036). `uv.lock` pins the runtime and `testing` closure, and the `build`/`publish`
dependency groups that `tox -e build` and `tox -e publish` install with `--only-group`. It does
**not** cover the PEP 517 build requirements (`[build-system] requires` — `setuptools`,
`setuptools_scm`), which are resolved fresh into an isolated environment on every wheel build —
see the coverage gaps below.

**Last reviewed:** 2026-08-27

## Active suppressions

| ID | Package | Installed | Fix version | Rationale | Review by |
|----|---------|-----------|-------------|-----------|-----------|
| PYSEC-2026-3447 | setuptools | 81.0.0 | 83.0.0 | Held at 81.0.0 by `uv.lock` and not independently upgradable: `torch` 2.12.1 declares `Requires-Dist: setuptools<82`, so the pinned torch caps setuptools below the fix version. Clearing this advisory therefore depends on the torch bump below, not on a separate setuptools bump. This is the runtime/`testing`-extra copy — the one `pip-audit` actually sees. | 2026-10-29 |
| PYSEC-2025-194 | torch | 2.12.1 | 2.13.0 | `torch` is held at 2.12.1 by `constraint-dependencies` in `pyproject.toml` for reproducibility of the committed evaluation results. Bumping to ≥2.13.0 shifts the whole ML stack and requires re-validating the pipeline and regenerating reported numbers; scheduled as a deliberate, separately-verified change. Load-bearing only on macOS — see the coverage gap below. | 2026-10-29 |

## Coverage gaps

Recorded so the gate is not read as claiming more coverage than it has.

| Package | Gap | Why | Mitigation |
|---------|-----|-----|------------|
| torch | The environment audit skips it on Linux and Windows — including the CI runner | `uv.lock` sources torch from the PyTorch CPU index, whose Linux/Windows wheels carry the local version `2.12.1+cpu`. `pip-audit` cannot match a local version against PyPI and skips it, reporting `Dependency not found on PyPI and could not be audited: torch (2.12.1+cpu)`, and still exits 0. The macOS wheels have no local version, so there torch *is* audited by the environment pass — which is why the PYSEC-2025-194 suppression above is still required. | **Closed in the gate.** `tox` runs `bin/audit-torch` straight after `pip-audit`; it strips the local version from the installed torch and audits that upstream version, so a new torch advisory fails the gate on every platform. Verify it is live by running `bin/audit-torch` with no arguments — it must exit non-zero on PYSEC-2025-194. |
| PEP 517 build requirements | Never audited | `setuptools` and `setuptools_scm` from `[build-system] requires` are resolved fresh into an isolated build environment; `setuptools_scm` does not appear in `uv.lock` at all. `pip-audit` only sees the installed gate environment. | Bounded by `exclude-newer`; re-check with `uv run --extra testing pip-audit -r` against the `[build-system]` pins when the build backend changes. |

The project itself (`colors-of-meaning`) is skipped for the same reason and is expected — it
is not published to PyPI.

## Fixed rather than suppressed

Advisories the gate surfaced against the locked closure and that were cleared by moving the
lock forward, recorded so the default is visibly "upgrade", not "suppress".

| ID | Package | Was | Now | Date |
|----|---------|-----|-----|------|
| PYSEC-2026-3716 | datasets | 5.0.0 | 5.0.1 | 2026-08-27 |
| PYSEC-2026-3721 | pip (transitive, via `pip-api` ← `pip-audit`) | 26.1.2 | 26.2.1 | 2026-08-27 |

## Pruned suppressions

Re-checked on 2026-07-20 with `pip-audit` (no ignores) against the locked closure; the
following no longer fire and were removed from `tox.ini`:

| ID | Reason for removal |
|----|--------------------|
| GHSA-4xh5-x5gv-qwph | No longer reported against the locked dependency versions. |
| CVE-2026-4539 | No longer reported against the locked dependency versions. |

If either advisory reappears (e.g. a transitive dependency reintroduces an affected
version), the gate will surface it again — that is the intended behaviour, not a regression.

## How to re-evaluate

```bash
# Show every advisory the gate would otherwise suppress, against the locked closure:
uv run --extra testing pip-audit

# Audit torch's upstream version, which the environment pass skips because of the +cpu local
# version. This is what the gate runs; with no --ignore-vuln it must fail on PYSEC-2025-194.
uv run --extra testing bin/audit-torch

# Confirm a specific ID no longer fires before pruning its --ignore-vuln from tox.ini.
```
