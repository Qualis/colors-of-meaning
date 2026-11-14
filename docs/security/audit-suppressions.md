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

**Last reviewed:** 2026-11-14

## Active suppressions

| ID | Package | Installed | Fix version | Rationale | Review by |
|----|---------|-----------|-------------|-----------|-----------|
| PYSEC-2026-3447 | setuptools | 81.0.0 | 83.0.0 | Held at 81.0.0 by `uv.lock` and not independently upgradable: `torch` 2.12.1 declares `Requires-Dist: setuptools<82`, so the pinned torch caps setuptools below the fix version. Clearing this advisory therefore depends on the torch bump below, not on a separate setuptools bump. This is the runtime/`testing`-extra copy — the one `pip-audit` actually sees. | 2027-01-11 |
| GHSA-xmj3-9gf7-h52w | setuptools | 81.0.0 | 84.1.0 | Same torch cap as PYSEC-2026-3447: the fix is above `setuptools<82`, so it clears with the torch bump, not separately. Low severity, Windows-only path escape in setuptools' legacy `Wheel.install_as_egg` path; the gate environment is installed by uv, which never routes a wheel through that path. | 2027-01-11 |
| PYSEC-2025-194 | torch | 2.12.1 | 2.13.0 | `torch` is held at 2.12.1 by `constraint-dependencies` in `pyproject.toml` for reproducibility of the committed evaluation results. Bumping to ≥2.13.0 shifts the whole ML stack and requires re-validating the pipeline and regenerating reported numbers; scheduled as a deliberate, separately-verified change. Load-bearing only on macOS — see the coverage gap below. | 2027-01-11 |

## Dependabot ignore list

`.github/dependabot.yml` ignores two packages. This is not a security suppression — the gate
still audits both and still fails on a new advisory against them — it exists because Dependabot
cannot see the pin and its update job hard-fails when it tries.

Dependabot rewrites a bare requirement to a pinned `<name>==<latest>` before re-resolving. For
these two that contradicts a constraint it has no visibility into, so `uv lock` reports
`No solution found` and the *entire* update job exits non-zero — taking the other dependency
updates in the group down with it.

| Package | Why Dependabot cannot update it | Un-ignore when |
|---------|--------------------------------|----------------|
| torch | `[tool.uv] constraint-dependencies = ["torch==2.12.1"]` in `pyproject.toml` holds torch at the version the committed evaluation results were produced with. A rewritten `torch==<latest>` contradicts it: `Because your project depends on torch==2.13.0 and torch==2.12.1 ... unsatisfiable`. | The deliberate torch bump lands and the constraint is removed. |
| setuptools | torch 2.12.1 declares `Requires-Dist: setuptools<82`, so the pinned torch caps setuptools. A rewritten `setuptools==<latest>` contradicts it: `Because torch>=2.12.1,<=2.12.1+cpu depends on setuptools<82 and colors-of-meaning[testing] depends on setuptools==84.0.0 ... incompatible`. | The torch cap lifts. Clearing PYSEC-2026-3447 depends on that bump, not on a separate setuptools bump. |

Both entries come back out together with the torch pin. Until then they are the reason
Dependabot's weekly run is green rather than red-with-a-created-PR.

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
| CVE-2026-111063, CVE-2026-111064, CVE-2026-111065, CVE-2026-111066, CVE-2026-111748, CVE-2026-111749, CVE-2026-111750, CVE-2026-123817, GHSA-2g87-pp6c-29x5 | aiohttp | 3.14.3 | 3.14.5 | 2026-11-14 |
| PYSEC-2026-4320 | certifi | 2026.7.22 | 2026.10.23 | 2026-11-14 |
| CVE-2026-112907, CVE-2026-112908, CVE-2026-112909, CVE-2026-112910, CVE-2026-112911, CVE-2026-112913, CVE-2026-112915, CVE-2026-112935, CVE-2026-112936, CVE-2026-112938, CVE-2026-112939, CVE-2026-112940 | filelock | 3.32.0 | 4.2.7 | 2026-11-14 |
| PYSEC-2026-4217 | fsspec | 2026.4.0 | 2026.6.0 | 2026-11-14 |
| CVE-2026-114613 | msgpack | 1.2.1 | 1.2.4 | 2026-11-14 |
| PYSEC-2026-4244 | multidict | 6.7.1 | 6.9.1 | 2026-11-14 |
| CVE-2026-96638, CVE-2026-96640, CVE-2026-96778, CVE-2026-112390, CVE-2026-112391, CVE-2026-112393, CVE-2026-112395, CVE-2026-112399, CVE-2026-112400, CVE-2026-112401, CVE-2026-112404, CVE-2026-112406, CVE-2026-112891, CVE-2026-112892, CVE-2026-112893, CVE-2026-112894, CVE-2026-112895, CVE-2026-112897, CVE-2026-112899, CVE-2026-112900, CVE-2026-128066, CVE-2026-128067, GHSA-78g4-xqwv-q3g8, PYSEC-2026-4326 – PYSEC-2026-4343, PYSEC-2026-4363 | pillow | 12.3.0 | 13.0.0 | 2026-11-14 |
| PYSEC-2026-4175, PYSEC-2026-4176, PYSEC-2026-4177 | urllib3 | 2.7.0 | 2.8.0 | 2026-11-14 |

filelock 4 and pillow 13 are major bumps the fixes require. Reaching them also moved the
packages their fix versions declare as floors — `virtualenv` 21.7.0 → 21.15.2 (the first line to
admit `filelock<5`), with its `packaging` 26.3 and `python-discovery` 1.9.1 floors, and `yarl`
1.24.5 → 1.25.1 for `aiohttp` — and nothing else in the closure.

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
