# Dependency vulnerability suppressions

`tox` runs `pip-audit` as a quality gate. A small number of advisories are suppressed with
`--ignore-vuln` in `tox.ini`. Each active suppression is recorded here with a rationale and
a review-by date so the decision is auditable and does not rot silently. Suppressions are
re-evaluated on every Dependabot dependency PR and at the latest by the review-by date.

The audited closure is the one resolved into `uv.lock` and installed by `uv sync --locked`
(feature 036). `uv.lock` pins the full runtime and `testing` closure. It does **not** cover the
PEP 517 build requirements (`[build-system] requires` — `setuptools`, `setuptools_scm`), which
are resolved fresh into an isolated environment on every wheel build, nor the `build`/`twine`
toolchains of `tox -e build` / `tox -e publish`, which install outside the lock. Those are
neither locked nor audited — see the coverage gaps below.

**Last reviewed:** 2026-07-29

## Active suppressions

| ID | Package | Installed | Fix version | Rationale | Review by |
|----|---------|-----------|-------------|-----------|-----------|
| PYSEC-2026-3447 | setuptools | 81.0.0 | 83.0.0 | Held at 81.0.0 by `uv.lock` and not independently upgradable: `torch` 2.12.1 declares `Requires-Dist: setuptools<82`, so the pinned torch caps setuptools below the fix version. Clearing this advisory therefore depends on the torch bump below, not on a separate setuptools bump. This is the runtime/`testing`-extra copy — the one `pip-audit` actually sees. | 2026-10-29 |
| PYSEC-2025-194 | torch | 2.12.1 | 2.13.0 | `torch` is held at 2.12.1 by `constraint-dependencies` in `pyproject.toml` for reproducibility of the committed evaluation results. Bumping to ≥2.13.0 shifts the whole ML stack and requires re-validating the pipeline and regenerating reported numbers; scheduled as a deliberate, separately-verified change. Load-bearing only on macOS — see the coverage gap below. | 2026-10-29 |

## Coverage gaps

Recorded so the gate is not read as claiming more coverage than it has.

| Package | Gap | Why | Mitigation |
|---------|-----|-----|------------|
| torch | Not audited by the gate on Linux and Windows — including the CI runner | `uv.lock` sources torch from the PyTorch CPU index, whose Linux/Windows wheels carry the local version `2.12.1+cpu`. `pip-audit` cannot match a local version against PyPI and skips it, reporting `Dependency not found on PyPI and could not be audited: torch (2.12.1+cpu)`, and still exits 0. The macOS wheels have no local version, so there torch *is* audited — which is why the PYSEC-2025-194 suppression above is still required. | Audit the upstream version explicitly (see below) whenever torch moves, and on each Dependabot PR. Nothing in `tox` enforces this yet; it is a manual step. |
| PEP 517 build requirements | Never audited | `setuptools` and `setuptools_scm` from `[build-system] requires` are resolved fresh into an isolated build environment; `setuptools_scm` does not appear in `uv.lock` at all. `pip-audit` only sees the installed gate environment. | Bounded by `exclude-newer`; re-check with `uv run --extra testing pip-audit -r` against the `[build-system]` pins when the build backend changes. |
| `build` / `twine` | Not covered by the lock | `tox -e build` and `tox -e publish` use `runner = uv-venv-runner` with `deps`, because the lock runner ignores `deps`. They install from PyPI (bounded by `exclude-newer`), not from `uv.lock`. | These envs do not run in CI; the released artifact is built manually. Re-check before a release. |

The project itself (`colors-of-meaning`) is skipped for the same reason and is expected — it
is not published to PyPI.

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

# Audit torch's upstream version, which the gate skips because of the +cpu local version.
# Strip the local segment from whatever uv.lock resolved, then audit that:
uv run --extra testing python -c "
from importlib.metadata import version
print('torch==' + version('torch').split('+')[0])
" > /tmp/torch-pin.txt
uv run --extra testing pip-audit -r /tmp/torch-pin.txt

# Confirm a specific ID no longer fires before pruning its --ignore-vuln from tox.ini.
```
