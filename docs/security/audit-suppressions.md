# Dependency vulnerability suppressions

`tox` runs `pip-audit` as a quality gate. A small number of advisories are suppressed with
`--ignore-vuln` in `tox.ini`. Each active suppression is recorded here with a rationale and
a review-by date so the decision is auditable and does not rot silently. Suppressions are
re-evaluated on every Dependabot dependency PR and at the latest by the review-by date.

**Last reviewed:** 2026-07-20

## Active suppressions

| ID | Package | Installed | Fix version | Rationale | Review by |
|----|---------|-----------|-------------|-----------|-----------|
| PYSEC-2026-3447 | setuptools | 81.0.0 | 83.0.0 | Build-time-only dependency (not shipped at runtime); not pinned in `requirements.lock`, so the resolved version tracks the build environment. Upgrade to ≥83.0.0 to be adopted via the Dependabot `pip` update rather than a manual pin. | 2026-10-20 |
| PYSEC-2025-194 | torch | 2.12.1 | 2.13.0 | `torch` is pinned at 2.12.1 in `requirements.lock` for reproducibility of the committed evaluation results. Bumping to ≥2.13.0 shifts the whole ML stack and requires re-validating the pipeline and regenerating reported numbers; scheduled as a deliberate, separately-verified change. | 2026-10-20 |

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
# Show every advisory the gate would otherwise suppress:
pip-audit
# Confirm a specific ID no longer fires before pruning its --ignore-vuln from tox.ini.
```
