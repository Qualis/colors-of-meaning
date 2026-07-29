# Plan: Adopt uv for dependency management and fast environments

## Implementation Strategy

Sequence the full adoption (Option B) as independently-committable, stop-safe stages so it can
halt at the safe subset (Option A) if the tox 3→4 migration proves too disruptive. Validate the
load-bearing risk — `tox-uv` requires tox 4 — **first**, before committing to the rest. Every
stage is verified by the full `tox` at 100% coverage. Build/tooling only; no `src` logic change.

Best landed after spec 035, so lint/format config already lives in `pyproject.toml` and the
metadata migration is a clean addition rather than a mixed edit.

## Layer Changes

No domain/application/infrastructure/interface/shared source changes. Build/tooling only.

### Stage 1 — metadata to `pyproject.toml [project]` (independently committable)

- Move from `setup.cfg [metadata]/[options]`: `name`, `description`, `readme = "README.MD"`,
  `requires-python = ">=3.11"`, `dependencies` (the abstract `install_requires`),
  `[project.optional-dependencies] testing = [...]`, `[project.scripts]` (the two
  `colors-of-meaning*` console entry points), and `dynamic = ["version"]` with
  `[tool.setuptools_scm]` retained. Keep `[tool.setuptools.packages.find] where = ["src"]`
  and package-data (`py.typed`, resources).
- Verify: `python -m build` + `twine check dist/*` pass; `pip install -e .` works; imports and
  the two entry points resolve. `setup.py` becomes a thin shim or is retired.

### Stage 2 — uv lock with explicit torch index (independently committable)

- Add `[tool.uv.sources]` pinning `torch`/`torchmetrics` to a CPU index, plus
  `[[tool.uv.index]] name = "pytorch-cpu" url = "https://download.pytorch.org/whl/cpu"
  explicit = true` (CUDA-by-marker optional, per SPEC).
- `uv lock` → commit `uv.lock`. Verify a CPU-only `uv sync` installs torch from the CPU index
  with **no** unsupported CUDA stack (the 030 `pip check` nvidia warning is gone), and the
  suite runs. At this point Option A's benefit (a real resolved lock) is already banked.

### Stage 3 — tox 4 + tox-uv (the risk gate; independently committable)

- Confirm `tox-uv`'s tox requirement; upgrade `tox.ini` to tox 4 syntax and add `tox-uv`.
  Re-verify **all** ~25 testenvs (the many ad-hoc `[testenv:*]` CLI/report envs), not just
  `[testenv]`. Retire `PIP_CONSTRAINT` (uv sync uses `uv.lock`).
- If tox 4 proves too disruptive, **stop here on Option A**: keep tox 3 + `PIP_CONSTRAINT`, but
  point it at a `uv pip compile`-generated `requirements.lock`.

### Stage 4 — reconcile dependents & docs (independently committable)

- `.github/workflows/pr-gate.yml`: install via `uv` (pin the uv version); drop the pip/lock
  install step. `.github/dependabot.yml`: uv ecosystem. `docs/security/audit-suppressions.md`:
  re-express suppressions for the uv lock. `CONTRIBUTING.rst` + `CLAUDE.md`: uv commands.
  Retire `requirements.lock` (Option B) or keep it uv-generated (Option A).

## Dependency Injection

None.

## Task List

1. [ ] risk-gate: confirm `tox-uv` ⇒ tox 4; spike the tox 3→4 upgrade on the env set; decide
   Option A vs B before committing beyond Stage 2.
2. [ ] stage 1: migrate metadata to `pyproject.toml [project]`; `build` + `twine check` green.
3. [ ] stage 2: `[tool.uv.sources]` torch CPU index + `uv lock`; commit `uv.lock`; verify a
   clean CPU install with no CUDA stack; `tox` green.
4. [ ] stage 3: tox 4 + `tox-uv`; re-verify all testenvs; retire `PIP_CONSTRAINT` (or stop at A).
5. [ ] stage 4: update `pr-gate.yml`, `dependabot.yml`, suppressions register, docs; retire or
   regenerate `requirements.lock`.
6. [ ] verify: full `tox` green at 100% coverage; second `uv lock` is a no-op (stable);
   CPU-only install carries no unsupported CUDA pins.

## Testing Strategy

No new `src` code, so no coverage delta. Validation is operational:

- **Lock correctness:** `uv lock` stable on re-run; CPU-only `uv sync` installs torch from the
  CPU index with no `nvidia-*` "not supported on this platform" warning.
- **Packaging:** `python -m build` + `twine check` pass after the `[project]` migration
  (entry points, version, readme, extras all intact).
- **Gate integrity:** `tox` green at 100% across all testenvs, before and after tox-uv.
- **CI:** a draft PR shows the uv-backed `pr-gate` green and faster.
- Verify with `tox`, never the tools directly, for the final check.

## Observability Plan

None.

## Risks and Mitigations

- **`tox-uv` forces tox 4 across ~25 ad-hoc testenvs.** → Task 1 validates it first; Stages
  1–2 deliver value without it; stop-safe on Option A if disruptive.
- **`setup.cfg` → `[project]` drops metadata** (entry points, setuptools_scm, package-data,
  py.typed). → Stage 1 gated on `build` + `twine check` + entry-point resolution before moving on.
- **uv torch index misconfigured** (wrong platform wheel). → Stage 2 verifies a real CPU
  install + suite run; CUDA is opt-in via markers.
- **Superseding 030/031 breaks the gate/CI.** → Stage 4 updates `pr-gate.yml`,
  `PIP_CONSTRAINT`, Dependabot, and the suppressions register together; a draft PR confirms.
- **Change-for-its-own-sake.** → The SPEC states plainly that 030's lock already works; uv is
  justified only by the CUDA-lock fix and CI speed, and Option A is the fallback if B's cost
  outweighs those.
- **uv version drift.** → Pin uv in CI and the lock.

## Validation against the spec

Cross-platform lock with explicit torch index, no hand-excluded nvidia pins → tasks 3,6;
tox green via uv → tasks 3–4,6; `[project]` metadata + build/twine → task 2; CPU install
without CUDA stack → tasks 3,6; 030/031 dependents + Dependabot + suppressions updated →
task 5; uv pinned → tasks 4–5; Option A fallback preserved → task 1/Stage 3.
