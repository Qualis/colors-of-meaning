# Plan: Retire API basic authentication

## Implementation Strategy

A deletion, not a refactor. The auth stack is a leaf: two whole packages, one CLI module, four
lines of container wiring, two settings fields, one dependency. Nothing imports it except its
own tests and the four lines in `interface/api/main.py`, so the removal order is outside-in —
drop the call sites first so no intermediate state has a dangling import, then delete the
packages, then re-point the tests that only *mention* the settings, then the prose, then re-lock.

Three edits are **not** pure deletion and carry the judgment in this change:

1. `test_architecture.py` — the `Security Module` rule **must** die with its package. This is not
   optional cleanup: verified against `pytest-archon` 0.0.7 in a scratch worktree, an empty match
   set produces `FAILED Rule 'Security Module': - NO CANDIDATES MATCHED`, raised from the
   `makereport` hook, so it presents as a red test with no traceback line in the file.
   `_does_not_import_argon2` is **widened** from `domain.*`/`application.*` to
   `colors_of_meaning.*`. It stops being a layer-boundary rule and becomes a regression guard:
   password hashing cannot reappear anywhere in the package without a failing test. It must stay
   a `.should(predicate, …)` rule — `should_not_import` is vacuous against external packages —
   which it already is.
2. `test_configuration.py` / `test_configuration_integration.py` — the settings *behaviours*
   (properties-file load, environment precedence over properties, defaults when the properties
   file is absent) are real and must survive. They are re-pointed at `experiment_config`, the one
   surviving field with no dedicated environment-precedence test today, so coverage of the
   behaviour is preserved rather than deleted along with the field it happened to use.
3. `test_docs_claims_consistency.py` — gains one guard, following the file's existing
   `_read_readme()` + term-list pattern, so a future contribution cannot silently re-add an
   authentication claim.

Prefer-editing-over-creating holds: the only new files are this spec pair.

## Layer Changes

### Interface Layer (`src/colors_of_meaning/interface/`) — done first

`api/main.py`
- Delete the import block, lines 27–31 (`BasicAuthenticator`, `SecurityDependency`,
  `get_basic_authenticator`).
- Delete lines 142–145 in `get_container()` (build authenticator, build security dependency, two
  `container[…]` registrations). Leave the blank-line rhythm of the function intact; the
  `health_checker` block follows directly after the use-case registrations.

`cli/hash_password.py`
- Delete the file. No `tox.ini` environment, no `[project.scripts]` entry, no `bin/` caller —
  verified before deleting.

### Domain Layer (`src/colors_of_meaning/domain/`)

- Delete `authentication/` entirely (`authenticator.py`, `__init__.py`).

### Application Layer (`src/colors_of_meaning/application/`)

- No change.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

- Delete `security/` entirely (`basic_authentication.py`, `__init__.py`).

### Shared Layer (`src/colors_of_meaning/shared/`)

`configuration.py`
- Remove `admin: str = "admin"` and `admin_password_hash: str = ""` from `ApplicationSettings`.
- `resources/application.properties` needs no edit — it already holds only `reload` and `host`.
- `_apply_property` is generic over `hasattr`, so no logic changes.

### Packaging

`pyproject.toml`
- Remove `"argon2-cffi",` from `[project] dependencies`.

`uv.lock`
- Regenerate with `uv lock` (never `--frozen`; `tox` runs `uv sync --locked` and fails on a stale
  lock). Commit the lock with the `pyproject.toml` change, per the repository's uv convention.
- Reverse-dependency check, done rather than assumed: `argon2-cffi` is required only by
  `colors-of-meaning` itself, and `argon2-cffi-bindings` only by `argon2-cffi`, so both leave the
  lock entirely. `cffi` **stays** — it is still required by `cryptography` ← `secretstorage` ←
  `keyring` ← `twine` in the `publish` group — but sheds the extra platform wheels that only
  `argon2-cffi-bindings`' broader marker set pulled in, so a handful of `cffi` wheel lines
  disappearing from the diff is expected, not a mistake.
- Dry-run result on a scratch copy: 169 → 167 packages, "Removed argon2-cffi v25.1.0 / Removed
  argon2-cffi-bindings v25.1.0", **94 deleted lines and zero added lines** — a pure removal with
  no incidental version churn. `uv lock --check` exits 0 against the committed lock today, so the
  baseline is clean. Because `[tool.uv] exclude-newer` is a *relative* seven-day window, still
  diff the regenerated lock and confirm nothing but the argon2 and cffi-wheel lines moved.

## Dependency Injection

Two registrations removed from `get_container()`:

- `container[BasicAuthenticator]`
- `container[SecurityDependency]`

Nothing added. No test container override exists for either key, so no test wiring changes.

## Task List

1. [ ] interface: strip the `infrastructure.security` import block and the four authenticator
   lines from `api/main.py`
2. [ ] interface: delete `cli/hash_password.py`
3. [ ] domain: delete `domain/authentication/`
4. [ ] infrastructure: delete `infrastructure/security/`
5. [ ] shared: drop `admin` and `admin_password_hash` from `ApplicationSettings`
6. [ ] packaging: drop `argon2-cffi` from `pyproject.toml`, run `uv lock`, commit both
7. [ ] tests: delete `tests/…/domain/authentication/`, `tests/…/infrastructure/security/`,
   `tests/…/interface/cli/test_hash_password.py`
8. [ ] tests: drop the four auth fixtures and the `basic_authentication` import from
   `tests/…/conftest.py` — cut **lines 8–34 exactly**. Do *not* also take 35–36: the surviving
   `@pytest.fixture` needs its two blank lines, and cutting them makes `ruff format --check`
   report `Would reformat` and fail the gate
9. [ ] tests: drop `test_should_emit_fail_closed_warning_at_most_once_when_module_is_imported`
   (lines 350–355) from `test_main_app.py`, **and** `import logging` at line 5 — it is used
   only by that test, and `ruff` fires `F401 'logging' imported but unused` otherwise
10. [ ] tests: remove the `Security Module` rule (lines 68–77) from `test_architecture.py` and
    widen `_does_not_import_argon2` to match `colors_of_meaning.*`. Apply this to
    `test_architecture.py` **only** — `tests/colors_of_meaning/test_synesthetic_architecture.py`
    is a second, near-identically-named archon file whose enumerated CLI rule does not list
    `hash_password`; it was checked and needs no edit
11. [ ] tests: re-point `test_configuration.py`. Each line range below **includes the `@patch`
    decorators**, which sit one line above the `def`; cutting from the `def` leaves an orphan
    decorator. Precisely:
    - **Delete** (they exist only to exercise a removed field, and each would raise
      `AttributeError` or read as a live security claim):
      `test_should_load_admin_setting_from_properties_file`,
      `test_should_use_admin_default_when_missing_properties_file`,
      `test_should_default_admin_password_hash_to_empty_when_unset`,
      `test_should_read_admin_password_hash_from_environment`,
      `test_should_overlay_admin_password_hash_from_properties_file`,
      `test_should_not_expose_plaintext_password_attribute`,
      `test_should_get_admin_setting_value` (redundant with `test_should_get_host_setting_value`).
    - **Re-point** (the behaviour is real and must not be lost with the field it happened to
      use): `test_should_use_environment_variables_over_properties` → `APP_EXPERIMENT_CONFIG`;
      `test_should_allow_setting_override` → `experiment_config`.
    - **Clean stale payloads** (currently green but misleading): drop the `admin` /
      `admin_password_hash` keys from the mocked properties dicts in
      `test_should_load_reload_setting_from_properties_file`,
      `test_should_load_host_setting_from_properties_file`, and the env-override test; drop the
      `mock_settings.admin = "admin"` lines from the surviving `TestApplicationSettingProvider`
      tests.
    - **Leave alone:** `TestLoadPropertiesFile`'s `"admin=testadmin\npassword=testpassword"`
      sample text. It exercises the generic key=value parser against arbitrary input and has no
      connection to the auth feature; rewriting it is churn, not cleanup.
12. [ ] tests: re-point both integration tests in `test_configuration_integration.py` from
    `APP_ADMIN` / `APP_ADMIN_PASSWORD_HASH` to `APP_EXPERIMENT_CONFIG`. Both currently run a
    temp script through `subprocess.run(..., check=True)` that calls `provider.get("admin")`;
    after the field is gone `_get_from_settings` raises `ValueError`, the child exits non-zero,
    and `check=True` raises. These are the only tests that exercise the settings path in a real
    subprocess, so they are re-pointed rather than deleted — they contribute no measured
    coverage, but they are the only end-to-end evidence that `.env`/environment resolution works
    outside the mocked unit tests.
13. [ ] tests: add the README authentication-claim guard to `test_docs_claims_consistency.py`,
    following the existing `UNIMPLEMENTED_INFRASTRUCTURE_TERMS` + `_..._terms_in()` shape.
    **Scope it to `README.MD` and `docs/*.md` only.** A repo-wide grep guard would flag its own
    spec: `.specs/012-…`, `.specs/ROADMAP.md`, and this feature's own SPEC and PLAN all contain
    the literal tokens as dated record. This is the trap already recorded from spec 027, where a
    guard matched the corrective phrasing describing the stale claim
14. [ ] docs: `README.MD` — delete the whole `### API authentication` H3 (lines 378–389, between
    `## Development` and `## Architecture`); drop `security,` from the Infrastructure architecture
    bullet (lines 398–400); add the one-sentence unauthenticated-posture statement
15. [ ] docs: `docs/cli.md` — delete the `### hash_password` H3 (lines 447–457). Keep the
    `## Utilities` H2: `### cli` still lives under it
16. [ ] docs: `docs/design.md` — delete line 69 (`authentication/   Authenticator`) and line 88
    (`security/         Argon2id basic authentication`) from the module map. The
    "Ports and their adapters" table at lines 37–51 never listed `Authenticator`, so it is
    already correct and needs no edit
17. [ ] docs: `.claude/CLAUDE.md` — structure tree (108–109, 155–156), domain structure bullet
    (199), infrastructure rule (226) and structure bullet (236), the
    `### Authentication & Authorization (implemented)` block (432–437, relocating the generic
    "Never commit credentials or secrets" hygiene line into `### Secrets Management` rather than
    losing it), the stale Secrets Management bullet (445 — after this change the only
    environment-loaded secret is `ANTHROPIC_API_KEY`), and the System Qualities security entry
    (462–465, where the auditing half of the claim stays true and the Argon2id half does not).
    Leave the `bandit`/`semgrep` gate rows and the "No secrets are committed" checklist item —
    they are static-analysis and generic hygiene, unrelated to this feature
18. [ ] skills: `.claude/skills/hexagonal-architecture-scaffolder/SKILL.md` — **highest priority
    of the doc set**, because it is a code generator, not prose: strip the
    `HTTPBasicCredentials` import (206), the `authentication_dependency` constructor parameter
    and assignment (222, 227), both `dependencies=[Depends(authentication_dependency)]` route
    kwargs (234, 242), the "Add authentication dependencies" rule (280), and the
    `security_dependency.authentication_dependency()` wiring in Step 10 (342); tidy the
    "persistence, security, etc." aside (27)
19. [ ] skills: `.claude/skills/test-generator/SKILL.md` (250, 289) and its
    `references/test-examples-from-codebase.md` (62–98) — drop the `authentication_dependency`
    kwarg, the `authentication_headers` fixture usage (which references a `conftest` fixture that
    never existed under that name), and the two auth bullets
20. [ ] skills: `.claude/skills/self-documenting-refactor/references/refactoring-examples-from-codebase.md`
    — delete the five worked examples built on `basic_authentication.py` (Authentication Methods
    42–62, Authentication Setup 90–113, Authentication Errors 160–178, Credential Verification
    244–272, Scenario 4 408–441), re-point the Quick Reference rows (531–532) and anti-pattern
    snippets (566–572, 588–589) at a non-auth example, drop the References bullet (611), and fix
    the Table of Contents and Scenario numbering. Each affected pattern retains at least one
    surviving example, so no pattern is left illustrated by nothing
21. [ ] skills: one-line edits to `.claude/skills/plan/SKILL.md` (55) and
    `.claude/skills/specify/SKILL.md` (51), both of which name the `security/` package
22. [ ] repo: generalise the `.gitignore` comment on line 61 ("Local secrets (developer-exported
    credential hashes)"); **keep** the `.env` entry on line 62
23. [ ] memory: annotate the agent-memory file `p2-2-hash-credentials.md` (and its `MEMORY.md`
    index line) as superseded by this feature. It currently asserts Argon2id basic auth as a
    present fact of this codebase, which would send a future session hunting for or re-adding it
    — the same failure this change exists to prevent in the docs. Keep the transferable gotchas
    it records (the archon `should_not_import`-is-vacuous note, the "inspect record `vars()` not
    `caplog.text`" note); drop the current-state claims
24. [ ] verify: `rm -rf src/*.egg-info` **and** the stale non-editable copy the local tox venv
    still holds at `/tmp/.tox/default/lib/python3.11/site-packages/colors_of_meaning/`, then
    `tox -e format`, then a full `tox`. Both can import the deleted modules and fake either a
    green gutted `conftest` or a coverage regression that CI will not reproduce

**Sequencing is load-bearing.** Tasks 1–13 (sources, tests, `pyproject.toml`, `uv.lock`) must
land in ONE commit: `--cov-fail-under=100` fails if the tests go first, and `uv sync --locked`
fails if `pyproject.toml` goes without the relock — which breaks *every* tox environment,
including `clean`. Tasks 14–23 (docs, skills, memory) can follow separately; no gate binds them.
Within the source change, task 8 (`conftest.py`) must not be deferred: its module-scope import
of `basic_authentication` means a deleted `src` module aborts collection of the **entire** suite
— 1,875 tests, not just the auth ones.

Local order of operations: `rm -rf src/*.egg-info /tmp/.tox/default` → delete → `uv lock` →
`tox`.

## Testing Strategy

- No new behaviour, so no new behavioural tests. The suite shrinks by 36 tests (24 + 9 + 2 in the
  deleted files, plus the fail-closed test in `test_main_app.py`) and gains one docs guard.
- One logical assertion per test, `assert_that` in the touched files, and the
  `test_should_<behaviour>_when_<condition>` naming for the new guard.
- Coverage stays at 100% by construction: every deleted line of production code has its tests
  deleted with it, and no surviving line loses its only exercise. The one line to watch is the
  `else:` warning branch in `get_basic_authenticator()` — it disappears with its module, so there
  is nothing to re-cover.
- `xenon` scans tests as well as `src`, so the new guard helper stays grade A — a module-level
  term tuple plus a one-line comprehension helper, matching
  `_unimplemented_infrastructure_terms_in` in the same file.
- Guard against the local-venv trap recorded for this repo: after deleting `src/` modules, stale
  `*.egg-info` can resurrect them in the local tox venv and fake a coverage drop. Remove the
  egg-info before reading a failure as real.
- Verification is `tox` in full, not `pytest` — the eight gates matter here, particularly `ruff`
  (unused imports after the deletions), `mypy` (the container registrations), and `pip-audit`
  (the re-locked closure).

## Observability Plan

Two log sites are deleted with their modules: the `credential verification` INFO record and the
`authentication unavailable: no admin password hash configured` WARNING. No new log records —
the absence of authentication is a documented property, not a per-import event. The remaining
`correlation_id` logging in `api/main.py` is untouched.

## Verification already performed

The predictions above are not read off the source. A scratch worktree was cut from `HEAD`, every
deletion in this plan was executed in it, and the real gate binaries were run from the existing
tox venv. Results:

| Gate | Result on the post-deletion tree |
|---|---|
| pytest + coverage | `6629 stmts, 0 missing, 914 branches, 0 partial, 100%`; `Required test coverage of 100% reached` |
| `ruff check` | one `F401` (`logging` in `test_main_app.py`); clean once dropped |
| `ruff format --check` | one file (`conftest.py`) from an over-eager blank-line cut; `343 files already formatted` with the exact cut above |
| `mypy src` | `Success: no issues found in 170 source files` — the two surviving `# type: ignore` in `main.py` are still needed |
| `bandit -r src` | exit 0 |
| `semgrep --config p/default` | `Ran 434 rules on 649 files: 0 findings` |
| `radon` / `xenon` | exit 0 |
| `test_docs_claims_consistency.py` | 8 passed after the README / cli.md / design.md edits |
| `uv lock` | −94 / +0 lines; `uv lock --check` then exits 0 |

`shared/configuration.py` finishes at 85 stmts / 12 branches / 0 missing, and
`interface/api/main.py` at 109 / 8 / 0 — both 100%, so there is no coverage hole to backfill.

Two further things were checked rather than assumed: `build/openapi.json` has no
`securitySchemes` key **before or after**, so the API contract `bin/create-service` ships with
the image is unchanged and no contract artifact needs regenerating; and `pip-audit` is
unaffected, since removing two distributions cannot add a finding and neither suppressed
advisory relates to argon2.

## Risks and Mitigations

- **Risk (materialised — this is a record, not a prediction):** the local tox venv installs the
  project **non-editable**, so it keeps a snapshot that does not track `src/` deletions. The
  first full `tox` run after the deletions reported `1 failed` on the widened Argon2 archrule and
  `99.92%` coverage with 6 missing statements — all of it from a stale
  `interface/cli/hash_password.py` still sitting in `/tmp/.tox/default/lib/python3.11/
  site-packages/`, which the archrule walked and coverage measured. The same run passed at 100%
  against `PYTHONPATH=src`. Hand-deleting individual files out of the venv makes it *worse*, not
  better, because it leaves the copy internally inconsistent.
  → **Mitigation:** `rm -rf /tmp/.tox src/*.egg-info` and re-run. Never diagnose a post-deletion
  gate failure before doing this. CI is unaffected — it builds fresh environments.
- **Risk:** a deleted setting has a consumer I have not found, and the API fails at import.
  → **Mitigation:** `admin` and `admin_password_hash` were grepped across `src/`, `tests/`,
  `bin/`, `configs/`, `infrastructure/`, `.github/` and the properties resource; the only reader
  is `get_basic_authenticator()`. `ApplicationSettingProvider.get` raises `ValueError` for an
  unknown key, so any missed consumer fails loudly and immediately at container construction —
  which `create_app()` performs at import, so the test suite catches it on collection.
- **Risk:** `uv lock` drifts unrelated dependencies and turns a small change into a large diff,
  or trips `pip-audit`. → **Mitigation:** review the lock diff and confirm it touches only the
  two argon2 entries and the `colors-of-meaning` requires-dist list; re-run if it does more.
- **Risk:** the widened architecture rule passes vacuously and looks like coverage it is not.
  → **Mitigation:** it is asserted against a real predicate over `all_imports`, the same
  mechanism that catches the rule today; the meaningful check is that it fails if `argon2` is
  reintroduced, which is verified by temporarily re-adding an import during development.
- **Risk:** a developer's git-ignored `.env` still carries `APP_ADMIN_PASSWORD_HASH`, and the API
  dies at startup with `ValidationError: extra_forbidden`. **Measured, both directions:**
  `ApplicationSettings` sets `env_file=".env"` and pydantic-settings 2.14.2 defaults to
  `extra="forbid"`, so a stale key **in the `.env` file** raises on every construction, while the
  same key **exported as an environment variable** is silently ignored and harmless. CI never
  sees it (no `.env`), and no `.env` exists on this machine, so no gate will catch it.
  → **Mitigation:** say so explicitly in the commit message and the README sentence. Setting
  `extra="ignore"` on `SettingsConfigDict` would immunise the app, but that is a real behaviour
  change to shared configuration — it also discards typo protection for every other key — so it
  is offered as a decision rather than folded in silently.
- **Risk:** removing the section leaves a reader assuming authentication exists elsewhere.
  → **Mitigation:** acceptance criterion requires one positive sentence stating the API is
  unauthenticated, rather than silent deletion.
- **Risk:** the change is later reverted piecemeal, restoring the code but not the docs.
  → **Mitigation:** this spec pair records the full inventory, and the README guard test fails if
  the claim returns without the wiring discussion this spec demands.
