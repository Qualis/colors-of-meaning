# Feature: Retire API basic authentication

## Overview

The repository ships an HTTP Basic authentication stack — an `Authenticator` port in the domain,
an Argon2id-backed `BasicAuthenticator` and `SecurityDependency` adapter in
`infrastructure/security/`, a `hash_password` CLI, two `ApplicationSettings` fields, an
`argon2-cffi` dependency, 35 tests across three files, two `pytest-archon` rules, and a README
section explaining how to export `APP_ADMIN_PASSWORD_HASH` — **that protects nothing**. No route on the FastAPI app
declares it. `get_container()` constructs a `BasicAuthenticator` and registers it plus a
`SecurityDependency` in the Lagom container, and nothing ever resolves either. The health and
palette-query controllers are mounted without a `dependencies=[…]` list, so every endpoint the
service exposes is, and always has been, open.

The README section is therefore the sharpest kind of documentation defect this repository cares
about: it is *accurate about the mechanism* and *wrong about the outcome*. A reader who follows
it exports a hash, restarts the API, and is authenticated against nothing. On a repository whose
stated purpose is claims-versus-reality honesty, an unwired security control described as a
working one costs more credibility than having no authentication at all.

This feature deletes the stack — port, adapter, CLI, settings, dependency, tests, architecture
rules, and every prose claim — and says plainly, once, that the API is unauthenticated and is
intended for local and trusted-network use. Nothing that runs today changes behaviour, because
nothing that runs today consults it. The removal is deliberately reversible: the implementation
is one `git revert` away, and this spec records what was removed and why, so restoring it later
means restoring it **wired to routes**, which is the only version worth having.

## User Stories

- As an evaluator reading the README, I want the security posture stated honestly, so that I do
  not credit the project with an access control it does not enforce.
- As an operator, I want no instructions that imply the API is protected, so that I do not deploy
  it to an untrusted network believing a credential stands in front of it.
- As a maintainer, I want the dead port, adapter, CLI, settings and dependency gone, so that the
  next reader of `interface/api/main.py` is not tracing container registrations that nothing
  resolves.
- As a maintainer, I want a regression guard, so that password hashing cannot be reintroduced by
  accident without the wiring that would make it real.

## Acceptance Criteria

- [ ] Given `README.MD`, when searched, then the `### API authentication` section is absent and
  no text mentions Argon2, `APP_ADMIN`, `APP_ADMIN_PASSWORD_HASH`, basic auth, or a credential.
- [ ] Given `README.MD`, when read, then a single sentence in the operational section states that
  the API is unauthenticated and intended for local or trusted-network use, so the removal is a
  stated posture rather than a silent omission.
- [ ] Given `src/colors_of_meaning/`, when searched, then `argon2` is imported nowhere,
  `domain/authentication/` and `infrastructure/security/` do not exist, and
  `interface/cli/hash_password.py` does not exist.
- [ ] Given `interface/api/main.py`, when read, then `get_container()` registers no
  `BasicAuthenticator` and no `SecurityDependency`, and the module imports nothing from
  `infrastructure.security`.
- [ ] Given `shared/configuration.py`, when read, then `ApplicationSettings` declares neither
  `admin` nor `admin_password_hash`; the surviving fields are `reload`, `host`, and
  `experiment_config`, which are the three that have consumers.
- [ ] Given the FastAPI app, when every route is exercised, then behaviour is byte-identical to
  before this change — the health endpoints and the palette-query endpoint respond exactly as
  they do today, because no route ever consulted the authenticator.
- [ ] Given `pyproject.toml`, when read, then `argon2-cffi` is absent from `dependencies`, and
  `uv.lock` has been regenerated with `uv lock` so `tox`'s `uv sync --locked` succeeds.
- [ ] Given `tests/`, when run, then the deleted modules' tests are gone
  (`domain/authentication/`, `infrastructure/security/`, `interface/cli/test_hash_password.py`),
  the four unused auth fixtures are gone from `conftest.py`, and no surviving test references
  `admin`, `admin_password_hash`, `APP_ADMIN`, or Argon2.
- [ ] Given `tests/colors_of_meaning/shared/test_configuration.py` and
  `test_configuration_integration.py`, when read, then the settings behaviours they cover —
  properties-file loading, environment-variable precedence, and defaults — are still covered,
  exercised against a surviving setting rather than deleted along with `admin`.
- [ ] Given `tests/colors_of_meaning/test_architecture.py`, when read, then the
  `Security Module` rule is gone (it matches a package that no longer exists) and the Argon2
  confinement rule is **widened**, not deleted: it now asserts that no module anywhere in
  `colors_of_meaning.*` imports `argon2`, which is a live regression guard rather than a
  vacuously-true one.
- [ ] Given `tests/colors_of_meaning/shared/test_docs_claims_consistency.py`, when run, then a
  new guard fails if `README.MD` reacquires an authentication claim.
- [ ] Given `docs/design.md`, `docs/cli.md`, and `.claude/CLAUDE.md`, when read, then the module
  map has no `authentication/` or `security/` line, the `hash_password` utility entry is gone,
  and the Security Requirements section states the honest posture instead of describing an
  Argon2id control.
- [ ] Given `.claude/skills/`, when a scaffold or test template is applied, then the generated
  code compiles: the hexagonal scaffolder no longer emits a controller taking an
  `authentication_dependency: Callable[[Optional[HTTPBasicCredentials]], None]` and attaching
  `dependencies=[Depends(authentication_dependency)]`, and the test generator no longer stubs it.
  Skill templates that generate uncompilable code are the same defect class as documentation that
  describes absent behaviour, and this repository has already treated them that way once
  (`728bceb fix(skills): replace the scaffolder's deleted reference feature`).
- [ ] Given `.gitignore`, when read, then the `.env` entry survives — only its comment, which
  currently justifies the ignore as "developer-exported credential hashes", is generalised.
- [ ] Given a developer whose local git-ignored `.env` still carries `APP_ADMIN_PASSWORD_HASH`,
  when the API starts, then the failure mode is known and documented. Measured, not assumed:
  `ApplicationSettings` sets `env_file=".env"` and pydantic-settings 2.14.2 defaults to
  `extra="forbid"`, so a **`.env` file** entry for a removed field raises
  `ValidationError: extra_forbidden` on every construction, while the same key exported as a
  plain **environment variable** is silently ignored and harmless. The README told developers to
  `export` it (safe); `.gitignore` line 61 describes `.env` as holding "developer-exported
  credential hashes" (unsafe). This is invisible to CI, which has no `.env`.
- [ ] Given `tox`, when run in full, then all eight quality gates pass and coverage is 100%, and
  the suite collects **1,835 tests, down from 1,875**. The arithmetic, so a mismatch is
  diagnosable rather than mysterious: −9 authenticator, −24 basic-authentication,
  −2 `hash_password` CLI, −6 settings, −1 archon (`Security Module`; the Argon2 rule is widened
  and kept, not deleted), −1 fail-closed, **+3** new docs guards. The two
  `test_configuration_integration.py` tests and two `test_configuration.py` tests are
  **re-pointed, not deleted**, so they do not appear in the delta.

### What is deliberately **not** touched

- `.specs/012-p2-2-hash-credentials/` and `.specs/ROADMAP.md` are historical records — `012`
  documents a change that was made, and the ROADMAP records an audit taken at a point in time.
  Rewriting either would falsify the history that makes them useful. This spec is the forward
  record; they remain the backward one.
- `docs/security/audit-suppressions.md` is unrelated despite its path: it records `pip-audit`
  `--ignore-vuln` rationales for `setuptools` and `torch`. It stays, and so does `docs/security/`.
- The pre-existing `coconut*` rot in the two skill reference files. Those examples cite a
  controller retired by spec `011`, and fixing them is its own cleanup. This feature removes the
  **auth** references it creates and leaves a clearly separable problem clearly separate.
- `TestLoadPropertiesFile`'s `"admin=testadmin\npassword=testpassword"` sample text — it
  exercises the generic `key=value` parser against arbitrary input, not the auth feature.
- `CHANGELOG.rst`, which has one stub entry and is not maintained per-feature.

## Hexagonal Layer Impact

### Domain Layer (`src/colors_of_meaning/domain/`)

Deleted: `authentication/authenticator.py` and `authentication/__init__.py` — the entire
`authentication/` package. `Authenticator` is a port with exactly one adapter and zero callers
once the adapter goes.

### Application Layer (`src/colors_of_meaning/application/`)

No change. No use case ever referenced authentication.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

Deleted: `security/basic_authentication.py` and `security/__init__.py` — the entire `security/`
package, comprising `hash_password`, `BasicAuthenticator`, `SecurityDependency`,
`get_basic_authenticator`, and `get_security_dependency`.

### Interface Layer (`src/colors_of_meaning/interface/`)

Deleted: `cli/hash_password.py`.

Modified: `api/main.py` — drop the `infrastructure.security` import block and the four lines of
`get_container()` that build and register the authenticator and security dependency. Nothing
else in the function changes; the codebook, distance calculator, use cases, and health checker
registrations are untouched.

### Shared Layer (`src/colors_of_meaning/shared/`)

Modified: `configuration.py` — remove the `admin` and `admin_password_hash` fields from
`ApplicationSettings`. Both exist solely for the authenticator; `resources/application.properties`
already carries only `reload` and `host`, so no resource file changes.

### Tests (`tests/colors_of_meaning/`)

Deleted: `domain/authentication/` (whole directory), `infrastructure/security/` (whole
directory), `interface/cli/test_hash_password.py`.

Modified: `conftest.py` (four fixtures and one import block removed — none is referenced by any
surviving test, and because it is a `conftest`, the dangling import would otherwise abort
collection of the **entire** suite); `test_architecture.py` (one rule removed, one widened);
`interface/api/test_main_app.py` (the fail-closed-warning test removed, since the warning it
counts is emitted by deleted code); `shared/test_configuration.py` and
`shared/test_configuration_integration.py` (re-pointed at surviving settings);
`shared/test_docs_claims_consistency.py` (new guard).

Two of these fail **silently** rather than red, which is why they are enumerated rather than
discovered by running the suite: the Argon2 confinement predicate is trivially true once the
dependency is gone, and `assert len(fail_closed) <= 1` is satisfied by zero matches.

The `Security Module` archon rule is the opposite and worth stating precisely, because the
repository's own lore predicts it wrongly. It does **not** pass vacuously — it hard-fails with
`Rule 'Security Module': - NO CANDIDATES MATCHED`, because `pytest-archon` calls `add_failure`
on an empty candidate set and converts that into a failed report from its `makereport` hook.
The failure therefore surfaces with no traceback line inside the test file. The familiar
"archon rules go vacuous" behaviour applies to `should_not_import` against *external* packages,
where there is nothing to match on the import side — not to an empty **match** set.

### Repository files

`pyproject.toml` and `uv.lock` (dependency removal), `.gitignore` (comment only),
`.claude/CLAUDE.md`, `README.MD`, `docs/design.md`, `docs/cli.md`, and five files under
`.claude/skills/` whose templates and reference examples are built on the auth stack.

## API Contracts

None changed. Every endpoint responds exactly as before — the change removes an unreferenced
container entry, not a request-path behaviour. The OpenAPI document is unaffected: no
`securitySchemes` entry was ever generated, because `HTTPBasic` was never attached to a route.

## CLI Impact

`python -m colors_of_meaning.interface.cli.hash_password` is removed. It had no `tox`
environment and no console-script entry point, so no `tox.ini` environment, `pyproject.toml`
`[project.scripts]` entry, or `bin/` script changes are required. `docs/cli.md` loses its
`### hash_password` entry under `## Utilities`.

## Dependency Injection

Two Lagom registrations are removed from `get_container()`: `container[BasicAuthenticator]` and
`container[SecurityDependency]`. No registration is added. Nothing resolved either key, so no
call site needs a replacement.

## Observability

Two log sites disappear with their modules: the `credential verification` INFO record (with
`correlation_id`, username, and outcome) and the `authentication unavailable: no admin password
hash configured` WARNING emitted at container construction. No new logging is added — an absent
authenticator is a documented property of the service, not a runtime event worth a record on
every import.

## Open Questions

- **Scope reading.** "Remove the basic-auth for the moment" is read as *delete the
  implementation*, not *comment it out* or *leave it and delete only the README section*. Leaving
  unreferenced security code in the tree while denying it in the docs would invert the current
  defect rather than fix it. Git history plus this spec is the restoration path.
- **`ApplicationSettings.admin` goes too.** It is read only by
  `get_basic_authenticator()`. Keeping an orphan `admin` username setting would leave the same
  kind of residue this change exists to remove. Reversible in one line if a future consumer
  appears.
- **The widened Argon2 architecture rule is a judgment call.** The alternative is deleting the
  rule outright. Widening costs five lines, keeps a test that already exists, and converts a
  boundary rule into a regression guard — the same pattern
  `test_docs_claims_consistency.py` already uses for prose. Trivially deletable if unwanted.
- **`.specs/ROADMAP.md` will read oddly and that is accepted.** It records `P2-2 security: hash +
  de-commit credentials` as a completed milestone for a feature that no longer exists. Specs
  `016`, `027` and `032` all left the ROADMAP untouched when they superseded its findings, and an
  audit taken on a date should keep saying what was true on that date. If a reconciliation is
  wanted, the honest edit is a one-line "superseded by `040`" note, never deleting the row.
- **The API's security posture after this change** is: unauthenticated, documented as such, and
  suitable for local or trusted-network use. If authentication is wanted later, the version worth
  building attaches the dependency to the routers in `create_app()` and has a contract test that
  a request without credentials receives `401` — which is precisely the test the deleted suite
  never had.
