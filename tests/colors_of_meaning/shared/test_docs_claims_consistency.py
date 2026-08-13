from pathlib import Path

from assertpy import assert_that

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
README_PATH = REPOSITORY_ROOT / "README.MD"
DESIGN_DOC_PATH = REPOSITORY_ROOT / "docs" / "design.md"
CLI_DOC_PATH = REPOSITORY_ROOT / "docs" / "cli.md"
TOX_CONFIGURATION_PATH = REPOSITORY_ROOT / "tox.ini"


def _read_readme() -> str:
    return README_PATH.read_text(encoding="utf-8")


def _read_design_doc() -> str:
    return DESIGN_DOC_PATH.read_text(encoding="utf-8")


def _read_cli_doc() -> str:
    return CLI_DOC_PATH.read_text(encoding="utf-8")


UNIMPLEMENTED_INFRASTRUCTURE_TERMS = (
    "Redis",
    "Vault",
    "Pub/Sub",
    "circuit breaker",
    "distributed tracing",
    "Terraform",
)


def _unimplemented_infrastructure_terms_in(document: str) -> list[str]:
    lowered = document.lower()
    return [term for term in UNIMPLEMENTED_INFRASTRUCTURE_TERMS if term.lower() in lowered]


RETIRED_AUTHENTICATION_TERMS = (
    "APP_ADMIN",
    "Argon2",
    "basic-auth",
    "hash_password",
)


def _retired_authentication_terms_in(document: str) -> list[str]:
    lowered = document.lower()
    return [term for term in RETIRED_AUTHENTICATION_TERMS if term.lower() in lowered]


STATUS_EMOJI = ("✅", "◑", "➕", "⬜", "✓", "\U0001f4c2", "\U0001f4c4")
NON_COMMAND_ENVIRONMENTS = frozenset({"format", "watch", "clean", "build", "publish"})
ENVIRONMENT_HEADER_PREFIX = "[testenv:"


def _status_emoji_in(document: str) -> list[str]:
    return [glyph for glyph in STATUS_EMOJI if glyph in document]


def _environment_headers() -> list[str]:
    lines = TOX_CONFIGURATION_PATH.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.startswith(ENVIRONMENT_HEADER_PREFIX)]


def _environment_name(header: str) -> str:
    return header[len(ENVIRONMENT_HEADER_PREFIX) : -1]


def _command_environment_names() -> list[str]:
    names = [_environment_name(header) for header in _environment_headers()]
    return [name for name in names if name not in NON_COMMAND_ENVIRONMENTS]


def _undocumented_command_environments() -> list[str]:
    cli_doc = _read_cli_doc()
    return [name for name in _command_environment_names() if f"`{name}`" not in cli_doc]


class TestReadmeClaimsConsistency:
    def test_should_not_cite_the_stale_73_book_count_when_reading_readme(self) -> None:
        assert_that(_read_readme()).does_not_contain("73 book")

    def test_should_state_the_committed_133_work_count_when_reading_readme(self) -> None:
        assert_that(_read_readme()).contains("133 works")


class TestDesignDocClaimsConsistency:
    def test_should_not_label_the_distance_wasserstein_2_when_reading_design_doc(self) -> None:
        assert_that(_read_design_doc()).does_not_contain("Wasserstein-2")

    def test_should_label_the_distance_wasserstein_1_when_reading_design_doc(self) -> None:
        assert_that(_read_design_doc()).contains("Wasserstein-1")

    def test_should_not_describe_training_as_random_targets_when_reading_design_doc(self) -> None:
        assert_that(_read_design_doc()).does_not_contain("random targets")


class TestReadmeDoesNotOverclaimInfrastructure:
    def test_should_not_advertise_unimplemented_infrastructure_when_reading_readme(self) -> None:
        assert_that(_unimplemented_infrastructure_terms_in(_read_readme())).is_empty()


class TestDocsDoNotClaimRetiredAuthentication:
    def test_should_not_describe_api_authentication_when_reading_readme(self) -> None:
        assert_that(_retired_authentication_terms_in(_read_readme())).is_empty()

    def test_should_not_describe_api_authentication_when_reading_cli_doc(self) -> None:
        assert_that(_retired_authentication_terms_in(_read_cli_doc())).is_empty()

    def test_should_not_describe_api_authentication_when_reading_design_doc(self) -> None:
        assert_that(_retired_authentication_terms_in(_read_design_doc())).is_empty()


class TestReadmeStaysSuccinct:
    def test_should_not_use_status_emoji_when_reading_readme(self) -> None:
        assert_that(_status_emoji_in(_read_readme())).is_empty()


class TestCommandReferenceCoversEveryEnvironment:
    def test_should_document_every_command_environment_when_reading_cli_doc(self) -> None:
        assert_that(_undocumented_command_environments()).is_empty()
