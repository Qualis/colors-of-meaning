from pathlib import Path

from assertpy import assert_that

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
README_PATH = REPOSITORY_ROOT / "README.MD"
DESIGN_DOC_PATH = REPOSITORY_ROOT / "docs" / "design.md"


def _read_readme() -> str:
    return README_PATH.read_text(encoding="utf-8")


def _read_design_doc() -> str:
    return DESIGN_DOC_PATH.read_text(encoding="utf-8")


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
