from assertpy import assert_that

from colors_of_meaning.domain.model.authorship_report import (
    AuthorshipCorpus,
    AuthorshipHeldOut,
    AuthorshipReport,
    AuthorshipScalingPoint,
)


def _corpus() -> AuthorshipCorpus:
    return AuthorshipCorpus(
        authors_works=[("austen", 6), ("darwin", 4)],
        split_counts=[("train", 100), ("validation", 20), ("test", 30)],
    )


class TestAuthorshipCorpus:
    def test_should_count_the_qualifying_authors(self) -> None:
        assert_that(_corpus().author_count()).is_equal_to(2)

    def test_should_sum_the_works_across_authors(self) -> None:
        assert_that(_corpus().total_works()).is_equal_to(10)


class TestAuthorshipScalingPoint:
    def test_should_report_accuracy_relative_to_chance(self) -> None:
        point = AuthorshipScalingPoint(
            paragraphs_per_work=60, train_paragraphs=5340, mean_accuracy=0.09, std_accuracy=0.016
        )

        assert_that(point.times_chance(0.045)).is_equal_to(2.0)


class TestAuthorshipHeldOut:
    def test_should_carry_the_method_accuracy(self) -> None:
        row = AuthorshipHeldOut(method="color", bits_per_token=12.0, accuracy=0.108, macro_f1=0.106)

        assert_that(row.accuracy).is_equal_to(0.108)


class TestAuthorshipReport:
    def test_should_carry_the_computed_chance(self) -> None:
        report = AuthorshipReport(corpus=_corpus(), held_out=[], chance=0.045, scaling=[])

        assert_that(report.chance).is_equal_to(0.045)
