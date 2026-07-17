from typing import List
from unittest.mock import Mock

from colors_of_meaning.application.use_case.authorship_report_use_case import (
    AuthorshipReportUseCase,
)
from colors_of_meaning.domain.model.authorship_report import (
    AuthorshipCorpus,
    AuthorshipScalingPoint,
)


def _corpus(authors: int = 2) -> AuthorshipCorpus:
    return AuthorshipCorpus(
        authors_works=[(f"author_{index}", 3) for index in range(authors)],
        split_counts=[("train", 90), ("validation", 20), ("test", 30)],
    )


def _result(accuracy: float, macro_f1: float) -> Mock:
    return Mock(accuracy=accuracy, macro_f1=macro_f1)


def _factory(results: dict) -> Mock:
    def build(method: str) -> Mock:
        use_case = Mock()
        use_case.execute.return_value = results[method]
        return use_case

    return Mock(side_effect=build)


def _sweep(points: List[AuthorshipScalingPoint]) -> Mock:
    return Mock(return_value=points)


def _manifest_points() -> List[AuthorshipScalingPoint]:
    return [
        AuthorshipScalingPoint(paragraphs_per_work=60, train_paragraphs=5340, mean_accuracy=0.09, std_accuracy=0.01)
    ]


def _execute(use_case: AuthorshipReportUseCase, refresh: bool, sweep: Mock):
    return use_case.execute(
        corpus=_corpus(),
        methods=["color", "tfidf"],
        bits_per_token={"color": 12.0, "tfidf": None},
        max_samples=None,
        seed=42,
        refresh_scaling=refresh,
        scaling_sweep=sweep,
    )


class TestAuthorshipReportUseCase:
    def test_should_compute_a_held_out_row_per_method(self) -> None:
        factory = _factory({"color": _result(0.108, 0.106), "tfidf": _result(0.372, 0.347)})
        repository = Mock()
        repository.load.return_value = _manifest_points()

        report = _execute(AuthorshipReportUseCase(factory, repository), False, _sweep([]))

        assert len(report.held_out) == 2

    def test_should_carry_the_method_accuracy_into_the_row(self) -> None:
        factory = _factory({"color": _result(0.108, 0.106), "tfidf": _result(0.372, 0.347)})
        repository = Mock()
        repository.load.return_value = _manifest_points()

        report = _execute(AuthorshipReportUseCase(factory, repository), False, _sweep([]))

        assert report.held_out[0].accuracy == 0.108

    def test_should_pass_the_method_bits_to_the_evaluator(self) -> None:
        color_use_case = Mock()
        color_use_case.execute.return_value = _result(0.1, 0.1)
        factory = Mock(return_value=color_use_case)
        repository = Mock()
        repository.load.return_value = _manifest_points()

        use_case = AuthorshipReportUseCase(factory, repository)
        use_case.execute(
            corpus=_corpus(),
            methods=["color"],
            bits_per_token={"color": 12.0},
            max_samples=None,
            seed=42,
            refresh_scaling=False,
            scaling_sweep=_sweep([]),
        )

        assert color_use_case.execute.call_args.kwargs["bits_per_token"] == 12.0

    def test_should_derive_chance_from_the_author_count(self) -> None:
        factory = _factory({"color": _result(0.1, 0.1), "tfidf": _result(0.1, 0.1)})
        repository = Mock()
        repository.load.return_value = _manifest_points()

        report = _execute(AuthorshipReportUseCase(factory, repository), False, _sweep([]))

        assert report.chance == 0.5

    def test_should_read_scaling_from_the_manifest_when_not_refreshing(self) -> None:
        factory = _factory({"color": _result(0.1, 0.1), "tfidf": _result(0.1, 0.1)})
        repository = Mock()
        repository.load.return_value = _manifest_points()
        sweep = _sweep([])

        _execute(AuthorshipReportUseCase(factory, repository), False, sweep)

        sweep.assert_not_called()

    def test_should_run_the_sweep_when_refreshing(self) -> None:
        factory = _factory({"color": _result(0.1, 0.1), "tfidf": _result(0.1, 0.1)})
        repository = Mock()
        sweep = _sweep(_manifest_points())

        report = _execute(AuthorshipReportUseCase(factory, repository), True, sweep)

        assert report.scaling == _manifest_points()

    def test_should_persist_the_swept_manifest_when_refreshing(self) -> None:
        factory = _factory({"color": _result(0.1, 0.1), "tfidf": _result(0.1, 0.1)})
        repository = Mock()
        points = _manifest_points()

        _execute(AuthorshipReportUseCase(factory, repository), True, _sweep(points))

        repository.save.assert_called_once_with(points)
