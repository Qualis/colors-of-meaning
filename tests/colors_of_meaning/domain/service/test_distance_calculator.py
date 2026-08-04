import numpy as np

from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.service.distance_calculator import DistanceCalculator


class FirstBinGapDistanceCalculator(DistanceCalculator):
    def compute_distance(self, doc1: ColoredDocument, doc2: ColoredDocument) -> float:
        return float(abs(doc1.histogram[0] - doc2.histogram[0]))

    def metric_name(self) -> str:
        return "first_bin_gap"


def _document(leading_mass: float) -> ColoredDocument:
    return ColoredDocument(histogram=np.array([leading_mass, 1.0 - leading_mass], dtype=np.float64))


def _queries() -> list:
    return [_document(0.1), _document(0.5)]


def _references() -> list:
    return [_document(0.2), _document(0.4), _document(0.9)]


class TestComputeDistanceMatrix:
    def test_should_return_a_query_by_reference_matrix_when_documents_are_given(self) -> None:
        matrix = FirstBinGapDistanceCalculator().compute_distance_matrix(_queries(), _references())

        assert matrix.shape == (2, 3)

    def test_should_match_the_per_pair_distance_in_every_cell_when_using_the_default(self) -> None:
        calculator = FirstBinGapDistanceCalculator()
        queries, references = _queries(), _references()

        matrix = calculator.compute_distance_matrix(queries, references)

        expected = [[calculator.compute_distance(query, reference) for reference in references] for query in queries]
        assert np.array_equal(matrix, np.array(expected))

    def test_should_return_an_empty_matrix_when_there_are_no_queries(self) -> None:
        matrix = FirstBinGapDistanceCalculator().compute_distance_matrix([], _references())

        assert matrix.shape == (0, 3)

    def test_should_return_an_empty_matrix_when_there_are_no_references(self) -> None:
        matrix = FirstBinGapDistanceCalculator().compute_distance_matrix(_queries(), [])

        assert matrix.shape == (2, 0)
