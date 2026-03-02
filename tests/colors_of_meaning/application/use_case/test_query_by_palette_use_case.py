import numpy as np
from unittest.mock import Mock

from colors_of_meaning.application.use_case.query_by_palette_use_case import (
    QueryByPaletteUseCase,
)
from colors_of_meaning.domain.model.lab_color import LabColor
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.colored_document import ColoredDocument


class TestQueryByPaletteUseCase:
    def test_should_query_by_palette(self) -> None:
        mock_compare = Mock()
        mock_compare.find_nearest_neighbors.return_value = [("doc_1", 0.5)]

        codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=2)

        use_case = QueryByPaletteUseCase(
            compare_use_case=mock_compare,
            codebook=codebook,
        )

        palette = [(LabColor(l=50, a=0, b=0), 1.0)]
        corpus = [ColoredDocument(histogram=np.ones(8, dtype=np.float64) / 8, document_id="doc_1")]

        results = use_case.execute(palette=palette, corpus_docs=corpus, k=1)

        assert len(results) == 1

    def test_should_delegate_to_compare_use_case(self) -> None:
        mock_compare = Mock()
        mock_compare.find_nearest_neighbors.return_value = []

        codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=2)

        use_case = QueryByPaletteUseCase(
            compare_use_case=mock_compare,
            codebook=codebook,
        )

        palette = [(LabColor(l=50, a=0, b=0), 1.0)]
        corpus = []

        use_case.execute(palette=palette, corpus_docs=corpus, k=5)

        mock_compare.find_nearest_neighbors.assert_called_once()

    def test_should_create_document_with_palette_query_id(self) -> None:
        mock_compare = Mock()
        mock_compare.find_nearest_neighbors.return_value = []

        codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=2)

        use_case = QueryByPaletteUseCase(
            compare_use_case=mock_compare,
            codebook=codebook,
        )

        palette = [(LabColor(l=50, a=0, b=0), 1.0)]
        use_case.execute(palette=palette, corpus_docs=[], k=5)

        call_args = mock_compare.find_nearest_neighbors.call_args
        query_doc = call_args.kwargs.get("query_doc") or call_args[1].get("query_doc") or call_args[0][0]

        assert query_doc.document_id == "palette_query"

    def test_should_create_normalized_histogram(self) -> None:
        mock_compare = Mock()
        mock_compare.find_nearest_neighbors.return_value = []

        codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=2)

        use_case = QueryByPaletteUseCase(
            compare_use_case=mock_compare,
            codebook=codebook,
        )

        palette = [
            (LabColor(l=50, a=0, b=0), 2.0),
            (LabColor(l=25, a=-64, b=-64), 1.0),
        ]
        doc = use_case._palette_to_document(palette)

        assert np.isclose(doc.histogram.sum(), 1.0)

    def test_should_handle_empty_weights_with_uniform_distribution(self) -> None:
        mock_compare = Mock()
        mock_compare.find_nearest_neighbors.return_value = []

        codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=2)

        use_case = QueryByPaletteUseCase(
            compare_use_case=mock_compare,
            codebook=codebook,
        )

        palette = [(LabColor(l=50, a=0, b=0), 0.0)]
        doc = use_case._palette_to_document(palette)

        assert np.isclose(doc.histogram.sum(), 1.0)
