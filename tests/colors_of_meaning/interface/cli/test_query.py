from unittest.mock import Mock, patch, mock_open
import pytest

from colors_of_meaning.interface.cli.query import main, QueryArgs, _parse_palette
from colors_of_meaning.domain.model.lab_color import LabColor


class TestParsePalette:
    def test_should_parse_single_color(self) -> None:
        json_str = '[{"l": 50, "a": 10, "b": -20, "weight": 1.0}]'

        palette = _parse_palette(json_str)

        assert len(palette) == 1

    def test_should_parse_color_values(self) -> None:
        json_str = '[{"l": 50, "a": 10, "b": -20, "weight": 1.0}]'

        palette = _parse_palette(json_str)
        color, weight = palette[0]

        assert isinstance(color, LabColor)
        assert color.l == 50.0

    def test_should_parse_weight(self) -> None:
        json_str = '[{"l": 50, "a": 10, "b": -20, "weight": 2.5}]'

        palette = _parse_palette(json_str)
        _, weight = palette[0]

        assert weight == 2.5

    def test_should_default_weight_to_one(self) -> None:
        json_str = '[{"l": 50, "a": 10, "b": -20}]'

        palette = _parse_palette(json_str)
        _, weight = palette[0]

        assert weight == 1.0

    def test_should_parse_multiple_colors(self) -> None:
        json_str = '[{"l": 50, "a": 0, "b": 0, "weight": 1.0}, {"l": 75, "a": 10, "b": -10, "weight": 0.5}]'

        palette = _parse_palette(json_str)

        assert len(palette) == 2


class TestQueryCLI:
    @patch("colors_of_meaning.interface.cli.query.FileColorCodebookRepository")
    @patch("colors_of_meaning.interface.cli.query.WassersteinDistanceCalculator")
    @patch("colors_of_meaning.interface.cli.query.CompareDocumentsUseCase")
    @patch("colors_of_meaning.interface.cli.query.QueryByPaletteUseCase")
    @patch("builtins.open", new_callable=mock_open)
    @patch("colors_of_meaning.interface.cli.query.pickle")
    @patch("builtins.print")
    def test_should_execute_query_workflow(
        self,
        mock_print: Mock,
        mock_pickle: Mock,
        mock_file: Mock,
        mock_query_class: Mock,
        mock_compare_class: Mock,
        mock_distance_class: Mock,
        mock_repo_class: Mock,
    ) -> None:
        mock_repo = Mock()
        mock_repo.load.return_value = Mock(num_bins=8)
        mock_repo_class.return_value = mock_repo

        mock_pickle.load.return_value = []

        mock_query = Mock()
        mock_query.execute.return_value = [("doc_1", 0.5)]
        mock_query_class.return_value = mock_query

        args = QueryArgs(
            palette_json='[{"l": 50, "a": 0, "b": 0, "weight": 1.0}]',
            encoded_documents="test.pkl",
            codebook_name="codebook_4096",
            k=5,
        )

        main(args)

        mock_query.execute.assert_called_once()

    @patch("colors_of_meaning.interface.cli.query.FileColorCodebookRepository")
    @patch("builtins.open", new_callable=mock_open)
    @patch("colors_of_meaning.interface.cli.query.pickle")
    @patch("builtins.print")
    def test_should_raise_when_codebook_not_found(
        self,
        mock_print: Mock,
        mock_pickle: Mock,
        mock_file: Mock,
        mock_repo_class: Mock,
    ) -> None:
        mock_repo = Mock()
        mock_repo.load.return_value = None
        mock_repo_class.return_value = mock_repo

        mock_pickle.load.return_value = []

        args = QueryArgs(
            palette_json='[{"l": 50, "a": 0, "b": 0, "weight": 1.0}]',
            encoded_documents="test.pkl",
            codebook_name="missing_codebook",
            k=5,
        )

        with pytest.raises(FileNotFoundError):
            main(args)
