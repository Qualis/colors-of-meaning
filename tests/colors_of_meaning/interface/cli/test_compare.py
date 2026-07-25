from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import numpy as np

from colors_of_meaning.interface.cli.compare import main, CompareArgs
from colors_of_meaning.domain.model.colored_document import ColoredDocument


class TestCompareCLI:
    @patch("colors_of_meaning.interface.cli.compare.SynestheticConfig")
    @patch("colors_of_meaning.interface.cli.compare.load_codebook")
    @patch("builtins.open", new_callable=mock_open)
    @patch("colors_of_meaning.interface.cli.compare.pickle")
    @patch("colors_of_meaning.interface.cli.compare.WassersteinDistanceCalculator")
    @patch("colors_of_meaning.interface.cli.compare.CompareDocumentsUseCase")
    @patch("builtins.print")
    def test_should_execute_compare_workflow_with_wasserstein(
        self,
        mock_print: Mock,
        mock_use_case_class: Mock,
        mock_calculator_class: Mock,
        mock_pickle: Mock,
        mock_file: Mock,
        mock_load_codebook: Mock,
        mock_config_class: Mock,
        tmp_path: Path,
    ) -> None:
        mock_config = Mock()
        mock_config.distance.metric = "wasserstein"
        mock_config_class.from_yaml.return_value = mock_config

        mock_codebook = Mock()
        mock_load_codebook.return_value = mock_codebook

        doc1 = ColoredDocument(histogram=np.array([0.5, 0.5], dtype=np.float64), document_id="doc1")
        doc2 = ColoredDocument(histogram=np.array([0.3, 0.7], dtype=np.float64), document_id="doc2")
        mock_pickle.load.return_value = [doc1, doc2]

        mock_calculator = Mock()
        mock_calculator_class.return_value = mock_calculator

        mock_use_case = Mock()
        mock_use_case.find_nearest_neighbors.return_value = [("doc2", 0.5)]
        mock_use_case_class.return_value = mock_use_case

        config_path = tmp_path / "config.yaml"
        encoded_path = tmp_path / "encoded.pkl"

        args = CompareArgs(
            config=str(config_path),
            encoded_documents=str(encoded_path),
            k=5,
            query_index=0,
        )

        main(args)

        mock_calculator_class.assert_called_once_with(
            codebook=mock_codebook, sinkhorn_reg=mock_config.distance.sinkhorn_reg
        )
        mock_use_case.find_nearest_neighbors.assert_called_once()

    @patch("colors_of_meaning.interface.cli.compare.SynestheticConfig")
    @patch("builtins.open", new_callable=mock_open)
    @patch("colors_of_meaning.interface.cli.compare.pickle")
    @patch("colors_of_meaning.interface.cli.compare.JensenShannonDistanceCalculator")
    @patch("colors_of_meaning.interface.cli.compare.CompareDocumentsUseCase")
    @patch("builtins.print")
    def test_should_execute_compare_workflow_with_jensen_shannon(
        self,
        mock_print: Mock,
        mock_use_case_class: Mock,
        mock_calculator_class: Mock,
        mock_pickle: Mock,
        mock_file: Mock,
        mock_config_class: Mock,
        tmp_path: Path,
    ) -> None:
        mock_config = Mock()
        mock_config.distance.metric = "jensen_shannon"
        mock_config.distance.smoothing_epsilon = 1e-8
        mock_config_class.from_yaml.return_value = mock_config

        doc1 = ColoredDocument(histogram=np.array([0.5, 0.5], dtype=np.float64), document_id="doc1")
        mock_pickle.load.return_value = [doc1]

        mock_calculator = Mock()
        mock_calculator_class.return_value = mock_calculator

        mock_use_case = Mock()
        mock_use_case.find_nearest_neighbors.return_value = []
        mock_use_case_class.return_value = mock_use_case

        args = CompareArgs(
            config=str(tmp_path / "config.yaml"),
            encoded_documents=str(tmp_path / "encoded.pkl"),
            k=5,
            query_index=0,
        )

        main(args)

        mock_calculator_class.assert_called_once_with(smoothing_epsilon=1e-8)

    @patch("colors_of_meaning.interface.cli.compare.SynestheticConfig")
    @patch("builtins.open", new_callable=mock_open)
    @patch("colors_of_meaning.interface.cli.compare.pickle")
    def test_should_raise_error_when_query_index_out_of_range(
        self,
        mock_pickle: Mock,
        mock_file: Mock,
        mock_config_class: Mock,
        tmp_path: Path,
    ) -> None:
        mock_config = Mock()
        mock_config.distance.metric = "wasserstein"
        mock_config_class.from_yaml.return_value = mock_config

        doc1 = ColoredDocument(histogram=np.array([0.5, 0.5], dtype=np.float64))
        mock_pickle.load.return_value = [doc1]

        args = CompareArgs(
            config=str(tmp_path / "config.yaml"),
            encoded_documents=str(tmp_path / "encoded.pkl"),
            query_index=5,
        )

        try:
            main(args)
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "out of range" in str(e)
