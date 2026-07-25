import json
from contextlib import ExitStack
from pathlib import Path
from typing import List, Tuple
from unittest.mock import Mock, patch

import numpy as np
import pytest
from assertpy import assert_that

from colors_of_meaning.domain.model.ablation_result import AblationResult
from colors_of_meaning.domain.model.evaluation_result import EvaluationResult
from colors_of_meaning.interface.cli.ablate import (
    AblateArgs,
    MAX_STRUCTURE_SAMPLES,
    main,
    _build_classifier_factory,
    _create_distance_calculator,
    _parse_codebook_specification,
    _structure_sample_budget,
)


def _scripted_results() -> List[AblationResult]:
    result = EvaluationResult(accuracy=0.81, macro_f1=0.79, recall_at_k={}, mrr=0.7)
    return [
        AblationResult("grid4096", "wasserstein", result, -0.6),
        AblationResult("grid4096", "cosine", result, -0.55),
    ]


def _run_main(tmp_path: Path, scripted_results: List[AblationResult]) -> Tuple[Mock, Path]:
    output_path = tmp_path / "sweep.json"
    args = AblateArgs(config="configs/base.yaml", dataset="ag_news", output_path=str(output_path))
    with ExitStack() as stack:
        config = Mock()
        config.training.seed = 42
        config.training.batch_size = 32
        config.dataset.test_split = "test"
        config.dataset.max_samples = 100
        stack.enter_context(
            patch("colors_of_meaning.interface.cli.ablate.SynestheticConfig")
        ).from_yaml.return_value = config

        dataset = Mock()
        sample = Mock()
        sample.text = "a sentence."
        dataset.get_samples.return_value = [sample]
        stack.enter_context(patch("colors_of_meaning.interface.cli.ablate.AGNewsDatasetAdapter")).return_value = dataset

        embedding_adapter = Mock()
        embedding_adapter.encode_batch.return_value = np.zeros((1, 3), dtype=np.float32)
        stack.enter_context(patch("colors_of_meaning.interface.cli.ablate.SentenceEmbeddingAdapter")).return_value = (
            embedding_adapter
        )

        stack.enter_context(patch("colors_of_meaning.interface.cli.ablate.create_color_mapper"))
        mock_load_codebook = stack.enter_context(patch("colors_of_meaning.interface.cli.ablate.load_codebook"))
        mock_load_codebook.return_value = Mock()
        stack.enter_context(patch("colors_of_meaning.interface.cli.ablate.SpearmanStructurePreservationEvaluator"))
        stack.enter_context(patch("colors_of_meaning.interface.cli.ablate.SklearnMetricsCalculator"))

        use_case_class = stack.enter_context(patch("colors_of_meaning.interface.cli.ablate.AblationSweepUseCase"))
        use_case_class.return_value.execute.return_value = scripted_results

        print_mock = stack.enter_context(patch("builtins.print"))
        main(args)
    return print_mock, output_path


def _printed_lines(print_mock: Mock) -> List[str]:
    return [str(call.args[0]) for call in print_mock.call_args_list if call.args]


class TestAblateCLI:
    def test_should_default_metrics_when_args_created(self) -> None:
        args = AblateArgs()

        assert args.metrics == ["wasserstein", "jensen_shannon", "cosine"]

    def test_should_default_codebooks_when_args_created(self) -> None:
        args = AblateArgs()

        assert args.codebooks == ["grid1024=codebook_1024", "grid4096=codebook_4096", "learned=codebook_learned"]

    def test_should_create_wasserstein_calculator_when_metric_is_wasserstein(self) -> None:
        with patch("colors_of_meaning.interface.cli.ablate.WassersteinDistanceCalculator") as calculator_class:
            calculator = _create_distance_calculator("wasserstein", Mock(), Mock())

        assert calculator is calculator_class.return_value

    def test_should_create_jensen_shannon_calculator_when_metric_is_jensen_shannon(self) -> None:
        with patch("colors_of_meaning.interface.cli.ablate.JensenShannonDistanceCalculator") as calculator_class:
            calculator = _create_distance_calculator("jensen_shannon", Mock(), Mock())

        assert calculator is calculator_class.return_value

    def test_should_create_cosine_calculator_when_metric_is_cosine(self) -> None:
        with patch("colors_of_meaning.interface.cli.ablate.CosineHistogramDistanceCalculator") as calculator_class:
            calculator = _create_distance_calculator("cosine", Mock(), Mock())

        assert calculator is calculator_class.return_value

    def test_should_raise_value_error_when_metric_is_unknown(self) -> None:
        with pytest.raises(ValueError, match="Unknown metric"):
            _create_distance_calculator("unknown", Mock(), Mock())

    def test_should_raise_value_error_when_codebook_specification_lacks_separator(self) -> None:
        with pytest.raises(ValueError, match="label=path"):
            _parse_codebook_specification("nolabel")

    def test_should_build_color_histogram_classifier_from_factory(self) -> None:
        with ExitStack() as stack:
            stack.enter_context(patch("colors_of_meaning.interface.cli.ablate.QuantizedColorMapper"))
            stack.enter_context(patch("colors_of_meaning.interface.cli.ablate.EncodeDocumentUseCase"))
            stack.enter_context(patch("colors_of_meaning.interface.cli.ablate.CosineHistogramDistanceCalculator"))
            classifier_class = stack.enter_context(
                patch("colors_of_meaning.interface.cli.ablate.ColorHistogramClassifier")
            )
            factory = _build_classifier_factory(Mock(), Mock(), Mock(), 5)

            classifier = factory(Mock(), "cosine")

        assert classifier is classifier_class.return_value

    def test_should_cap_structure_samples_when_dataset_has_no_limit(self) -> None:
        config = Mock()
        config.dataset.max_samples = None

        assert _structure_sample_budget(config) == MAX_STRUCTURE_SAMPLES

    def test_should_respect_dataset_limit_when_smaller_than_cap(self) -> None:
        config = Mock()
        config.dataset.max_samples = 10

        assert _structure_sample_budget(config) == 10

    def test_should_write_artifact_with_row_per_cell_when_main_runs(self, tmp_path: Path) -> None:
        _, output_path = _run_main(tmp_path, _scripted_results())

        written = json.loads(output_path.read_text())

        assert_that(written).is_length(2)

    def test_should_print_row_per_cell_when_main_runs(self, tmp_path: Path) -> None:
        print_mock, _ = _run_main(tmp_path, _scripted_results())

        data_rows = [line for line in _printed_lines(print_mock) if line.startswith("grid4096 |")]

        assert len(data_rows) == 2
