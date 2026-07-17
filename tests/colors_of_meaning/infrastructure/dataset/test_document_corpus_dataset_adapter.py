import hashlib
import logging
from pathlib import Path

import numpy as np
import pytest

from colors_of_meaning.application.use_case.encode_document_use_case import (
    EncodeDocumentUseCase,
)
from colors_of_meaning.application.use_case.evaluate_use_case import EvaluateUseCase
from colors_of_meaning.application.use_case.rate_distortion_sweep_use_case import (
    RateDistortionSweepUseCase,
)
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.service.color_mapper import QuantizedColorMapper
from colors_of_meaning.infrastructure.dataset.document_corpus_dataset_adapter import (
    DocumentCorpusDatasetAdapter,
)
from colors_of_meaning.infrastructure.evaluation.color_histogram_classifier import (
    ColorHistogramClassifier,
)
from colors_of_meaning.infrastructure.evaluation.sklearn_metrics_calculator import (
    SklearnMetricsCalculator,
)
from colors_of_meaning.infrastructure.ml.color_vq_compression_baseline import (
    ColorVqCompressionBaseline,
)
from colors_of_meaning.infrastructure.ml.pytorch_color_mapper import PyTorchColorMapper
from colors_of_meaning.infrastructure.ml.wasserstein_distance_calculator import (
    WassersteinDistanceCalculator,
)

EMBED_DIM = 8


def _write(root: Path, author: str, work: str, text: str) -> None:
    author_dir = root / author
    author_dir.mkdir(parents=True, exist_ok=True)
    (author_dir / f"{work}.txt").write_text(text, encoding="utf-8")


def _paragraphs(prefix: str, count: int) -> str:
    return "\n\n".join(f"{prefix} paragraph {index} here" for index in range(count))


def _all_samples(adapter: DocumentCorpusDatasetAdapter) -> list:
    return (
        adapter.get_samples("train", seed=0)
        + adapter.get_samples("validation", seed=0)
        + adapter.get_samples("test", seed=0)
    )


def _texts_with_prefix(samples: list, prefix: str) -> set:
    return {sample.text for sample in samples if sample.text.startswith(prefix)}


class TestAuthorLabelling:
    def test_should_return_authors_in_sorted_order(self, tmp_path: Path) -> None:
        _write(tmp_path, "zebra", "w1", _paragraphs("zzz", 3))
        _write(tmp_path, "alpha", "w1", _paragraphs("aaa", 3))

        adapter = DocumentCorpusDatasetAdapter(str(tmp_path), min_paragraph_chars=5)

        assert adapter.get_label_names() == ["alpha", "zebra"]

    def test_should_assign_label_zero_to_the_first_author(self, tmp_path: Path) -> None:
        _write(tmp_path, "zebra", "w1", _paragraphs("zzz", 3))
        _write(tmp_path, "alpha", "w1", _paragraphs("aaa", 3))

        adapter = DocumentCorpusDatasetAdapter(str(tmp_path), min_paragraph_chars=5)

        assert {sample.label for sample in _all_samples(adapter) if sample.text.startswith("aaa")} == {0}

    def test_should_count_classes_as_the_qualifying_authors(self, tmp_path: Path) -> None:
        _write(tmp_path, "alpha", "w1", _paragraphs("aaa", 3))
        _write(tmp_path, "beta", "w1", _paragraphs("bbb", 3))

        adapter = DocumentCorpusDatasetAdapter(str(tmp_path), min_paragraph_chars=5)

        assert adapter.get_num_classes() == 2

    def test_should_share_one_label_across_an_authors_multiple_works(self, tmp_path: Path) -> None:
        for work in ("origin", "voyage", "descent"):
            _write(tmp_path, "darwin", work, _paragraphs(work, 1))

        adapter = DocumentCorpusDatasetAdapter(str(tmp_path), min_paragraph_chars=5)

        assert {sample.label for sample in _all_samples(adapter)} == {0}


class TestWorkLevelSplit:
    def _three_work_author(self, root: Path) -> None:
        _write(root, "darwin", "a_origin", _paragraphs("origin", 2))
        _write(root, "darwin", "b_voyage", _paragraphs("voyage", 2))
        _write(root, "darwin", "c_descent", _paragraphs("descent", 2))

    def test_should_not_overlap_any_split_for_work_level_holdout(self, tmp_path: Path) -> None:
        self._three_work_author(tmp_path)

        adapter = DocumentCorpusDatasetAdapter(str(tmp_path), min_paragraph_chars=5)

        train = {s.text for s in adapter.get_samples("train", seed=0)}
        validation = {s.text for s in adapter.get_samples("validation", seed=0)}
        test = {s.text for s in adapter.get_samples("test", seed=0)}
        assert len(train) + len(validation) + len(test) == len(train | validation | test)

    def test_should_hold_out_a_whole_work_for_test(self, tmp_path: Path) -> None:
        self._three_work_author(tmp_path)

        adapter = DocumentCorpusDatasetAdapter(str(tmp_path), min_paragraph_chars=5)

        assert all(sample.text.startswith("descent") for sample in adapter.get_samples("test", seed=0))

    def test_should_hold_out_a_whole_work_for_validation(self, tmp_path: Path) -> None:
        self._three_work_author(tmp_path)

        adapter = DocumentCorpusDatasetAdapter(str(tmp_path), min_paragraph_chars=5)

        assert all(sample.text.startswith("voyage") for sample in adapter.get_samples("validation", seed=0))

    def test_should_produce_identical_samples_on_repeated_loads(self, tmp_path: Path) -> None:
        _write(tmp_path, "alpha", "w1", _paragraphs("aaa", 6))
        _write(tmp_path, "beta", "w1", _paragraphs("bbb", 6))

        first = DocumentCorpusDatasetAdapter(str(tmp_path), min_paragraph_chars=5).get_samples("train", seed=42)
        second = DocumentCorpusDatasetAdapter(str(tmp_path), min_paragraph_chars=5).get_samples("train", seed=42)
        assert first == second


class TestParagraphSplit:
    def test_should_populate_train_for_a_single_work_author(self, tmp_path: Path) -> None:
        _write(tmp_path, "solo", "only", _paragraphs("p", 6))

        adapter = DocumentCorpusDatasetAdapter(str(tmp_path), min_paragraph_chars=5)

        assert len(adapter.get_samples("train", seed=0)) >= 1

    def test_should_populate_validation_for_a_single_work_author(self, tmp_path: Path) -> None:
        _write(tmp_path, "solo", "only", _paragraphs("p", 6))

        adapter = DocumentCorpusDatasetAdapter(str(tmp_path), min_paragraph_chars=5)

        assert len(adapter.get_samples("validation", seed=0)) >= 1

    def test_should_populate_test_for_a_single_work_author(self, tmp_path: Path) -> None:
        _write(tmp_path, "solo", "only", _paragraphs("p", 6))

        adapter = DocumentCorpusDatasetAdapter(str(tmp_path), min_paragraph_chars=5)

        assert len(adapter.get_samples("test", seed=0)) >= 1

    def test_should_place_same_work_paragraphs_in_multiple_splits_under_paragraph_strategy(
        self, tmp_path: Path
    ) -> None:
        _write(tmp_path, "darwin", "a_origin", _paragraphs("origin", 6))
        _write(tmp_path, "darwin", "b_voyage", _paragraphs("voyage", 6))

        adapter = DocumentCorpusDatasetAdapter(str(tmp_path), min_paragraph_chars=5, split_strategy="paragraph")

        train_voyage = _texts_with_prefix(adapter.get_samples("train", seed=0), "voyage")
        test_voyage = _texts_with_prefix(adapter.get_samples("test", seed=0), "voyage")
        assert train_voyage and test_voyage


class TestSampling:
    def _two_author_corpus(self, root: Path) -> None:
        _write(root, "alpha", "w", _paragraphs("alpha", 10))
        _write(root, "beta", "w", _paragraphs("beta", 10))

    def test_should_limit_to_the_max_samples_budget(self, tmp_path: Path) -> None:
        self._two_author_corpus(tmp_path)

        adapter = DocumentCorpusDatasetAdapter(str(tmp_path), min_paragraph_chars=5)

        assert len(adapter.get_samples("train", max_samples=4, seed=1)) == 4

    def test_should_subsample_reproducibly_under_the_same_seed(self, tmp_path: Path) -> None:
        self._two_author_corpus(tmp_path)

        adapter = DocumentCorpusDatasetAdapter(str(tmp_path), min_paragraph_chars=5)

        first = adapter.get_samples("train", max_samples=4, seed=1)
        second = adapter.get_samples("train", max_samples=4, seed=1)
        assert first == second

    def test_should_stratify_the_budget_across_authors(self, tmp_path: Path) -> None:
        self._two_author_corpus(tmp_path)

        adapter = DocumentCorpusDatasetAdapter(str(tmp_path), min_paragraph_chars=5)

        labels = [sample.label for sample in adapter.get_samples("train", max_samples=4, seed=1)]
        assert sorted(labels) == [0, 0, 1, 1]

    def test_should_cap_the_paragraphs_taken_per_work(self, tmp_path: Path) -> None:
        _write(tmp_path, "alpha", "w1", _paragraphs("alpha", 20))
        _write(tmp_path, "beta", "w1", _paragraphs("beta", 20))

        adapter = DocumentCorpusDatasetAdapter(str(tmp_path), min_paragraph_chars=5, paragraphs_per_work=6)

        assert len(_texts_with_prefix(_all_samples(adapter), "alpha")) == 6

    def test_should_order_the_full_split_deterministically_without_a_seed(self, tmp_path: Path) -> None:
        self._two_author_corpus(tmp_path)

        adapter = DocumentCorpusDatasetAdapter(str(tmp_path), min_paragraph_chars=5)

        first = [sample.text for sample in adapter.get_samples("train")]
        second = [sample.text for sample in adapter.get_samples("train")]
        assert first == second


class TestWorksPerAuthor:
    def test_should_report_the_work_count_for_each_qualifying_author(self, tmp_path: Path) -> None:
        _write(tmp_path, "alpha", "w1", _paragraphs("alpha", 6))
        _write(tmp_path, "beta", "w1", _paragraphs("beta", 6))
        _write(tmp_path, "beta", "w2", _paragraphs("beta_two", 6))

        adapter = DocumentCorpusDatasetAdapter(str(tmp_path), min_paragraph_chars=5)

        assert adapter.works_per_author() == [("alpha", 1), ("beta", 2)]


class TestSkipping:
    def test_should_ignore_non_txt_files(self, tmp_path: Path) -> None:
        _write(tmp_path, "alpha", "w1", _paragraphs("aaa", 3))
        (tmp_path / "alpha" / "notes.md").write_text("ignored markdown content here", encoding="utf-8")

        adapter = DocumentCorpusDatasetAdapter(str(tmp_path), min_paragraph_chars=5)

        assert all("markdown" not in sample.text for sample in _all_samples(adapter))

    def test_should_skip_empty_work_files(self, tmp_path: Path) -> None:
        _write(tmp_path, "alpha", "good", _paragraphs("aaa", 3))
        _write(tmp_path, "alpha", "empty", "")

        adapter = DocumentCorpusDatasetAdapter(str(tmp_path), min_paragraph_chars=5)

        assert len(_all_samples(adapter)) == 3

    def test_should_skip_a_directory_named_like_a_txt_work_without_raising(self, tmp_path: Path) -> None:
        _write(tmp_path, "alpha", "w1", _paragraphs("aaa", 3))
        (tmp_path / "alpha" / "trap.txt").mkdir()

        adapter = DocumentCorpusDatasetAdapter(str(tmp_path), min_paragraph_chars=5)

        assert adapter.get_label_names() == ["alpha"]

    def test_should_skip_an_author_with_too_few_paragraphs_to_split(self, tmp_path: Path) -> None:
        _write(tmp_path, "alpha", "w1", _paragraphs("aaa", 3))
        _write(tmp_path, "tiny", "w1", _paragraphs("ttt", 2))

        adapter = DocumentCorpusDatasetAdapter(str(tmp_path), min_paragraph_chars=5)

        assert adapter.get_label_names() == ["alpha"]

    def test_should_return_no_labels_when_the_directory_is_missing(self, tmp_path: Path) -> None:
        adapter = DocumentCorpusDatasetAdapter(str(tmp_path / "absent"), min_paragraph_chars=5)

        assert adapter.get_label_names() == []

    def test_should_return_no_samples_when_the_directory_is_missing(self, tmp_path: Path) -> None:
        adapter = DocumentCorpusDatasetAdapter(str(tmp_path / "absent"))

        assert adapter.get_samples("train") == []


class TestDiscoveryLogging:
    def test_should_log_the_discovered_author_count(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        _write(tmp_path, "alpha", "w1", _paragraphs("aaa", 3))
        _write(tmp_path, "beta", "w1", _paragraphs("bbb", 3))

        with caplog.at_level(logging.INFO):
            DocumentCorpusDatasetAdapter(str(tmp_path), min_paragraph_chars=5)

        record = next(record for record in caplog.records if record.msg == "Discovered document corpus")
        assert record.authors == 2

    def test_should_log_the_validation_split_size(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        _write(tmp_path, "alpha", "w1", _paragraphs("aaa", 6))

        with caplog.at_level(logging.INFO):
            DocumentCorpusDatasetAdapter(str(tmp_path), min_paragraph_chars=5)

        record = next(record for record in caplog.records if record.msg == "Discovered document corpus")
        assert record.validation_size >= 1


class TestDeduplication:
    def test_should_drop_a_paragraph_repeated_across_works(self, tmp_path: Path) -> None:
        shared = "this shared passage is reprinted across two works of the same author here"
        _write(tmp_path, "smith", "essays", shared + "\n\n" + _paragraphs("essays", 3))
        _write(tmp_path, "smith", "theory", shared + "\n\n" + _paragraphs("theory", 3))

        adapter = DocumentCorpusDatasetAdapter(str(tmp_path), min_paragraph_chars=5)

        assert len([sample for sample in _all_samples(adapter) if sample.text == shared]) == 1


class _ClusteredEmbeddingAdapter:
    def encode_document_sentences(self, document: str, batch_size: int = 32) -> np.ndarray:
        return self.encode_batch([document])

    def encode_batch(self, texts: list, batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        return np.array([self._vector(text) for text in texts], dtype=np.float32)

    def _vector(self, text: str) -> np.ndarray:
        cluster = self._seeded(text.split()[0]) * 6.0
        jitter = self._seeded(text) * 0.2
        return (cluster + jitter).astype(np.float32)

    @staticmethod
    def _seeded(key: str) -> np.ndarray:
        digest = hashlib.sha256(key.encode("utf-8")).digest()[:4]
        return np.random.default_rng(int.from_bytes(digest, "big")).standard_normal(EMBED_DIM)


def _corpus_tree(root: Path) -> None:
    for author in ("alpha", "beta"):
        for work in ("one", "two", "three"):
            _write(root, author, work, _paragraphs(f"{author} {work}", 4))


def _color_vq_baseline_factory(color_mapper: PyTorchColorMapper):  # type: ignore[no-untyped-def]
    def build(method: str, budget: int) -> ColorVqCompressionBaseline:
        codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=budget)
        return ColorVqCompressionBaseline(codebook=codebook, color_mapper=color_mapper)

    return build


@pytest.mark.integration
class TestDocumentCorpusAuthorAccuracy:
    def _evaluate_use_case(self, root: Path) -> EvaluateUseCase:
        adapter = DocumentCorpusDatasetAdapter(str(root), min_paragraph_chars=5)
        mapper = PyTorchColorMapper(
            input_dim=EMBED_DIM, hidden_dim_1=16, hidden_dim_2=8, dropout_rate=0.0, device="cpu"
        )
        codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=2)
        encode_use_case = EncodeDocumentUseCase(QuantizedColorMapper(mapper, codebook))
        classifier = ColorHistogramClassifier(
            _ClusteredEmbeddingAdapter(), encode_use_case, WassersteinDistanceCalculator(codebook=codebook), k=3
        )
        return EvaluateUseCase(classifier, SklearnMetricsCalculator(), adapter)

    def test_should_measure_author_accuracy_as_a_valid_fraction(self, tmp_path: Path) -> None:
        _corpus_tree(tmp_path)

        result = self._evaluate_use_case(tmp_path).execute(seed=42)

        assert 0.0 <= result.accuracy <= 1.0

    def test_should_classify_authors_identically_on_repeated_runs(self, tmp_path: Path) -> None:
        _corpus_tree(tmp_path)
        use_case = self._evaluate_use_case(tmp_path)

        first = use_case.execute(seed=42)
        second = use_case.execute(seed=42)

        assert first.accuracy == second.accuracy


@pytest.mark.integration
class TestDocumentCorpusRateDistortion:
    def _sweep(self, root: Path):  # type: ignore[no-untyped-def]
        adapter = DocumentCorpusDatasetAdapter(str(root), min_paragraph_chars=5)
        texts = [sample.text for sample in adapter.get_samples("test", seed=0)]
        embeddings = _ClusteredEmbeddingAdapter().encode_batch(texts)
        mapper = PyTorchColorMapper(
            input_dim=EMBED_DIM, hidden_dim_1=16, hidden_dim_2=8, dropout_rate=0.0, device="cpu"
        )
        return RateDistortionSweepUseCase(_color_vq_baseline_factory(mapper)), embeddings

    def test_should_produce_a_distortion_frontier_over_document_embeddings(self, tmp_path: Path) -> None:
        _corpus_tree(tmp_path)
        use_case, embeddings = self._sweep(tmp_path)

        frontier = use_case.execute(embeddings, budgets=[2, 4], methods=["color_vq"])

        assert len(frontier.points) == 2

    def test_should_produce_identical_frontiers_on_repeated_runs(self, tmp_path: Path) -> None:
        _corpus_tree(tmp_path)
        use_case, embeddings = self._sweep(tmp_path)

        first = use_case.execute(embeddings, budgets=[2, 4], methods=["color_vq"])
        second = use_case.execute(embeddings, budgets=[2, 4], methods=["color_vq"])

        assert first.points == second.points
