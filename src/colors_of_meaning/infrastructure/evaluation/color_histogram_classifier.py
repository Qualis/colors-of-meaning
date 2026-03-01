from typing import List, Optional, Any
from collections import Counter

import numpy as np

from colors_of_meaning.domain.service.classifier import Classifier
from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.service.distance_calculator import DistanceCalculator
from colors_of_meaning.application.use_case.encode_document_use_case import EncodeDocumentUseCase
from colors_of_meaning.infrastructure.embedding.sentence_embedding_adapter import (
    SentenceEmbeddingAdapter,
)


class ColorHistogramClassifier(Classifier):
    def __init__(
        self,
        embedding_adapter: SentenceEmbeddingAdapter,
        encode_use_case: EncodeDocumentUseCase,
        distance_calculator: DistanceCalculator,
        k: int = 5,
        num_candidates: int = 100,
        M: int = 16,  # noqa: N803
        ef_construction: int = 200,
        ef: int = 50,
    ) -> None:
        self.embedding_adapter = embedding_adapter
        self.encode_use_case = encode_use_case
        self.distance_calculator = distance_calculator
        self.k = k
        self.num_candidates = num_candidates
        self.M = M  # noqa: N803
        self.ef_construction = ef_construction
        self.ef = ef
        self.training_docs: List[ColoredDocument] = []
        self.training_labels: List[int] = []
        self.index: Optional[Any] = None
        self.histogram_dim: Optional[int] = None

    def fit(self, samples: List[EvaluationSample]) -> None:
        import hnswlib  # type: ignore

        self.training_docs = []
        self.training_labels = []
        histograms = []

        for idx, sample in enumerate(samples):
            embeddings = self.embedding_adapter.encode_document_sentences(sample.text)
            colored_doc = self.encode_use_case.execute(embeddings, document_id=f"train_{idx}")
            self.training_docs.append(colored_doc)
            histograms.append(colored_doc.histogram.astype(np.float32))
            self.training_labels.append(sample.label)

        histogram_matrix = np.array(histograms, dtype=np.float32)
        self.histogram_dim = histogram_matrix.shape[1]
        num_elements = histogram_matrix.shape[0]

        self.index = hnswlib.Index(space="cosine", dim=self.histogram_dim)
        self.index.init_index(
            max_elements=num_elements,
            ef_construction=self.ef_construction,
            M=self.M,
            random_seed=100,
        )
        self.index.add_items(histogram_matrix, np.arange(num_elements))
        self.index.set_ef(self.ef)

    def predict(self, samples: List[EvaluationSample]) -> List[int]:
        if self.index is None:
            raise RuntimeError("Classifier must be fitted before prediction")

        predictions = []

        for idx, sample in enumerate(samples):
            embeddings = self.embedding_adapter.encode_document_sentences(sample.text)
            query_doc = self.encode_use_case.execute(embeddings, document_id=f"test_{idx}")
            query_histogram = query_doc.histogram.astype(np.float32).reshape(1, -1)
            neighbor_labels = self._find_k_nearest_labels(query_doc, query_histogram)
            predicted_label = self._majority_vote(neighbor_labels)
            predictions.append(predicted_label)

        return predictions

    def _find_k_nearest_labels(self, query_doc: ColoredDocument, query_histogram: np.ndarray) -> List[int]:
        if self.k <= 0:
            return []

        candidate_indices = self._retrieve_candidate_indices(query_histogram)
        if not candidate_indices:
            return []

        return self._rerank_by_distance(query_doc, candidate_indices)

    def _retrieve_candidate_indices(self, query_histogram: np.ndarray) -> List[int]:
        effective_candidates = min(self.num_candidates, len(self.training_labels))
        if effective_candidates <= 0:
            return []

        indices, _ = self.index.knn_query(query_histogram, k=effective_candidates)  # type: ignore[union-attr]
        return [int(i) for i in indices[0] if i >= 0]

    def _rerank_by_distance(self, query_doc: ColoredDocument, candidate_indices: List[int]) -> List[int]:
        distances = [
            (self.distance_calculator.compute_distance(query_doc, self.training_docs[idx]), self.training_labels[idx])
            for idx in candidate_indices
        ]
        distances.sort(key=lambda x: x[0])
        effective_k = min(self.k, len(distances))
        return [label for _, label in distances[:effective_k]]

    def _majority_vote(self, labels: List[int]) -> int:
        if not labels:
            return 0
        counter = Counter(labels)
        return counter.most_common(1)[0][0]
