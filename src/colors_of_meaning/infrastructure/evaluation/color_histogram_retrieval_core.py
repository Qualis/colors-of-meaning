from typing import Any, List, Optional, Tuple

import numpy as np

from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample
from colors_of_meaning.domain.service.distance_calculator import DistanceCalculator
from colors_of_meaning.application.use_case.encode_document_use_case import EncodeDocumentUseCase
from colors_of_meaning.infrastructure.embedding.sentence_embedding_adapter import (
    SentenceEmbeddingAdapter,
)


class ColorHistogramRetrievalCore:
    def __init__(
        self,
        embedding_adapter: SentenceEmbeddingAdapter,
        encode_use_case: EncodeDocumentUseCase,
        distance_calculator: DistanceCalculator,
        num_candidates: int = 100,
        M: int = 16,  # noqa: N803
        ef_construction: int = 200,
        ef: int = 50,
    ) -> None:
        self.embedding_adapter = embedding_adapter
        self.encode_use_case = encode_use_case
        self.distance_calculator = distance_calculator
        self.num_candidates = num_candidates
        self.M = M  # noqa: N803
        self.ef_construction = ef_construction
        self.ef = ef
        self.training_samples: List[EvaluationSample] = []
        self.training_docs: List[ColoredDocument] = []
        self.index: Optional[Any] = None
        self.histogram_dim: Optional[int] = None

    def fit(self, samples: List[EvaluationSample]) -> None:
        import hnswlib  # type: ignore

        self.training_samples = list(samples)
        self.training_docs = []
        histograms = []

        for idx, sample in enumerate(self.training_samples):
            embeddings = self.embedding_adapter.encode_document_sentences(sample.text)
            colored_doc = self.encode_use_case.execute(embeddings, document_id=f"train_{idx}")
            self.training_docs.append(colored_doc)
            histograms.append(colored_doc.histogram.astype(np.float32))

        histogram_matrix = np.array(histograms, dtype=np.float32)
        self.histogram_dim = histogram_matrix.shape[1]
        num_elements = histogram_matrix.shape[0]

        self.index = hnswlib.Index(space="cosine", dim=self.histogram_dim)
        self.index.set_num_threads(1)
        self.index.init_index(
            max_elements=num_elements,
            ef_construction=self.ef_construction,
            M=self.M,
            random_seed=100,
        )
        self.index.add_items(histogram_matrix, np.arange(num_elements))
        self.index.set_ef(max(self.ef, self.num_candidates))

    def rank(self, query: EvaluationSample, k: int, document_id: str) -> List[Tuple[int, float]]:
        if k <= 0:
            return []

        query_doc = self.encode_use_case.execute(
            self.embedding_adapter.encode_document_sentences(query.text), document_id=document_id
        )
        query_histogram = query_doc.histogram.astype(np.float32).reshape(1, -1)
        candidate_indices = self._retrieve_candidate_indices(query_histogram, k)
        if not candidate_indices:
            return []

        return self._rerank_by_distance(query_doc, candidate_indices, k)

    def _retrieve_candidate_indices(self, query_histogram: np.ndarray, k: int) -> List[int]:
        effective_candidates = min(max(self.num_candidates, k), len(self.training_samples))
        if effective_candidates <= 0:
            return []

        indices, _ = self.index.knn_query(query_histogram, k=effective_candidates)  # type: ignore[union-attr]
        return [int(i) for i in indices[0] if i >= 0]

    def _rerank_by_distance(
        self, query_doc: ColoredDocument, candidate_indices: List[int], k: int
    ) -> List[Tuple[int, float]]:
        scored = [
            (self.distance_calculator.compute_distance(query_doc, self.training_docs[idx]), idx)
            for idx in candidate_indices
        ]
        scored.sort(key=lambda pair: pair[0])
        effective_k = min(k, len(scored))
        return [(idx, distance) for distance, idx in scored[:effective_k]]
