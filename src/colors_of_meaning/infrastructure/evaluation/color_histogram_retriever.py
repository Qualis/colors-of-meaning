from typing import List, Tuple

from colors_of_meaning.domain.service.retriever import Retriever
from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample
from colors_of_meaning.domain.service.distance_calculator import DistanceCalculator
from colors_of_meaning.application.use_case.encode_document_use_case import EncodeDocumentUseCase
from colors_of_meaning.infrastructure.embedding.sentence_embedding_adapter import (
    SentenceEmbeddingAdapter,
)
from colors_of_meaning.infrastructure.evaluation.color_histogram_retrieval_core import (
    ColorHistogramRetrievalCore,
)


class ColorHistogramRetriever(Retriever):
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
        self.core = ColorHistogramRetrievalCore(
            embedding_adapter,
            encode_use_case,
            distance_calculator,
            num_candidates=num_candidates,
            M=M,
            ef_construction=ef_construction,
            ef=ef,
        )

    def fit(self, samples: List[EvaluationSample]) -> None:
        self.core.fit(samples)

    def search(self, query: EvaluationSample, k: int) -> List[Tuple[EvaluationSample, float]]:
        if self.core.index is None:
            raise RuntimeError("Retriever must be fitted before search")

        ranked = self.core.rank(query, k, document_id="query")
        return [(self.core.training_samples[index], distance) for index, distance in ranked]
