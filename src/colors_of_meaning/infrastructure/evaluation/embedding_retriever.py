from typing import Any, List, Optional, Tuple

import numpy as np

from colors_of_meaning.domain.service.retriever import Retriever
from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample
from colors_of_meaning.infrastructure.embedding.sentence_embedding_adapter import (
    SentenceEmbeddingAdapter,
)


class EmbeddingRetriever(Retriever):
    def __init__(
        self,
        embedding_adapter: SentenceEmbeddingAdapter,
        M: int = 16,  # noqa: N803
        ef_construction: int = 200,
        ef: int = 50,
    ) -> None:
        self.embedding_adapter = embedding_adapter
        self.M = M  # noqa: N803
        self.ef_construction = ef_construction
        self.ef = ef
        self.index: Optional[Any] = None
        self.training_samples: List[EvaluationSample] = []
        self.dimension: Optional[int] = None

    def fit(self, samples: List[EvaluationSample]) -> None:
        import hnswlib  # type: ignore

        self.training_samples = list(samples)
        texts = [sample.text for sample in self.training_samples]

        embeddings = self.embedding_adapter.encode_batch(texts, batch_size=32, show_progress=False)
        embeddings_array = np.array(embeddings, dtype=np.float32)

        self.dimension = embeddings_array.shape[1]
        num_elements = embeddings_array.shape[0]

        self.index = hnswlib.Index(space="l2", dim=self.dimension)
        self.index.set_num_threads(1)
        self.index.init_index(
            max_elements=num_elements,
            ef_construction=self.ef_construction,
            M=self.M,
            random_seed=100,
        )
        self.index.add_items(embeddings_array, np.arange(num_elements))
        self.index.set_ef(max(self.ef, num_elements))

    def search(self, query: EvaluationSample, k: int) -> List[Tuple[EvaluationSample, float]]:
        if self.index is None:
            raise RuntimeError("Retriever must be fitted before search")
        if k <= 0:
            return []

        effective_k = min(k, len(self.training_samples))
        embedding = np.array(
            self.embedding_adapter.encode_batch([query.text], batch_size=32, show_progress=False),
            dtype=np.float32,
        )
        self.index.set_ef(max(self.ef, effective_k))
        indices, distances = self.index.knn_query(embedding, k=effective_k)
        return [
            (self.training_samples[int(i)], float(distance))
            for i, distance in zip(indices[0], distances[0], strict=True)
            if i >= 0
        ]
