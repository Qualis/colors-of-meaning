import math

import numpy as np
import numpy.typing as npt
from sklearn.cluster import MiniBatchKMeans  # type: ignore[import-untyped]

from colors_of_meaning.domain.service.compression_baseline import (
    CompressionBaseline,
    CompressedResult,
)


class PQCompressionBaseline(CompressionBaseline):
    def __init__(
        self,
        num_subspaces: int = 48,
        num_centroids: int = 256,
    ) -> None:
        self.num_subspaces = num_subspaces
        self.num_centroids = num_centroids

    def compress(self, embeddings: npt.NDArray) -> CompressedResult:
        embeddings = embeddings.astype(np.float32)
        num_samples, embedding_dim = embeddings.shape

        num_subspaces = min(self.num_subspaces, embedding_dim)
        subspace_dim = embedding_dim // num_subspaces
        remainder = embedding_dim % num_subspaces

        raw_bytes = embeddings.tobytes()
        original_size_bits = len(raw_bytes) * 8

        total_reconstruction_error = 0.0
        bits_per_code = int(math.ceil(math.log2(max(self.num_centroids, 2))))
        compressed_size_bits = num_samples * num_subspaces * bits_per_code

        offset = 0
        for s in range(num_subspaces):
            current_dim = subspace_dim + (1 if s < remainder else 0)
            subspace_data = embeddings[:, offset : offset + current_dim]
            offset += current_dim

            num_centroids = min(self.num_centroids, num_samples)
            kmeans = MiniBatchKMeans(
                n_clusters=num_centroids, random_state=42, n_init=1, batch_size=min(256, num_samples)
            )
            kmeans.fit(subspace_data)

            codes = kmeans.predict(subspace_data)
            reconstructed = kmeans.cluster_centers_[codes]
            total_reconstruction_error += float(np.sum((subspace_data - reconstructed) ** 2))

        reconstruction_error = total_reconstruction_error / (num_samples * embedding_dim)

        return CompressedResult(
            compressed_size_bits=compressed_size_bits,
            original_size_bits=original_size_bits,
            reconstruction_error=reconstruction_error,
        )

    def name(self) -> str:
        return f"pq_m{self.num_subspaces}_k{self.num_centroids}"
