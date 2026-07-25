import tyro
import pickle  # nosec B403
from dataclasses import dataclass
from typing import List

from colors_of_meaning.shared.synesthetic_config import SynestheticConfig
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.service.distance_calculator import DistanceCalculator
from colors_of_meaning.infrastructure.ml.wasserstein_distance_calculator import (
    WassersteinDistanceCalculator,
)
from colors_of_meaning.infrastructure.ml.jensen_shannon_distance_calculator import (
    JensenShannonDistanceCalculator,
)
from colors_of_meaning.application.use_case.compare_documents_use_case import (
    CompareDocumentsUseCase,
)
from colors_of_meaning.interface.cli.codebook_loading import load_codebook

DEFAULT_CODEBOOK_NAME = "codebook_4096"


@dataclass
class CompareArgs:
    config: str = "configs/base.yaml"
    encoded_documents: str = "artifacts/encoded/test_documents.pkl"
    k: int = 5
    query_index: int = 0


def _create_distance_calculator(config: SynestheticConfig) -> DistanceCalculator:
    if config.distance.metric != "wasserstein":
        return JensenShannonDistanceCalculator(smoothing_epsilon=config.distance.smoothing_epsilon)
    codebook = load_codebook(DEFAULT_CODEBOOK_NAME)
    return WassersteinDistanceCalculator(codebook=codebook, sinkhorn_reg=config.distance.sinkhorn_reg)


def main(args: CompareArgs) -> None:
    config = SynestheticConfig.from_yaml(args.config)

    print(f"Loading encoded documents from {args.encoded_documents}...")
    with open(args.encoded_documents, "rb") as f:
        documents: List[ColoredDocument] = pickle.load(f)  # nosec B301 nosemgrep

    if args.query_index >= len(documents):
        raise ValueError(f"Query index {args.query_index} out of range")

    distance_calculator = _create_distance_calculator(config)
    use_case = CompareDocumentsUseCase(distance_calculator)

    query_doc = documents[args.query_index]
    print(f"\nQuery document: {query_doc.document_id}")

    nearest_neighbors = use_case.find_nearest_neighbors(query_doc, documents, k=args.k)

    print(f"\nTop {args.k} nearest neighbors:")
    for rank, (doc_id, distance) in enumerate(nearest_neighbors, 1):
        print(f"{rank}. {doc_id}: distance = {distance:.4f}")


if __name__ == "__main__":
    main(tyro.cli(CompareArgs))
