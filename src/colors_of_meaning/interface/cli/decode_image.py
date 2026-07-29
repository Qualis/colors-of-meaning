import tyro
import pickle  # nosec B403
from dataclasses import dataclass
from typing import List

import numpy as np

from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.infrastructure.ml.wasserstein_distance_calculator import (
    WassersteinDistanceCalculator,
)
from colors_of_meaning.infrastructure.visualization.pillow_document_image_renderer import (
    PillowDocumentImageRenderer,
)
from colors_of_meaning.application.use_case.compare_documents_use_case import (
    CompareDocumentsUseCase,
)
from colors_of_meaning.application.use_case.decode_image_to_document_use_case import (
    DecodeImageToDocumentUseCase,
    DocumentMatch,
)
from colors_of_meaning.interface.cli.codebook_loading import load_codebook


@dataclass
class DecodeImageArgs:
    image_path: str = "reports/figures/document_a4.png"
    codebook_name: str = "codebook_4096"
    encoded_documents: str = ""
    k: int = 5


def _load_corpus(encoded_documents: str) -> List[ColoredDocument]:
    if not encoded_documents:
        return []
    with open(encoded_documents, "rb") as f:
        corpus: List[ColoredDocument] = pickle.load(f)  # nosec B301  # nosemgrep
    return corpus


def _print_summary(document: ColoredDocument, neighbors: List[DocumentMatch]) -> None:
    recovered_cells = len(document.color_sequence or [])
    distinct_colors = int(np.count_nonzero(document.histogram))
    print(f"Recovered {recovered_cells} cells across {distinct_colors} distinct colors")
    for rank, (doc_id, distance) in enumerate(neighbors, 1):
        print(f"  {rank}. {doc_id} (distance: {distance:.4f})")


def main(args: DecodeImageArgs) -> None:
    codebook = load_codebook(args.codebook_name)
    corpus = _load_corpus(args.encoded_documents)

    distance_calculator = WassersteinDistanceCalculator(codebook=codebook)
    compare_use_case = CompareDocumentsUseCase(distance_calculator)
    use_case = DecodeImageToDocumentUseCase(PillowDocumentImageRenderer(), compare_use_case)

    document, neighbors = use_case.execute(args.image_path, codebook, corpus, k=args.k)
    _print_summary(document, neighbors)


if __name__ == "__main__":
    main(tyro.cli(DecodeImageArgs))
