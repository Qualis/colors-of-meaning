import json
import tyro
import pickle  # nosec B403
from dataclasses import dataclass
from typing import List

from colors_of_meaning.domain.model.lab_color import LabColor
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.infrastructure.ml.wasserstein_distance_calculator import (
    WassersteinDistanceCalculator,
)
from colors_of_meaning.infrastructure.persistence.file_color_codebook_repository import (
    FileColorCodebookRepository,
)
from colors_of_meaning.application.use_case.compare_documents_use_case import (
    CompareDocumentsUseCase,
)
from colors_of_meaning.application.use_case.query_by_palette_use_case import (
    QueryByPaletteUseCase,
)


@dataclass
class QueryArgs:
    palette_json: str = '[{"l": 50, "a": 0, "b": 0, "weight": 1.0}]'
    encoded_documents: str = "artifacts/encoded/test_documents.pkl"
    codebook_name: str = "codebook_4096"
    k: int = 5


def _parse_palette(palette_json: str) -> list:
    raw = json.loads(palette_json)
    palette = []
    for entry in raw:
        color = LabColor(
            l=float(entry["l"]),
            a=float(entry["a"]),
            b=float(entry["b"]),
        )
        weight = float(entry.get("weight", 1.0))
        palette.append((color, weight))
    return palette


def main(args: QueryArgs) -> None:
    print(f"Loading encoded documents from {args.encoded_documents}...")
    with open(args.encoded_documents, "rb") as f:
        documents: List[ColoredDocument] = pickle.load(f)  # nosec B301 nosemgrep

    print(f"Loading codebook {args.codebook_name}...")
    codebook_repo = FileColorCodebookRepository()
    codebook_result = codebook_repo.load(args.codebook_name)
    if codebook_result is None:
        raise FileNotFoundError(f"Codebook {args.codebook_name} not found")
    codebook: ColorCodebook = codebook_result

    palette = _parse_palette(args.palette_json)
    print(f"Querying with {len(palette)} colors, k={args.k}...")

    distance_calculator = WassersteinDistanceCalculator()
    compare_use_case = CompareDocumentsUseCase(distance_calculator=distance_calculator)
    query_use_case = QueryByPaletteUseCase(
        compare_use_case=compare_use_case,
        codebook=codebook,
    )

    results = query_use_case.execute(
        palette=palette,
        corpus_docs=documents,
        k=args.k,
    )

    print(f"\nTop {args.k} matches:")
    print("-" * 40)
    for rank, (doc_id, distance) in enumerate(results, 1):
        print(f"  {rank}. {doc_id} (distance: {distance:.4f})")


if __name__ == "__main__":
    main(tyro.cli(QueryArgs))
