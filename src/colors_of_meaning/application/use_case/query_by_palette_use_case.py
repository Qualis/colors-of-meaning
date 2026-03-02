from typing import List, Tuple

import numpy as np

from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.model.lab_color import LabColor
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.application.use_case.compare_documents_use_case import (
    CompareDocumentsUseCase,
)


class QueryByPaletteUseCase:
    def __init__(
        self,
        compare_use_case: CompareDocumentsUseCase,
        codebook: ColorCodebook,
    ) -> None:
        self.compare_use_case = compare_use_case
        self.codebook = codebook

    def execute(
        self,
        palette: List[Tuple[LabColor, float]],
        corpus_docs: List[ColoredDocument],
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        query_doc = self._palette_to_document(palette)
        return self.compare_use_case.find_nearest_neighbors(
            query_doc=query_doc,
            corpus_docs=corpus_docs,
            k=k,
        )

    def _palette_to_document(self, palette: List[Tuple[LabColor, float]]) -> ColoredDocument:
        histogram = np.zeros(self.codebook.num_bins, dtype=np.float64)

        for color, weight in palette:
            bin_index = self.codebook.quantize(color)
            histogram[bin_index] += weight

        total = histogram.sum()
        if total > 0:
            histogram = histogram / total
        else:
            histogram = np.ones(self.codebook.num_bins, dtype=np.float64) / self.codebook.num_bins

        return ColoredDocument(
            histogram=histogram,
            document_id="palette_query",
        )
