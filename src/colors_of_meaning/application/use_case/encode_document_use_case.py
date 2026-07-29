from typing import List
import numpy.typing as npt

from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.service.color_mapper import QuantizedColorMapper


class EncodeDocumentUseCase:
    def __init__(self, quantized_mapper: QuantizedColorMapper) -> None:
        self.quantized_mapper = quantized_mapper

    def execute(self, sentence_embeddings: npt.NDArray, document_id: str) -> ColoredDocument:
        color_bins = self.quantized_mapper.embed_batch_to_bins(sentence_embeddings)

        return ColoredDocument.from_color_sequence(
            color_sequence=color_bins,
            num_bins=self.quantized_mapper.codebook.num_bins,
            document_id=document_id,
        )

    def execute_batch(
        self, sentence_embeddings_list: List[npt.NDArray], document_ids: List[str]
    ) -> List[ColoredDocument]:
        if len(sentence_embeddings_list) != len(document_ids):
            raise ValueError("Mismatch between embeddings and document IDs")

        return [
            self.execute(embeddings, doc_id)
            for embeddings, doc_id in zip(sentence_embeddings_list, document_ids, strict=True)
        ]
