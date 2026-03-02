from typing import List, Tuple

from fastapi import APIRouter, status

from colors_of_meaning.application.use_case.query_by_palette_use_case import (
    QueryByPaletteUseCase,
)
from colors_of_meaning.domain.model.lab_color import LabColor
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.interface.api.data_transfer_object.palette_query_dto import (
    PaletteQueryRequestDTO,
    PaletteQueryResponseDTO,
    PaletteMatchDTO,
)


def create_query_controller(
    query_use_case: QueryByPaletteUseCase,
    corpus_docs: List[ColoredDocument],
) -> APIRouter:
    router = APIRouter(tags=["query"])

    async def query_by_palette(request: PaletteQueryRequestDTO) -> PaletteQueryResponseDTO:
        palette: List[Tuple[LabColor, float]] = [(LabColor(l=c.l, a=c.a, b=c.b), c.weight) for c in request.colors]

        results = query_use_case.execute(
            palette=palette,
            corpus_docs=corpus_docs,
            k=request.k,
        )

        matches = [PaletteMatchDTO(document_id=doc_id, distance=distance) for doc_id, distance in results]

        return PaletteQueryResponseDTO(
            matches=matches,
            query_colors=len(request.colors),
        )

    router.add_api_route(
        "/query/palette",
        query_by_palette,
        methods=["POST"],
        summary="Query documents by color palette",
        description="Find documents matching a specified color distribution",
        status_code=status.HTTP_200_OK,
    )

    return router
