import pytest
from pydantic import ValidationError

from colors_of_meaning.interface.api.data_transfer_object.palette_query_dto import (
    PaletteColorDTO,
    PaletteQueryRequestDTO,
    PaletteQueryResponseDTO,
    PaletteMatchDTO,
)


class TestPaletteColorDTO:
    def test_should_create_valid_color(self) -> None:
        color = PaletteColorDTO(l=50.0, a=10.0, b=-20.0, weight=1.0)

        assert color.l == 50.0

    def test_should_default_weight_to_one(self) -> None:
        color = PaletteColorDTO(l=50.0, a=0.0, b=0.0)

        assert color.weight == 1.0

    def test_should_reject_invalid_lightness(self) -> None:
        with pytest.raises(ValidationError):
            PaletteColorDTO(l=101.0, a=0.0, b=0.0)

    def test_should_reject_negative_weight(self) -> None:
        with pytest.raises(ValidationError):
            PaletteColorDTO(l=50.0, a=0.0, b=0.0, weight=-1.0)


class TestPaletteQueryRequestDTO:
    def test_should_create_valid_request(self) -> None:
        request = PaletteQueryRequestDTO(
            colors=[PaletteColorDTO(l=50.0, a=0.0, b=0.0)],
            k=5,
        )

        assert len(request.colors) == 1

    def test_should_default_k_to_five(self) -> None:
        request = PaletteQueryRequestDTO(
            colors=[PaletteColorDTO(l=50.0, a=0.0, b=0.0)],
        )

        assert request.k == 5

    def test_should_reject_empty_colors(self) -> None:
        with pytest.raises(ValidationError):
            PaletteQueryRequestDTO(colors=[], k=5)


class TestPaletteMatchDTO:
    def test_should_create_match(self) -> None:
        match = PaletteMatchDTO(document_id="doc_1", distance=0.5)

        assert match.document_id == "doc_1"
        assert match.distance == 0.5


class TestPaletteQueryResponseDTO:
    def test_should_create_response(self) -> None:
        response = PaletteQueryResponseDTO(
            matches=[PaletteMatchDTO(document_id="doc_1", distance=0.5)],
            query_colors=1,
        )

        assert len(response.matches) == 1
        assert response.query_colors == 1
