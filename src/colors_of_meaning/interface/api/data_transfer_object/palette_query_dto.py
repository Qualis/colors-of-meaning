from pydantic import BaseModel, Field
from typing import List


class PaletteColorDTO(BaseModel):
    l: float = Field(..., ge=0.0, le=100.0)
    a: float = Field(..., ge=-128.0, le=127.0)
    b: float = Field(..., ge=-128.0, le=127.0)
    weight: float = Field(default=1.0, gt=0.0)


class PaletteQueryRequestDTO(BaseModel):
    colors: List[PaletteColorDTO] = Field(..., min_length=1)
    k: int = Field(default=5, ge=1, le=100)


class PaletteMatchDTO(BaseModel):
    document_id: str
    distance: float


class PaletteQueryResponseDTO(BaseModel):
    matches: List[PaletteMatchDTO]
    query_colors: int
