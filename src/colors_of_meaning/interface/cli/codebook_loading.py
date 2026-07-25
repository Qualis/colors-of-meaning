from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.infrastructure.persistence.file_color_codebook_repository import (
    FileColorCodebookRepository,
)


def load_codebook(codebook_name: str) -> ColorCodebook:
    codebook = FileColorCodebookRepository().load(codebook_name)
    if codebook is None:
        raise FileNotFoundError(f"Codebook not found: {codebook_name}")
    return codebook
