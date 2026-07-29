import pickle  # nosec B403
from pathlib import Path
from typing import Optional, cast

from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.repository.color_codebook_repository import (
    ColorCodebookRepository,
)


class FileColorCodebookRepository(ColorCodebookRepository):
    def __init__(self, base_path: str = "artifacts/codebooks") -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(self, codebook: ColorCodebook, name: str) -> None:
        file_path = self._get_file_path(name)
        with open(file_path, "wb") as f:
            pickle.dump(codebook, f)  # nosemgrep: python.lang.security.deserialization.pickle.avoid-pickle

    def load(self, name: str) -> Optional[ColorCodebook]:
        file_path = self._get_file_path(name)
        if not file_path.exists():
            return None

        with open(file_path, "rb") as f:
            return cast(ColorCodebook, pickle.load(f))  # nosec B301  # nosemgrep

    def exists(self, name: str) -> bool:
        return self._get_file_path(name).exists()

    def delete(self, name: str) -> None:
        file_path = self._get_file_path(name)
        if file_path.exists():
            file_path.unlink()

    def _get_file_path(self, name: str) -> Path:
        if not name.endswith(".pkl"):
            name = f"{name}.pkl"
        return self.base_path / name
