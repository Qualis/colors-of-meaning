from functools import partial
from pathlib import Path
from unittest.mock import patch

import pytest

from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.lab_color import LabColor
from colors_of_meaning.infrastructure.persistence.file_color_codebook_repository import (
    FileColorCodebookRepository,
)
from colors_of_meaning.interface.cli.codebook_loading import load_codebook

MODULE = "colors_of_meaning.interface.cli.codebook_loading"


def _repository_rooted_at(base_path: Path) -> partial:
    return partial(FileColorCodebookRepository, base_path=str(base_path))


class TestLoadCodebook:
    def test_should_return_the_saved_codebook_when_the_artifact_exists(self, tmp_path: Path) -> None:
        codebook = ColorCodebook(colors=[LabColor(l=50.0, a=1.0, b=2.0)], num_bins=1)
        FileColorCodebookRepository(base_path=str(tmp_path)).save(codebook, "stored_codebook")

        with patch(f"{MODULE}.FileColorCodebookRepository", _repository_rooted_at(tmp_path)):
            loaded = load_codebook("stored_codebook")

        assert loaded.get_color(0) == LabColor(l=50.0, a=1.0, b=2.0)

    def test_should_raise_file_not_found_when_the_artifact_is_missing(self, tmp_path: Path) -> None:
        with patch(f"{MODULE}.FileColorCodebookRepository", _repository_rooted_at(tmp_path)):
            with pytest.raises(FileNotFoundError, match="Codebook not found: missing_codebook"):
                load_codebook("missing_codebook")
