import json
from pathlib import Path

from colors_of_meaning.domain.model.authorship_report import AuthorshipScalingPoint
from colors_of_meaning.infrastructure.persistence.json_authorship_scaling_repository import (
    JsonAuthorshipScalingRepository,
)


def _point() -> AuthorshipScalingPoint:
    return AuthorshipScalingPoint(
        paragraphs_per_work=60, train_paragraphs=5340, mean_accuracy=0.092, std_accuracy=0.016
    )


class TestJsonAuthorshipScalingRepository:
    def test_should_load_points_from_a_committed_manifest(self, tmp_path: Path) -> None:
        manifest = tmp_path / "scaling.json"
        manifest.write_text(json.dumps({"points": [_to_entry(_point())]}), encoding="utf-8")

        points = JsonAuthorshipScalingRepository(str(manifest)).load()

        assert points == [_point()]

    def test_should_write_points_the_manifest_reader_can_load(self, tmp_path: Path) -> None:
        manifest = tmp_path / "nested" / "scaling.json"
        repository = JsonAuthorshipScalingRepository(str(manifest))

        repository.save([_point()])

        assert repository.load() == [_point()]

    def test_should_create_the_manifest_parent_directory_on_save(self, tmp_path: Path) -> None:
        manifest = tmp_path / "reports" / "data" / "scaling.json"

        JsonAuthorshipScalingRepository(str(manifest)).save([_point()])

        assert manifest.exists()


def _to_entry(point: AuthorshipScalingPoint) -> dict:
    return {
        "paragraphs_per_work": point.paragraphs_per_work,
        "train_paragraphs": point.train_paragraphs,
        "mean_accuracy": point.mean_accuracy,
        "std_accuracy": point.std_accuracy,
    }
