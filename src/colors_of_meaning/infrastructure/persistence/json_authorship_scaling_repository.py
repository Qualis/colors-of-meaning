import json
from pathlib import Path
from typing import Any, Dict, List

from colors_of_meaning.domain.model.authorship_report import AuthorshipScalingPoint
from colors_of_meaning.domain.repository.authorship_scaling_repository import (
    AuthorshipScalingRepository,
)


class JsonAuthorshipScalingRepository(AuthorshipScalingRepository):
    def __init__(self, manifest_path: str) -> None:
        self.manifest_path = Path(manifest_path)

    def load(self) -> List[AuthorshipScalingPoint]:
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return [self._to_point(entry) for entry in payload["points"]]

    def save(self, points: List[AuthorshipScalingPoint]) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"points": [self._to_entry(point) for point in points]}
        self.manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _to_point(entry: Dict[str, Any]) -> AuthorshipScalingPoint:
        return AuthorshipScalingPoint(
            paragraphs_per_work=int(entry["paragraphs_per_work"]),
            train_paragraphs=int(entry["train_paragraphs"]),
            mean_accuracy=float(entry["mean_accuracy"]),
            std_accuracy=float(entry["std_accuracy"]),
        )

    @staticmethod
    def _to_entry(point: AuthorshipScalingPoint) -> Dict[str, Any]:
        return {
            "paragraphs_per_work": point.paragraphs_per_work,
            "train_paragraphs": point.train_paragraphs,
            "mean_accuracy": point.mean_accuracy,
            "std_accuracy": point.std_accuracy,
        }
