from typing import List

import pytest
from assertpy import assert_that

from colors_of_meaning.domain.model.authorship_report import AuthorshipScalingPoint
from colors_of_meaning.domain.repository.authorship_scaling_repository import (
    AuthorshipScalingRepository,
)


class InMemoryScalingRepository(AuthorshipScalingRepository):
    def __init__(self) -> None:
        self.points: List[AuthorshipScalingPoint] = []

    def load(self) -> List[AuthorshipScalingPoint]:
        return self.points

    def save(self, points: List[AuthorshipScalingPoint]) -> None:
        self.points = points


class TestAuthorshipScalingRepositoryPort:
    def test_should_not_instantiate_the_abstract_repository(self) -> None:
        with pytest.raises(TypeError):
            AuthorshipScalingRepository()  # type: ignore

    def test_should_round_trip_points_through_a_concrete_repository(self) -> None:
        repository = InMemoryScalingRepository()
        point = AuthorshipScalingPoint(
            paragraphs_per_work=60, train_paragraphs=5340, mean_accuracy=0.09, std_accuracy=0.01
        )

        repository.save([point])

        assert_that(repository.load()).is_equal_to([point])
