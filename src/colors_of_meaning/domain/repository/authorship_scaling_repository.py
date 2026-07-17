from abc import ABC, abstractmethod
from typing import List

from colors_of_meaning.domain.model.authorship_report import AuthorshipScalingPoint


class AuthorshipScalingRepository(ABC):
    @abstractmethod
    def load(self) -> List[AuthorshipScalingPoint]:
        raise NotImplementedError

    @abstractmethod
    def save(self, points: List[AuthorshipScalingPoint]) -> None:
        raise NotImplementedError
