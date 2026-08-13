from typing import List
from unittest.mock import Mock

import pytest

from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample
from colors_of_meaning.domain.repository.dataset_repository import DatasetRepository


@pytest.fixture
def mock_dataset_repository() -> Mock:
    mock_repository = Mock(spec=DatasetRepository)
    return mock_repository


@pytest.fixture
def sample_evaluation_samples() -> List[EvaluationSample]:
    return [
        EvaluationSample(text="This is a train sample", label=0, split="train"),
        EvaluationSample(text="Another train sample", label=1, split="train"),
        EvaluationSample(text="This is a test sample", label=0, split="test"),
        EvaluationSample(text="Another test sample", label=1, split="test"),
    ]


@pytest.fixture
def sample_label_names() -> List[str]:
    return ["class_0", "class_1"]
