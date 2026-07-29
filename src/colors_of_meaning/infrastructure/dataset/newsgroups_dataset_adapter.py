from typing import List, Optional

from sklearn.datasets import fetch_20newsgroups  # type: ignore

from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample
from colors_of_meaning.domain.repository.dataset_repository import DatasetRepository
from colors_of_meaning.infrastructure.dataset.seeded_sampler import seeded_subsample


class NewsgroupsDatasetAdapter(DatasetRepository):
    def __init__(self) -> None:
        self._label_names = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes")).target_names

    def get_samples(
        self,
        split: str,
        max_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[EvaluationSample]:
        sklearn_split = "train" if split == "train" else "test"

        newsgroups = fetch_20newsgroups(subset=sklearn_split, remove=("headers", "footers", "quotes"))

        samples = [
            EvaluationSample(text=text, label=int(label), split=split)
            for text, label in zip(newsgroups.data, newsgroups.target, strict=True)
            if text.strip()
        ]

        return seeded_subsample(samples, max_samples, seed)

    def get_label_names(self) -> List[str]:
        return list(self._label_names)

    def get_num_classes(self) -> int:
        return len(self._label_names)
