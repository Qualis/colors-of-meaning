import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample
from colors_of_meaning.domain.repository.dataset_repository import DatasetRepository
from colors_of_meaning.infrastructure.dataset.seeded_sampler import seeded_subsample
from colors_of_meaning.shared.document_corpus import (
    discover_author_works,
    extract_paragraphs,
    parse_author_work,
    strip_gutenberg_boilerplate,
)

logger = logging.getLogger(__name__)

DEFAULT_MIN_PARAGRAPH_CHARS = 200
DEFAULT_PARAGRAPHS_PER_WORK = 60
DEFAULT_VALIDATION_FRACTION = 0.2
DEFAULT_TEST_FRACTION = 0.2
DEFAULT_SAMPLE_SEED = 42
WORK_SPLIT_STRATEGY = "work"
MINIMUM_WORKS_FOR_WORK_SPLIT = 3
MINIMUM_PARAGRAPHS_FOR_SPLIT = 3

ThreeWaySplit = Tuple[List[str], List[str], List[str]]
AuthorSplit = Tuple[str, int, List[str], List[str], List[str]]


class DocumentCorpusDatasetAdapter(DatasetRepository):
    def __init__(
        self,
        documents_dir: str,
        min_paragraph_chars: int = DEFAULT_MIN_PARAGRAPH_CHARS,
        paragraphs_per_work: int = DEFAULT_PARAGRAPHS_PER_WORK,
        split_strategy: str = WORK_SPLIT_STRATEGY,
        validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
        test_fraction: float = DEFAULT_TEST_FRACTION,
    ) -> None:
        qualifying = _scan_qualifying_authors(
            Path(documents_dir),
            min_paragraph_chars,
            paragraphs_per_work,
            split_strategy,
            validation_fraction,
            test_fraction,
        )
        self._label_names = [author for author, _works, _train, _val, _test in qualifying]
        self._authors_works = [(author, works) for author, works, _train, _val, _test in qualifying]
        self._samples_by_split = _assemble_samples(qualifying)
        _log_discovery(qualifying, split_strategy)

    def get_samples(
        self,
        split: str,
        max_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[EvaluationSample]:
        resolved_seed = seed if seed is not None else DEFAULT_SAMPLE_SEED
        return seeded_subsample(self._samples_by_split.get(split, []), max_samples, resolved_seed)

    def get_label_names(self) -> List[str]:
        return list(self._label_names)

    def get_num_classes(self) -> int:
        return len(self._label_names)

    def works_per_author(self) -> List[Tuple[str, int]]:
        return list(self._authors_works)


def _scan_qualifying_authors(
    root: Path,
    min_chars: int,
    cap: int,
    strategy: str,
    val_fraction: float,
    test_fraction: float,
) -> List[AuthorSplit]:
    grouped = _group_works_by_author(root, min_chars, cap)
    candidates = [
        (author, len(works), _split_author(works, strategy, val_fraction, test_fraction))
        for author, works in sorted(grouped.items())
    ]
    return [
        (author, works, train, val, test) for author, works, (train, val, test) in candidates if all((train, val, test))
    ]


def _group_works_by_author(root: Path, min_chars: int, cap: int) -> Dict[str, List[List[str]]]:
    grouped: Dict[str, List[List[str]]] = {}
    seen: Set[str] = set()
    for work_path in _discover_works(root):
        author, _work = parse_author_work(work_path)
        paragraphs = _deduplicate(_read_work_paragraphs(work_path, min_chars, cap), seen)
        if paragraphs:
            grouped.setdefault(author, []).append(paragraphs)
    return grouped


def _deduplicate(paragraphs: List[str], seen: Set[str]) -> List[str]:
    fresh = [paragraph for paragraph in paragraphs if paragraph not in seen]
    seen.update(fresh)
    return fresh


def _discover_works(root: Path) -> List[Path]:
    return discover_author_works(root)


def _read_work_paragraphs(work_path: Path, min_chars: int, cap: int) -> List[str]:
    if not work_path.is_file():
        return []
    text = work_path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        return []
    return extract_paragraphs(strip_gutenberg_boilerplate(text), min_chars)[:cap]


def _split_author(works: List[List[str]], strategy: str, val_fraction: float, test_fraction: float) -> ThreeWaySplit:
    if strategy == WORK_SPLIT_STRATEGY and len(works) >= MINIMUM_WORKS_FOR_WORK_SPLIT:
        return _split_by_work(works, val_fraction, test_fraction)
    return _split_by_paragraph(_flatten(works), val_fraction, test_fraction)


def _split_by_work(works: List[List[str]], val_fraction: float, test_fraction: float) -> ThreeWaySplit:
    train_works, validation_works = _holdout_counts(len(works), val_fraction, test_fraction)
    return (
        _flatten(works[:train_works]),
        _flatten(works[train_works : train_works + validation_works]),
        _flatten(works[train_works + validation_works :]),
    )


def _split_by_paragraph(paragraphs: List[str], val_fraction: float, test_fraction: float) -> ThreeWaySplit:
    if len(paragraphs) < MINIMUM_PARAGRAPHS_FOR_SPLIT:
        return [], [], []
    train_count, validation_count = _holdout_counts(len(paragraphs), val_fraction, test_fraction)
    return (
        paragraphs[:train_count],
        paragraphs[train_count : train_count + validation_count],
        paragraphs[train_count + validation_count :],
    )


def _holdout_counts(total: int, val_fraction: float, test_fraction: float) -> Tuple[int, int]:
    test_count = min(max(1, int(total * test_fraction)), total - 2)
    validation_count = min(max(1, int(total * val_fraction)), total - 1 - test_count)
    return total - validation_count - test_count, validation_count


def _flatten(works: List[List[str]]) -> List[str]:
    return [paragraph for work in works for paragraph in work]


def _assemble_samples(qualifying: List[AuthorSplit]) -> Dict[str, List[EvaluationSample]]:
    samples_by_split: Dict[str, List[EvaluationSample]] = {"train": [], "validation": [], "test": []}
    for label, (_author, _works, train, validation, test) in enumerate(qualifying):
        samples_by_split["train"].extend(_samples(train, label, "train"))
        samples_by_split["validation"].extend(_samples(validation, label, "validation"))
        samples_by_split["test"].extend(_samples(test, label, "test"))
    return samples_by_split


def _samples(paragraphs: List[str], label: int, split: str) -> List[EvaluationSample]:
    return [EvaluationSample(text=paragraph, label=label, split=split) for paragraph in paragraphs]


def _log_discovery(qualifying: List[AuthorSplit], strategy: str) -> None:
    train_size = sum(len(train) for _author, _works, train, _val, _test in qualifying)
    validation_size = sum(len(val) for _author, _works, _train, val, _test in qualifying)
    test_size = sum(len(test) for _author, _works, _train, _val, test in qualifying)
    logger.info(
        "Discovered document corpus",
        extra={
            "correlation_id": str(uuid.uuid4()),
            "authors": len(qualifying),
            "works_per_author": {author: works for author, works, _train, _val, _test in qualifying},
            "total_paragraphs": train_size + validation_size + test_size,
            "train_size": train_size,
            "validation_size": validation_size,
            "test_size": test_size,
            "split_strategy": strategy,
        },
    )
