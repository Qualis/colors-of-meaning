from functools import cached_property
from typing import List

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix  # type: ignore
from scipy.spatial.distance import jensenshannon  # type: ignore
from scipy.special import xlogy  # type: ignore

from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.service.distance_calculator import DistanceCalculator

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

MISMATCHED_BINS_MESSAGE = "Documents must have the same number of bins"
QUERY_CHUNK_SIZE = 512


def squared_distance_contribution(left: FloatArray, right: FloatArray) -> FloatArray:
    midpoint = (left + right) / 2.0
    contribution = 0.5 * xlogy(left, left) + 0.5 * xlogy(right, right) - xlogy(midpoint, midpoint)
    return np.asarray(contribution, dtype=np.float64)


def _smoothed_totals(histograms: List[FloatArray], num_bins: int, smoothing_epsilon: float) -> FloatArray:
    return np.array([histogram.sum() for histogram in histograms]) + num_bins * smoothing_epsilon


def _support_sizes(supports: List[IntArray]) -> FloatArray:
    return np.array([len(support) for support in supports], dtype=np.float64)


def _smoothed_support_values(
    histograms: List[FloatArray], supports: List[IntArray], totals: FloatArray, smoothing_epsilon: float
) -> FloatArray:
    return np.concatenate(
        [
            (histogram[support] + smoothing_epsilon) / total
            for histogram, support, total in zip(histograms, supports, totals, strict=True)
        ]
    )


class SmoothedHistogramBatch:
    def __init__(self, documents: List[ColoredDocument], smoothing_epsilon: float) -> None:
        self.histograms = [document.histogram for document in documents]
        self.size = len(self.histograms)
        self.num_bins = documents[0].num_bins
        self.totals = _smoothed_totals(self.histograms, self.num_bins, smoothing_epsilon)
        self.background = smoothing_epsilon / self.totals
        self.supports = [np.nonzero(histogram)[0] for histogram in self.histograms]
        self.support_sizes = _support_sizes(self.supports)
        self.support_bins = np.concatenate(self.supports).astype(np.int64)
        self.support_values = _smoothed_support_values(self.histograms, self.supports, self.totals, smoothing_epsilon)
        self.expander = self._build_expander()

    @cached_property
    def membership(self) -> npt.NDArray[np.bool_]:
        membership = np.zeros((self.size, self.num_bins), dtype=bool)
        for row, support in enumerate(self.supports):
            membership[row, support] = True
        return membership

    def gather_smoothed(self, bins: IntArray, smoothing_epsilon: float) -> FloatArray:
        gathered = np.array([histogram[bins] for histogram in self.histograms], dtype=np.float64)
        return (gathered.T + smoothing_epsilon) / self.totals[None, :]

    def _build_expander(self) -> csr_matrix:
        entry_count = len(self.support_bins)
        document_of_entry = np.repeat(np.arange(self.size), [len(support) for support in self.supports])
        return csr_matrix(
            (np.ones(entry_count), (np.arange(entry_count), document_of_entry)),
            shape=(entry_count, self.size),
        )


class JensenShannonDistanceCalculator(DistanceCalculator):
    def __init__(self, smoothing_epsilon: float = 1e-8) -> None:
        self.smoothing_epsilon = smoothing_epsilon

    def compute_distance(self, doc1: ColoredDocument, doc2: ColoredDocument) -> float:
        if doc1.num_bins != doc2.num_bins:
            raise ValueError(MISMATCHED_BINS_MESSAGE)

        hist1_smoothed = doc1.histogram + self.smoothing_epsilon
        hist2_smoothed = doc2.histogram + self.smoothing_epsilon

        hist1_normalized = hist1_smoothed / hist1_smoothed.sum()
        hist2_normalized = hist2_smoothed / hist2_smoothed.sum()

        distance = jensenshannon(hist1_normalized, hist2_normalized)

        return float(distance)

    def compute_distance_matrix(self, queries: List[ColoredDocument], references: List[ColoredDocument]) -> FloatArray:
        if not queries or not references:
            return np.empty((len(queries), len(references)), dtype=np.float64)

        self._reject_mismatched_bins(queries + references)
        reference_batch = SmoothedHistogramBatch(references, self.smoothing_epsilon)
        distances = np.empty((len(queries), len(references)), dtype=np.float64)

        for start in range(0, len(queries), QUERY_CHUNK_SIZE):
            stop = min(start + QUERY_CHUNK_SIZE, len(queries))
            distances[start:stop] = self._distance_block(queries[start:stop], reference_batch)

        return distances

    def metric_name(self) -> str:
        return "jensen_shannon"

    def _distance_block(self, queries: List[ColoredDocument], reference_batch: SmoothedHistogramBatch) -> FloatArray:
        query_batch = SmoothedHistogramBatch(queries, self.smoothing_epsilon)
        squared_distances = self._query_support_contribution(query_batch, reference_batch)
        shared_support_sizes = self._add_reference_only_contribution(query_batch, reference_batch, squared_distances)
        squared_distances += self._background_contribution(query_batch, reference_batch, shared_support_sizes)
        return np.sqrt(np.maximum(squared_distances, 0.0))

    @staticmethod
    def _reject_mismatched_bins(documents: List[ColoredDocument]) -> None:
        if len({document.num_bins for document in documents}) > 1:
            raise ValueError(MISMATCHED_BINS_MESSAGE)

    def _query_support_contribution(
        self, query_batch: SmoothedHistogramBatch, reference_batch: SmoothedHistogramBatch
    ) -> FloatArray:
        reference_values = reference_batch.gather_smoothed(query_batch.support_bins, self.smoothing_epsilon)
        entry_contributions = squared_distance_contribution(query_batch.support_values[:, None], reference_values)
        return np.asarray(query_batch.expander.T @ entry_contributions, dtype=np.float64)

    @staticmethod
    def _add_reference_only_contribution(
        query_batch: SmoothedHistogramBatch,
        reference_batch: SmoothedHistogramBatch,
        squared_distances: FloatArray,
    ) -> FloatArray:
        shared = query_batch.membership[:, reference_batch.support_bins]
        entry_contributions = squared_distance_contribution(
            query_batch.background[:, None], reference_batch.support_values[None, :]
        )
        squared_distances += np.asarray(np.where(shared, 0.0, entry_contributions) @ reference_batch.expander)
        return np.asarray(shared.astype(np.float64) @ reference_batch.expander, dtype=np.float64)

    @staticmethod
    def _background_contribution(
        query_batch: SmoothedHistogramBatch,
        reference_batch: SmoothedHistogramBatch,
        shared_support_sizes: FloatArray,
    ) -> FloatArray:
        background_bins = (
            query_batch.num_bins
            - query_batch.support_sizes[:, None]
            - reference_batch.support_sizes[None, :]
            + shared_support_sizes
        )
        return background_bins * squared_distance_contribution(
            query_batch.background[:, None], reference_batch.background[None, :]
        )
