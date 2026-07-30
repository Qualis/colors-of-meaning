from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.model.narrative_arc import NarrativeArc
from colors_of_meaning.domain.model.rate_distortion_point import RateDistortionFrontier


class FigureRenderer(ABC):
    @abstractmethod
    def render_codebook_palette(self, codebook: ColorCodebook, output_path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def render_document_histograms(
        self,
        documents: List[ColoredDocument],
        labels: List[int],
        label_names: List[str],
        output_path: str,
        samples_per_class: int = 2,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def render_tsne_projection(
        self,
        documents: List[ColoredDocument],
        labels: List[int],
        label_names: List[str],
        output_path: str,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def render_confusion_matrix(
        self,
        y_true: List[int],
        y_pred: List[int],
        label_names: List[str],
        output_path: str,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def render_corpus_signatures(
        self,
        documents: List[ColoredDocument],
        labels: List[int],
        label_names: List[str],
        codebook: ColorCodebook,
        output_path: str,
        top_colors: int = 24,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def render_rate_distortion(self, frontier: RateDistortionFrontier, output_path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def render_narrative_arc(self, arc: NarrativeArc, output_path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def render_a4_gallery(
        self,
        sheet_paths: List[str],
        output_path: str,
        columns: int = 12,
        captions: Optional[List[str]] = None,
        title: str = "Per-book A4 colour signatures",
        max_tile_pixels: Optional[int] = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def render_image_comparison(self, panels: List[Tuple[str, str]], title: str, output_path: str) -> None:
        raise NotImplementedError
