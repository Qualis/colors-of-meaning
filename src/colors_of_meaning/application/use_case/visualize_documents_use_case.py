from typing import List

from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.service.figure_renderer import FigureRenderer


class VisualizeDocumentsUseCase:
    def __init__(self, figure_renderer: FigureRenderer) -> None:
        self.figure_renderer = figure_renderer

    def execute_histograms(
        self,
        documents: List[ColoredDocument],
        labels: List[int],
        label_names: List[str],
        output_path: str,
        samples_per_class: int = 2,
    ) -> None:
        self.figure_renderer.render_document_histograms(documents, labels, label_names, output_path, samples_per_class)

    def execute_projection(
        self,
        documents: List[ColoredDocument],
        labels: List[int],
        label_names: List[str],
        output_path: str,
    ) -> None:
        self.figure_renderer.render_tsne_projection(documents, labels, label_names, output_path)

    def execute_confusion_matrix(
        self,
        y_true: List[int],
        y_pred: List[int],
        label_names: List[str],
        output_path: str,
    ) -> None:
        self.figure_renderer.render_confusion_matrix(y_true, y_pred, label_names, output_path)

    def execute_corpus_signatures(
        self,
        documents: List[ColoredDocument],
        labels: List[int],
        label_names: List[str],
        codebook: ColorCodebook,
        output_path: str,
        top_colors: int = 24,
    ) -> None:
        self.figure_renderer.render_corpus_signatures(documents, labels, label_names, codebook, output_path, top_colors)

    def execute_a4_gallery(self, sheet_paths: List[str], output_path: str, columns: int = 12) -> None:
        self.figure_renderer.render_a4_gallery(sheet_paths, output_path, columns)
