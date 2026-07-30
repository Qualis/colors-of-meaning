import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest

from PIL import Image

from colors_of_meaning.domain.model.lab_color import LabColor
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.model.narrative_arc import (
    ArcPoint,
    NarrativeArc,
    NarrativeBeat,
)
from colors_of_meaning.domain.model.rate_distortion_point import (
    RateDistortionFrontier,
    RateDistortionPoint,
)
from colors_of_meaning.domain.service.figure_renderer import FigureRenderer
from colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer import (
    FIGURE_DPI,
    _load_tile_image,
    NARRATIVE_ARC_FIGURE_SIZE,
    RATE_DISTORTION_FIGURE_SIZE,
    MatplotlibFigureRenderer,
)

MODULE = "colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer"


class TestMatplotlibFigureRendererInterface:
    def test_should_implement_figure_renderer_interface(self) -> None:
        renderer = MatplotlibFigureRenderer()

        assert isinstance(renderer, FigureRenderer)


class TestRenderCodebookPalette:
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.lab_to_rgb")
    def test_should_create_figure_for_codebook(
        self,
        mock_lab_to_rgb: Mock,
        mock_plt: Mock,
        tmp_path: Path,
    ) -> None:
        mock_lab_to_rgb.return_value = (128, 128, 128)
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        colors = [LabColor(l=50.0, a=0.0, b=0.0) for _ in range(4)]
        codebook = ColorCodebook(colors=colors, num_bins=4)

        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "palette.png")
        renderer.render_codebook_palette(codebook, output_path)

        mock_ax.imshow.assert_called_once()

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.lab_to_rgb")
    def test_should_set_title_with_codebook_size(
        self,
        mock_lab_to_rgb: Mock,
        mock_plt: Mock,
        tmp_path: Path,
    ) -> None:
        mock_lab_to_rgb.return_value = (128, 128, 128)
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        colors = [LabColor(l=50.0, a=0.0, b=0.0) for _ in range(4)]
        codebook = ColorCodebook(colors=colors, num_bins=4)

        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "palette.png")
        renderer.render_codebook_palette(codebook, output_path)

        mock_ax.set_title.assert_called_once_with("Color Codebook (4 colors)")

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.lab_to_rgb")
    def test_should_hide_axes_for_codebook_palette(
        self,
        mock_lab_to_rgb: Mock,
        mock_plt: Mock,
        tmp_path: Path,
    ) -> None:
        mock_lab_to_rgb.return_value = (128, 128, 128)
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        colors = [LabColor(l=50.0, a=0.0, b=0.0) for _ in range(4)]
        codebook = ColorCodebook(colors=colors, num_bins=4)

        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "palette.png")
        renderer.render_codebook_palette(codebook, output_path)

        mock_ax.axis.assert_called_once_with("off")

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.lab_to_rgb")
    def test_should_save_codebook_palette_to_file(
        self,
        mock_lab_to_rgb: Mock,
        mock_plt: Mock,
        tmp_path: Path,
    ) -> None:
        mock_lab_to_rgb.return_value = (128, 128, 128)
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        colors = [LabColor(l=50.0, a=0.0, b=0.0) for _ in range(4)]
        codebook = ColorCodebook(colors=colors, num_bins=4)

        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "palette.png")
        renderer.render_codebook_palette(codebook, output_path)

        mock_fig.savefig.assert_called_once_with(output_path, dpi=150, bbox_inches="tight")

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.lab_to_rgb")
    def test_should_close_figure_after_saving(
        self,
        mock_lab_to_rgb: Mock,
        mock_plt: Mock,
        tmp_path: Path,
    ) -> None:
        mock_lab_to_rgb.return_value = (128, 128, 128)
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        colors = [LabColor(l=50.0, a=0.0, b=0.0) for _ in range(4)]
        codebook = ColorCodebook(colors=colors, num_bins=4)

        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "palette.png")
        renderer.render_codebook_palette(codebook, output_path)

        mock_plt.close.assert_called_once_with(mock_fig)


class TestRenderDocumentHistograms:
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    def test_should_create_subplots_for_document_histograms(
        self,
        mock_plt: Mock,
        tmp_path: Path,
    ) -> None:
        mock_fig = Mock()
        mock_axes = np.array([[Mock(), Mock()]])
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        histogram = np.array([0.5, 0.3, 0.2])
        documents = [
            ColoredDocument(histogram=histogram, document_id="doc_0"),
            ColoredDocument(histogram=histogram, document_id="doc_1"),
        ]
        labels = [0, 1]
        label_names = ["Class A", "Class B"]

        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "histograms.png")
        renderer.render_document_histograms(documents, labels, label_names, output_path, samples_per_class=1)

        mock_plt.subplots.assert_called_once()

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    def test_should_handle_single_document_histogram(
        self,
        mock_plt: Mock,
        tmp_path: Path,
    ) -> None:
        mock_fig = Mock()
        mock_single_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_single_ax)

        histogram = np.array([0.5, 0.3, 0.2])
        documents = [ColoredDocument(histogram=histogram, document_id="doc_0")]
        labels = [0]
        label_names = ["Class A"]

        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "histograms.png")
        renderer.render_document_histograms(documents, labels, label_names, output_path, samples_per_class=1)

        mock_fig.suptitle.assert_called_once()

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    def test_should_hide_unused_axes_in_grid(
        self,
        mock_plt: Mock,
        tmp_path: Path,
    ) -> None:
        mock_fig = Mock()
        mock_unused_ax = Mock()
        mock_axes = np.array([[Mock(), Mock()], [Mock(), mock_unused_ax]])
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        histogram = np.array([0.5, 0.3, 0.2])
        documents = [
            ColoredDocument(histogram=histogram, document_id="doc_0"),
            ColoredDocument(histogram=histogram, document_id="doc_1"),
            ColoredDocument(histogram=histogram, document_id="doc_2"),
        ]
        labels = [0, 1, 2]
        label_names = ["Class A", "Class B", "Class C"]

        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "histograms.png")
        renderer.render_document_histograms(documents, labels, label_names, output_path, samples_per_class=1)

        mock_unused_ax.axis.assert_called_once_with("off")

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    def test_should_set_suptitle_for_histograms(
        self,
        mock_plt: Mock,
        tmp_path: Path,
    ) -> None:
        mock_fig = Mock()
        mock_axes = np.array([[Mock(), Mock()]])
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        histogram = np.array([0.5, 0.3, 0.2])
        documents = [
            ColoredDocument(histogram=histogram, document_id="doc_0"),
            ColoredDocument(histogram=histogram, document_id="doc_1"),
        ]
        labels = [0, 1]
        label_names = ["Class A", "Class B"]

        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "histograms.png")
        renderer.render_document_histograms(documents, labels, label_names, output_path, samples_per_class=1)

        mock_fig.suptitle.assert_called_once_with("Document Color Histograms by Class")

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    def test_should_save_histogram_figure_to_file(
        self,
        mock_plt: Mock,
        tmp_path: Path,
    ) -> None:
        mock_fig = Mock()
        mock_axes = np.array([[Mock(), Mock()]])
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        histogram = np.array([0.5, 0.3, 0.2])
        documents = [
            ColoredDocument(histogram=histogram, document_id="doc_0"),
            ColoredDocument(histogram=histogram, document_id="doc_1"),
        ]
        labels = [0, 1]
        label_names = ["Class A", "Class B"]

        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "histograms.png")
        renderer.render_document_histograms(documents, labels, label_names, output_path, samples_per_class=1)

        mock_fig.savefig.assert_called_once_with(output_path, dpi=150, bbox_inches="tight")


class TestRenderTsneProjection:
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.matplotlib.colormaps")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.TSNE")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    def test_should_clamp_tsne_perplexity_below_sample_count(
        self,
        mock_plt: Mock,
        mock_tsne_class: Mock,
        mock_colormaps: Mock,
        tmp_path: Path,
    ) -> None:
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_cmap = MagicMock()
        mock_cmap.return_value = np.array([[1, 0, 0, 1], [0, 1, 0, 1]])
        mock_colormaps.__getitem__.return_value = mock_cmap

        mock_tsne = Mock()
        mock_tsne.fit_transform.return_value = np.array([[1.0, 2.0], [3.0, 4.0]])
        mock_tsne_class.return_value = mock_tsne

        histogram = np.array([0.5, 0.3, 0.2])
        documents = [
            ColoredDocument(histogram=histogram),
            ColoredDocument(histogram=histogram),
        ]
        labels = [0, 1]
        label_names = ["Class A", "Class B"]

        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "projection.png")
        renderer.render_tsne_projection(documents, labels, label_names, output_path)

        mock_tsne_class.assert_called_once_with(n_components=2, perplexity=1, random_state=42)

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.matplotlib.colormaps")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.TSNE")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    def test_should_create_scatter_plot_for_each_class(
        self,
        mock_plt: Mock,
        mock_tsne_class: Mock,
        mock_colormaps: Mock,
        tmp_path: Path,
    ) -> None:
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_cmap = MagicMock()
        mock_cmap.return_value = np.array([[1, 0, 0, 1], [0, 1, 0, 1]])
        mock_colormaps.__getitem__.return_value = mock_cmap

        mock_tsne = Mock()
        mock_tsne.fit_transform.return_value = np.array([[1.0, 2.0], [3.0, 4.0]])
        mock_tsne_class.return_value = mock_tsne

        histogram = np.array([0.5, 0.3, 0.2])
        documents = [
            ColoredDocument(histogram=histogram),
            ColoredDocument(histogram=histogram),
        ]
        labels = [0, 1]
        label_names = ["Class A", "Class B"]

        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "projection.png")
        renderer.render_tsne_projection(documents, labels, label_names, output_path)

        assert mock_ax.scatter.call_count == 2

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.matplotlib.colormaps")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.TSNE")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    def test_should_set_title_and_labels_for_projection(
        self,
        mock_plt: Mock,
        mock_tsne_class: Mock,
        mock_colormaps: Mock,
        tmp_path: Path,
    ) -> None:
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_cmap = MagicMock()
        mock_cmap.return_value = np.array([[1, 0, 0, 1], [0, 1, 0, 1]])
        mock_colormaps.__getitem__.return_value = mock_cmap

        mock_tsne = Mock()
        mock_tsne.fit_transform.return_value = np.array([[1.0, 2.0], [3.0, 4.0]])
        mock_tsne_class.return_value = mock_tsne

        histogram = np.array([0.5, 0.3, 0.2])
        documents = [
            ColoredDocument(histogram=histogram),
            ColoredDocument(histogram=histogram),
        ]
        labels = [0, 1]
        label_names = ["Class A", "Class B"]

        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "projection.png")
        renderer.render_tsne_projection(documents, labels, label_names, output_path)

        mock_ax.set_title.assert_called_once_with("t-SNE Projection of Color Histograms")

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.matplotlib.colormaps")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.TSNE")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    def test_should_save_projection_figure_to_file(
        self,
        mock_plt: Mock,
        mock_tsne_class: Mock,
        mock_colormaps: Mock,
        tmp_path: Path,
    ) -> None:
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_cmap = MagicMock()
        mock_cmap.return_value = np.array([[1, 0, 0, 1], [0, 1, 0, 1]])
        mock_colormaps.__getitem__.return_value = mock_cmap

        mock_tsne = Mock()
        mock_tsne.fit_transform.return_value = np.array([[1.0, 2.0], [3.0, 4.0]])
        mock_tsne_class.return_value = mock_tsne

        histogram = np.array([0.5, 0.3, 0.2])
        documents = [
            ColoredDocument(histogram=histogram),
            ColoredDocument(histogram=histogram),
        ]
        labels = [0, 1]
        label_names = ["Class A", "Class B"]

        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "projection.png")
        renderer.render_tsne_projection(documents, labels, label_names, output_path)

        mock_fig.savefig.assert_called_once_with(output_path, dpi=150, bbox_inches="tight")


class TestRenderConfusionMatrix:
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.matplotlib.colormaps")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.confusion_matrix")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    def test_should_compute_confusion_matrix(
        self,
        mock_plt: Mock,
        mock_cm_fn: Mock,
        mock_colormaps: Mock,
        tmp_path: Path,
    ) -> None:
        mock_fig = Mock()
        mock_ax = Mock()
        mock_ax.figure = mock_fig
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.setp = Mock()

        cm_array = np.array([[10, 2], [3, 15]])
        mock_cm_fn.return_value = cm_array

        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "confusion.png")
        renderer.render_confusion_matrix([0, 0, 1, 1], [0, 1, 0, 1], ["A", "B"], output_path)

        mock_cm_fn.assert_called_once_with([0, 0, 1, 1], [0, 1, 0, 1])

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.matplotlib.colormaps")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.confusion_matrix")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    def test_should_display_confusion_matrix_as_heatmap(
        self,
        mock_plt: Mock,
        mock_cm_fn: Mock,
        mock_colormaps: Mock,
        tmp_path: Path,
    ) -> None:
        mock_fig = Mock()
        mock_ax = Mock()
        mock_ax.figure = mock_fig
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.setp = Mock()

        cm_array = np.array([[10, 2], [3, 15]])
        mock_cm_fn.return_value = cm_array

        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "confusion.png")
        renderer.render_confusion_matrix([0, 0, 1, 1], [0, 1, 0, 1], ["A", "B"], output_path)

        mock_ax.imshow.assert_called_once()

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.matplotlib.colormaps")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.confusion_matrix")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    def test_should_annotate_confusion_matrix_cells(
        self,
        mock_plt: Mock,
        mock_cm_fn: Mock,
        mock_colormaps: Mock,
        tmp_path: Path,
    ) -> None:
        mock_fig = Mock()
        mock_ax = Mock()
        mock_ax.figure = mock_fig
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.setp = Mock()

        cm_array = np.array([[10, 2], [3, 15]])
        mock_cm_fn.return_value = cm_array

        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "confusion.png")
        renderer.render_confusion_matrix([0, 0, 1, 1], [0, 1, 0, 1], ["A", "B"], output_path)

        assert mock_ax.text.call_count == 4

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.matplotlib.colormaps")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.confusion_matrix")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    def test_should_save_confusion_matrix_figure_to_file(
        self,
        mock_plt: Mock,
        mock_cm_fn: Mock,
        mock_colormaps: Mock,
        tmp_path: Path,
    ) -> None:
        mock_fig = Mock()
        mock_ax = Mock()
        mock_ax.figure = mock_fig
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.setp = Mock()

        cm_array = np.array([[10, 2], [3, 15]])
        mock_cm_fn.return_value = cm_array

        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "confusion.png")
        renderer.render_confusion_matrix([0, 0, 1, 1], [0, 1, 0, 1], ["A", "B"], output_path)

        mock_fig.savefig.assert_called_once_with(output_path, dpi=150, bbox_inches="tight")


class TestRenderCorpusSignatures:
    def _make_inputs(self) -> tuple:
        histogram = np.array([0.4, 0.3, 0.2, 0.1])
        documents = [
            ColoredDocument(histogram=histogram, document_id="darwin_0"),
            ColoredDocument(histogram=histogram, document_id="smith_0"),
        ]
        labels = [0, 1]
        label_names = ["Darwin", "Smith"]
        colors = [LabColor(l=50.0, a=0.0, b=0.0) for _ in range(4)]
        codebook = ColorCodebook(colors=colors, num_bins=4)
        return documents, labels, label_names, codebook

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.lab_to_rgb")
    def test_should_create_one_subplot_row_per_corpus(
        self,
        mock_lab_to_rgb: Mock,
        mock_plt: Mock,
        tmp_path: Path,
    ) -> None:
        mock_lab_to_rgb.return_value = (128, 128, 128)
        mock_plt.subplots.return_value = (Mock(), np.array([Mock(), Mock()]))
        documents, labels, label_names, codebook = self._make_inputs()

        renderer = MatplotlibFigureRenderer()
        renderer.render_corpus_signatures(documents, labels, label_names, codebook, str(tmp_path / "sig.png"))

        mock_plt.subplots.assert_called_once()

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.lab_to_rgb")
    def test_should_draw_color_bars_for_each_corpus(
        self,
        mock_lab_to_rgb: Mock,
        mock_plt: Mock,
        tmp_path: Path,
    ) -> None:
        mock_lab_to_rgb.return_value = (128, 128, 128)
        mock_ax = Mock()
        mock_plt.subplots.return_value = (Mock(), np.array([mock_ax, Mock()]))
        documents, labels, label_names, codebook = self._make_inputs()

        renderer = MatplotlibFigureRenderer()
        renderer.render_corpus_signatures(documents, labels, label_names, codebook, str(tmp_path / "sig.png"))

        assert mock_ax.barh.call_count == 4

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.lab_to_rgb")
    def test_should_set_suptitle_with_top_colors(
        self,
        mock_lab_to_rgb: Mock,
        mock_plt: Mock,
        tmp_path: Path,
    ) -> None:
        mock_lab_to_rgb.return_value = (128, 128, 128)
        mock_fig = Mock()
        mock_plt.subplots.return_value = (mock_fig, np.array([Mock(), Mock()]))
        documents, labels, label_names, codebook = self._make_inputs()

        renderer = MatplotlibFigureRenderer()
        renderer.render_corpus_signatures(documents, labels, label_names, codebook, str(tmp_path / "sig.png"))

        mock_fig.suptitle.assert_called_once_with("Per-corpus color signature (top 24 colors)")

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.lab_to_rgb")
    def test_should_save_corpus_signatures_to_file(
        self,
        mock_lab_to_rgb: Mock,
        mock_plt: Mock,
        tmp_path: Path,
    ) -> None:
        mock_lab_to_rgb.return_value = (128, 128, 128)
        mock_fig = Mock()
        mock_plt.subplots.return_value = (mock_fig, np.array([Mock(), Mock()]))
        documents, labels, label_names, codebook = self._make_inputs()

        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "sig.png")
        renderer.render_corpus_signatures(documents, labels, label_names, codebook, output_path)

        mock_fig.savefig.assert_called_once_with(output_path, dpi=150, bbox_inches="tight")

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.lab_to_rgb")
    def test_should_color_each_corpus_by_its_own_dominant_bin(
        self,
        mock_lab_to_rgb: Mock,
        mock_plt: Mock,
        tmp_path: Path,
    ) -> None:
        mock_lab_to_rgb.side_effect = lambda color: (int(color.l), 0, 0)
        mock_ax_a = Mock()
        mock_ax_b = Mock()
        mock_plt.subplots.return_value = (Mock(), np.array([mock_ax_a, mock_ax_b]))
        documents = [
            ColoredDocument(histogram=np.array([0.7, 0.1, 0.1, 0.1]), document_id="a_0"),
            ColoredDocument(histogram=np.array([0.1, 0.1, 0.1, 0.7]), document_id="b_0"),
        ]
        labels = [0, 1]
        label_names = ["A", "B"]
        codebook = ColorCodebook(colors=[LabColor(l=float(i * 10), a=0.0, b=0.0) for i in range(4)], num_bins=4)

        renderer = MatplotlibFigureRenderer()
        renderer.render_corpus_signatures(documents, labels, label_names, codebook, str(tmp_path / "sig.png"))

        first_bin_colors = (
            mock_ax_a.barh.call_args_list[0].kwargs["color"][0],
            mock_ax_b.barh.call_args_list[0].kwargs["color"][0],
        )
        assert first_bin_colors[0] != first_bin_colors[1]

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.lab_to_rgb")
    def test_should_limit_bars_to_top_colors(
        self,
        mock_lab_to_rgb: Mock,
        mock_plt: Mock,
        tmp_path: Path,
    ) -> None:
        mock_lab_to_rgb.return_value = (128, 128, 128)
        mock_ax = Mock()
        mock_plt.subplots.return_value = (Mock(), np.array([mock_ax]))
        documents = [ColoredDocument(histogram=np.array([0.4, 0.3, 0.2, 0.1]), document_id="a_0")]
        codebook = ColorCodebook(colors=[LabColor(l=50.0, a=0.0, b=0.0) for _ in range(4)], num_bins=4)

        renderer = MatplotlibFigureRenderer()
        renderer.render_corpus_signatures(documents, [0], ["A"], codebook, str(tmp_path / "sig.png"), top_colors=2)

        assert mock_ax.barh.call_count == 2

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.lab_to_rgb")
    def test_should_stack_bars_left_to_right_by_cumulative_width(
        self,
        mock_lab_to_rgb: Mock,
        mock_plt: Mock,
        tmp_path: Path,
    ) -> None:
        mock_lab_to_rgb.return_value = (128, 128, 128)
        mock_ax = Mock()
        mock_plt.subplots.return_value = (Mock(), np.array([mock_ax]))
        documents = [ColoredDocument(histogram=np.array([0.4, 0.3, 0.2, 0.1]), document_id="a_0")]
        codebook = ColorCodebook(colors=[LabColor(l=50.0, a=0.0, b=0.0) for _ in range(4)], num_bins=4)

        renderer = MatplotlibFigureRenderer()
        renderer.render_corpus_signatures(documents, [0], ["A"], codebook, str(tmp_path / "sig.png"))

        lefts = [call.kwargs["left"] for call in mock_ax.barh.call_args_list]
        assert np.allclose(lefts, [0.0, 0.4, 0.7, 0.9])

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.lab_to_rgb")
    def test_should_skip_corpus_with_no_documents(
        self,
        mock_lab_to_rgb: Mock,
        mock_plt: Mock,
        tmp_path: Path,
    ) -> None:
        mock_lab_to_rgb.return_value = (128, 128, 128)
        mock_ax_a = Mock()
        mock_ax_b = Mock()
        mock_plt.subplots.return_value = (Mock(), np.array([mock_ax_a, mock_ax_b]))
        documents = [ColoredDocument(histogram=np.array([0.4, 0.3, 0.2, 0.1]), document_id="a_0")]
        codebook = ColorCodebook(colors=[LabColor(l=50.0, a=0.0, b=0.0) for _ in range(4)], num_bins=4)

        renderer = MatplotlibFigureRenderer()
        renderer.render_corpus_signatures(documents, [0], ["A", "B"], codebook, str(tmp_path / "sig.png"))

        assert mock_ax_b.barh.call_count == 0


class TestSelectSamplesPerClass:
    def test_should_select_correct_number_of_samples_per_class(self) -> None:
        labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]

        selected = MatplotlibFigureRenderer._select_samples_per_class(labels, 3, 2)

        assert len(selected) == 6

    def test_should_select_from_each_class(self) -> None:
        labels = [0, 0, 1, 1, 2, 2]

        selected = MatplotlibFigureRenderer._select_samples_per_class(labels, 3, 1)

        assert selected == [0, 2, 4]

    def test_should_handle_fewer_samples_than_requested(self) -> None:
        labels = [0, 1]

        selected = MatplotlibFigureRenderer._select_samples_per_class(labels, 2, 3)

        assert len(selected) == 2


class TestRenderRateDistortion:
    def _frontier(self) -> RateDistortionFrontier:
        return RateDistortionFrontier(
            [
                RateDistortionPoint("color_vq", 3.0, 5.0, 0.70),
                RateDistortionPoint("color_vq", 12.0, 1.0, 0.85),
                RateDistortionPoint("pq", 3.0, 0.02, None),
                RateDistortionPoint("pq", 12.0, 0.005, None),
                RateDistortionPoint("gzip", 48.0, 0.0, None),
            ]
        )

    def test_should_write_a_non_empty_rate_distortion_png(self, tmp_path: Path) -> None:
        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "rate_distortion.png")

        renderer.render_rate_distortion(self._frontier(), output_path)

        assert os.path.getsize(output_path) > 0

    def test_should_render_rate_distortion_at_exact_uncropped_size(self, tmp_path: Path) -> None:
        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "rate_distortion.png")

        renderer.render_rate_distortion(self._frontier(), output_path)

        expected = (int(RATE_DISTORTION_FIGURE_SIZE[0] * FIGURE_DPI), int(RATE_DISTORTION_FIGURE_SIZE[1] * FIGURE_DPI))
        with Image.open(output_path) as image:
            rendered_size = image.size
        assert rendered_size == expected


def _narrative_arc(beats: int = 3) -> NarrativeArc:
    points = [
        ArcPoint(
            beat=NarrativeBeat(index=index, title=f"Beat {index}", text="prose"),
            colour=LabColor(l=40.0 + index * 5.0, a=12.0, b=-8.0),
            histogram=ColoredDocument(histogram=np.full(4, 0.25, dtype=np.float64)),
        )
        for index in range(beats)
    ]
    return NarrativeArc(points=points, drift_series=[0.1] * (beats - 1), coherence_series=[0.2] * beats)


class TestRenderNarrativeArc:
    def test_should_write_a_non_empty_narrative_arc_png(self, tmp_path: Path) -> None:
        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "story_compass.png")

        renderer.render_narrative_arc(_narrative_arc(), output_path)

        assert os.path.getsize(output_path) > 0

    def test_should_render_narrative_arc_at_exact_uncropped_size(self, tmp_path: Path) -> None:
        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "story_compass.png")

        renderer.render_narrative_arc(_narrative_arc(), output_path)

        expected = (int(NARRATIVE_ARC_FIGURE_SIZE[0] * FIGURE_DPI), int(NARRATIVE_ARC_FIGURE_SIZE[1] * FIGURE_DPI))
        with Image.open(output_path) as image:
            rendered_size = image.size
        assert rendered_size == expected

    def test_should_render_a_single_beat_arc_without_crashing(self, tmp_path: Path) -> None:
        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "single.png")

        renderer.render_narrative_arc(_narrative_arc(beats=1), output_path)

        assert os.path.getsize(output_path) > 0

    @patch(f"{MODULE}.plt")
    @patch(f"{MODULE}.lab_to_rgb")
    def test_should_create_five_stacked_panels(self, mock_lab_to_rgb: Mock, mock_plt: Mock, tmp_path: Path) -> None:
        mock_lab_to_rgb.return_value = (128, 128, 128)
        mock_plt.subplots.return_value = (Mock(), np.array([Mock() for _ in range(5)]))

        MatplotlibFigureRenderer().render_narrative_arc(_narrative_arc(), str(tmp_path / "arc.png"))

        mock_plt.subplots.assert_called_once_with(5, 1, figsize=NARRATIVE_ARC_FIGURE_SIZE)

    @patch(f"{MODULE}.plt")
    @patch(f"{MODULE}.lab_to_rgb")
    def test_should_draw_one_swatch_per_beat(self, mock_lab_to_rgb: Mock, mock_plt: Mock, tmp_path: Path) -> None:
        mock_lab_to_rgb.return_value = (128, 128, 128)
        axes = np.array([Mock() for _ in range(5)])
        mock_plt.subplots.return_value = (Mock(), axes)

        MatplotlibFigureRenderer().render_narrative_arc(_narrative_arc(beats=3), str(tmp_path / "arc.png"))

        assert axes[4].barh.call_count == 3

    @patch(f"{MODULE}.plt")
    @patch(f"{MODULE}.lab_to_rgb")
    def test_should_save_narrative_arc_without_cropping(
        self, mock_lab_to_rgb: Mock, mock_plt: Mock, tmp_path: Path
    ) -> None:
        mock_lab_to_rgb.return_value = (128, 128, 128)
        mock_fig = Mock()
        mock_plt.subplots.return_value = (mock_fig, np.array([Mock() for _ in range(5)]))
        output_path = str(tmp_path / "arc.png")

        MatplotlibFigureRenderer().render_narrative_arc(_narrative_arc(), output_path)

        mock_fig.savefig.assert_called_once_with(output_path, dpi=150)


def _write_sheet(directory: Path, name: str, shade: int) -> str:
    path = directory / name
    Image.new("RGB", (20, 28), (shade, shade, shade)).save(path, format="PNG")
    return str(path)


def _sheet_paths(directory: Path, count: int) -> list:
    return [_write_sheet(directory, f"sheet_{index}.png", 10 * index) for index in range(count)]


class TestRenderA4Gallery:
    def test_should_write_a_non_empty_gallery_png(self, tmp_path: Path) -> None:
        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "gallery.png")

        renderer.render_a4_gallery(_sheet_paths(tmp_path, 5), output_path, columns=3)

        assert os.path.getsize(output_path) > 0

    def test_should_raise_when_no_sheet_paths_are_provided(self, tmp_path: Path) -> None:
        renderer = MatplotlibFigureRenderer()

        with pytest.raises(ValueError, match="at least one sheet"):
            renderer.render_a4_gallery([], str(tmp_path / "gallery.png"))

    def test_should_render_identical_bytes_across_runs(self, tmp_path: Path) -> None:
        renderer = MatplotlibFigureRenderer()
        sheets = _sheet_paths(tmp_path, 4)
        first = str(tmp_path / "first.png")
        second = str(tmp_path / "second.png")

        renderer.render_a4_gallery(sheets, first, columns=2)
        renderer.render_a4_gallery(sheets, second, columns=2)

        assert (tmp_path / "first.png").read_bytes() == (tmp_path / "second.png").read_bytes()

    @patch(f"{MODULE}.plt")
    def test_should_clamp_columns_to_the_sheet_count(self, mock_plt: Mock, tmp_path: Path) -> None:
        mock_plt.subplots.return_value = (Mock(), np.array([Mock(), Mock()]))
        mock_plt.imread.return_value = np.zeros((2, 2, 3))

        MatplotlibFigureRenderer().render_a4_gallery(["a.png", "b.png"], str(tmp_path / "g.png"), columns=12)

        assert mock_plt.subplots.call_args.args == (1, 2)

    def test_should_write_a_non_empty_gallery_png_when_captions_are_provided(self, tmp_path: Path) -> None:
        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "captioned.png")

        renderer.render_a4_gallery(_sheet_paths(tmp_path, 4), output_path, 2, ["a/w", "b/x", "c/y", "d/z"])

        assert os.path.getsize(output_path) > 0

    def test_should_raise_when_the_caption_count_does_not_match_the_sheet_count(self, tmp_path: Path) -> None:
        renderer = MatplotlibFigureRenderer()

        with pytest.raises(ValueError, match="one caption per sheet"):
            renderer.render_a4_gallery(_sheet_paths(tmp_path, 3), str(tmp_path / "g.png"), 2, ["only", "two"])

    @patch(f"{MODULE}.plt")
    def test_should_draw_a_caption_on_each_tile_when_captions_are_provided(
        self, mock_plt: Mock, tmp_path: Path
    ) -> None:
        axes = np.array([Mock(), Mock()])
        mock_plt.subplots.return_value = (Mock(), axes)
        mock_plt.imread.return_value = np.zeros((2, 2, 3))

        MatplotlibFigureRenderer().render_a4_gallery(["a.png", "b.png"], str(tmp_path / "g.png"), 2, ["a/w", "b/x"])

        assert [axis.text.call_args.args[2] for axis in axes] == ["a/w", "b/x"]

    @patch(f"{MODULE}.plt")
    def test_should_use_the_supplied_title_for_the_gallery(self, mock_plt: Mock, tmp_path: Path) -> None:
        figure = Mock()
        mock_plt.subplots.return_value = (figure, np.array([Mock()]))
        mock_plt.imread.return_value = np.zeros((2, 2, 3))

        MatplotlibFigureRenderer().render_a4_gallery(["a.png"], str(tmp_path / "g.png"), 1, None, "Barcodes")

        figure.suptitle.assert_called_once_with("Barcodes")

    def test_should_downsample_tiles_when_a_pixel_bound_is_given(self, tmp_path: Path) -> None:
        renderer = MatplotlibFigureRenderer()
        wide = str(tmp_path / "wide.png")
        Image.new("RGB", (800, 800), (10, 20, 30)).save(wide, format="PNG")

        renderer.render_a4_gallery([wide], str(tmp_path / "bounded.png"), 1, None, "Bounded", 100)

        assert os.path.getsize(tmp_path / "bounded.png") > 0

    def test_should_leave_tiles_at_full_size_when_no_pixel_bound_is_given(self, tmp_path: Path) -> None:
        loaded = _load_tile_image(_write_sheet(tmp_path, "full.png", 40), None)

        assert loaded.shape[:2] == (28, 20)

    def test_should_shrink_the_tile_to_the_pixel_bound(self, tmp_path: Path) -> None:
        loaded = _load_tile_image(_write_sheet(tmp_path, "bounded_tile.png", 40), 7)

        assert max(loaded.shape[:2]) <= 7


class TestRenderImageComparison:
    def test_should_write_a_non_empty_comparison_png(self, tmp_path: Path) -> None:
        renderer = MatplotlibFigureRenderer()
        panels = [(path, f"caption {index}") for index, path in enumerate(_sheet_paths(tmp_path, 2))]
        output_path = str(tmp_path / "two_ways.png")

        renderer.render_image_comparison(panels, "Two ways", output_path)

        assert os.path.getsize(output_path) > 0

    def test_should_raise_when_no_panels_are_provided(self, tmp_path: Path) -> None:
        renderer = MatplotlibFigureRenderer()

        with pytest.raises(ValueError, match="at least one panel"):
            renderer.render_image_comparison([], "Two ways", str(tmp_path / "two_ways.png"))

    @patch(f"{MODULE}.plt")
    def test_should_create_one_axis_per_panel(self, mock_plt: Mock, tmp_path: Path) -> None:
        mock_plt.subplots.return_value = (Mock(), np.array([Mock(), Mock()]))
        mock_plt.imread.return_value = np.zeros((2, 2, 3))

        panels = [("a.png", "meaning"), ("b.png", "bytes")]
        MatplotlibFigureRenderer().render_image_comparison(panels, "Two ways", str(tmp_path / "c.png"))

        assert mock_plt.subplots.call_args.args == (1, 2)

    @patch(f"{MODULE}.plt")
    def test_should_caption_each_panel(self, mock_plt: Mock, tmp_path: Path) -> None:
        axes = np.array([Mock(), Mock()])
        mock_plt.subplots.return_value = (Mock(), axes)
        mock_plt.imread.return_value = np.zeros((2, 2, 3))

        panels = [("a.png", "meaning"), ("b.png", "bytes")]
        MatplotlibFigureRenderer().render_image_comparison(panels, "Two ways", str(tmp_path / "c.png"))

        assert [axis.text.call_args.args[2] for axis in axes] == ["meaning", "bytes"]

    @patch(f"{MODULE}.plt")
    def test_should_set_the_comparison_title_on_the_figure(self, mock_plt: Mock, tmp_path: Path) -> None:
        figure = Mock()
        mock_plt.subplots.return_value = (figure, np.array([Mock()]))
        mock_plt.imread.return_value = np.zeros((2, 2, 3))

        MatplotlibFigureRenderer().render_image_comparison([("a.png", "meaning")], "Two ways", str(tmp_path / "c.png"))

        assert figure.suptitle.call_args.args == ("Two ways",)


class TestSaveFigure:
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    def test_should_create_output_directory_if_not_exists(
        self,
        mock_plt: Mock,
        tmp_path: Path,
    ) -> None:
        mock_fig = Mock()
        output_path = str(tmp_path / "subdir" / "figure.png")

        MatplotlibFigureRenderer._save_figure(mock_fig, output_path)

        assert os.path.isdir(str(tmp_path / "subdir"))

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    def test_should_save_with_correct_dpi(
        self,
        mock_plt: Mock,
        tmp_path: Path,
    ) -> None:
        mock_fig = Mock()
        output_path = str(tmp_path / "figure.png")

        MatplotlibFigureRenderer._save_figure(mock_fig, output_path)

        mock_fig.savefig.assert_called_once_with(output_path, dpi=150, bbox_inches="tight")

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    def test_should_handle_filename_without_directory(
        self,
        mock_plt: Mock,
    ) -> None:
        mock_fig = Mock()

        MatplotlibFigureRenderer._save_figure(mock_fig, "figure.png")

        mock_fig.savefig.assert_called_once_with("figure.png", dpi=150, bbox_inches="tight")
