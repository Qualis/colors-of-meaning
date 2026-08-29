import math
import os
from typing import Any, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from sklearn.manifold import TSNE  # type: ignore
from sklearn.metrics import confusion_matrix  # type: ignore

from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.model.lab_color import LabColor
from colors_of_meaning.domain.model.narrative_arc import NarrativeArc
from colors_of_meaning.domain.model.objective_comparison import (
    ObjectiveArmResult,
    ObjectiveComparison,
)
from colors_of_meaning.domain.model.rate_distortion_point import (
    RateDistortionFrontier,
    RateDistortionPoint,
)
from colors_of_meaning.domain.service.figure_renderer import FigureRenderer
from colors_of_meaning.shared.lab_utils import lab_to_rgb

matplotlib.use("Agg")

FIGURE_DPI = 150
RATE_DISTORTION_FIGURE_SIZE = (11.0, 7.0)
NARRATIVE_ARC_FIGURE_SIZE = (11.0, 9.0)
OBJECTIVE_COMPARISON_FIGURE_SIZE = (10.0, 6.5)
OBJECTIVE_COMPARISON_BAR_COLOR = "#4c72b0"
OBJECTIVE_COMPARISON_ERROR_CAPSIZE = 4.0
OBJECTIVE_COMPARISON_LABEL_ROTATION = 20
GALLERY_TILE_WIDTH_INCHES = 1.3
GALLERY_TILE_HEIGHT_INCHES = 1.84
GALLERY_CAPTION_OFFSET = -0.02
GALLERY_CAPTION_FONT_SIZE = 5
COMPARISON_PANEL_WIDTH_INCHES = 4.7
COMPARISON_PANEL_HEIGHT_INCHES = 6.4
COMPARISON_CAPTION_OFFSET = -0.02
COMPARISON_CAPTION_FONT_SIZE = 11
COMPARISON_TITLE_FONT_SIZE = 15


def _entry_at(entries: Optional[List[str]], position: int) -> Optional[str]:
    if entries is None or position >= len(entries):
        return None
    return entries[position]


def _load_tile_image(sheet_path: str, max_tile_pixels: Optional[int]) -> npt.NDArray:
    image = plt.imread(sheet_path)
    if max_tile_pixels is None:
        return np.asarray(image)
    step = max(1, int(math.ceil(max(image.shape[0], image.shape[1]) / max_tile_pixels)))
    return np.asarray(image[::step, ::step])


class MatplotlibFigureRenderer(FigureRenderer):
    def render_codebook_palette(self, codebook: ColorCodebook, output_path: str) -> None:
        grid_size = int(math.ceil(math.sqrt(codebook.num_bins)))
        image = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)

        for i in range(codebook.num_bins):
            row = i // grid_size
            col = i % grid_size
            rgb = lab_to_rgb(codebook.colors[i])
            image[row, col] = rgb

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image, interpolation="nearest")
        ax.set_title(f"Color Codebook ({codebook.num_bins} colors)")
        ax.axis("off")

        self._save_figure(fig, output_path)

    def render_document_histograms(
        self,
        documents: List[ColoredDocument],
        labels: List[int],
        label_names: List[str],
        output_path: str,
        samples_per_class: int = 2,
    ) -> None:
        selected_indices = self._select_samples_per_class(labels, len(label_names), samples_per_class)

        num_plots = len(selected_indices)
        cols = min(num_plots, 2)
        rows = int(math.ceil(num_plots / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows))
        if num_plots == 1:
            axes = np.array([axes])
        axes = np.atleast_2d(axes)

        for plot_idx, doc_idx in enumerate(selected_indices):
            row = plot_idx // cols
            col = plot_idx % cols
            ax = axes[row, col]
            document = documents[doc_idx]
            label = labels[doc_idx]

            nonzero_mask = document.histogram > 0
            nonzero_bins = np.where(nonzero_mask)[0]
            nonzero_values = document.histogram[nonzero_mask]

            ax.bar(range(len(nonzero_bins)), nonzero_values, width=1.0)
            ax.set_title(f"{label_names[label]} (doc {doc_idx})")
            ax.set_xlabel("Color bin")
            ax.set_ylabel("Frequency")

        for plot_idx in range(num_plots, rows * cols):
            row = plot_idx // cols
            col = plot_idx % cols
            axes[row, col].axis("off")

        fig.suptitle("Document Color Histograms by Class")
        fig.tight_layout()
        self._save_figure(fig, output_path)

    def render_tsne_projection(
        self,
        documents: List[ColoredDocument],
        labels: List[int],
        label_names: List[str],
        output_path: str,
    ) -> None:
        histograms = np.array([doc.histogram for doc in documents])

        perplexity = min(30, len(documents) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        projections = tsne.fit_transform(histograms)

        fig, ax = plt.subplots(figsize=(10, 8))

        unique_labels = sorted(set(labels))
        cmap = matplotlib.colormaps["tab10"]
        colors = cmap(np.linspace(0, 1, len(unique_labels)))

        for label_idx, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            ax.scatter(
                projections[mask, 0],
                projections[mask, 1],
                c=[colors[label_idx]],
                label=label_names[label],
                alpha=0.6,
                s=20,
            )

        ax.set_title("t-SNE Projection of Color Histograms")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend()

        fig.tight_layout()
        self._save_figure(fig, output_path)

    def render_confusion_matrix(
        self,
        y_true: List[int],
        y_pred: List[int],
        label_names: List[str],
        output_path: str,
    ) -> None:
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(cm, interpolation="nearest", cmap=matplotlib.colormaps["Blues"])
        ax.figure.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=label_names,
            yticklabels=label_names,
            ylabel="True label",
            xlabel="Predicted label",
            title="Confusion Matrix",
        )

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        fig.tight_layout()
        self._save_figure(fig, output_path)

    def render_corpus_signatures(
        self,
        documents: List[ColoredDocument],
        labels: List[int],
        label_names: List[str],
        codebook: ColorCodebook,
        output_path: str,
        top_colors: int = 24,
    ) -> None:
        fig, axes = plt.subplots(len(label_names), 1, figsize=(12, 1.5 * len(label_names)))
        axes = np.atleast_1d(axes)

        for label_index, name in enumerate(label_names):
            histograms = self._corpus_histograms(documents, labels, label_index)
            if not histograms:
                continue
            ax = axes[label_index]
            self._draw_color_signature(ax, np.mean(histograms, axis=0), codebook, top_colors)
            ax.set_ylabel(name, rotation=0, ha="right", va="center")

        fig.suptitle(f"Per-corpus color signature (top {top_colors} colors)")
        fig.tight_layout()
        self._save_figure(fig, output_path)

    def render_rate_distortion(self, frontier: RateDistortionFrontier, output_path: str) -> None:
        fig, distortion_axis = plt.subplots(figsize=RATE_DISTORTION_FIGURE_SIZE)
        fig.subplots_adjust(right=0.85)
        accuracy_axis = distortion_axis.twinx()

        for method in sorted({point.method for point in frontier.points}):
            self._plot_method_series(distortion_axis, accuracy_axis, self._method_points(frontier, method), method)

        distortion_axis.set_xscale("symlog")
        distortion_axis.set_yscale("symlog")
        distortion_axis.set_xlabel("Bits per token (symlog scale)")
        distortion_axis.set_ylabel("Reconstruction error, symlog (native: ΔE or MSE)")
        accuracy_axis.set_ylabel("Downstream accuracy")
        accuracy_axis.set_ylim(0.0, 1.0)
        distortion_axis.set_title("Rate-distortion frontier for semantic color compression")
        self._merge_legends(distortion_axis, accuracy_axis)
        self._save_figure(fig, output_path, tight=False)

    def render_objective_comparison(self, comparison: ObjectiveComparison, output_path: str) -> None:
        arms = list(comparison.results)
        fig, axis = plt.subplots(figsize=OBJECTIVE_COMPARISON_FIGURE_SIZE)
        positions = list(range(len(arms)))

        axis.bar(
            positions,
            [arm.mean_rho for arm in arms],
            yerr=[arm.stdev_rho for arm in arms],
            capsize=OBJECTIVE_COMPARISON_ERROR_CAPSIZE,
            color=OBJECTIVE_COMPARISON_BAR_COLOR,
        )
        axis.set_xticks(positions)
        axis.set_xticklabels([arm.arm for arm in arms], rotation=OBJECTIVE_COMPARISON_LABEL_ROTATION, ha="right")
        self._draw_control_references(axis, list(comparison.controls))
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_ylabel("Held-out Spearman rho (embedding cosine vs Lab delta E)")
        axis.set_title("Structure preservation by training objective (mean +/- sd over seeds)")
        self._save_figure(fig, output_path)

    @staticmethod
    def _draw_control_references(axis: Any, controls: List[ObjectiveArmResult]) -> None:
        for index, control in enumerate(controls):
            axis.axhline(
                control.mean_rho,
                linestyle="--",
                color=f"C{index + 1}",
                label=f"{control.arm} ({control.mean_rho:.4f})",
            )
        if controls:
            axis.legend(loc="best")

    def render_a4_gallery(
        self,
        sheet_paths: List[str],
        output_path: str,
        columns: int = 12,
        captions: Optional[List[str]] = None,
        title: str = "Per-book A4 colour signatures",
        max_tile_pixels: Optional[int] = None,
    ) -> None:
        if not sheet_paths:
            raise ValueError("render_a4_gallery requires at least one sheet path")
        if captions is not None and len(captions) != len(sheet_paths):
            raise ValueError(
                f"render_a4_gallery needs one caption per sheet, got {len(captions)} for {len(sheet_paths)}"
            )

        column_count = min(columns, len(sheet_paths))
        row_count = int(math.ceil(len(sheet_paths) / column_count))
        fig, axes = plt.subplots(
            row_count,
            column_count,
            figsize=(column_count * GALLERY_TILE_WIDTH_INCHES, row_count * GALLERY_TILE_HEIGHT_INCHES),
        )
        flat_axes = np.atleast_1d(axes).ravel()

        for position, axis in enumerate(flat_axes):
            self._draw_gallery_tile(
                axis, _entry_at(sheet_paths, position), _entry_at(captions, position), max_tile_pixels
            )

        fig.suptitle(title)
        self._save_figure(fig, output_path)

    @staticmethod
    def _draw_gallery_tile(
        axis: Any,
        sheet_path: Optional[str],
        caption: Optional[str] = None,
        max_tile_pixels: Optional[int] = None,
    ) -> None:
        axis.axis("off")
        if sheet_path is None:
            return
        axis.imshow(_load_tile_image(sheet_path, max_tile_pixels), interpolation="nearest")
        if caption is not None:
            axis.text(
                0.0,
                GALLERY_CAPTION_OFFSET,
                caption,
                transform=axis.transAxes,
                fontsize=GALLERY_CAPTION_FONT_SIZE,
                va="top",
            )

    def render_image_comparison(self, panels: List[Tuple[str, str]], title: str, output_path: str) -> None:
        if not panels:
            raise ValueError("render_image_comparison requires at least one panel")

        fig, axes = plt.subplots(
            1,
            len(panels),
            figsize=(len(panels) * COMPARISON_PANEL_WIDTH_INCHES, COMPARISON_PANEL_HEIGHT_INCHES),
        )
        for axis, panel in zip(np.atleast_1d(axes).ravel(), panels, strict=True):
            self._draw_comparison_panel(axis, panel)

        fig.suptitle(title, fontsize=COMPARISON_TITLE_FONT_SIZE)
        self._save_figure(fig, output_path)

    @staticmethod
    def _draw_comparison_panel(axis: Any, panel: Tuple[str, str]) -> None:
        image_path, caption = panel
        axis.axis("off")
        axis.imshow(plt.imread(image_path), interpolation="nearest")
        axis.text(
            0.0,
            COMPARISON_CAPTION_OFFSET,
            caption,
            transform=axis.transAxes,
            fontsize=COMPARISON_CAPTION_FONT_SIZE,
            va="top",
        )

    def render_narrative_arc(self, arc: NarrativeArc, output_path: str) -> None:
        fig, axes = plt.subplots(5, 1, figsize=NARRATIVE_ARC_FIGURE_SIZE)
        beat_indices = list(range(arc.beat_count))
        self._plot_curve(axes[0], beat_indices, arc.lightness_series, "Lightness (sentiment)", "tab:orange")
        self._plot_curve(axes[1], beat_indices, arc.chroma_series, "Chroma (concreteness)", "tab:green")
        self._plot_curve(axes[2], beat_indices, arc.hue_series, "Hue deg (topic)", "tab:blue")
        self._plot_drift(axes[3], arc.drift_series)
        self._plot_swatch_strip(axes[4], arc.colours)
        fig.suptitle("Narrative color compass")
        fig.tight_layout()
        self._save_figure(fig, output_path, tight=False)

    @staticmethod
    def _plot_curve(ax: Any, beat_indices: List[int], values: List[float], ylabel: str, color: str) -> None:
        ax.plot(beat_indices, values, marker="o", color=color)
        ax.set_ylabel(ylabel)
        ax.set_xticks(beat_indices)

    @staticmethod
    def _plot_drift(ax: Any, drift_series: List[float]) -> None:
        positions = [index + 0.5 for index in range(len(drift_series))]
        ax.plot(positions, drift_series, marker="s", color="tab:red")
        ax.set_ylabel("Drift (beat to beat)")

    @staticmethod
    def _plot_swatch_strip(ax: Any, colours: List[LabColor]) -> None:
        for position, colour in enumerate(colours):
            rgb = np.array(lab_to_rgb(colour), dtype=float) / 255.0
            ax.barh(0, 1.0, left=position, height=1.0, color=rgb)
        ax.set_xlim(0, len(colours))
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlabel("Beat colours (left to right)")

    @staticmethod
    def _method_points(frontier: RateDistortionFrontier, method: str) -> List[RateDistortionPoint]:
        method_points = [point for point in frontier.points if point.method == method]
        return sorted(method_points, key=lambda point: point.bits_per_token)

    def _plot_method_series(
        self, distortion_axis: Any, accuracy_axis: Any, points: List[RateDistortionPoint], method: str
    ) -> None:
        bits = [point.bits_per_token for point in points]
        distortion = [point.reconstruction_error for point in points]
        distortion_axis.plot(bits, distortion, marker="o", label=f"{method} distortion")
        self._plot_accuracy_series(accuracy_axis, points, method)

    def _plot_accuracy_series(self, accuracy_axis: Any, points: List[RateDistortionPoint], method: str) -> None:
        measured = self._points_with_accuracy(points)
        if not measured:
            return
        accuracy_axis.plot(
            [point.bits_per_token for point in measured],
            [point.accuracy for point in measured],
            marker="s",
            linestyle="--",
            label=f"{method} accuracy",
        )

    @staticmethod
    def _points_with_accuracy(points: List[RateDistortionPoint]) -> List[RateDistortionPoint]:
        return [point for point in points if point.accuracy is not None]

    @staticmethod
    def _merge_legends(primary_axis: Any, secondary_axis: Any) -> None:
        primary_handles, primary_labels = primary_axis.get_legend_handles_labels()
        secondary_handles, secondary_labels = secondary_axis.get_legend_handles_labels()
        primary_axis.legend(primary_handles + secondary_handles, primary_labels + secondary_labels, loc="best")

    @staticmethod
    def _corpus_histograms(
        documents: List[ColoredDocument], labels: List[int], label_index: int
    ) -> List[npt.NDArray[np.float64]]:
        return [documents[i].histogram for i in range(len(documents)) if labels[i] == label_index]

    @staticmethod
    def _draw_color_signature(ax: Any, mean_histogram: Any, codebook: ColorCodebook, top_colors: int) -> None:
        top_bins = np.argsort(mean_histogram)[::-1][:top_colors]
        left = 0.0
        for bin_index in top_bins:
            width = float(mean_histogram[bin_index])
            rgb = np.array(lab_to_rgb(codebook.colors[int(bin_index)]), dtype=float) / 255.0
            ax.barh(0, width, left=left, height=1.0, color=rgb)
            left += width
        ax.set_xlim(0, left)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xticks([])

    @staticmethod
    def _select_samples_per_class(labels: List[int], num_classes: int, samples_per_class: int) -> List[int]:
        selected: List[int] = []
        for class_idx in range(num_classes):
            class_indices = [i for i, label in enumerate(labels) if label == class_idx]
            selected.extend(class_indices[:samples_per_class])
        return selected

    @staticmethod
    def _save_figure(fig: plt.Figure, output_path: str, tight: bool = True) -> None:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        if tight:
            fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
        else:
            fig.savefig(output_path, dpi=FIGURE_DPI)
        plt.close(fig)
