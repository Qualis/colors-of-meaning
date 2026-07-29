import os
from math import ceil, sqrt
from typing import Callable, Dict, List, Tuple

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw

from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.service.cell_geometry import CellBox, cell_center, tile_canvas, tile_grid
from colors_of_meaning.domain.service.document_image_renderer import (
    DocumentImageLayout,
    DocumentImageRenderer,
)
from colors_of_meaning.shared.lab_utils import lab_to_rgb, rgb_to_lab

A4_WIDTH_MM = 210
A4_HEIGHT_MM = 297
CELL_SIZE_MM = 10
MILLIMETERS_PER_INCH = 25.4
BACKGROUND_RGB: Tuple[int, int, int] = (255, 255, 255)

Rgb = Tuple[int, int, int]
SignatureBand = Tuple[int, int, int]


class PillowDocumentImageRenderer(DocumentImageRenderer):
    def render_document_image(
        self,
        document: ColoredDocument,
        codebook: ColorCodebook,
        layout: DocumentImageLayout,
        output_path: str,
        dpi: int,
    ) -> None:
        width, height = self._canvas_pixels(dpi)
        image = Image.new("RGB", (width, height), BACKGROUND_RGB)

        layouts: Dict[str, Callable[[Image.Image, ColoredDocument, ColorCodebook], None]] = {
            "score": self._render_score,
            "signature": self._render_signature,
            "mosaic": self._render_mosaic,
        }
        renderer = layouts.get(layout)
        if renderer is None:
            raise ValueError(f"Unknown layout: {layout}")
        renderer(image, document, codebook)

        self._save(image, output_path, dpi)

    def decode_document_image(self, input_path: str, codebook: ColorCodebook) -> List[int]:
        with Image.open(input_path) as opened:
            pixels = np.asarray(opened.convert("RGB"))

        boxes = self._score_boxes(pixels.shape[1], pixels.shape[0])
        cell_colors = [self._sample_center(pixels, box) for box in boxes]
        content_colors = self._trim_trailing_background(cell_colors)
        return [codebook.quantize(rgb_to_lab(*color)) for color in content_colors]

    def _render_score(self, image: Image.Image, document: ColoredDocument, codebook: ColorCodebook) -> None:
        draw = ImageDraw.Draw(image)
        boxes = self._score_boxes(image.width, image.height)
        sequence = document.color_sequence or []
        if len(sequence) > len(boxes):
            raise ValueError(f"score layout needs {len(sequence)} cells but the A4 canvas fits {len(boxes)}")
        for bin_index, box in zip(sequence, boxes, strict=False):
            draw.rectangle(box, fill=lab_to_rgb(codebook.get_color(bin_index)))

    def _render_signature(self, image: Image.Image, document: ColoredDocument, codebook: ColorCodebook) -> None:
        draw = ImageDraw.Draw(image)
        for bin_index, top, bottom in self._signature_bands(document.histogram, image.height):
            draw.rectangle((0, top, image.width, bottom), fill=lab_to_rgb(codebook.get_color(bin_index)))

    def _render_mosaic(self, image: Image.Image, document: ColoredDocument, codebook: ColorCodebook) -> None:
        draw = ImageDraw.Draw(image)
        grid_size = ceil(sqrt(codebook.num_bins))
        boxes = tile_grid(image.width, image.height, grid_size, grid_size)
        peak_frequency = float(document.histogram.max())
        for bin_index in range(codebook.num_bins):
            intensity = float(document.histogram[bin_index]) / peak_frequency
            tinted = self._tint(lab_to_rgb(codebook.get_color(bin_index)), intensity)
            draw.rectangle(boxes[bin_index], fill=tinted)

    @staticmethod
    def _signature_bands(histogram: npt.NDArray[np.float64], height: int) -> List[SignatureBand]:
        ordered_bins = sorted(
            (index for index in range(len(histogram)) if histogram[index] > 0),
            key=lambda index: (-float(histogram[index]), index),
        )
        bands: List[SignatureBand] = []
        cumulative = 0.0
        top = 0
        for bin_index in ordered_bins:
            cumulative += float(histogram[bin_index])
            bottom = round(cumulative * height)
            bands.append((bin_index, top, bottom))
            top = bottom
        return bands

    @staticmethod
    def _tint(color: Rgb, intensity: float) -> Rgb:
        def blend(channel: int, background: int) -> int:
            return round(channel * intensity + background * (1.0 - intensity))

        return (
            blend(color[0], BACKGROUND_RGB[0]),
            blend(color[1], BACKGROUND_RGB[1]),
            blend(color[2], BACKGROUND_RGB[2]),
        )

    @staticmethod
    def _sample_center(pixels: npt.NDArray, box: CellBox) -> Rgb:
        center_x, center_y = cell_center(box)
        pixel = pixels[center_y, center_x]
        return (int(pixel[0]), int(pixel[1]), int(pixel[2]))

    @staticmethod
    def _trim_trailing_background(cell_colors: List[Rgb]) -> List[Rgb]:
        last_content = -1
        for index, color in enumerate(cell_colors):
            if color != BACKGROUND_RGB:
                last_content = index
        return cell_colors[: last_content + 1]

    def _score_boxes(self, width: int, height: int) -> List[CellBox]:
        return tile_canvas(width, height, self._cell_size_pixels(width))

    @staticmethod
    def _cell_size_pixels(width: int) -> int:
        return max(1, round(width * CELL_SIZE_MM / A4_WIDTH_MM))

    @staticmethod
    def _canvas_pixels(dpi: int) -> Tuple[int, int]:
        width = round(A4_WIDTH_MM / MILLIMETERS_PER_INCH * dpi)
        height = round(A4_HEIGHT_MM / MILLIMETERS_PER_INCH * dpi)
        return (width, height)

    @staticmethod
    def _save(image: Image.Image, output_path: str, dpi: int) -> None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        image.save(output_path, format="PNG", dpi=(dpi, dpi))
