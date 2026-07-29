import colorsys
import os
from math import ceil
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from PIL import Image

from colors_of_meaning.domain.service.cell_geometry import a4_canvas_pixels, cell_center, tile_grid
from colors_of_meaning.domain.service.data_image_codec import DataImageCodec
from colors_of_meaning.domain.service.data_payload import (
    HEADER_SIZE,
    compress_text,
    decompress_text,
    frame_page,
    payload_length,
    reassemble,
    split_into_pages,
)

DEFAULT_CELL_SIZE = 4
DEFAULT_MAX_PAGES = 1000
PALETTE_SIZE = 256
GOLDEN_RATIO_CONJUGATE = 0.61803398875

GridShape = Tuple[int, int, int, int]


def _build_data_palette() -> List[int]:
    channels: List[int] = []
    for index in range(PALETTE_SIZE):
        hue = (index * GOLDEN_RATIO_CONJUGATE) % 1.0
        saturation = 0.55 + 0.35 * (index % 3) / 2.0
        value = 0.45 + 0.5 * (index % 2)
        red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
        channels.extend([round(red * 255), round(green * 255), round(blue * 255)])
    return channels


DATA_PALETTE = _build_data_palette()


class PillowDataImageCodec(DataImageCodec):
    def __init__(self, cell_size: int = DEFAULT_CELL_SIZE, max_pages: int = DEFAULT_MAX_PAGES) -> None:
        if cell_size <= 0:
            raise ValueError(f"cell_size must be positive, got {cell_size}")
        self.cell_size = cell_size
        self.max_pages = max_pages

    def encode(self, text: str, output_path: str, dpi: int) -> List[str]:
        columns, max_rows, width, height = self._grid_shape(dpi)
        capacity = columns * max_rows - HEADER_SIZE
        chunks = split_into_pages(compress_text(text), capacity)
        page_count = len(chunks)
        if page_count > self.max_pages:
            raise ValueError(f"document needs {page_count} pages, exceeding the cap of {self.max_pages}")
        paths = _page_paths(output_path, page_count)
        for index, (chunk, path) in enumerate(zip(chunks, paths, strict=True)):
            self._paint_page(frame_page(chunk, index, page_count), columns, width, height, path, dpi)
        return paths

    def decode(self, input_paths: List[str]) -> str:
        pages = [self._read_page(path) for path in input_paths]
        return decompress_text(reassemble(pages))

    def _grid_shape(self, dpi: int) -> GridShape:
        width, height = a4_canvas_pixels(dpi)
        columns = width // self.cell_size
        max_rows = height // self.cell_size
        if columns < HEADER_SIZE:
            raise ValueError(
                f"cell_size {self.cell_size} leaves {columns} columns; the header needs at least {HEADER_SIZE}"
            )
        return (columns, max_rows, width, height)

    def _paint_page(self, framed: bytes, columns: int, width: int, height: int, path: str, dpi: int) -> None:
        rows = ceil(len(framed) / columns)
        padded = framed + bytes(columns * rows - len(framed))
        grid = Image.frombytes("P", (columns, rows), padded)
        grid.putpalette(DATA_PALETTE)
        canvas = grid.resize((width, height), Image.Resampling.NEAREST)
        canvas.putpalette(DATA_PALETTE)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        canvas.save(path, format="PNG", dpi=(dpi, dpi))

    def _read_page(self, path: str) -> bytes:
        with Image.open(path) as opened:
            indices = np.asarray(opened)
        columns = indices.shape[1] // self.cell_size
        return self._sample_cells(indices, columns, self._recover_rows(indices, columns))

    def _recover_rows(self, indices: npt.NDArray, columns: int) -> int:
        header_boxes = tile_grid(indices.shape[1], indices.shape[0], columns, 1)[:HEADER_SIZE]
        header = bytes(int(indices[0, cell_center(box)[0]]) for box in header_boxes)
        return ceil((HEADER_SIZE + payload_length(header)) / columns)

    @staticmethod
    def _sample_cells(indices: npt.NDArray, columns: int, rows: int) -> bytes:
        centers = [cell_center(box) for box in tile_grid(indices.shape[1], indices.shape[0], columns, rows)]
        horizontal = np.fromiter((center[0] for center in centers), dtype=np.int64, count=len(centers))
        vertical = np.fromiter((center[1] for center in centers), dtype=np.int64, count=len(centers))
        return bytes(indices[vertical, horizontal].astype(np.uint8).tobytes())


def _page_paths(output_path: str, page_count: int) -> List[str]:
    if page_count == 1:
        return [output_path]
    base, extension = os.path.splitext(output_path)
    return [f"{base}_p{index:02d}{extension}" for index in range(1, page_count + 1)]
