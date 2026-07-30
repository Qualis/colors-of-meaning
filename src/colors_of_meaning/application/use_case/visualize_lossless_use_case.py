from typing import List, Tuple

from colors_of_meaning.domain.service.figure_renderer import FigureRenderer


class VisualizeLosslessUseCase:
    def __init__(self, figure_renderer: FigureRenderer) -> None:
        self.figure_renderer = figure_renderer

    def execute_gallery(
        self,
        page_paths: List[str],
        captions: List[str],
        title: str,
        output_path: str,
        columns: int = 12,
        max_tile_pixels: int = 200,
    ) -> None:
        self.figure_renderer.render_a4_gallery(page_paths, output_path, columns, captions, title, max_tile_pixels)

    def execute_comparison(self, panels: List[Tuple[str, str]], title: str, output_path: str) -> None:
        self.figure_renderer.render_image_comparison(panels, title, output_path)
