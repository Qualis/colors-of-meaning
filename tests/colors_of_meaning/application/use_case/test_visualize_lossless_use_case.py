from unittest.mock import Mock

from colors_of_meaning.application.use_case.visualize_lossless_use_case import (
    VisualizeLosslessUseCase,
)


class TestVisualizeLosslessUseCaseGallery:
    def test_should_delegate_gallery_rendering_to_figure_renderer(self) -> None:
        mock_renderer = Mock()

        use_case = VisualizeLosslessUseCase(mock_renderer)
        use_case.execute_gallery(["a.png"], ["austen/emma"], "Barcodes", "/output/gallery.png")

        mock_renderer.render_a4_gallery.assert_called_once_with(
            ["a.png"], "/output/gallery.png", 12, ["austen/emma"], "Barcodes", 200
        )

    def test_should_pass_custom_columns_to_the_renderer(self) -> None:
        mock_renderer = Mock()

        use_case = VisualizeLosslessUseCase(mock_renderer)
        use_case.execute_gallery(["a.png"], ["austen/emma"], "Barcodes", "/output/gallery.png", columns=4)

        assert mock_renderer.render_a4_gallery.call_args.args[2] == 4


class TestVisualizeLosslessUseCaseComparison:
    def test_should_delegate_comparison_rendering_to_figure_renderer(self) -> None:
        mock_renderer = Mock()
        panels = [("sheet.png", "meaning"), ("barcode.png", "bytes")]

        use_case = VisualizeLosslessUseCase(mock_renderer)
        use_case.execute_comparison(panels, "Two ways", "/output/two_ways.png")

        mock_renderer.render_image_comparison.assert_called_once_with(panels, "Two ways", "/output/two_ways.png")
