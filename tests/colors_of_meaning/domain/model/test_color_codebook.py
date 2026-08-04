import numpy as np
import pytest
from colors_of_meaning.domain.model.lab_color import LabColor
from colors_of_meaning.domain.model.color_codebook import ColorCodebook


class TestColorCodebook:
    def test_should_create_codebook_with_valid_colors(self) -> None:
        colors = [
            LabColor(l=0.0, a=0.0, b=0.0),
            LabColor(l=50.0, a=0.0, b=0.0),
            LabColor(l=100.0, a=0.0, b=0.0),
        ]

        codebook = ColorCodebook(colors=colors, num_bins=3)

        assert len(codebook.colors) == 3
        assert codebook.num_bins == 3

    def test_should_raise_error_when_color_count_mismatches_num_bins(self) -> None:
        colors = [LabColor(l=0.0, a=0.0, b=0.0)]

        with pytest.raises(ValueError, match="Expected 3 colors"):
            ColorCodebook(colors=colors, num_bins=3)

    def test_should_quantize_color_to_nearest_bin(self) -> None:
        colors = [
            LabColor(l=0.0, a=0.0, b=0.0),
            LabColor(l=50.0, a=0.0, b=0.0),
            LabColor(l=100.0, a=0.0, b=0.0),
        ]
        codebook = ColorCodebook(colors=colors, num_bins=3)

        bin_index = codebook.quantize(LabColor(l=45.0, a=0.0, b=0.0))

        assert bin_index == 1

    def test_should_match_first_minimum_when_distances_tie(self) -> None:
        colors = [
            LabColor(l=40.0, a=0.0, b=0.0),
            LabColor(l=60.0, a=0.0, b=0.0),
        ]
        codebook = ColorCodebook(colors=colors, num_bins=2)

        bin_index = codebook.quantize(LabColor(l=50.0, a=0.0, b=0.0))

        assert bin_index == 0

    def test_should_quantize_a_batch_of_colors_to_their_nearest_bins(self) -> None:
        codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=8)
        colors = [codebook.colors[index] for index in (0, 100, 511)]

        assert list(codebook.quantize_batch(colors)) == [0, 100, 511]

    def test_should_match_the_single_color_path_when_quantizing_a_batch(self) -> None:
        codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=8)
        generator = np.random.default_rng(5)
        samples = generator.uniform([0.0, -128.0, -128.0], [100.0, 127.0, 127.0], size=(1500, 3))
        colors = [LabColor(l=float(row[0]), a=float(row[1]), b=float(row[2])) for row in samples]

        assert list(codebook.quantize_batch(colors)) == [codebook.quantize(color) for color in colors]

    def test_should_return_an_empty_result_when_quantizing_no_colors(self) -> None:
        codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=2)

        assert codebook.quantize_batch([]).shape == (0,)

    def test_should_match_first_minimum_when_batch_distances_tie(self) -> None:
        colors = [LabColor(l=40.0, a=0.0, b=0.0), LabColor(l=60.0, a=0.0, b=0.0)]
        codebook = ColorCodebook(colors=colors, num_bins=2)

        assert list(codebook.quantize_batch([LabColor(l=50.0, a=0.0, b=0.0)])) == [0]

    def test_should_quantize_identically_to_uniform_grid_baseline_when_color_in_range(self) -> None:
        codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=8)
        query = LabColor(l=37.0, a=11.0, b=-23.0)
        palette = np.array([color.to_tuple() for color in codebook.colors], dtype=np.float64)
        expected = int(np.argmin(np.sum((palette - np.array(query.to_tuple())) ** 2, axis=1)))

        bin_index = codebook.quantize(query)

        assert bin_index == expected

    def test_should_get_color_at_bin_index(self) -> None:
        colors = [
            LabColor(l=0.0, a=0.0, b=0.0),
            LabColor(l=50.0, a=0.0, b=0.0),
        ]
        codebook = ColorCodebook(colors=colors, num_bins=2)

        color = codebook.get_color(1)

        assert color.l == 50.0

    def test_should_raise_error_when_bin_index_is_out_of_range(self) -> None:
        colors = [LabColor(l=0.0, a=0.0, b=0.0)]
        codebook = ColorCodebook(colors=colors, num_bins=1)

        with pytest.raises(ValueError, match="bin_index must be in"):
            codebook.get_color(5)

    def test_should_create_uniform_grid_codebook(self) -> None:
        codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=16)

        assert codebook.num_bins == 4096
        assert len(codebook.colors) == 4096

    def test_should_raise_error_when_num_bins_is_not_positive(self) -> None:
        colors = []

        with pytest.raises(ValueError, match="num_bins must be positive"):
            ColorCodebook(colors=colors, num_bins=0)
