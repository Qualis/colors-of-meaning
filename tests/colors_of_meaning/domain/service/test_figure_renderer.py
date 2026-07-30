from typing import Any, Callable, Dict, List, Tuple, Type

import pytest

from colors_of_meaning.domain.service.figure_renderer import FigureRenderer

RENDER_METHOD_NAMES: Tuple[str, ...] = (
    "render_codebook_palette",
    "render_document_histograms",
    "render_tsne_projection",
    "render_confusion_matrix",
    "render_corpus_signatures",
    "render_rate_distortion",
    "render_narrative_arc",
    "render_a4_gallery",
    "render_image_comparison",
)


def _accept_anything(_self: Any, *_args: Any, **_kwargs: Any) -> None:
    return None


def _renderer_implementing_all_but(omitted_method_name: str) -> Type[FigureRenderer]:
    implementations: Dict[str, Callable[..., None]] = {
        name: _accept_anything for name in RENDER_METHOD_NAMES if name != omitted_method_name
    }
    return type("PartialFigureRenderer", (FigureRenderer,), implementations)


def _is_rejected_at_instantiation(renderer_type: Type[FigureRenderer]) -> bool:
    try:
        renderer_type()
    except TypeError:
        return True
    return False


def _method_names_rejected_when_omitted() -> List[str]:
    return [name for name in RENDER_METHOD_NAMES if _is_rejected_at_instantiation(_renderer_implementing_all_but(name))]


class TestFigureRenderer:
    def test_should_not_instantiate_abstract_class(self) -> None:
        with pytest.raises(TypeError):
            FigureRenderer()  # type: ignore

    def test_should_reject_a_renderer_when_any_render_method_is_omitted(self) -> None:
        assert _method_names_rejected_when_omitted() == list(RENDER_METHOD_NAMES)

    def test_should_allow_concrete_implementation(self) -> None:
        class ConcreteFigureRenderer(FigureRenderer):
            def render_codebook_palette(self, codebook, output_path):  # type: ignore
                pass

            def render_document_histograms(self, documents, labels, label_names, output_path, samples_per_class=2):  # type: ignore  # noqa: E501
                pass

            def render_tsne_projection(self, documents, labels, label_names, output_path):  # type: ignore
                pass

            def render_confusion_matrix(self, y_true, y_pred, label_names, output_path):  # type: ignore
                pass

            def render_corpus_signatures(self, documents, labels, label_names, codebook, output_path, top_colors=24):  # type: ignore  # noqa: E501
                pass

            def render_rate_distortion(self, frontier, output_path):  # type: ignore
                pass

            def render_narrative_arc(self, arc, output_path):  # type: ignore
                pass

            def render_a4_gallery(
                self, sheet_paths, output_path, columns=12, captions=None, title="", max_tile_pixels=None
            ):  # type: ignore  # noqa: E501
                pass

            def render_image_comparison(self, panels, title, output_path):  # type: ignore
                pass

        renderer = ConcreteFigureRenderer()

        assert isinstance(renderer, FigureRenderer)
