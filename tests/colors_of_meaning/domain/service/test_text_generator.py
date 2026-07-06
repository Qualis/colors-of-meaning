from typing import List, Optional

from assertpy import assert_that

from colors_of_meaning.domain.model.book_specification import BeatBrief, BookSpecification
from colors_of_meaning.domain.model.generated_book import TokenUsage
from colors_of_meaning.domain.service.text_generator import GenerationError, TextGenerator


class _StubTextGenerator(TextGenerator):
    def generate_outline(self, specification: BookSpecification) -> List[BeatBrief]:
        return [BeatBrief(index=0, title="Opening", synopsis="A calm morning.")]

    def generate_beat_prose(
        self, brief: BeatBrief, preceding_context: str, corrective_note: Optional[str] = None
    ) -> str:
        return "Prose."

    def consumed_tokens(self) -> TokenUsage:
        return TokenUsage(input_tokens=5, output_tokens=7)


class TestTextGenerator:
    def test_should_generate_outline_through_the_port(self) -> None:
        generator = _StubTextGenerator()

        outline = generator.generate_outline(BookSpecification(summary="A town.", num_beats=1, words_per_beat=100))

        assert_that(outline[0].title).is_equal_to("Opening")

    def test_should_generate_prose_through_the_port(self) -> None:
        generator = _StubTextGenerator()

        prose = generator.generate_beat_prose(BeatBrief(index=0, title="Opening", synopsis="s"), preceding_context="")

        assert_that(prose).is_equal_to("Prose.")

    def test_should_report_consumed_tokens_through_the_port(self) -> None:
        generator = _StubTextGenerator()

        assert_that(generator.consumed_tokens().total_tokens).is_equal_to(12)


class TestGenerationError:
    def test_should_be_an_exception(self) -> None:
        assert_that(issubclass(GenerationError, Exception)).is_true()
