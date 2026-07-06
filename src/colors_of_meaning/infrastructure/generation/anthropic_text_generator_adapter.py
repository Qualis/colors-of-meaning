from typing import List, Optional, cast

import anthropic
from anthropic.types import Message, OutputConfigParam, TextBlockParam, ThinkingConfigParam
from anthropic.types.parsed_message import ParsedMessage
from pydantic import BaseModel

from colors_of_meaning.domain.model.book_specification import BeatBrief, BookSpecification
from colors_of_meaning.domain.model.generated_book import TokenUsage
from colors_of_meaning.domain.service.text_generator import GenerationError, TextGenerator
from colors_of_meaning.shared.synesthetic_config import GenerationConfig

_ADAPTIVE_THINKING: ThinkingConfigParam = {"type": "adaptive"}


class _BeatBriefSchema(BaseModel):
    index: int
    title: str
    synopsis: str


class _OutlineSchema(BaseModel):
    beats: List[_BeatBriefSchema]


class AnthropicTextGeneratorAdapter(TextGenerator):
    def __init__(self, config: GenerationConfig, client: Optional[anthropic.Anthropic] = None) -> None:
        self._config = config
        self._client = client if client is not None else anthropic.Anthropic()
        self._input_tokens = 0
        self._output_tokens = 0
        self._bible = ""
        self._words_per_beat = config.words_per_beat

    def generate_outline(self, specification: BookSpecification) -> List[BeatBrief]:
        response = self._request_outline(specification)
        self._accumulate(response.usage.input_tokens, response.usage.output_tokens)
        _guard_refusal(response.stop_reason)
        outline = self._parse_outline(response.parsed_output, specification)
        self._bible = self._book_bible(specification, outline)
        self._words_per_beat = specification.words_per_beat
        return outline

    def generate_beat_prose(
        self,
        brief: BeatBrief,
        preceding_context: str,
        corrective_note: Optional[str] = None,
    ) -> str:
        message = self._request_prose(brief, preceding_context, corrective_note)
        self._accumulate(message.usage.input_tokens, message.usage.output_tokens)
        _guard_refusal(message.stop_reason)
        return _extract_text(message)

    def consumed_tokens(self) -> TokenUsage:
        return TokenUsage(input_tokens=self._input_tokens, output_tokens=self._output_tokens)

    def _request_outline(self, specification: BookSpecification) -> ParsedMessage[_OutlineSchema]:
        try:
            return self._client.messages.parse(
                model=self._config.model,
                max_tokens=self._config.outline_max_tokens,
                thinking=_ADAPTIVE_THINKING,
                output_config=self._output_config(),
                system=[_cached_text_block(self._premise_text(specification))],
                messages=[{"role": "user", "content": self._outline_instruction(specification)}],
                output_format=_OutlineSchema,
            )
        except anthropic.APIError as error:
            raise GenerationError(f"outline request failed: {error}") from error

    def _request_prose(self, brief: BeatBrief, preceding_context: str, corrective_note: Optional[str]) -> Message:
        try:
            with self._client.messages.stream(
                model=self._config.model,
                max_tokens=self._config.prose_max_tokens,
                thinking=_ADAPTIVE_THINKING,
                output_config=self._output_config(),
                system=[_cached_text_block(self._bible)],
                messages=[
                    {
                        "role": "user",
                        "content": self._prose_instruction(brief, preceding_context, corrective_note),
                    }
                ],
            ) as stream:
                return stream.get_final_message()
        except anthropic.APIError as error:
            raise GenerationError(f"prose request failed: {error}") from error

    def _output_config(self) -> OutputConfigParam:
        return cast(OutputConfigParam, {"effort": self._config.effort})

    def _accumulate(self, input_tokens: int, output_tokens: int) -> None:
        self._input_tokens += input_tokens
        self._output_tokens += output_tokens

    def _parse_outline(self, parsed: Optional[_OutlineSchema], specification: BookSpecification) -> List[BeatBrief]:
        if parsed is None:
            raise GenerationError("outline response did not contain structured output")
        return [self._to_brief(beat, specification) for beat in parsed.beats]

    @staticmethod
    def _to_brief(beat: _BeatBriefSchema, specification: BookSpecification) -> BeatBrief:
        return BeatBrief(
            index=beat.index,
            title=beat.title,
            synopsis=beat.synopsis,
            target_tone=specification.tone_target_for(beat.index),
        )

    def _premise_text(self, specification: BookSpecification) -> str:
        tone = specification.tone or "unspecified"
        return f"BOOK PREMISE\n{specification.summary}\n\nTARGET TONE: {tone}"

    def _outline_instruction(self, specification: BookSpecification) -> str:
        return (
            f"Plan a {specification.num_beats}-beat outline for a book built from the premise above. "
            f"Each beat becomes one chapter of roughly {specification.words_per_beat} words. "
            "Return the beats in reading order; give every beat a title and a one-paragraph synopsis."
        )

    def _book_bible(self, specification: BookSpecification, outline: List[BeatBrief]) -> str:
        lines = [self._premise_text(specification), "", "OUTLINE"]
        lines.extend(f"{brief.index + 1}. {brief.title}: {brief.synopsis}" for brief in outline)
        return "\n".join(lines)

    def _prose_instruction(self, brief: BeatBrief, preceding_context: str, corrective_note: Optional[str]) -> str:
        sections = [
            f'Write chapter {brief.index + 1}, titled "{brief.title}".',
            f"Beat synopsis: {brief.synopsis}",
            f"Write approximately {self._words_per_beat} words of polished prose that continues the book.",
        ]
        if preceding_context:
            sections.append(f"PRECEDING CHAPTERS SO FAR:\n{preceding_context}")
        if brief.target_tone is not None:
            tone = brief.target_tone
            sections.append(
                "Aim for this tone signature — " f"lightness {tone.lightness}, chroma {tone.chroma}, hue {tone.hue}."
            )
        if corrective_note:
            sections.append(corrective_note)
        return "\n\n".join(sections)


def _guard_refusal(stop_reason: Optional[str]) -> None:
    if stop_reason == "refusal":
        raise GenerationError("generation was refused by the safety classifier before content was produced")


def _extract_text(message: Message) -> str:
    chunks = [block.text for block in message.content if block.type == "text"]
    if not chunks:
        raise GenerationError("generation response contained no text content")
    return "".join(chunks)


def _cached_text_block(text: str) -> TextBlockParam:
    return {"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}
