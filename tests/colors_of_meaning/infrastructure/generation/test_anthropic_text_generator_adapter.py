from types import SimpleNamespace
from typing import List, Optional
from unittest.mock import MagicMock, patch

import httpx
import pytest

import anthropic

from colors_of_meaning.domain.model.book_specification import (
    BeatBrief,
    BookSpecification,
    ToneTarget,
)
from colors_of_meaning.domain.service.text_generator import GenerationError
from colors_of_meaning.infrastructure.generation.anthropic_text_generator_adapter import (
    AnthropicTextGeneratorAdapter,
    _OutlineSchema,
)
from colors_of_meaning.shared.synesthetic_config import GenerationConfig

_MODULE = "colors_of_meaning.infrastructure.generation.anthropic_text_generator_adapter"
_SPEC = BookSpecification(summary="A quiet town.", num_beats=2, words_per_beat=300)
_TONED_SPEC = BookSpecification(
    summary="A quiet town.",
    num_beats=1,
    words_per_beat=300,
    tone="sombre",
    beat_tone_targets=[ToneTarget(lightness=20.0)],
)


def _usage(input_tokens: int, output_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)


def _outline_beat(index: int, title: str, synopsis: str) -> SimpleNamespace:
    return SimpleNamespace(index=index, title=title, synopsis=synopsis)


def _parsed(
    beats: Optional[List[SimpleNamespace]] = None,
    stop_reason: str = "end_turn",
    input_tokens: int = 5,
    output_tokens: int = 20,
) -> SimpleNamespace:
    parsed_output = None if beats is None else SimpleNamespace(beats=beats)
    return SimpleNamespace(
        usage=_usage(input_tokens, output_tokens),
        stop_reason=stop_reason,
        parsed_output=parsed_output,
    )


def _text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _message(
    content: Optional[List[SimpleNamespace]] = None,
    stop_reason: str = "end_turn",
    input_tokens: int = 10,
    output_tokens: int = 50,
) -> SimpleNamespace:
    blocks = content if content is not None else [_text_block("Chapter text.")]
    return SimpleNamespace(
        usage=_usage(input_tokens, output_tokens),
        stop_reason=stop_reason,
        content=blocks,
    )


def _client(
    parsed: Optional[SimpleNamespace] = None,
    message: Optional[SimpleNamespace] = None,
    parse_side_effect: Optional[BaseException] = None,
    stream_side_effect: Optional[BaseException] = None,
) -> MagicMock:
    client = MagicMock()
    if parse_side_effect is not None:
        client.messages.parse.side_effect = parse_side_effect
    else:
        client.messages.parse.return_value = (
            parsed if parsed is not None else _parsed([_outline_beat(0, "Opening", "s")])
        )
    if stream_side_effect is not None:
        client.messages.stream.side_effect = stream_side_effect
    else:
        client.messages.stream.return_value.__enter__.return_value.get_final_message.return_value = (
            message if message is not None else _message()
        )
    return client


def _adapter(client: MagicMock) -> AnthropicTextGeneratorAdapter:
    return AnthropicTextGeneratorAdapter(GenerationConfig(), client=client)


def _brief(target_tone: Optional[ToneTarget] = None) -> BeatBrief:
    return BeatBrief(index=0, title="Opening", synopsis="A calm morning.", target_tone=target_tone)


def _rate_limit_error() -> anthropic.RateLimitError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.RateLimitError("rate limited", response=httpx.Response(429, request=request), body=None)


def _api_status_error() -> anthropic.APIStatusError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIStatusError("boom", response=httpx.Response(500, request=request), body=None)


def _connection_error() -> anthropic.APIConnectionError:
    return anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))


class TestOutlineRequest:
    def test_should_request_the_configured_model(self) -> None:
        client = _client()

        _adapter(client).generate_outline(_SPEC)

        assert client.messages.parse.call_args.kwargs["model"] == "claude-opus-4-8"

    def test_should_request_adaptive_thinking(self) -> None:
        client = _client()

        _adapter(client).generate_outline(_SPEC)

        assert client.messages.parse.call_args.kwargs["thinking"] == {"type": "adaptive"}

    def test_should_request_high_effort(self) -> None:
        client = _client()

        _adapter(client).generate_outline(_SPEC)

        assert client.messages.parse.call_args.kwargs["output_config"] == {"effort": "high"}

    def test_should_request_the_structured_outline_schema(self) -> None:
        client = _client()

        _adapter(client).generate_outline(_SPEC)

        assert client.messages.parse.call_args.kwargs["output_format"] is _OutlineSchema

    def test_should_prompt_cache_the_premise_system_block(self) -> None:
        client = _client()

        _adapter(client).generate_outline(_SPEC)

        assert client.messages.parse.call_args.kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}

    def test_should_return_beat_briefs_from_the_structured_output(self) -> None:
        client = _client(parsed=_parsed([_outline_beat(0, "Opening", "A calm morning.")]))

        outline = _adapter(client).generate_outline(_SPEC)

        assert outline[0].title == "Opening"

    def test_should_attach_tone_targets_to_the_briefs(self) -> None:
        client = _client(parsed=_parsed([_outline_beat(0, "Opening", "A calm morning.")]))

        outline = _adapter(client).generate_outline(_TONED_SPEC)

        assert outline[0].target_tone is not None


class TestProseRequest:
    def test_should_return_the_streamed_message_text(self) -> None:
        client = _client(message=_message([_text_block("Chapter prose.")]))

        prose = _adapter(client).generate_beat_prose(_brief(), preceding_context="")

        assert prose == "Chapter prose."

    def test_should_stream_prose_with_the_configured_model(self) -> None:
        client = _client()

        _adapter(client).generate_beat_prose(_brief(), preceding_context="")

        assert client.messages.stream.call_args.kwargs["model"] == "claude-opus-4-8"

    def test_should_prompt_cache_the_bible_system_block(self) -> None:
        client = _client()
        adapter = _adapter(client)
        adapter.generate_outline(_SPEC)

        adapter.generate_beat_prose(_brief(), preceding_context="")

        assert client.messages.stream.call_args.kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}

    def test_should_carry_the_outline_into_the_cached_bible(self) -> None:
        client = _client(parsed=_parsed([_outline_beat(0, "Opening", "A calm morning.")]))
        adapter = _adapter(client)
        adapter.generate_outline(_SPEC)

        adapter.generate_beat_prose(_brief(), preceding_context="")

        assert "OUTLINE" in client.messages.stream.call_args.kwargs["system"][0]["text"]

    def test_should_include_the_corrective_note_in_the_prompt(self) -> None:
        client = _client()

        _adapter(client).generate_beat_prose(_brief(), preceding_context="", corrective_note="stay sombre")

        assert "stay sombre" in client.messages.stream.call_args.kwargs["messages"][0]["content"]

    def test_should_include_preceding_context_in_the_prompt(self) -> None:
        client = _client()

        _adapter(client).generate_beat_prose(_brief(), preceding_context="Earlier chapters.")

        assert "PRECEDING CHAPTERS" in client.messages.stream.call_args.kwargs["messages"][0]["content"]

    def test_should_include_target_tone_guidance_in_the_prompt(self) -> None:
        client = _client()

        _adapter(client).generate_beat_prose(_brief(ToneTarget(lightness=15.0)), preceding_context="")

        assert "tone signature" in client.messages.stream.call_args.kwargs["messages"][0]["content"]

    def test_should_extract_text_skipping_non_text_blocks(self) -> None:
        message = _message([SimpleNamespace(type="thinking", thinking="hmm"), _text_block("Real prose.")])
        client = _client(message=message)

        prose = _adapter(client).generate_beat_prose(_brief(), preceding_context="")

        assert prose == "Real prose."


class TestTokenAccounting:
    def test_should_accumulate_tokens_across_outline_and_prose(self) -> None:
        client = _client(
            parsed=_parsed([_outline_beat(0, "Opening", "s")], input_tokens=5, output_tokens=20),
            message=_message(input_tokens=10, output_tokens=50),
        )
        adapter = _adapter(client)
        adapter.generate_outline(_SPEC)
        adapter.generate_beat_prose(_brief(), preceding_context="")

        assert adapter.consumed_tokens().total_tokens == 85


class TestRefusalAndErrors:
    def test_should_raise_when_the_outline_is_refused(self) -> None:
        client = _client(parsed=_parsed(stop_reason="refusal"))

        with pytest.raises(GenerationError):
            _adapter(client).generate_outline(_SPEC)

    def test_should_raise_when_prose_is_refused(self) -> None:
        client = _client(message=_message(stop_reason="refusal"))

        with pytest.raises(GenerationError):
            _adapter(client).generate_beat_prose(_brief(), preceding_context="")

    def test_should_raise_when_the_outline_has_no_structured_output(self) -> None:
        client = _client(parsed=_parsed(beats=None))

        with pytest.raises(GenerationError):
            _adapter(client).generate_outline(_SPEC)

    def test_should_raise_when_prose_has_no_text_content(self) -> None:
        client = _client(message=_message([SimpleNamespace(type="thinking", thinking="hmm")]))

        with pytest.raises(GenerationError):
            _adapter(client).generate_beat_prose(_brief(), preceding_context="")

    def test_should_map_rate_limit_error_to_generation_error(self) -> None:
        client = _client(parse_side_effect=_rate_limit_error())

        with pytest.raises(GenerationError):
            _adapter(client).generate_outline(_SPEC)

    def test_should_map_api_status_error_to_generation_error(self) -> None:
        client = _client(parse_side_effect=_api_status_error())

        with pytest.raises(GenerationError):
            _adapter(client).generate_outline(_SPEC)

    def test_should_map_connection_error_to_generation_error(self) -> None:
        client = _client(stream_side_effect=_connection_error())

        with pytest.raises(GenerationError):
            _adapter(client).generate_beat_prose(_brief(), preceding_context="")


class TestClientConstruction:
    @patch(f"{_MODULE}.anthropic.Anthropic")
    def test_should_build_the_client_from_environment_when_none_is_injected(self, mock_anthropic: MagicMock) -> None:
        adapter = AnthropicTextGeneratorAdapter(GenerationConfig())

        assert adapter._client is mock_anthropic.return_value
