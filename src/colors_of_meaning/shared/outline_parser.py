import re
from typing import List, Tuple

from colors_of_meaning.domain.model.narrative_arc import NarrativeBeat

_HEADING_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s+(.*\S)\s*$")
_BLANK_LINE_PATTERN = re.compile(r"\n\s*\n")


def parse_outline(text: str, min_chars: int) -> List[NarrativeBeat]:
    sections = _split_into_sections(text)
    kept = [(title, body) for title, body in sections if len(body) >= min_chars]
    return [NarrativeBeat(index=index, title=title, text=body) for index, (title, body) in enumerate(kept)]


def _split_into_sections(text: str) -> List[Tuple[str, str]]:
    if _has_heading(text):
        return _split_by_headings(text)
    return _split_by_blank_lines(text)


def _has_heading(text: str) -> bool:
    return any(_HEADING_PATTERN.match(line) for line in text.splitlines())


def _heading_lines(lines: List[str]) -> List[Tuple[int, str]]:
    matches: List[Tuple[int, str]] = []
    for index, line in enumerate(lines):
        match = _HEADING_PATTERN.match(line)
        if match:
            matches.append((index, match.group(1).strip()))
    return matches


def _split_by_headings(text: str) -> List[Tuple[str, str]]:
    lines = text.splitlines()
    headings = _heading_lines(lines)
    ends = [start for start, _ in headings][1:] + [len(lines)]
    return [(title, "\n".join(lines[start + 1 : end]).strip()) for (start, title), end in zip(headings, ends)]


def _split_by_blank_lines(text: str) -> List[Tuple[str, str]]:
    blocks = _BLANK_LINE_PATTERN.split(text.strip())
    return [("", block.strip()) for block in blocks if block.strip()]
