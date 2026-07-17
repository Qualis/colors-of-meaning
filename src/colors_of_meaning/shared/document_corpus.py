from pathlib import Path
from typing import List, Tuple


def strip_gutenberg_boilerplate(text: str) -> str:
    start = text.find("*** START OF")
    start = text.find("\n", start) + 1 if start != -1 else 0
    end = text.find("*** END OF")
    end = end if end != -1 else len(text)
    return text[start:end]


def extract_paragraphs(text: str, min_chars: int) -> List[str]:
    paragraphs: List[str] = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        joined = " ".join(line.strip() for line in block.split("\n") if line.strip())
        if len(joined) >= min_chars:
            paragraphs.append(joined)
    return paragraphs


def parse_author_work(path: Path) -> Tuple[str, str]:
    return path.parent.name, path.stem


def discover_author_works(root: Path) -> List[Path]:
    if not root.is_dir():
        return []
    return sorted(root.glob("*/*.txt"), key=lambda path: (path.parent.name, path.name))
