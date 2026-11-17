"""Document loading and overlapping text chunking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md"}


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """A text segment with its source file and zero-based chunk index."""

    source: str
    chunk_index: int
    text: str


def split_text(
    text: str,
    chunk_size: int = 120,
    chunk_overlap: int = 20,
) -> list[str]:
    """Split text into word-based chunks with optional overlap.

    ``chunk_size`` and ``chunk_overlap`` are measured in whitespace-separated
    words. The overlap preserves context between adjacent chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    step = chunk_size - chunk_overlap
    chunks: list[str] = []
    start = 0
    while start < len(words):
        chunk_words = words[start : start + chunk_size]
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks


def load_documents(
    directory: str | Path,
    chunk_size: int = 120,
    chunk_overlap: int = 20,
) -> list[DocumentChunk]:
    """Load supported Markdown and text files recursively from a directory."""

    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Document directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Document path is not a directory: {root}")

    chunks: list[DocumentChunk] = []
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    for file_path in files:
        relative_source = file_path.relative_to(root).as_posix()
        content = file_path.read_text(encoding="utf-8", errors="replace")
        for index, chunk in enumerate(
            split_text(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ):
            chunks.append(
                DocumentChunk(
                    source=relative_source,
                    chunk_index=index,
                    text=chunk,
                )
            )

    return chunks
