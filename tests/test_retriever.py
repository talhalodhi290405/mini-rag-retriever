from __future__ import annotations

from pathlib import Path

import pytest

from mini_rag_retriever.documents import DocumentChunk, load_documents, split_text
from mini_rag_retriever.retrieval import retrieve


def test_split_text_returns_no_chunks_for_empty_text() -> None:
    assert split_text("   \n  ") == []


def test_split_text_creates_expected_chunk_count_and_overlap() -> None:
    words = [f"word{index}" for index in range(11)]

    chunks = split_text(" ".join(words), chunk_size=5, chunk_overlap=2)

    assert len(chunks) == 3
    assert chunks[0].split() == words[0:5]
    assert chunks[1].split() == words[3:8]
    assert chunks[2].split() == words[6:11]


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [
        (0, 0),
        (5, -1),
        (5, 5),
        (5, 6),
    ],
)
def test_split_text_rejects_invalid_chunk_configuration(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    with pytest.raises(ValueError):
        split_text("some words to split", chunk_size, chunk_overlap)


def test_load_documents_reads_txt_and_markdown_recursively(tmp_path: Path) -> None:
    (tmp_path / "guide.md").write_text("agentic AI tools", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("Python retrieval systems", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "details.md").write_text("document search", encoding="utf-8")
    (tmp_path / "ignored.csv").write_text("not indexed", encoding="utf-8")

    chunks = load_documents(tmp_path, chunk_size=20, chunk_overlap=2)

    assert [chunk.source for chunk in chunks] == [
        "guide.md",
        "nested/details.md",
        "notes.txt",
    ]
    assert all(chunk.chunk_index == 0 for chunk in chunks)
    assert "not indexed" not in [chunk.text for chunk in chunks]


def test_load_documents_reports_missing_directory(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        load_documents(missing)


def test_retrieve_ranks_lexically_relevant_chunk_first() -> None:
    chunks = [
        DocumentChunk(
            source="python.md",
            chunk_index=0,
            text="Python is useful for scripting and data processing.",
        ),
        DocumentChunk(
            source="agents.md",
            chunk_index=0,
            text="Agentic AI systems use tools to complete multi-step tasks.",
        ),
        DocumentChunk(
            source="cooking.md",
            chunk_index=0,
            text="Bread recipes use flour, yeast, water, and salt.",
        ),
    ]

    results = retrieve("agentic AI tools", chunks, top_k=2)

    assert results
    assert results[0].source == "agents.md"
    assert results[0].score > 0
    assert len(results) <= 2


def test_retrieve_omits_chunks_without_lexical_overlap() -> None:
    chunks = [
        DocumentChunk(
            source="one.md",
            chunk_index=0,
            text="The database stores customer records.",
        ),
        DocumentChunk(
            source="two.md",
            chunk_index=0,
            text="A bicycle has two wheels and pedals.",
        ),
    ]

    results = retrieve("quantum mechanics", chunks)

    assert results == []


def test_retrieve_returns_empty_list_for_empty_corpus() -> None:
    assert retrieve("document search", []) == []


@pytest.mark.parametrize(
    ("query", "top_k"),
    [
        ("   ", 3),
        ("valid query", 0),
        ("valid query", -1),
    ],
)
def test_retrieve_rejects_invalid_search_arguments(query: str, top_k: int) -> None:
    chunks = [
        DocumentChunk(
            source="example.md",
            chunk_index=0,
            text="A sample document about retrieval.",
        )
    ]

    with pytest.raises(ValueError):
        retrieve(query, chunks, top_k=top_k)


def test_retrieve_uses_deterministic_order_for_equal_scores() -> None:
    chunks = [
        DocumentChunk(
            source="zeta.md",
            chunk_index=0,
            text="Python automation tools",
        ),
        DocumentChunk(
            source="alpha.md",
            chunk_index=0,
            text="Python automation tools",
        ),
    ]

    results = retrieve("Python automation", chunks, top_k=2)

    assert [result.source for result in results] == ["alpha.md", "zeta.md"]
