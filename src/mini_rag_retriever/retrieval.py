"""TF-IDF cosine-similarity retrieval for small local document collections."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from mini_rag_retriever.documents import DocumentChunk

_TOKEN_PATTERN = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A ranked matching chunk returned by the retriever."""

    source: str
    chunk_index: int
    score: float
    text: str


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.casefold())


def _tfidf_vectors(
    texts: list[str],
) -> tuple[list[dict[str, float]], Counter[str]]:
    """Create sparse TF-IDF vectors and return corpus document frequencies."""

    token_lists = [_tokenize(text) for text in texts]
    document_frequency: Counter[str] = Counter()
    for tokens in token_lists:
        document_frequency.update(set(tokens))

    document_count = len(texts)
    vectors: list[dict[str, float]] = []
    for tokens in token_lists:
        frequencies = Counter(tokens)
        vector: dict[str, float] = {}
        for token, frequency in frequencies.items():
            inverse_frequency = math.log(
                (document_count + 1) / (document_frequency[token] + 1)
            ) + 1.0
            vector[token] = (1.0 + math.log(frequency)) * inverse_frequency
        vectors.append(vector)

    return vectors, document_frequency


def _query_vector(
    query: str,
    document_frequency: Counter[str],
    document_count: int,
) -> dict[str, float]:
    """Weight query terms with the same inverse-document-frequency values."""

    frequencies = Counter(_tokenize(query))
    vector: dict[str, float] = {}
    for token, frequency in frequencies.items():
        inverse_frequency = math.log(
            (document_count + 1) / (document_frequency[token] + 1)
        ) + 1.0
        vector[token] = (1.0 + math.log(frequency)) * inverse_frequency
    return vector


def _cosine_similarity(
    first: dict[str, float],
    second: dict[str, float],
) -> float:
    """Calculate cosine similarity between two sparse vectors."""

    if not first or not second:
        return 0.0

    dot_product = sum(value * second.get(term, 0.0) for term, value in first.items())
    first_norm = math.sqrt(sum(value * value for value in first.values()))
    second_norm = math.sqrt(sum(value * value for value in second.values()))
    if first_norm == 0.0 or second_norm == 0.0:
        return 0.0
    return dot_product / (first_norm * second_norm)


def retrieve(
    query: str,
    chunks: Iterable[DocumentChunk],
    top_k: int = 3,
) -> list[SearchResult]:
    """Rank document chunks by TF-IDF cosine similarity to a query.

    Results with no lexical overlap are omitted. Ties are resolved by source
    path and then chunk index to keep output deterministic.
    """

    if not query.strip():
        raise ValueError("query cannot be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    chunk_list = list(chunks)
    if not chunk_list:
        return []

    document_vectors, document_frequency = _tfidf_vectors(
        [chunk.text for chunk in chunk_list]
    )
    query_vector = _query_vector(query, document_frequency, len(chunk_list))

    ranked: list[SearchResult] = []
    for chunk, vector in zip(chunk_list, document_vectors, strict=True):
        score = _cosine_similarity(query_vector, vector)
        if score > 0.0:
            ranked.append(
                SearchResult(
                    source=chunk.source,
                    chunk_index=chunk.chunk_index,
                    score=round(score, 6),
                    text=chunk.text,
                )
            )

    ranked.sort(key=lambda item: (-item.score, item.source, item.chunk_index))
    return ranked[:top_k]
