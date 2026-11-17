"""A lightweight local document retrieval package."""

from mini_rag_retriever.documents import DocumentChunk, load_documents, split_text
from mini_rag_retriever.retrieval import SearchResult, retrieve

__all__ = [
    "DocumentChunk",
    "SearchResult",
    "load_documents",
    "retrieve",
    "split_text",
]
