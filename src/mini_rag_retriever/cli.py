"""Command-line interface for Mini RAG Retriever."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from mini_rag_retriever.documents import load_documents
from mini_rag_retriever.retrieval import retrieve


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""

    parser = argparse.ArgumentParser(
        prog="mini-rag",
        description="Search local TXT and Markdown files using TF-IDF retrieval.",
    )
    parser.add_argument("query", help="question or keywords to search for")
    parser.add_argument(
        "--docs",
        default="examples/documents",
        help="directory containing .txt and .md files (default: examples/documents)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="maximum number of matching chunks to return (default: 3)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=120,
        help="number of words per chunk (default: 120)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=20,
        help="overlapping words between chunks (default: 20)",
    )
    parser.add_argument(
        "--output",
        help="optional JSON output file; results print to the terminal when omitted",
    )
    return parser


def main() -> int:
    """Load documents, run retrieval, and print or save JSON results."""

    parser = build_parser()
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero")

    try:
        chunks = load_documents(
            args.docs,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        results = retrieve(args.query, chunks, top_k=args.top_k)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    payload = {
        "query": args.query,
        "chunks_indexed": len(chunks),
        "results_count": len(results),
        "results": [asdict(result) for result in results],
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Found {len(results)} matching chunks. JSON saved to {output_path}")
    else:
        print(rendered)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
