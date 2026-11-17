# Mini RAG Retriever

A lightweight, dependency-free Python document retrieval project. It loads local Markdown and plain-text files, splits them into overlapping chunks, ranks those chunks against a search query using TF-IDF and cosine similarity, and returns source-aware JSON results.

> **Scope:** This project implements the retrieval stage of a Retrieval-Augmented Generation (RAG) workflow. It does not call an LLM or generate answers. It demonstrates document ingestion, chunking, lexical ranking, and traceable retrieval in a small, easy-to-run codebase.

## Features

- Recursively load `.md` and `.txt` files from a local directory.
- Split documents into configurable word-based chunks.
- Preserve context between adjacent chunks with configurable overlap.
- Rank chunks with TF-IDF weighting and cosine similarity.
- Return source file, chunk index, relevance score, and matching text.
- Produce deterministic rankings when scores are tied.
- Print JSON results in the terminal or save them to a file.
- Use only the Python standard library at runtime.

## Requirements

- Python 3.10 or newer
- pip

There are no runtime third-party dependencies. The optional development dependency is pytest.

## Installation

Clone the repository:

```bash
git clone https://github.com/talhalodhi290405/mini-rag-retriever.git
cd mini-rag-retriever
```

Create and activate a virtual environment.

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the package in editable mode with the test dependency:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Quick start

The repository includes a sample Markdown document in `examples/documents/`.

Search it from the project root:

```bash
mini-rag "agentic AI tools"
```

Search a different directory and return up to five matching chunks:

```bash
mini-rag "retrieval augmented generation" --docs ./my-documents --top-k 5
```

Save results as JSON:

```bash
mini-rag "cosine similarity" --output reports/search-results.json
```

Display all available options:

```bash
mini-rag --help
```

### CLI options

| Argument | Description | Default |
| --- | --- | --- |
| `query` | Search query or question | Required |
| `--docs` | Directory containing Markdown and text documents | `examples/documents` |
| `--top-k` | Maximum number of matching chunks to return | `3` |
| `--chunk-size` | Maximum words per chunk | `120` |
| `--chunk-overlap` | Number of words shared between adjacent chunks | `20` |
| `--output` | Optional path for saving JSON results | Print to terminal |

The chunk overlap must be non-negative and smaller than the chunk size. The top-k value must be greater than zero.

## How it works

The retrieval pipeline has four stages:

1. **Document ingestion** — recursively find supported `.md` and `.txt` files and read them as UTF-8 text.
2. **Chunking** — split each document into word-based segments. The default chunk size is 120 words with a 20-word overlap.
3. **Vectorization and scoring** — tokenize text, calculate TF-IDF weights, and compare the query vector with each chunk using cosine similarity.
4. **Ranking and output** — sort matches by descending similarity, use source and chunk index as deterministic tie-breakers, and return the top-k results as JSON.

### TF-IDF and cosine similarity

Term Frequency–Inverse Document Frequency (TF-IDF) gives more weight to terms that are frequent in a chunk but less common across the indexed chunks. Cosine similarity compares the query and chunk vectors, producing a score used to rank lexical matches.

Chunks with no shared query terms are omitted from the results. Scores are relative lexical relevance signals, not calibrated probabilities or factual confidence values.

### Architecture

```text
Local .md / .txt files
        |
        v
   Document loader
        |
        v
 Word-based chunking
        |
        v
  TF-IDF vectors  <---- Search query
        |                    |
        +---- Cosine --------+
                 |
                 v
          Ranked chunks
                 |
                 v
             JSON output
```

## Example output

The following is illustrative. Actual scores and matching passages depend on the documents and query being searched.

```json
{
  "query": "agentic AI tools",
  "chunks_indexed": 1,
  "results_count": 1,
  "results": [
    {
      "source": "agentic_ai.md",
      "chunk_index": 0,
      "score": 0.426731,
      "text": "Agentic AI systems use an agent to pursue a goal through a sequence of decisions and actions..."
    }
  ]
}
```

## Run tests

Install the development dependency and run the test suite:

```bash
python -m pytest -v
```

The tests cover chunk generation and overlap, invalid chunk settings, recursive document loading, unsupported file filtering, missing directories, retrieval ranking, no-match behavior, empty corpora, argument validation, and deterministic tie-breaking.

## Project structure

```text
mini-rag-retriever/
├── examples/
│   └── documents/
│       └── agentic_ai.md
├── src/
│   └── mini_rag_retriever/
│       ├── __init__.py
│       ├── cli.py
│       ├── documents.py
│       └── retrieval.py
├── tests/
│   └── test_retriever.py
├── .gitignore
├── LICENSE
├── README.md
└── pyproject.toml
```

## Limitations and future improvements

This is an intentionally small lexical retrieval prototype. It currently:

- Uses exact token overlap rather than semantic embeddings.
- Recalculates TF-IDF vectors for each query instead of persisting an index.
- Processes files and ranks chunks synchronously in memory.
- Supports Markdown and plain text only.
- Retrieves passages but does not generate natural-language answers.

Possible extensions include persisted indexes, embedding-based semantic search, hybrid retrieval, metadata filters, PDF ingestion, and an optional LLM answer-generation layer with source citations.

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
