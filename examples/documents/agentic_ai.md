# Agentic AI Fundamentals

Agentic AI systems use an agent to pursue a goal through a sequence of decisions and actions. An agent commonly observes its environment, reasons about the next step, selects a tool, executes that tool, and evaluates the result.

Tool calling allows an AI model to request specific functions such as searching documents, calculating values, or retrieving information from an API. A tool should have a clear description, a constrained input schema, predictable output, and appropriate error handling.

Retrieval-augmented generation, often called RAG, combines information retrieval with language generation. A typical RAG pipeline loads source documents, splits them into smaller chunks, converts those chunks into searchable representations, retrieves relevant passages for a query, and provides the passages to a language model as context.

TF-IDF is a lexical information-retrieval technique. Term frequency measures how often a word appears in a document, while inverse document frequency reduces the weight of words that occur in many documents. Cosine similarity compares the resulting vectors and can rank text passages by their lexical relevance to a query.

A reliable agentic workflow should validate tool inputs, handle failures, preserve useful execution traces, and avoid claiming that an action succeeded before receiving confirmation. Tests should cover successful calls, invalid input, empty results, and error paths.
