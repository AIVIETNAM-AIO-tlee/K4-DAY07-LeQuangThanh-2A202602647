from __future__ import annotations

from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self._store = store
        self._llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        """
        Retrieve relevant chunks from store, construct prompt with context, and call LLM.
        """
        if self._store.get_collection_size() == 0:
            return "No relevant information found in the knowledge base."

        chunks = self._store.search(question, top_k=top_k)
        if not chunks:
            return "No relevant information found in the knowledge base."

        context_blocks = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("metadata", {}).get("source", chunk.get("id", f"doc{i}"))
            text = chunk.get("content", chunk.get("chunk", ""))
            context_blocks.append(f"[{i}] (Source: {source})\n{text}")

        context = "\n\n".join(context_blocks)
        prompt = (
            "Answer the question based only on the following context. "
            "Cite sources using [1], [2], etc., when possible. "
            "If the answer cannot be found in the context, state that clearly.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        return self._llm_fn(prompt)