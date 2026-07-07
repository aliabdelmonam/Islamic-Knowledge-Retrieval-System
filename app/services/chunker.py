"""
Token-based parent-child chunker.
Splits each parent document into token-sized child chunks using the embedding
model's own tokenizer for accurate Arabic token counting.
"""
from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def split_by_tokens(
    tokenizer: AutoTokenizer,
    text: str,
    chunk_size: int = 350,
    chunk_overlap: int = 32,
) -> list[str]:
    """Split *text* into token-count-limited chunks with overlap."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks: list[str] = []
    step = max(chunk_size - chunk_overlap, 1)
    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i : i + chunk_size]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        if chunk_text.strip():
            chunks.append(chunk_text.strip())
    return chunks


def build_child_chunks(
    raw_docs: list[Document],
    model_name: str,
    chunk_size: int = 350,
    chunk_overlap: int = 32,
) -> tuple[list[Document], list[Document]]:
    """
    Build token-based child chunks from parent documents.

    Returns
    -------
    child_docs : list[Document]
        Child chunks to index into the vector store.
    parent_store : list[Document]
        Original parent documents (indexed by parent_id).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    parent_store: list[Document] = []
    child_docs: list[Document] = []

    for parent_id, parent_doc in enumerate(raw_docs):
        parent_store.append(parent_doc)
        for local_idx, chunk_text in enumerate(
            split_by_tokens(tokenizer, parent_doc.page_content, chunk_size, chunk_overlap)
        ):
            global_idx = len(child_docs)
            child_meta = {
                "idx": global_idx,
                "parent_id": parent_id,
                "chunk_index": local_idx,
                **parent_doc.metadata,
            }
            child_docs.append(Document(page_content=chunk_text, metadata=child_meta))

    logger.info(
        "Chunking complete — parents: %d | child chunks: %d",
        len(parent_store),
        len(child_docs),
    )

    # sanity check
    assert all(
        c.metadata["idx"] == i for i, c in enumerate(child_docs)
    ), "chunk idx/position mismatch!"

    return child_docs, parent_store
