from __future__ import annotations

import time

from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_sni_puil.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBED_BATCH_SIZE,
    PDF_SOURCE_DIR,
    PERSIST_DIRECTORY,
    embeddings,
    load_env,
    vector_store,
)


def ingest(*, reset: bool = False) -> Chroma:
    load_env()
    pdf_paths = sorted(PDF_SOURCE_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {PDF_SOURCE_DIR}")

    pages: list[Document] = []
    for pdf_path in pdf_paths:
        print(f"Loading {pdf_path.name}...", flush=True)
        loaded = PyPDFLoader(str(pdf_path)).load()
        print(f"Loaded {len(loaded)} pages from {pdf_path.name}", flush=True)
        pages.extend(loaded)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_documents(pages)
    ids = [_chunk_id(chunk, index) for index, chunk in enumerate(chunks)]
    print(f"Split into {len(chunks)} chunks", flush=True)

    PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)
    store = vector_store()
    if reset:
        store.reset_collection()
        pending_chunks = chunks
        pending_ids = ids
    else:
        pending_chunks, pending_ids = _pending(store, chunks, ids)
        print(
            f"Already indexed {len(chunks) - len(pending_chunks)} chunks",
            flush=True,
        )

    total = len(pending_chunks)
    if total == 0:
        print("Vector store is already up to date", flush=True)
        return store

    print(
        f"Embedding with OpenAI {embeddings().model} into collection {COLLECTION_NAME}",
        flush=True,
    )
    for start in range(0, total, EMBED_BATCH_SIZE):
        batch = pending_chunks[start : start + EMBED_BATCH_SIZE]
        batch_ids = pending_ids[start : start + EMBED_BATCH_SIZE]
        _add_with_retry(store, batch, batch_ids)
        done = min(start + EMBED_BATCH_SIZE, total)
        print(f"Embedded {done}/{total} remaining", flush=True)

    stored = _stored_count(store)
    print(
        f"Saved {stored} vectors to {PERSIST_DIRECTORY} (collection={COLLECTION_NAME})",
        flush=True,
    )
    return store


def _pending(
    store: Chroma,
    chunks: list[Document],
    ids: list[str],
) -> tuple[list[Document], list[str]]:
    stored = store.get(include=["documents"])
    existing_ids = set(stored.get("ids") or [])
    existing_texts = set(stored.get("documents") or [])
    pending_chunks: list[Document] = []
    pending_ids: list[str] = []
    for chunk, chunk_id in zip(chunks, ids, strict=True):
        if chunk_id in existing_ids or chunk.page_content in existing_texts:
            continue
        pending_chunks.append(chunk)
        pending_ids.append(chunk_id)
    return pending_chunks, pending_ids


def _stored_count(store: Chroma) -> int:
    return len(store.get(include=[]).get("ids") or [])


def _chunk_id(chunk: Document, index: int) -> str:
    source = Path(str(chunk.metadata.get("source", "doc"))).stem
    page = chunk.metadata.get("page", "na")
    start_index = chunk.metadata.get("start_index", index)
    if source == "sni_puil_2011_pdf":
        return f"sni-puil-{page}-{start_index}-{index}"
    return f"{source}-{page}-{start_index}-{index}"


def _add_with_retry(
    store: Chroma,
    batch: list[Document],
    ids: list[str],
    retries: int = 6,
) -> None:
    delay = 5.0
    for attempt in range(retries):
        try:
            store.add_documents(batch, ids=ids)
            return
        except Exception as exc:
            if attempt == retries - 1:
                raise
            print(f"Embed batch failed ({exc!r}); retrying in {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 60)
