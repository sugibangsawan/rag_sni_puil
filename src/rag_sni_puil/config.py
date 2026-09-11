from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

ROOT = Path(__file__).resolve().parents[2]
PDF_SOURCE_DIR = ROOT / "PDF_SOURCE"
PERSIST_DIRECTORY = ROOT / "chroma_db"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_BATCH_SIZE = 100
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-5.6-luna"
COLLECTION_NAME = "sni-puil-openai"


def load_env() -> None:
    load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing from the environment or .env")


def embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def llm() -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL, temperature=0)


def vector_store() -> Chroma:
    load_env()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings(),
        persist_directory=str(PERSIST_DIRECTORY),
        collection_metadata={"hnsw:space": "cosine"},
    )
