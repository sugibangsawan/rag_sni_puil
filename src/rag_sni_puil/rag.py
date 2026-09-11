from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from rag_sni_puil.config import LLM_MODEL, llm, vector_store

SYSTEM_PROMPT = """Anda adalah asisten untuk SNI PUIL 2011 dan spesifikasi kabel Sutrado.
Jawab hanya berdasarkan kutipan dokumen yang diberikan.
Jika jawaban tidak ada di konteks, katakan bahwa itu tidak ditemukan di dokumen.
Sebutkan nama dokumen dan nomor halaman jika tersedia."""


def ask(
    question: str,
    *,
    k: int = 6,
    history: list[dict[str, str]] | None = None,
    store: Chroma | None = None,
) -> str:
    store = store or vector_store()
    docs = store.similarity_search(question, k=k)
    if not docs:
        return "Tidak ada konteks di vector store. Jalankan ingest dulu."

    context_parts = []
    for doc in docs:
        source = Path(str(doc.metadata.get("source", ""))).name or "dokumen"
        page = doc.metadata.get("page")
        page_label = (
            f"halaman {page + 1}" if isinstance(page, int) else "halaman tidak diketahui"
        )
        context_parts.append(f"[{source}, {page_label}]\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    messages: list = [
        SystemMessage(content=f"{SYSTEM_PROMPT}\n\nKonteks:\n{context}"),
    ]
    for turn in (history or [])[-10:]:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        elif turn["role"] == "assistant":
            messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=question))

    response = llm().invoke(messages)
    print(f"(model={LLM_MODEL}, retrieved={len(docs)})", flush=True)
    return str(response.content)
