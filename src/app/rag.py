from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

DEFAULT_COLLECTION = "jobs_sections_v1"
DEFAULT_SECTIONS = ["requirements", "responsibilities", "preferred"]


@dataclass
class RetrievedChunk:
    company: str
    job_role_raw: str
    section: str
    url: str
    text: str


def _get_persist_dir() -> Path:
    # src/app/rag.py -> parents[1] == src
    return Path(__file__).resolve().parents[1] / "db" / "chroma_jobs"


def _roles_contains(meta: dict, role: str) -> bool:
    roles_joined = (meta.get("roles_joined") or "").strip()
    roles = roles_joined.split("|") if roles_joined else []
    return role in roles


def get_vectorstore(collection_name: str = DEFAULT_COLLECTION) -> Chroma:
    persist_dir = _get_persist_dir()
    embeddings = OpenAIEmbeddings()
    return Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )


def retrieve_job_chunks(
    role: str,
    query: str,
    k: int = 6,
    sections: Optional[List[str]] = None,
    collection_name: str = DEFAULT_COLLECTION,
) -> List[RetrievedChunk]:
    """
    role: canonical role 예) "backend", "data_engineer"
    query: 사용자 질문
    """
    sections = sections or DEFAULT_SECTIONS
    vs = get_vectorstore(collection_name=collection_name)

    raw_docs = vs.similarity_search(query, k=max(k * 4, 20))

    out: List[RetrievedChunk] = []
    for d in raw_docs:
        meta = d.metadata or {}
        sec = (meta.get("section") or "").strip()

        if sec and sec not in sections:
            continue
        if not _roles_contains(meta, role):
            continue

        out.append(
            RetrievedChunk(
                company=str(meta.get("company") or ""),
                job_role_raw=str(meta.get("job_role_raw") or ""),
                section=sec,
                url=str(meta.get("job_url") or ""),
                text=d.page_content,
            )
        )
        if len(out) >= k:
            break

    return out


def format_chunks_for_prompt(chunks: List[RetrievedChunk], max_chars_each: int = 650) -> str:
    if not chunks:
        return ""

    lines = []
    for i, c in enumerate(chunks, 1):
        excerpt = c.text.strip().replace("\n", " ")
        if len(excerpt) > max_chars_each:
            excerpt = excerpt[:max_chars_each] + "…"
        lines.append(
            f"[근거 {i}] 회사: {c.company} | 라벨: {c.job_role_raw} | 섹션: {c.section}\n"
            f"- url: {c.url}\n"
            f"- 내용: {excerpt}"
        )
    return "\n\n".join(lines)


#url만 따로 뽑기.
def extract_source_urls(chunks: List[RetrievedChunk], max_urls: int = 5) -> List[str]:
    seen = set()
    out: List[str] = []
    for c in chunks:
        u = (c.url or "").strip()
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(u)
        if len(out) >= max_urls:
            break
    return out
def format_source_urls(urls: List[str]) -> str:
    if not urls:
        return ""
    return "\n".join([f"- {u}" for u in urls])
