# src/db/test_retrieval.py
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 너희 ingest 출력 기준
BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIR = BASE_DIR / "chroma_jobs"
COLLECTION_NAME = "jobs_sections_v1"


def roles_contains(metadata: dict, role: str) -> bool:
    roles_joined = (metadata.get("roles_joined") or "").strip()
    roles = roles_joined.split("|") if roles_joined else []
    return role in roles


def main():
    embeddings = OpenAIEmbeddings()

    vs = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
    )

    role = "backend"  # "data_engineer" / "devops" 등으로 바꿔 테스트 가능
    query = "요즘 백엔드 공고에서 자주 요구하는 기술 스택과 키워드를 알려줘. AWS, 쿠버네티스, MSA 관점으로."
    k = 12  # 넉넉하게 가져온 다음 post-filter

    docs = vs.similarity_search(query, k=k)

    # role post-filter
    filtered = [d for d in docs if roles_contains(d.metadata, role)]

    print("=== RETRIEVAL TEST ===")
    print(f"- persist_dir: {PERSIST_DIR}")
    print(f"- collection: {COLLECTION_NAME}")
    print(f"- query: {query}")
    print(f"- raw hits: {len(docs)}")
    print(f"- role='{role}' hits: {len(filtered)}")

    for i, d in enumerate(filtered[:6], 1):
        company = d.metadata.get("company")
        section = d.metadata.get("section")
        job_role_raw = d.metadata.get("job_role_raw")
        url = d.metadata.get("job_url")
        snippet = d.page_content[:350].replace("\n", " ")
        print(f"\n[{i}] {company} | {job_role_raw} | section={section}")
        print(f"    {snippet} ...")
        print(f"    url: {url}")


if __name__ == "__main__":
    main()
