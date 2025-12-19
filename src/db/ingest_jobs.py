# src/app/db/ingest_jobs.py
from __future__ import annotations
import shutil
import json
import re
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from dotenv import load_dotenv

load_dotenv()

# LangChain imports (버전 차이 대비)
try:
    from langchain_openai import OpenAIEmbeddings
except Exception as e:
    raise ImportError("langchain_openai가 필요합니다. pip install langchain-openai") from e

try:
    # 최신 권장
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    # 구버전 fallback
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

try:
    from langchain_core.documents import Document
except Exception:
    from langchain.schema import Document  # type: ignore


from langchain_chroma import Chroma 


BASE_DIR = Path(__file__).resolve().parent
JOBS_PATH = BASE_DIR / "jobs.json"
PERSIST_DIR = BASE_DIR / "chroma_jobs"
COLLECTION_NAME = "jobs_sections_v1"


# -------------------------
# 1) job_role 정규화
# -------------------------
# 너희 프로젝트에 맞게 계속 늘려가면 됨
ROLE_RULES: List[Tuple[str, str]] = [
    (r"(데이터\s*엔지니어|data\s*engineer|dw|etl|airflow|spark|hadoop)", "data_engineer"),
    (r"(데이터\s*분석|data\s*analyst|bi\s*analyst|analytics)", "data_analyst"),
    (r"(머신러닝|ml\s*engineer|ai\s*engineer|딥러닝)", "ml_engineer"),
    (r"(백엔드|backend|server|spring|java|kotlin|node\.?js|django|fastapi)", "backend"),
    (r"(프론트|frontend|react|vue|next\.?js)", "frontend"),
    (r"(앱\s*개발|android|ios|flutter|react\s*native)", "mobile"),
    (r"(devops|sre|kubernetes|docker|ci/cd|terraform|aws|gcp|azure)", "devops"),
    (r"(보안|security|pentest|취약점)", "security"),
    (r"(qa|테스트|quality)", "qa"),
    (r"(pm|po|기획|product)", "pm_po"),
    (r"(mes|스마트\s*팩토리|smart\s*factory|제조)", "manufacturing_it"),
]

CANONICAL_ORDER = [
    "data_engineer", "data_analyst", "ml_engineer", "backend", "frontend",
    "mobile", "devops", "security", "qa", "pm_po", "manufacturing_it"
]


def normalize_roles(job_role_raw: str, raw_text: str) -> List[str]:
    text = f"{job_role_raw}\n{raw_text}".lower()
    found = set()

    for pattern, canonical in ROLE_RULES:
        if re.search(pattern, text, flags=re.IGNORECASE):
            found.add(canonical)

    # 복합 라벨인데 아무것도 못 잡는 경우 fallback
    if not found:
        # 괄호/슬래시 기반으로 약하게 토큰화 후 키워드 매칭
        tokens = re.split(r"[()/,·\-\s]+", job_role_raw.lower())
        token_text = " ".join(tokens)
        for pattern, canonical in ROLE_RULES:
            if re.search(pattern, token_text, flags=re.IGNORECASE):
                found.add(canonical)

    # 그래도 없으면 "other"
    if not found:
        return ["other"]

    # 안정적인 순서
    ordered = [r for r in CANONICAL_ORDER if r in found]
    # 혹시 order에 없는 값이 있으면 뒤에 붙이기
    ordered += sorted(list(found - set(ordered)))
    return ordered


# -------------------------
# 2) 섹션 파싱/추출
# -------------------------
def extract_preferred_from_raw(raw: str) -> str:
    """
    raw_description에서 [우대 사항] 블록을 최대한 뽑아오기.
    없으면 빈 문자열.
    """
    if not raw:
        return ""

    # [우대 사항] ~ 다음 [ ] 또는 끝
    m = re.search(r"\[우대\s*사항\]\s*(.*?)(?=\n\[[^\]]+\]|\Z)", raw, flags=re.DOTALL)
    if not m:
        return ""
    return m.group(1).strip()


def build_job_id(job: Dict[str, Any]) -> str:
    url = (job.get("job_url") or "").strip()
    if url:
        return hashlib.md5(url.encode("utf-8")).hexdigest()
    # url 없으면 회사+role+raw로 대충
    base = f"{job.get('company','')}-{job.get('job_role','')}-{job.get('raw_description','')[:200]}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()


def clean_text(s: Optional[str]) -> str:
    if not s:
        return ""
    # 너무 지저분한 공백 정리
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


# -------------------------
# 3) Documents 생성
# -------------------------
def job_to_section_docs(job: Dict[str, Any]) -> List[Document]:
    job_role_raw = clean_text(job.get("job_role"))
    company = clean_text(job.get("company"))
    requirements = clean_text(job.get("requirements_text"))
    responsibilities = clean_text(job.get("responsibilities_text"))
    raw = clean_text(job.get("raw_description"))
    preferred = extract_preferred_from_raw(raw)
    url = clean_text(job.get("job_url"))

    job_id = build_job_id(job)
    roles = normalize_roles(job_role_raw, raw)

    roles_joined = "|".join(roles)

    base_meta = {
        "job_id": job_id,
        "company": company,
        "job_role_raw": job_role_raw,
        "roles_joined": roles_joined,   # ✅ 문자열로 저장 (Chroma 호환)
        "job_url": url,
    }
    docs: List[Document] = []

    def add(section: str, text: str):
        text = clean_text(text)
        if not text:
            return
        content = (
            f"[회사] {company}\n"
            f"[직무라벨] {job_role_raw}\n"
            f"[섹션] {section}\n\n"
            f"{text}"
        )
        docs.append(Document(page_content=content, metadata={**base_meta, "section": section}))

    add("responsibilities", responsibilities)
    add("requirements", requirements)
    add("preferred", preferred)
    # raw는 너무 길 수 있어 optional인데, 있으면 추가(검색 recall용)
    add("raw", raw)

    return docs


# -------------------------
# 4) Chunking + Chroma 저장
# -------------------------
def chunk_documents(docs: List[Document]) -> List[Document]:
    """
    섹션 단위 docs를 조금 더 잘게 쪼개서 검색 품질 상승.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", "•", "-", " ", ""],
    )
    out: List[Document] = []
    for d in docs:
        # 섹션별로 적당히 쪼개기
        chunks = splitter.split_text(d.page_content)
        for i, c in enumerate(chunks):
            meta = dict(d.metadata)
            meta["chunk_index"] = i
            out.append(Document(page_content=c, metadata=meta))
    return out


def load_jobs(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ingest() -> None:
    if not JOBS_PATH.exists():
        raise FileNotFoundError(f"jobs.json not found: {JOBS_PATH}")

    jobs = load_jobs(JOBS_PATH)

    # ✅ A) DB 완전 초기화(가장 안전)
    if PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR)
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    # ✅ B) jobs.json 중복 제거 (job_id 기준)
    seen = set()
    deduped_jobs = []
    dup_count = 0
    for job in jobs:
        jid = build_job_id(job)
        if jid in seen:
            dup_count += 1
            continue
        seen.add(jid)
        deduped_jobs.append(job)

    # 1) docs 생성
    section_docs: List[Document] = []
    for job in deduped_jobs:
        section_docs.extend(job_to_section_docs(job))

    # 2) chunk
    chunked_docs = chunk_documents(section_docs)

    embeddings = OpenAIEmbeddings()

    vs = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
    )

    ids = []
    for d in chunked_docs:
        jid = d.metadata.get("job_id", "")
        sec = d.metadata.get("section", "")
        idx = d.metadata.get("chunk_index", 0)
        ids.append(f"{jid}:{sec}:{idx}")

    # ✅ C) 이번 실행 내에서도 혹시 남아있을 중복 id 방지(안전장치)
    uniq = {}
    for doc, _id in zip(chunked_docs, ids):
        if _id not in uniq:
            uniq[_id] = doc

    vs.add_documents(list(uniq.values()), ids=list(uniq.keys()))

    try:
        vs.persist()
    except Exception:
        pass

    print("✅ Ingest done.")
    print(f"- jobs(raw): {len(jobs)}")
    print(f"- jobs(deduped): {len(deduped_jobs)} (skipped duplicates: {dup_count})")
    print(f"- section_docs: {len(section_docs)}")
    print(f"- chunked_docs: {len(chunked_docs)}")
    print(f"- persisted_docs: {len(uniq)}")
    print(f"- persist_dir: {PERSIST_DIR}")
    print(f"- collection: {COLLECTION_NAME}")


# -------------------------
# 5) Retriever 예시 (role filter)
# -------------------------
def get_retriever(role: str, k: int = 6, sections: Optional[List[str]] = None):
    """
    role: canonical role 예) "data_engineer"
    sections: ["requirements","responsibilities","preferred"] 등
    """
    embeddings = OpenAIEmbeddings()
    vs = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
    )

    # Chroma metadata filter: 배열 필터가 환경에 따라 다를 수 있어서,
    # safest는 job_roles를 문자열로 하나 더 만들어두는 건데(roles_joined),
    # 지금은 우선 raw filter로 시도.
    # 만약 배열 필터가 안 먹으면 -> roles_joined로 바꿔주면 됨(아래 참고).
    where: Dict[str, Any] = {}
    if sections:
        # Chroma where에서 $in 지원 여부가 구현/버전에 따라 다름
        # 안 되면 retriever 후에 section으로 post-filter 하자.
        where["section"] = {"$in": sections}

    # job_roles 배열에 role이 포함되는 필터가 안 먹는 경우가 많아서,
    # 일단 후보를 더 뽑고 post-filter하는 방식도 추천.
    retriever = vs.as_retriever(search_kwargs={"k": max(k * 3, 10), "filter": where})
    return retriever


def demo_query():
    role = "backend"
    query = "요즘 백엔드 공고에서 요구하는 기술 스택이 뭐야? 특히 AWS랑 쿠버네티스 관련"
    retriever = get_retriever(role=role, k=6, sections=["requirements", "preferred", "responsibilities"])
    docs = retriever.invoke(query)

    # post-filter: role 포함만 남기기
    filtered = []
    for d in docs:
        roles_joined = (d.metadata.get("roles_joined") or "")
        roles = roles_joined.split("|") if roles_joined else []
        if role in roles:
            filtered.append(d)
    filtered = filtered[:6]


    print(f"\n=== DEMO ({role}) ===")
    for i, d in enumerate(filtered, 1):
        print(f"\n[{i}] {d.metadata.get('company')} | {d.metadata.get('job_role_raw')} | {d.metadata.get('section')}")
        print(d.page_content[:300], "...")

print("[INGEST persist_dir]", PERSIST_DIR)

if __name__ == "__main__":
    ingest()
    # demo_query()
