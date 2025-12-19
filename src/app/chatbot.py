from __future__ import annotations

import json
import re
import uuid
import os
from dataclasses import dataclass
from typing import List, Optional, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory  
from langchain.output_parsers import PydanticOutputParser

from .rag import (
    retrieve_job_chunks,
    format_chunks_for_prompt,
    extract_source_urls,     
    format_source_urls,      
)

from ..prompts.prompt import DIALOGUE_SYSTEM_PROMPT, EXTRACT_SYSTEM_PROMPT

load_dotenv()

DEBUG_STATE = os.getenv("DEBUG_STATE", "0") == "1"
DEBUG_STATE_IN_CHAT = os.getenv("DEBUG_STATE_IN_CHAT", "0") == "1"

# =========================
# 0) Pydantic 스키마 (6슬롯)
# =========================
SkillLevel = Literal["beginner", "intermediate", "advanced", "expert", "unknown"]


class ProjectItem(BaseModel):
    title: Optional[str] = Field(default=None, description="프로젝트 이름/주제")
    summary: Optional[str] = Field(default=None, description="프로젝트 한줄 요약")
    tech_stack: List[str] = Field(default_factory=list, description="사용 기술/도구")
    domain: Optional[str] = Field(default=None, description="도메인(제조/금융/헬스 등)")

    def is_meaningful(self) -> bool:
        t = (self.title or "").strip()
        s = (self.summary or "").strip()
        d = (self.domain or "").strip()
        tech = any((x or "").strip() for x in self.tech_stack)
        return bool(t or s or d or tech)


class LanguageSkill(BaseModel):
    name: str = Field(..., description="언어/기술 이름 (예: Python, SQL, Airflow)")
    level: SkillLevel = Field(default="unknown", description="숙련도")


class UserProfile(BaseModel):
    project_experience: List[ProjectItem] = Field(default_factory=list)
    project_role: List[str] = Field(default_factory=list)
    languages: List[LanguageSkill] = Field(default_factory=list)
    preferred_work: List[str] = Field(default_factory=list)
    interests: List[str] = Field(default_factory=list)
    major: Optional[str] = Field(default=None)


class ProfileUpdate(BaseModel):
    project_experience: Optional[List[ProjectItem]] = Field(default=None)
    project_role: Optional[List[str]] = Field(default=None)
    languages: Optional[List[LanguageSkill]] = Field(default=None)
    preferred_work: Optional[List[str]] = Field(default=None)
    interests: Optional[List[str]] = Field(default=None)
    major: Optional[str] = Field(default=None)


class ExtractionResult(BaseModel):
    profile_update: ProfileUpdate = Field(default_factory=ProfileUpdate)


def _dedup_str_list(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        x = (x or "").strip()
        if not x:
            continue
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def merge_profile(profile: UserProfile, update: ProfileUpdate) -> UserProfile:
    p = profile.model_copy(deep=True)

    if update.project_experience is not None:
        meaningful = [it for it in update.project_experience if it and it.is_meaningful()]
        if meaningful:
            existing = set()
            for it in p.project_experience:
                key = (
                    (it.title or "").strip().lower(),
                    (it.summary or "").strip().lower(),
                    ",".join(sorted([x.strip().lower() for x in it.tech_stack if (x or "").strip()])),
                )
                existing.add(key)
            for it in meaningful:
                key = (
                    (it.title or "").strip().lower(),
                    (it.summary or "").strip().lower(),
                    ",".join(sorted([x.strip().lower() for x in it.tech_stack if (x or "").strip()])),
                )
                if key != ("", "", "") and key in existing:
                    continue
                existing.add(key)
                p.project_experience.append(it)

    if update.project_role is not None and update.project_role:
        p.project_role = _dedup_str_list(p.project_role + update.project_role)

    if update.languages is not None and update.languages:
        order = {"unknown": 0, "beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}
        by_name = {ls.name.strip().lower(): ls for ls in p.languages if (ls.name or "").strip()}
        for ls in update.languages:
            k = (ls.name or "").strip().lower()
            if not k:
                continue
            if k not in by_name:
                by_name[k] = ls
            else:
                cur = by_name[k]
                if order.get(ls.level, 0) > order.get(cur.level, 0):
                    cur.level = ls.level
        p.languages = list(by_name.values())

    if update.preferred_work is not None and update.preferred_work:
        p.preferred_work = _dedup_str_list(p.preferred_work + update.preferred_work)

    if update.interests is not None and update.interests:
        p.interests = _dedup_str_list(p.interests + update.interests)

    if update.major is not None and update.major:
        if not p.major:
            p.major = update.major.strip()

    return p


def compute_filled_and_missing(profile: UserProfile) -> tuple[int, List[str]]:
    missing = []
    has_project = any(it.is_meaningful() for it in profile.project_experience)
    if not has_project:
        missing.append("project_experience")
    if not profile.project_role:
        missing.append("project_role")
    if not profile.languages:
        missing.append("languages")
    if not profile.preferred_work:
        missing.append("preferred_work")
    if not profile.interests:
        missing.append("interests")
    if not profile.major:
        missing.append("major")
    return 6 - len(missing), missing


# =========================
# 1) 세션 상태 저장소
# =========================
_store: dict[str, ChatMessageHistory] = {}
_profile_store: dict[str, UserProfile] = {}


@dataclass
class Flags:
    reco_consent: bool = False
    pending_reco_permission: bool = False

    # ✅ 갭분석(부족역량/로드맵) 동의/대기
    gap_consent: bool = False
    pending_gap_permission: bool = False

    turns_since_question: int = 99


_flag_store: dict[str, Flags] = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]


def get_profile(session_id: str) -> UserProfile:
    if session_id not in _profile_store:
        _profile_store[session_id] = UserProfile()
    return _profile_store[session_id]


def get_flags(session_id: str) -> Flags:
    if session_id not in _flag_store:
        _flag_store[session_id] = Flags()
    return _flag_store[session_id]


# =========================
# 2) intent / affirm 감지
# =========================
ASK_RECOMMEND_KEYWORDS = [
    "추천해줘", "추천해", "직무 추천", "직무 뭐가", "뭐가 맞아", "탑3", "top3", "결과 보여줘", "직무 뽑아줘"
]
ASK_GAP_KEYWORDS = [
    "부족", "역량", "갭", "gap", "로드맵", "공부", "학습", "준비", "뭘 공부", "어떻게 공부",
    "스킬", "기술 스택", "커리큘럼"
]
GREET_PATTERNS = [
    r"^안녕[!.~ ]*$",
    r"^안녕하세요[!.~ ]*$"
]


def detect_intent(text: str) -> str:
    t = (text or "").strip().lower()
    if re.match(r"^안녕", t):
        return "GREET"

    if any(re.match(p, t) for p in GREET_PATTERNS):
        return "GREET"
    if any(k in t for k in ASK_RECOMMEND_KEYWORDS):
        return "ASK_RECOMMEND"
    if any(k in (text or "") for k in ASK_GAP_KEYWORDS):
        return "ASK_GAP"
    return "CAREER"


def is_affirm(text: str) -> bool:
    t = (text or "").strip().lower()
    affirm_words = ["네", "예", "응", "ㅇㅇ", "좋아", "그래", "해주세요", "해줘", "추천해줘", "추천해"]
    if t in affirm_words:
        return True
    if "추천" in t and ("해" in t or "부탁" in t):
        return True
    if "로드맵" in t and ("해" in t or "부탁" in t):
        return True
    if "부족" in t and ("해" in t or "부탁" in t):
        return True
    return False


# =========================
# 3) 프로필 요약(허락 질문 전에 사용)
# =========================
def profile_brief(profile: UserProfile) -> str:
    parts = []

    if profile.major:
        parts.append(f"- 전공: {profile.major}")

    if profile.languages:
        langs = ", ".join([f"{x.name}({x.level})" for x in profile.languages[:4]])
        parts.append(f"- 사용해본 기술/언어: {langs}")

    if profile.interests:
        intr = ", ".join(profile.interests[:4])
        parts.append(f"- 관심 분야: {intr}")

    if profile.preferred_work:
        pref = ", ".join(profile.preferred_work[:4])
        parts.append(f"- 선호 업무/스타일: {pref}")

    meaningful_projects = [p for p in profile.project_experience if p.is_meaningful()]
    if meaningful_projects:
        p0 = meaningful_projects[0]
        label = p0.title or p0.summary or "프로젝트 경험"
        parts.append(f"- 프로젝트: {label}")

    if not parts:
        return "(아직 뚜렷하게 정리된 정보가 많지 않아요.)"

    return "\n".join(parts)


# =========================
# 4) 대화용 체인 + 추출용 체인
# =========================temp
chat_llm = ChatOpenAI(model="gpt-4o", temperature=0.6)

dialogue_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", DIALOGUE_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)
dialogue_chain = dialogue_prompt | chat_llm

runnable = RunnableWithMessageHistory(
    dialogue_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

extract_llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
extract_parser = PydanticOutputParser(pydantic_object=ExtractionResult)

extract_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", EXTRACT_SYSTEM_PROMPT),
        ("human", "user_message: {user_input}\ncurrent_profile_json: {current_profile_json}"),
    ]
)
extract_chain = extract_prompt | extract_llm


def _parse_extraction(raw: str) -> ExtractionResult:
    s = (raw or "").strip()
    if s.startswith("{") and s.endswith("}"):
        return extract_parser.parse(s)

    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return extract_parser.parse(s)
    return extract_parser.parse(m.group(0))


def _assistant_asked_permission_mode(mode: str) -> bool:
    return mode == "ASK_PERMISSION"


def _assistant_contains_question(text: str) -> bool:
    if not text:
        return False
    if "?" in text:
        return True
    return bool(re.search(r"(까요|실래요|인가요|어때요)\s*[.!~]?\s*$", text.strip()))


def _assistant_offered_gap(text: str) -> bool:
    """
    추천 직후에 “부족역량/로드맵 해드릴까요?” 같은 제안을 했는지 감지해서
    pending_gap_permission을 자동으로 켜기 위한 휴리스틱.
    """
    if not text:
        return False
    return bool(re.search(r"(부족|역량|로드맵|공부).*(정리해드릴까요|추천해드릴까요|해드릴까요|비교해드릴까요)", text))


# =========================
# 5) 정책(policy) 결정
# =========================
QUESTION_PRIORITY = ["interests", "preferred_work", "languages", "major", "project_experience", "project_role"]


def decide_mode(intent: str, filled_count: int, flags: Flags) -> str:
    if intent == "GREET":
        return "ASK_PERMISSION"  

    if intent == "ASK_RECOMMEND" and filled_count < 2:
        return "COUNSEL"   

    if filled_count >= 3 and flags.reco_consent:
        return "RECOMMEND"

    if filled_count >= 3 and (not flags.reco_consent) and (not flags.pending_reco_permission):
        return "ASK_PERMISSION"

    if intent == "ASK_RECOMMEND":
        return "ASK_RECOMMEND_FLOW"

    return "COUNSEL"



# =========================
# 6) 직무 선택 저장/감지
# =========================
_selected_role_store: dict[str, str] = {}

ROLE_KEYWORDS = {
    "backend": ["백엔드", "서버", "spring", "스프링", "java", "kotlin", "node", "fastapi", "django"],
    "data_engineer": ["데이터 엔지니어", "데이터엔지니어", "etl", "dw", "airflow", "spark", "스파크", "파이프라인"],
    "data_analyst": ["데이터 분석", "데이터분석", "bi", "대시보드", "리포트", "sql"],
    "ml_engineer": ["머신러닝", "딥러닝", "ai", "모델링", "ml"],
    "devops": ["devops", "sre", "쿠버네티스", "kubernetes", "docker", "ci/cd", "terraform", "테라폼", "aws"],
    "frontend": ["프론트", "frontend", "react", "vue", "next"],
    "mobile": ["안드로이드", "ios", "flutter", "react native", "앱 개발"],
}


def get_selected_role(session_id: str) -> str:
    return _selected_role_store.get(session_id, "")


def set_selected_role(session_id: str, role: str) -> None:
    _selected_role_store[session_id] = role


def detect_role_from_text(text: str) -> str:
    t = (text or "").lower()
    for role, keys in ROLE_KEYWORDS.items():
        if any(k.lower() in t for k in keys):
            return role
    return ""


# =========================
# 7) URL 추출(안정형)
# =========================


# =========================
# 8) 외부에서 사용할 함수
# =========================
def get_chat_response(
    session_id: str,
    user_input: str,
    role_name: str = "",
    retrieved_docs: str = "",
    selected_role: str = "",
    
) -> str:
    """
    - (1) 추출 체인으로 프로필 업데이트(조용히)
    - (2) filled_count 계산
    - (3) 추천 동의/갭분석 동의 상태 업데이트
    - (4) 필요할 때만 RAG(설명 or 갭분석)
    - (5) 대화 LLM 호출
    """

    profile = get_profile(session_id)
    flags = get_flags(session_id)


    intent = detect_intent(user_input)
    if not selected_role:
        selected_role = get_selected_role(session_id)

    mentioned = detect_role_from_text(user_input)
    if mentioned:
        selected_role = mentioned
        set_selected_role(session_id, selected_role)

    # 1) 추천 동의 처리
    if flags.pending_reco_permission and is_affirm(user_input):
        flags.reco_consent = True
        flags.pending_reco_permission = False

    # 2) ✅ 갭분석 동의 처리: role이 있을 때만 consent 확정
    if flags.pending_gap_permission and is_affirm(user_input):
        if selected_role:
            flags.gap_consent = True
            flags.pending_gap_permission = False
        else:
            # role이 비어있으면 갭분석을 바로 못 하니, 계속 대기 상태 유지
            flags.pending_gap_permission = True
            flags.gap_consent = False


    # 3) 프로필 추출(조용히)
    try:
        raw_extract = extract_chain.invoke(
            {
                "user_input": user_input,
                "current_profile_json": json.dumps(profile.model_dump(), ensure_ascii=False),
                "format_instructions": extract_parser.get_format_instructions(),
            }
        ).content
        extraction = _parse_extraction(raw_extract)
        profile = merge_profile(profile, extraction.profile_update)
        _profile_store[session_id] = profile
    except Exception:
        pass

    # 4) 상태 계산
    filled_count, missing = compute_filled_and_missing(profile)

    print(f"[DEBUG] session={session_id} intent={intent} filled_count={filled_count} missing={missing}", flush=True)

    mode = decide_mode(intent, filled_count, flags)
    if mode == "ASK_PERMISSION" and (not flags.pending_reco_permission):
        flags.pending_reco_permission = True

    if DEBUG_STATE:
        print(f"[DEBUG] intent={intent} mode={mode} filled_count={filled_count} missing={missing} selected_role={selected_role} reco={flags.reco_consent} gap={flags.gap_consent}")


    # 6) ✅ RAG 실행 조건
    # - 갭분석(ASK_GAP 또는 gap_consent) 이면 설명 트리거 없이도 RAG
    need_gap = (intent == "ASK_GAP") or flags.gap_consent

    # - 직무 설명/공고 기반 질문이면 RAG
    rag_triggers = ["뭐해", "무슨 일", "자세히", "설명", "요구", "자격", "우대", "공고", "스택", "기술", "알려줘", "궁금", "상세"]
    need_explain = any(k in user_input for k in rag_triggers)

    need_rag = need_gap or need_explain
    response_mode = "EXPLAIN" if (need_explain and selected_role) else mode

     # ✅ 프롬프트 변수 누락 방지: 항상 문자열로 유지
    retrieved_docs = retrieved_docs or ""
    source_urls = ""  # 프롬프트에 항상 주입

    if selected_role and need_rag:
        chunks = retrieve_job_chunks(
            role=selected_role,
            query=user_input.strip() if user_input.strip() else f"{selected_role} 직무 요구 역량/기술 스택",
            k=6,
            sections=["requirements", "preferred", "responsibilities"],
        )

        # 디버그 로그(원하면 유지)
        print(f"RAG HIT: role={selected_role}, chunks={len(chunks)}")

        retrieved_docs = format_chunks_for_prompt(chunks)

        # ✅ RetrievedChunk.url 기반으로 URL 추출 (rag.py 함수 사용)
        urls = extract_source_urls(chunks, max_urls=5)
        if (not urls) and retrieved_docs:
            urls = re.findall(r"- url:\s*(https?://\S+)", retrieved_docs)[:5]

        source_urls = format_source_urls(urls)

        source_urls = format_source_urls(urls)

    # 7) 대화용 LLM 호출
    vars_for_system = {
        "user_intent": intent if mode != "ASK_PERMISSION" else "CAREER",
        "filled_count": filled_count,
        "missing_slots": json.dumps(missing, ensure_ascii=False),
        "reco_consent": "true" if flags.reco_consent else "false",
        "pending_reco_permission": "true" if flags.pending_reco_permission else "false",

        # ✅ 갭분석 상태
        "gap_consent": "true" if flags.gap_consent else "false",
        "pending_gap_permission": "true" if flags.pending_gap_permission else "false",

        "profile_brief": profile_brief(profile),

        "role_name": role_name,
        "selected_role": selected_role,
        "retrieved_docs": retrieved_docs,

        # ✅ URL 근거 링크(프롬프트에서 [근거 링크] 섹션 출력)
        "source_urls": source_urls,
        "response_mode": response_mode,
    }

    response = runnable.invoke(
        {"input": user_input, **vars_for_system},
        config={"configurable": {"session_id": session_id}},
    )
    assistant_text = response.content

    # 8) pending 플래그 업데이트
    if _assistant_asked_permission_mode(mode):
        flags.pending_reco_permission = True
        flags.turns_since_question = 0
    else:
        if _assistant_contains_question(assistant_text):
            flags.turns_since_question = 0
        else:
            flags.turns_since_question += 1

    # ✅ 갭분석 제안을 assistant가 했으면 pending_gap_permission 켜기
    if _assistant_offered_gap(assistant_text) and (not flags.gap_consent):
        flags.pending_gap_permission = True

    _flag_store[session_id] = flags
    return assistant_text
    

def create_session_id() -> str:
    return str(uuid.uuid4())
