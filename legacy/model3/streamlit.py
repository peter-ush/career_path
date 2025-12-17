# streamlit.py
import os
import json
import re
from typing import Optional

import streamlit as st

from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from src.app.schemas import (
    UserProfile,
    ExtractionResult,
    merge_profile,
    compute_filled_and_missing,
)
from dotenv import load_dotenv

load_dotenv()


# -------------------------
# Prompts (2-pass)
# -------------------------
DIALOGUE_SYSTEM_PROMPT = """\
너는 한국어로 대화하는 IT 진로/직무 상담 챗봇이다.
사용자가 '면접/설문'처럼 느끼지 않게 자연스럽게 상담해야 한다.

[대화 톤/구조(강제)]
- 매 턴 기본 구조는 E-S-V 이다.
  E(공감/받아주기) 1문장
  S(요약/정리) 1문장: 사용자가 말한 것만 요약(상상 금지)
  V(가치 제공) 2~4문장: 방향 제시/선택지/짧은 전략
- Q(질문)는 policy.ask_question == true일 때만 "딱 1개" 한다.
  질문은 가볍고 열린 질문으로, 예시는 최대 2개까지만.
  절대 여러 질문을 한꺼번에 하지 마라.

[추천 규칙(강제)]
- 사용자가 명시적으로 추천을 요구한 경우(user_intent=ASK_RECOMMEND)에만
  직무 후보(Top3/탐색용)를 '목록 형태'로 제시할 수 있다.
- 그 외(user_intent=GREET/CAREER)에는 추천 목록을 먼저 던지지 말 것.
  대신 상담(방향 잡기/기준 세우기/다음 액션) 중심으로 말한다.

[현재 상태]
- user_intent: {user_intent}  # GREET / CAREER / ASK_RECOMMEND
- filled_count: {filled_count}
- missing_slots: {missing_slots}
- current_profile_json: {current_profile_json}

[policy]
- ask_question: {ask_question}  # true/false
- question_focus: {question_focus}  # 어떤 주제로 1개 질문할지 (없으면 "none")
- reco_mode: {reco_mode}  # NONE / EXPLORE / TOP3_CANDIDATES / TOP3_FINAL
  * reco_mode는 ASK_RECOMMEND일 때만 NONE이 아닐 수 있음

출력은 자연어 텍스트만. (JSON 출력 금지)
"""

EXTRACT_SYSTEM_PROMPT = """\
너는 사용자 메시지에서 '명시적으로 말한 정보만' 구조화해서 추출하는 정보추출기다.

[추출 대상 슬롯(6개)]
- project_experience (ProjectItem): title/summary/tech_stack/domain 중 하나라도 있으면 추출 가능
- project_role (list[str])
- languages (list[LanguageSkill]): name + level(unknown/beginner/intermediate/advanced/expert)
- preferred_work (list[str])
- interests (list[str])
- major (str)

[중요 규칙]
- 사용자가 말하지 않은 정보는 절대 추측해서 채우지 마라.
- 애매하면 None으로 둬라.
- user_message만 근거로 삼아라. assistant 메시지에서 유추하지 마라.

아래 스키마를 만족하는 JSON만 출력하라. JSON 밖 텍스트 금지.

{format_instructions}
"""

# -------------------------
# Intent + Policy
# -------------------------
ASK_RECOMMEND_KEYWORDS = [
    "추천해줘", "추천해", "직무 추천", "직무 뭐가", "뭐가 맞아", "탑3", "top3", "결과 보여줘", "직무 뽑아줘"
]
GREET_PATTERNS = [
    r"^안녕[!.~ ]*$",
    r"^안녕하세요[!.~ ]*$",
    r"^(ㅎㅇ|하이|hi|hello|hey)[!.~ ]*$",
]


def detect_intent(text: str) -> str:
    t = (text or "").strip().lower()
    if any(re.match(p, t) for p in GREET_PATTERNS):
        return "GREET"
    if any(k in t for k in ASK_RECOMMEND_KEYWORDS):
        return "ASK_RECOMMEND"
    return "CAREER"


QUESTION_PRIORITY = ["interests", "preferred_work", "major", "languages", "project_experience", "project_role"]


def pick_question_focus(missing_slots: list[str]) -> str:
    for k in QUESTION_PRIORITY:
        if k in missing_slots:
            return k
    return "none"


def compute_policy(intent: str, filled_count: int, missing_slots: list[str], turns_since_question: int) -> dict:
    policy = {"ask_question": False, "question_focus": "none", "reco_mode": "NONE"}

    if intent == "GREET":
        policy["ask_question"] = True
        policy["question_focus"] = "고민"
        return policy

    if intent != "ASK_RECOMMEND":
        # CAREER: 추천은 안 함. 질문은 2~3턴에 1번 정도.
        if turns_since_question >= 2:
            policy["ask_question"] = True
            policy["question_focus"] = pick_question_focus(missing_slots)
        return policy

    # ASK_RECOMMEND: 추천 요청이 온 경우에만 추천 모드 활성화
    if filled_count == 0:
        policy["reco_mode"] = "EXPLORE"
        policy["ask_question"] = True
        policy["question_focus"] = pick_question_focus(missing_slots)
    elif filled_count == 1:
        policy["reco_mode"] = "TOP3_CANDIDATES"
        policy["ask_question"] = True
        policy["question_focus"] = pick_question_focus(missing_slots)
    elif filled_count >= 2:
        policy["reco_mode"] = "TOP3_FINAL"
        policy["ask_question"] = turns_since_question >= 1
        policy["question_focus"] = "선택/우선순위"

    return policy


# -------------------------
# Extraction parsing helper
# -------------------------
def extract_json_block(s: str) -> Optional[str]:
    if not s:
        return None
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    return m.group(0) if m else None


def parse_extraction(parser: PydanticOutputParser, raw: str) -> ExtractionResult:
    try:
        return parser.parse(raw)
    except Exception:
        block = extract_json_block(raw)
        if not block:
            raise
        return parser.parse(block)


# -------------------------
# Streamlit App
# -------------------------
st.set_page_config(page_title="Career Chatbot MVP", page_icon="💬", layout="wide")
st.title("💬 Career Chatbot MVP (2-pass: 상담 대화 + 조용한 정보추출)")

with st.sidebar:
    st.subheader("설정")
    api_key = st.text_input(
        "OPENAI_API_KEY (환경변수가 있으면 비워도 됨)",
        type="password",
        value="",
        help="로컬에서 환경변수 OPENAI_API_KEY를 이미 설정했으면 비워도 됩니다.",
    )
    show_debug = st.toggle("디버그 보기", value=True)
    show_profile = st.toggle("프로필 JSON 보기", value=False)
    if st.button("대화 초기화"):
        st.session_state.clear()
        st.rerun()

# API Key 처리
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

# 세션 상태 초기화
if "profile" not in st.session_state:
    st.session_state.profile = UserProfile()
if "history" not in st.session_state:
    st.session_state.history = []  # langchain messages
if "chat" not in st.session_state:
    st.session_state.chat = []  # display messages: {"role": "user"/"assistant", "content": "..."}
if "turns_since_question" not in st.session_state:
    st.session_state.turns_since_question = 99

# LLM 리소스 캐시
@st.cache_resource
def get_llms():
    chat_llm = ChatOpenAI(model="gpt-5.0", temperature=0.6)
    extract_llm = ChatOpenAI(model="gpt-5.0", temperature=0.0)
    return chat_llm, extract_llm


chat_llm, extract_llm = get_llms()
extract_parser = PydanticOutputParser(pydantic_object=ExtractionResult)

dialogue_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", DIALOGUE_SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{user_input}"),
    ]
)

extract_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", EXTRACT_SYSTEM_PROMPT),
        ("human", "user_message: {user_input}\n\ncurrent_profile_json: {current_profile_json}"),
    ]
)

# 기존 채팅 렌더
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 입력
user_input = st.chat_input("메시지를 입력하세요…")
if user_input:
    # 화면에 유저 메시지
    st.session_state.chat.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    profile: UserProfile = st.session_state.profile
    history = st.session_state.history

    intent = detect_intent(user_input)
    filled_count, missing = compute_filled_and_missing(profile)

    policy = compute_policy(intent, filled_count, missing, st.session_state.turns_since_question)

    sys_vars = {
        "user_intent": intent,
        "filled_count": filled_count,
        "missing_slots": json.dumps(missing, ensure_ascii=False),
        "current_profile_json": json.dumps(profile.model_dump(), ensure_ascii=False),
        "ask_question": "true" if policy["ask_question"] else "false",
        "question_focus": policy["question_focus"],
        "reco_mode": policy["reco_mode"],
    }

    # (A) 상담 대화 생성
    messages = dialogue_prompt.format_messages(history=history[-6:], user_input=user_input, **sys_vars)
    assistant_text = chat_llm.invoke(messages).content.strip()

    # 질문 빈도 제어용(간단 휴리스틱)
    asked = ("?" in assistant_text) or assistant_text.strip().endswith(("까", "요", "니"))
    st.session_state.turns_since_question = 0 if (policy["ask_question"] and asked) else (st.session_state.turns_since_question + 1)

    # 화면에 어시스턴트 메시지
    st.session_state.chat.append({"role": "assistant", "content": assistant_text})
    with st.chat_message("assistant"):
        st.markdown(assistant_text)

    # history 업데이트
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=assistant_text))
    st.session_state.history = history

    # (B) 조용한 정보추출
    try:
        extract_vars = {
            "format_instructions": extract_parser.get_format_instructions(),
            "user_input": user_input,
            "current_profile_json": json.dumps(profile.model_dump(), ensure_ascii=False),
        }
        raw_extract = extract_llm.invoke(extract_prompt.format_messages(**extract_vars)).content
        extraction = parse_extraction(extract_parser, raw_extract)

        profile = merge_profile(profile, extraction.profile_update)
        st.session_state.profile = profile

    except Exception as e:
        if show_debug:
            st.sidebar.error(f"추출 파싱 오류: {e}")

    # 디버그/프로필 표시
    if show_debug:
        filled_count2, missing2 = compute_filled_and_missing(st.session_state.profile)
        st.sidebar.write(
            {
                "intent": intent,
                "filled_count": filled_count2,
                "missing_slots": missing2,
                "policy": policy,
                "turns_since_question": st.session_state.turns_since_question,
            }
        )

    if show_profile:
        st.sidebar.subheader("current_profile_json")
        st.sidebar.code(json.dumps(st.session_state.profile.model_dump(), ensure_ascii=False, indent=2), language="json")
