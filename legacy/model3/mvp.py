# src/app/mvp.py
import json
import re
from typing import Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from src.app.schemas import (
    UserProfile,
    ProfileUpdate,
    ExtractionResult,
    merge_profile,
    compute_filled_and_missing,
)

from dotenv import load_dotenv

load_dotenv()


# -------------------------
# 0) Prompt (대화용 / 추출용) - 파일 안에 내장
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
# 1) intent 감지
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


# -------------------------
# 2) policy 계산 (질문을 매턴 하지 않게 하는 핵심)
# -------------------------
QUESTION_PRIORITY = ["interests", "preferred_work", "major", "languages", "project_experience", "project_role"]


def pick_question_focus(missing_slots: list[str]) -> str:
    for k in QUESTION_PRIORITY:
        if k in missing_slots:
            return k
    return "none"


def compute_policy(intent: str, filled_count: int, missing_slots: list[str], turns_since_question: int) -> dict:
    """
    자연스러움 목표:
    - 매 턴 질문하지 않기 (2~3턴에 1번 정도)
    - GREET에서는 고민을 묻는 1개 질문은 OK
    - 추천은 ASK_RECOMMEND일 때만 (reco_mode로 제어)
    """
    policy = {
        "ask_question": False,
        "question_focus": "none",
        "reco_mode": "NONE",
    }

    if intent == "GREET":
        policy["ask_question"] = True
        policy["question_focus"] = "고민"
        return policy

    if intent != "ASK_RECOMMEND":
        # CAREER: 추천은 안 함. 질문은 '자주' 하지 않게.
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
    elif filled_count == 2:
        policy["reco_mode"] = "TOP3_FINAL"
        # 추천을 이미 해주므로, 질문은 매번 안 해도 됨 (부담 줄이기)
        policy["ask_question"] = turns_since_question >= 1
        policy["question_focus"] = "선택/우선순위"
    else:
        policy["reco_mode"] = "TOP3_FINAL"
        policy["ask_question"] = turns_since_question >= 1
        policy["question_focus"] = "선택/우선순위"

    return policy


# -------------------------
# 3) JSON 파싱 보정 (추출용)
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
# 4) main
# -------------------------
def main():
    # 대화용: 자연스러움(약간의 온도)
    chat_llm = ChatOpenAI(model="gpt-5.2", temperature=0.6)
    # 추출용: 안정적(낮은 온도)
    extract_llm = ChatOpenAI(model="gpt-5.2", temperature=0.0)

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

    profile = UserProfile()
    history = []  # HumanMessage/AIMessage list
    turns_since_question = 99  # 처음은 질문 허용

    print("=== Career Chatbot MVP (2-pass: Dialogue + Extraction) ===")
    print("종료하려면 'exit' 입력\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break

        intent = detect_intent(user_input)
        filled_count, missing = compute_filled_and_missing(profile)

        policy = compute_policy(intent, filled_count, missing, turns_since_question)

        # -----------------
        # (A) 대화 생성 (자연스러운 상담)
        # -----------------
        sys_vars = {
            "user_intent": intent,
            "filled_count": filled_count,
            "missing_slots": json.dumps(missing, ensure_ascii=False),
            "current_profile_json": json.dumps(profile.model_dump(), ensure_ascii=False),
            "ask_question": "true" if policy["ask_question"] else "false",
            "question_focus": policy["question_focus"],
            "reco_mode": policy["reco_mode"],
        }

        messages = dialogue_prompt.format_messages(history=history[-6:], user_input=user_input, **sys_vars)
        assistant_text = chat_llm.invoke(messages).content.strip()

        print(f"\nBot: {assistant_text}")

        # 질문을 했는지(대충) 체크해서 빈도 조절
        asked = "?" in assistant_text or "까" in assistant_text[-6:]  # 완벽하진 않지만 충분
        turns_since_question = 0 if (policy["ask_question"] and asked) else (turns_since_question + 1)

        # history 업데이트
        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=assistant_text))

        # -----------------
        # (B) 추출 (조용히 슬롯 업데이트)
        # -----------------
        extract_vars = {
            "format_instructions": extract_parser.get_format_instructions(),
            "user_input": user_input,
            "current_profile_json": json.dumps(profile.model_dump(), ensure_ascii=False),
        }
        raw_extract = extract_llm.invoke(extract_prompt.format_messages(**extract_vars)).content
        extraction: ExtractionResult = parse_extraction(extract_parser, raw_extract)

        profile = merge_profile(profile, extraction.profile_update)

        # 디버그 로그
        filled_count, missing = compute_filled_and_missing(profile)
        print(f"(debug) intent={intent}, filled={filled_count}, missing={missing}, policy={policy}\n")


if __name__ == "__main__":
    main()
