# router.py
from dataclasses import dataclass, field
from typing import Literal, Dict, Any, Tuple, List
import re

Intent = Literal["info_collect", "explain", "job_trend", "fallback"]

@dataclass
class ProfileState:
    goal_role: str | None = None
    experience_level: str | None = None
    skills: List[str] = field(default_factory=list)
    domain: str | None = None
    constraints: List[str] = field(default_factory=list)

def route_intent(state: ProfileState, text: str) -> Dict[str, Any]:
    t = text.strip().lower()

    # 1) 공고/트렌드
    if any(k in t for k in ["공고", "채용", "트렌드", "요즘", "많이 뽑", "우대사항", "자격요건"]):
        return {"intent": "job_trend", "confidence": 0.75, "slots": {}, "next_action": "answer_now"}

    # 2) 설명/개념
    if any(k in t for k in ["뭐야", "설명", "차이", "왜", "어떻게", "원리", "개념", "스택", "기술"]):
        return {"intent": "explain", "confidence": 0.70, "slots": {}, "next_action": "answer_now"}

    # 3) 정보수집(백그라운드 기반 추천)
    if any(k in t for k in ["어떤 직무", "뭘 할 수", "추천", "진로", "커리어", "나한테 맞", "가능할까"]):
        return {"intent": "info_collect", "confidence": 0.70, "slots": {}, "next_action": "ask_one_question"}

    # 4) 예외
    return {"intent": "fallback", "confidence": 0.40, "slots": {}, "next_action": "redirect"}

# -------- handlers (모듈 역할) --------
def handle_info_collect(state: ProfileState, text: str, route: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    # 지금은 질문 1개만 던지는 MVP
    reply = (
        "좋아. 너 상황에 맞게 추천하려면 딱 2가지만 알려줘!\n"
        "1) 지금까지 해본 프로젝트/경험(없으면 수업/과제라도)\n"
        "2) 관심 있는 방향: 백엔드 / 데이터 / AI 중 어디에 더 끌려?"
    )
    return reply, {}

def handle_explain(state: ProfileState, text: str, route: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    reply = (
        "설명해줄게. 어떤 직무/스택/개념을 말하는 거야?\n"
        "예) 데이터 엔지니어, 백엔드, Airflow, ETL/ELT, RAG"
    )
    return reply, {}

def handle_job_trend(state: ProfileState, text: str, route: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    reply = (
        "요즘 공고 트렌드는 직무마다 달라서, 먼저 목표 직무를 하나만 찍자.\n"
        "백엔드 / 데이터 엔지니어 / 데이터 분석 / ML 중 어디 트렌드가 궁금해?"
    )
    return reply, {}

def handle_fallback(state: ProfileState, text: str, route: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    reply = (
        "지금 질문은 IT 커리어 상담 범위를 조금 벗어난 것 같아.\n"
        "대신 아래 중 하나로 말해주면 바로 도와줄게!\n"
        "1) 내 백그라운드로 가능한 직무 추천\n"
        "2) 직무/기술 개념 설명\n"
        "3) 요즘 채용 공고 트렌드"
    )
    return reply, {}

HANDLERS = {
    "info_collect": handle_info_collect,
    "explain": handle_explain,
    "job_trend": handle_job_trend,
    "fallback": handle_fallback,
}

def run_turn(state: ProfileState, user_message: str) -> Tuple[ProfileState, Dict[str, Any], str]:
    route = route_intent(state, user_message)
    handler = HANDLERS.get(route["intent"], handle_fallback)
    reply, update = handler(state, user_message, route)

    # state 업데이트(지금은 비워둬도 됨)
    for k, v in update.items():
        setattr(state, k, v)

    return state, route, reply
