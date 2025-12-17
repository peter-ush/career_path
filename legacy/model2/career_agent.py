import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()


# -----------------------------
# 0) 간단한 상태(Profile) 저장
# -----------------------------
@dataclass
class ProfileState:
    goal_role: Optional[str] = None            # ex) "backend", "data_engineer"
    experience_level: Optional[str] = None     # ex) "student", "intern", "junior"
    skills: List[str] = field(default_factory=list)
    domain: Optional[str] = None               # ex) "manufacturing", "finance"
    constraints: List[str] = field(default_factory=list)

PROFILE = ProfileState()


def _add_skill(skill: str) -> None:
    skill = skill.lower().strip()
    if skill and skill not in PROFILE.skills:
        PROFILE.skills.append(skill)


def _detect_goal_role(text: str) -> Optional[str]:
    t = text.lower()
    # 한국어/영어 섞어도 대충 잡히게
    if any(k in t for k in ["백엔드", "backend", "spring", "서버"]):
        return "backend"
    if any(k in t for k in ["데이터 엔지니어", "data engineer", "데이터 파이프라인", "etl", "airflow", "spark"]):
        return "data_engineer"
    if any(k in t for k in ["데이터 분석", "data analyst", "분석가", "bi", "대시보드", "tableau", "power bi"]):
        return "data_analyst"
    if any(k in t for k in ["ml", "머신러닝", "딥러닝", "모델", "추천시스템", "mle", "ml engineer"]):
        return "ml_engineer"
    return None


def _detect_experience_level(text: str) -> Optional[str]:
    t = text.lower()
    if any(k in t for k in ["인턴", "intern"]):
        return "intern"
    if any(k in t for k in ["신입", "주니어", "junior", "0년", "경력없"]):
        return "junior"
    if any(k in t for k in ["학생", "학부", "2학년", "3학년", "4학년", "대학생"]):
        return "student"
    return None


def _extract_skills(text: str) -> List[str]:
    # MVP: 키워드 기반 (나중에 LLM 추출로 교체 가능)
    t = text.lower()
    known = [
        "python", "pandas", "numpy", "sql", "mysql", "postgresql", "oracle",
        "spark", "airflow", "dbt", "kafka",
        "docker", "kubernetes", "aws", "gcp", "azure",
        "java", "spring", "c#", ".net", "javascript", "typescript", "react",
        "fastapi", "django"
    ]
    found = []
    for k in known:
        if k in t:
            found.append(k.replace(".net", "dotnet"))
    # 한글 매핑 일부
    if "파이썬" in text: found.append("python")
    if "스프링" in text: found.append("spring")
    if "에어플로우" in text: found.append("airflow")
    if "도커" in text: found.append("docker")
    if "쿠버" in text or "쿠버네티스" in text: found.append("kubernetes")
    if "자바" in text: found.append("java")
    if "리액트" in text: found.append("react")
    if "오라클" in text: found.append("oracle")
    if "마이에스큐엘" in text or "mysql" in t: found.append("mysql")

    # 중복 제거
    return sorted(set(s.lower().strip() for s in found if s))


def _detect_domain(text: str) -> Optional[str]:
    t = text.lower()
    if any(k in t for k in ["제조", "mes", "스마트팩토리", "공정", "품질", "설비"]):
        return "manufacturing"
    if any(k in t for k in ["금융", "핀테크", "은행", "증권", "트레이딩"]):
        return "finance"
    if any(k in t for k in ["커머스", "이커머스", "쇼핑", "리테일"]):
        return "commerce"
    return None


# -----------------------------
# 1) Tool 4개 = 의도 4개
# -----------------------------
class CollectProfileArgs(BaseModel):
    user_message: str = Field(..., description="사용자 메시지(자연어). 여기서 프로필 슬롯을 추출한다.")


def collect_profile(user_message: str) -> Dict[str, Any]:
    # 슬롯 추출
    new_goal = _detect_goal_role(user_message)
    new_level = _detect_experience_level(user_message)
    new_domain = _detect_domain(user_message)
    new_skills = _extract_skills(user_message)

    if new_goal and not PROFILE.goal_role:
        PROFILE.goal_role = new_goal
    if new_level and not PROFILE.experience_level:
        PROFILE.experience_level = new_level
    if new_domain and not PROFILE.domain:
        PROFILE.domain = new_domain
    for s in new_skills:
        _add_skill(s)

    # 부족한 슬롯 기준(아주 단순 MVP)
    missing = []
    if not PROFILE.goal_role:
        missing.append("goal_role")
    if not PROFILE.experience_level:
        missing.append("experience_level")
    if not PROFILE.skills:
        missing.append("skills")

    # 다음 질문 1개만
    if "goal_role" in missing:
        next_q = "지금 관심 있는 방향이 있어? 예: 백엔드 / 데이터 엔지니어 / 데이터 분석 / ML"
    elif "experience_level" in missing:
        next_q = "현재 단계가 어떻게 돼? 예: 학생(학년) / 인턴 / 신입 / 경력"
    elif "skills" in missing:
        next_q = "지금까지 써본 기술/언어를 3~5개만 적어줘. (예: Python, SQL, Spring)"
    else:
        next_q = None

    return {
        "state": {
            "goal_role": PROFILE.goal_role,
            "experience_level": PROFILE.experience_level,
            "skills": PROFILE.skills,
            "domain": PROFILE.domain,
            "constraints": PROFILE.constraints,
        },
        "missing_slots": missing,
        "next_question": next_q,
    }


class ExplainArgs(BaseModel):
    question: str = Field(..., description="직무/스택/개념 질문(자연어)")


def explain_it(question: str) -> Dict[str, Any]:
    # MVP: 여기서는 “툴”이 그냥 설명을 생산해도 되지만,
    # 실제론 RAG 붙일 자리라서 구조만 유지
    return {
        "answer_hint": (
            "설명 모듈(MVP)입니다. RAG 붙이기 전에는 일반 설명으로 응답하세요.\n"
            f"질문: {question}"
        )
    }


class JobTrendArgs(BaseModel):
    role: Optional[str] = Field(default=None, description="트렌드를 보고 싶은 직무(없으면 추론)")
    region: Optional[str] = Field(default="KR", description="지역/시장 (예: KR, US)")
    experience_level: Optional[str] = Field(default=None, description="신입/주니어/경력 등")


def job_trend(role: Optional[str] = None, region: str = "KR", experience_level: Optional[str] = None) -> Dict[str, Any]:
    # MVP: 하드코딩 요약(나중에 RAG/크롤러로 교체)
    r = role or PROFILE.goal_role or "unspecified"
    lvl = experience_level or PROFILE.experience_level or "unspecified"

    trends = {
        "backend": [
            "Spring/Java 또는 Node 기반 서버 개발 경험",
            "RDB 설계/쿼리(SQL) + 트랜잭션/성능 튜닝 기초",
            "Docker/클라우드(AWS 등) + CI/CD 감각",
        ],
        "data_engineer": [
            "SQL/데이터 모델링 + 파이프라인(ETL/ELT) 설계",
            "Airflow/dbt/Spark 같은 워크플로/분산 처리",
            "클라우드 스토리지/웨어하우스(예: S3/BigQuery/Redshift 계열) 감각",
        ],
        "data_analyst": [
            "SQL + 대시보드(Tableau/Power BI) + 지표 설계",
            "실험/통계(가설검정, A/B) 또는 분석 리포팅 능력",
            "비즈니스 문제를 분석 질문으로 바꾸는 능력",
        ],
        "ml_engineer": [
            "모델 학습 파이프라인 + 배포(MLOps) 기초",
            "Python + ML 프레임워크 + 서빙(API/Batch) 경험",
            "데이터 품질/피처 관리/모니터링 관점",
        ],
        "unspecified": [
            "직무를 먼저 정하면 트렌드를 더 정확히 요약할 수 있어요(백엔드/데이터/분석/ML)."
        ]
    }

    return {
        "region": region,
        "role": r,
        "experience_level": lvl,
        "top_trends": trends.get(r, trends["unspecified"]),
        "note": "현재는 MVP라 일반화된 요약입니다. 추후 공고 RAG/크롤링 붙이면 최신 기반으로 바뀝니다."
    }


class FallbackArgs(BaseModel):
    user_message: str = Field(..., description="1~3 범위 밖 입력. 커리어 대화로 유도한다.")


def fallback_redirect(user_message: str) -> Dict[str, Any]:
    return {
        "redirect": (
            "지금 입력은 IT 커리어 상담 범위(직무/스택/공고 트렌드)와 조금 다를 수 있어.\n"
            "원하면 아래 중 하나로 다시 말해줘!\n"
            "1) 내 백그라운드로 가능한 직무 추천\n"
            "2) 특정 직무/기술 개념 설명\n"
            "3) 요즘 채용 공고 트렌드"
        )
    }


# Tool 등록
collect_tool = StructuredTool.from_function(
    func=collect_profile,
    name="collect_profile",
    description="사용자 메시지에서 커리어 상담에 필요한 프로필 슬롯(관심직무/경험수준/기술스택/도메인 등)을 추출하고 상태를 갱신한다. 부족한 슬롯이 있으면 다음 질문 1개를 제안한다.",
    args_schema=CollectProfileArgs,
)

explain_tool = StructuredTool.from_function(
    func=explain_it,
    name="explain_it",
    description="직무/기술/개념 질문에 대한 설명을 생성한다. (추후 RAG로 근거 보강 예정)",
    args_schema=ExplainArgs,
)

trend_tool = StructuredTool.from_function(
    func=job_trend,
    name="job_trend",
    description="요즘 공고에서 많이 요구되는 역량/스택 트렌드를 직무 기준으로 요약한다. (현재는 MVP 요약, 추후 공고 RAG/크롤러로 교체 예정)",
    args_schema=JobTrendArgs,
)

fallback_tool = StructuredTool.from_function(
    func=fallback_redirect,
    name="fallback_redirect",
    description="범위 밖 입력을 커리어 상담 흐름으로 부드럽게 리다이렉트한다.",
    args_schema=FallbackArgs,
)


# -----------------------------
# 2) 에이전트(=라우팅 포함) 구성
# -----------------------------
SYSTEM_RULES = """당신은 한국어 IT 커리어 상담 챗봇입니다.
사용자의 입력을 보고 스스로 판단하여 아래 도구 중 '필요한 것 하나'를 호출해 문제를 해결하세요.

도구:
- collect_profile: 사용자의 백그라운드/관심직무/스택 정보를 추출하고, 부족한 정보가 있으면 다음 질문 1개를 제안
- explain_it: 직무/스택/개념 설명
- job_trend: 공고 트렌드 요약(현재는 MVP 일반 요약)
- fallback_redirect: 범위 밖 입력을 커리어 상담으로 유도

규칙:
- 가능하면 매 턴 도구는 1개만 호출하세요(디버깅/일관성 목적).
- 사용자가 '추천/뭘 하면 좋을까/가능할까' 류면 먼저 collect_profile을 호출해 상태를 채우세요.
- 사용자가 '설명/차이/뭐야/어떻게' 류면 explain_it을 호출하세요.
- 사용자가 '요즘/공고/트렌드/많이 뽑' 류면 job_trend를 호출하세요.
- 위에 해당하지 않으면 fallback_redirect를 호출하세요.
- 도구 결과(JSON/딕셔너리)를 그대로 보여주지 말고, 사용자가 읽기 쉬운 자연어로 정리해서 답하세요.
- 정보가 부족하면 '질문 1개'만 하세요.
"""

def build_executor(verbose: bool = True) -> AgentExecutor:
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY가 필요합니다. .env에 OPENAI_API_KEY=... 를 넣어주세요.")

    llm = ChatOpenAI(model="gpt-5.2", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_RULES),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    tools = [collect_tool, explain_tool, trend_tool, fallback_tool]
    agent = create_openai_tools_agent(llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=verbose)
