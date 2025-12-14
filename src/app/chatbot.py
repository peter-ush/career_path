# chatbot_core.py
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from dotenv import load_dotenv
import uuid

from .prompts import (
    FULL_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    BACKGROUND_ASK_TEMPLATE,
    BACKGROUND_TO_ROLE_TEMPLATE,
    ROLE_DETAIL_TEMPLATE,
    GAP_ANALYSIS_TEMPLATE,
)

load_dotenv()

# 1) LLM 생성
llm = ChatOpenAI(model="gpt-5-mini")


# 3) 프롬프트 템플릿: system + history + human
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", FULL_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

# 4) 체인 구성
chain = prompt | llm

# 5) 세션별 히스토리 저장소
_store: dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    """세션 ID별로 LangChain 대화 히스토리 관리"""
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]


# 6) RunnableWithMessageHistory 설정 
runnable = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


# 7) 외부에서 사용할 함수: session_id + user_input → 모델 응답 문자열
def get_chat_response(
    session_id: str,
    user_input: str,
    role_name: str = "",
    retrieved_docs: str = "",
) -> str:
    """
    Streamlit 등 외부에서 호출할 진입점 함수.

    - session_id: 유저/브라우저마다 고유하게 관리
    - user_input: 사용자가 보낸 채팅 문자열
    - role_name, retrieved_docs: 나중에 RAG/직무선택 연동 시 확장용 (지금은 빈값 가능)
    """
    response = runnable.invoke(
        {
            "input": user_input,
            "role_name": role_name,
            "retrieved_docs": retrieved_docs,
        },
        config={"configurable": {"session_id": session_id}},
    )
    return response.content


# (옵션) session_id 하나 랜덤 생성해주는 유틸
def create_session_id() -> str:
    return str(uuid.uuid4())
