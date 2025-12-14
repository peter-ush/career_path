# streamlit.py
import streamlit as st
import uuid

from src.app.chatbot import get_chat_response, create_session_id

st.set_page_config(
    page_title="IT 진로/커리어 상담 챗봇",
    page_icon="💻",
)

st.title("💻 IT 진로/커리어 상담 챗봇")
st.caption("프로그래밍/프로젝트/관심 분야를 기반으로 진로를 같이 정리해줘요.")


# 1) 세션 상태 초기화
if "session_id" not in st.session_state:
    # chatbot_core의 유틸 써도 되고, 그냥 uuid4 바로 써도 됨
    st.session_state["session_id"] = create_session_id()

if "messages" not in st.session_state:
    # [{"role": "user"/"assistant", "content": "..."}] 구조로 저장
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "안녕하세요! 😊\n\n"
                       "IT 진로/커리어 상담 도와드릴게요.\n"
                       "지금 본인 상황(전공/학년, 프로그래밍 경험 등)을 편하게 얘기해주셔도 좋고,\n"
                       "막연하게 진로가 고민된다고 말해주셔도 괜찮아요.",
        }
    ]


# 2) 지금까지 대화 내용 표시
for msg in st.session_state["messages"]:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])


# 3) 사용자 입력 받기
user_input = st.chat_input("메시지를 입력해 주세요.")

if user_input:
    # (1) 화면에 유저 메시지 추가
    st.session_state["messages"].append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # (2) 모델 응답 얻기
    session_id = st.session_state["session_id"]

    # 나중에 직무 이름, RAG 결과 등을 붙이고 싶으면
    # 아래에 role_name, retrieved_docs를 추가로 세션 상태에서 꺼내서 넘겨도 됨.
    assistant_reply = get_chat_response(
        session_id=session_id,
        user_input=user_input,
        role_name="",        # 지금은 비워두고, 나중에 확장
        retrieved_docs="",   # 나중에 RAG 붙이면 여기 채우기
    )

    # (3) 화면에 모델 응답 추가
    st.session_state["messages"].append(
        {"role": "assistant", "content": assistant_reply}
    )
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)
