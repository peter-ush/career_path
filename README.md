it-career-path-chatbot/
├─ README.md
├─ requirements.txt
├─ data/
│  ├─ raw/          # API로 받아온 원본 데이터 (csv/json)
│  └─ processed/    # 전처리 후 DB에 넣기 전 데이터
├─ src/
│  ├─ crawler/      # 1. 채용 공고 API/크롤러 코드
│  │  └─ fetch_jobs.py
│  ├─ db/           # 2. DB 스키마 & 적재 코드
│  │  ├─ models.py
│  │  └─ init_db.py
│  ├─ rag/          # 3. RAG 관련 코드
│  │  ├─ vectorstore.py
│  │  └─ qa_chain.py
│  └─ app/          # 4. LLM 대화/로직 (백엔드)
│     └─ chatbot_core.py
├─ streamlit_app.py   # 4. Streamlit UI 엔트리 포인트
#123