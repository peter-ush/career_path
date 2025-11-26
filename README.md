# IT Career Path Chatbot (RAG 기반 IT 직무 추천 챗봇)

본 프로젝트는 **IT 전공자들의 진로 선택을 돕는 AI 기반 챗봇 서비스**입니다.  
사용자의 학업·프로젝트 경험·기술 스택을 분석하여 적합한 IT 직무를 추천하고,  
실제 채용 공고/합격 스펙 데이터를 기반으로 현실적인 커리어 정보를 제공합니다.

---

## 🚀 프로젝트 개요

### 💡 기획 배경
IT 전공은 진출 분야가 매우 넓지만,  
그만큼 학생들이 어떤 직무를 목표로 해야 할지 혼란을 겪는 경우가 많습니다.

이 프로젝트는 사용자의 **백그라운드 분석 + 최신 채용 동향 + 합격 스펙 데이터**를 통합하여  
**개인 맞춤 직무 추천**을 제공하는 것을 목표로 합니다.

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


## 🔨 실행 방법

### 1. 환경 변수 설정 (.env)
Root 경로에 `.env` 파일 생성:
OPENAI_API_KEY=sk-xxxx

### 2. 라이브러리 설치
pip install -r requirements.txt

3. Streamlit 실행
streamlit run streamlit_app.py
