# IT Career Path Chatbot  
**RAG 기반 IT 직무 추천 · 커리어 가이드 챗봇**

본 프로젝트는 **IT 전공자(특히 진로가 아직 명확하지 않은 학생)**가 “어떤 직무를 목표로 해야 할지”를 정리할 수 있도록 돕는 AI 챗봇입니다.  
사용자의 **프로젝트/기술 스택/관심사**를 대화 중 자연스럽게 수집하고, **채용 공고 기반 근거(RAG)**로 직무 설명·요구 역량·학습 로드맵(갭분석)을 제공합니다.

---

## 핵심 아이디어

- **상담처럼 자연스럽게 대화**하지만, 내부적으로는 사용자의 정보를 “6개 슬롯” 형태로 지속 수집합니다.
- **채용 공고를 섹션 단위(업무/자격/우대)**로 쪼개어 Chroma(Vector DB)에 적재하고,
- 사용자가 선택한 직무(또는 추천된 직무)에 대해 **공고 근거를 검색(RAG)**하여 답변합니다.
- 추천/갭분석은 **사용자 동의 기반(Consent-based)**으로 동작하도록 설계했습니다.

---

## 주요 기능

### 1) 대화형 사용자 프로필링 (6슬롯)
- 프로젝트 경험 / 프로젝트 역할 / 언어·기술 및 숙련도 / 선호 업무 / 관심 분야 / 전공
- `PydanticOutputParser`로 **사용자 발화에서 “명시된 정보만”** 구조화 추출
- 질문은 **한 번에 1개만** 하도록 프롬프트 레벨에서 강제

### 2) 채용 공고 RAG (Chroma)
- `src/db/jobs.json`(샘플 공고 데이터)을 섹션별로 분해하여 chunking 후 벡터 저장
- 검색 결과를 **직무 canonical label + 섹션**으로 post-filter 하여 안정성 강화
- 답변 하단에 **근거 URL**을 함께 제공(가능한 경우)

### 3) 직무 추천 & 갭분석(로드맵)
- 일정 수준 이상 정보가 모이면 Top 2~3 직무 추천(동의 후)
- 선택 직무 기준으로 공고에서 자주 보이는 요구 역량을 요약하고,
  사용자 현재 상태와 비교해 **부족한 역량 TOP + 학습 로드맵** 제시(동의 후)

### 4) Streamlit UI
- `streamlit_app.py`에서 Chat UI로 바로 실행 가능
- 세션 단위로 대화 이력 유지(간단 메모리)

---

## 기술 스택

- Python, Streamlit
- LangChain (`langchain-openai`, `langchain-chroma`, `langchain-text-splitters`)
- OpenAI API (Embeddings / Chat)
- ChromaDB (Vector Store)
- Pydantic, python-dotenv

---

## 폴더 구조 (현재 develop 기준)

```text
career_path/
├─ README.md
├─ requirements.txt
├─ requirements.lock.txt
├─ streamlit_app.py                # Streamlit 엔트리
│
├─ src/
│  ├─ app/
│  │  ├─ chatbot.py                # 대화 로직 + 프로필 추출/상태 관리
│  │  └─ rag.py                    # Chroma 검색 + 포맷 + URL 추출
│  │
│  ├─ db/
│  │  ├─ jobs.json                 # 공고 데이터(샘플)
│  │  ├─ ingest_jobs.py            # jobs.json → Chroma 적재 스크립트
│  │  └─ chroma_jobs/              # persist된 벡터DB 디렉토리
│  │
│  └─ prompts/
│     └─ prompt.py                 # 시스템/추출 프롬프트
│
└─ legacy/                         # 과거 실험/프로토타입(참고용)
```

---

<img width="5607" height="1772" alt="image" src="https://github.com/user-attachments/assets/4493ddaa-2b8e-40c3-af9e-6967ce897c52" />


## 빠른 실행 (Quickstart)

### 0) 준비 사항
- Python **3.10+** 권장
- OpenAI API Key 필요

### 1) 환경 변수 설정 (.env)
프로젝트 루트에 `.env` 파일을 만들고 아래를 추가합니다.

```bash
OPENAI_API_KEY=sk-xxxx
```

선택(디버그 출력):
```bash
DEBUG_STATE=1
DEBUG_STATE_IN_CHAT=0
```

### 2) 라이브러리 설치
```bash
pip install -r requirements.txt
```

> ⚠️ 현재 `requirements.txt`에는 Streamlit이 포함되어 있지 않습니다.  
> 아래 중 하나로 설치해주세요.
```bash
pip install streamlit
# 또는 requirements.txt에 streamlit==버전 추가
```

### 3)벡터DB 생성 (jobs.json → Chroma)
`src/db/jobs.json`을 수정했거나, 새로 적재하고 싶을 때 실행합니다.

```bash
python -m src.db.ingest_jobs
# 또는
python src/db/ingest_jobs.py
```

실행 후 `src/db/chroma_jobs/`에 벡터DB가 생성됩니다.

### 4) Streamlit 실행
```bash
streamlit run streamlit_app.py
```

---

## 사용 흐름 예시

1. “안녕 / 진로 고민돼”처럼 가볍게 시작  
2. 챗봇이 “실제 공고 기반으로 2~3개 직무 추천 가능”을 안내하고 **추천 동의**를 받음  
3. 대화 중 프로젝트/기술/관심 등을 자연스럽게 수집  
4. 정보가 충분해지면 직무 Top 2~3 추천  
5. 원하면 선택 직무 기준으로 **갭분석 + 로드맵** 제공(공고 근거 기반)


---

## 자주 겪는 문제 (Troubleshooting)

- **OpenAI API Key 오류**
  - `.env`가 루트에 있는지 확인
  - `OPENAI_API_KEY`가 정확한지 확인

- **Streamlit 모듈 없음**
  - `pip install streamlit` 실행

- **벡터 검색 결과가 비어있음**
  - `python -m src.db.ingest_jobs`로 벡터DB를 먼저 생성했는지 확인
  - `src/db/chroma_jobs/`가 존재하는지 확인

---
