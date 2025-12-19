<img width="5607" height="1772" alt="image" src="https://github.com/user-attachments/assets/e296a83f-1eff-4f99-967a-34c890c24b28" /># IT Career Path Chatbot (RAG 기반 IT 직무 추천 챗봇)

본 프로젝트는 **IT 전공자들의 진로 선택을 돕는 AI 기반 챗봇 서비스**입니다.  
사용자의 학업·프로젝트 경험·기술 스택을 분석하여 적합한 IT 직무를 추천하고,  
실제 채용 공고/합격 스펙 데이터를 기반으로 현실적인 커리어 정보를 제공합니다.

### 기획 배경
IT 전공은 진출 분야가 매우 넓지만,  
그만큼 학생들이 어떤 직무를 목표로 해야 할지 혼란을 겪는 경우가 많습니다.

이 프로젝트는 사용자의 **백그라운드 분석 + 최신 채용 동향 + 합격 스펙 데이터**를 통합하여  
**개인 맞춤 직무 추천**을 제공하는 것을 목표로 합니다.

##  주요 기능

### 1️⃣ 사용자 배경 분석
- 사용자의 학력, 수강 과목, 기술 스택, 프로젝트 경험 입력
- LLM이 부족한 정보 자동 보완 질문
- 역량을 기술 스택별로 정리하여 직무 후보 생성

### 2️⃣ 채용 공고 기반 직무 트렌드 분석
- IT 관련 최신 채용 공고 API 크롤링
- 공고별 스킬 태그/요구 연차/직무 조건 분석
- RAG 기반으로 요약 제공

### 3️⃣ 합격 스펙 기반 현실 조언
- 자소서/합격 후기/기술 스택 분포 정보 RAG 검색
- 실제 합격자들 스펙 근거 기반 답변

### 4️⃣ Streamlit 인터페이스
- 간단한 UI를 통해 직무 질문/추천 결과 확인
- 추천 직무와 필요한 스킬 로드맵을 시각적으로 제공

<img width="5607" height="1772" alt="image" src="https://github.com/user-attachments/assets/8836261a-e421-4a68-9751-15f12826cbe4" />


## 📁 폴더 구조

```text
career_path/
├─ README.md                # 프로젝트 설명 파일
├─ requirements.txt         # 필요한 라이브러리 목록

├─ data/
│  ├─ raw/                  # API로 받아온 원본 데이터 (csv/json)
│  └─ processed/            # 전처리 데이터 (DB insert 전)

├─ src/
│  ├─ crawler/              # 1. 채용 공고 수집 모듈
│  │  └─ fetch_jobs.py
│
│  ├─ db/                   # 2. 데이터베이스 작업 모듈
│  │  ├─ models.py          # DB 모델 정의
│  │  └─ init_db.py         # DB 초기화 및 적재 로직
│
│  ├─ rag/                  # 3. RAG(벡터 검색) 관련 코드
│  │  ├─ vectorstore.py     # 벡터스토어 관리
│  │  └─ qa_chain.py        # 문서 검색 + 답변
│
│  └─ app/                  # 4. 챗봇 로직 / 사용자 분석 로직
│     └─ chatbot_core.py
│
├─ streamlit_app.py         # Streamlit UI 엔트리 포인트
└─ .env                     # API Key 등 환경 변수 (gitignore 대상)
```


## 🔨 실행 방법

### 1. 환경 변수 설정 (.env)
Root 경로에 `.env` 파일 생성:
OPENAI_API_KEY=sk-xxxx

### 2. 라이브러리 설치
pip install -r requirements.txt

### 3. Streamlit 실행
streamlit run streamlit_app.py
