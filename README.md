# 가입설계 챗봇 (동양생명 FC 지원 AI 어시스턴트) — 프로토타입

동양생명 전속 설계사(FC)가 상담 현장에서 상품·특약·가입설계에 대해 즉시
정확한 답변을 받을 수 있도록 돕는 RAG(검색증강생성) 기반 챗봇 프로토타입입니다.

실제 보험 약관 PDF를 벡터 검색으로 찾아 그 내용에 근거해서만 답변하고,
근거가 없으면 "확인이 필요합니다"라고 답해 할루시네이션(사실이 아닌 내용을
지어내는 것)을 차단합니다.

## 기술 스택

| 영역 | 사용 기술 |
|---|---|
| 프론트/백엔드 | [Streamlit](https://streamlit.io/) (단일 앱) |
| 오케스트레이션 | [LangChain](https://www.langchain.com/) |
| LLM (답변 생성) | Google Gemini API (`gemini-flash-latest`) |
| 임베딩 (문서/질문 벡터화) | Google Gemini Embedding API (`gemini-embedding-001`) |
| 벡터스토어 | [Chroma](https://www.trychroma.com/) (로컬 영속 저장) |
| 배포 | Streamlit Community Cloud |

### 데이터 흐름

```
사용자 질문
  → Streamlit UI
  → LangChain
      ├─ Chroma 벡터 검색 (약관 PDF에서 관련 근거 청크 조회)
      └─ Gemini API (검색된 근거 + 대화 기록을 바탕으로 답변 생성)
  → 답변 + 참고 페이지 표시
```

## 주요 기능

- **약관 기반 RAG 응답**: `docs/` 폴더의 보험 약관 PDF를 미리 청크 단위로
  나눠 임베딩하고 Chroma에 저장. 질문이 들어오면 유사도 검색으로 관련 조항을
  찾아 그 내용에 근거해서만 답변합니다.
- **할루시네이션 차단**: 검색된 근거가 없거나 질문과 관련성이 낮으면(거리
  임계값 기준) "확인이 필요합니다. 정확한 내용은 약관 원문이나 상품 담당
  부서를 통해 확인해주세요."라고 답합니다. 인사말 등 일반 대화는 근거 없이도
  자연스럽게 응답합니다.
- **근거 페이지 표시**: 답변에 참고한 약관 페이지 번호를 함께 보여줘 설계사가
  원문을 바로 대조할 수 있습니다.
- **멀티턴 대화**: 이전 대화 맥락을 유지하며 이어지는 질문에 답합니다.

## 로컬 실행 방법

### 1. 저장소 클론 및 가상환경 준비

```bash
git clone https://github.com/WooTaek-Oh/insurance-design-assistant.git
cd insurance-design-assistant
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.12 기준으로 개발/테스트되었습니다.

### 2. Gemini API 키 발급 및 설정

1. [Google AI Studio](https://aistudio.google.com/apikey)에서 무료 API 키를 발급받습니다.
2. 프로젝트 루트에 `.streamlit/secrets.toml` 파일을 만들고 아래처럼 키를 넣습니다.

```toml
GOOGLE_API_KEY = "발급받은_키_값"
```

이 파일은 `.gitignore`에 포함되어 있어 git에는 올라가지 않습니다. 절대
키 값을 커밋하지 마세요.

### 3. 앱 실행

```bash
streamlit run streamlit_app.py
```

브라우저에서 `http://localhost:8501` 로 접속하면 됩니다. 이미 인덱싱된
`chroma_db/`가 저장소에 포함되어 있어 별도 인덱싱 없이 바로 실행 가능합니다.

## 약관 문서 재인덱싱 (문서를 추가/교체했을 때만)

`docs/` 폴더의 PDF를 추가하거나 교체한 경우에만 아래 스크립트를 다시
실행하면 됩니다. 평소 앱을 실행할 때는 필요하지 않습니다.

```bash
python scripts/build_index.py
```

- 기존에 이미 임베딩된 청크는 건너뛰고 이어서 진행합니다 (재실행 안전).
- 처음부터 다시 만들고 싶다면 `python scripts/build_index.py --rebuild`
  를 실행하세요.

> **참고**: Gemini 임베딩 API 무료 등급은 하루 1,000건의 요청 한도가
> 있습니다. 문서 분량이 많으면 하루 안에 다 끝나지 않을 수 있고, 그 경우
> 스크립트를 다시 실행하면 남은 부분부터 이어서 진행됩니다. 앱을 실제로
> 사용할 때(질문 1건당 임베딩 1회)는 이 한도에 거의 영향을 받지 않습니다.

## 프로젝트 구조

```
.
├── streamlit_app.py       # 메인 앱 (채팅 UI + RAG 응답 로직)
├── rag.py                 # RAG 공용 모듈 (PDF 로딩/청킹/임베딩/검색)
├── scripts/
│   └── build_index.py     # 약관 PDF 인덱싱 스크립트 (1회성/재실행 가능)
├── docs/                  # 인덱싱 대상 약관 PDF
├── chroma_db/             # 미리 빌드해둔 벡터 인덱스 (git에 포함)
├── requirements.txt
└── .streamlit/
    └── secrets.toml       # API 키 (로컬 전용, git에 미포함)
```

## Streamlit Cloud 배포

1. GitHub 저장소를 Streamlit Community Cloud에 연결합니다.
2. Main file path를 `streamlit_app.py`로 지정합니다.
3. App settings → Secrets에 아래 내용을 등록합니다.

```toml
GOOGLE_API_KEY = "발급받은_키_값"
```

4. `main` 브랜치에 push하면 자동으로 재배포됩니다. 벡터 인덱스(`chroma_db/`)가
   저장소에 이미 포함되어 있어 배포 시 재임베딩 없이 바로 로드됩니다.

## 알려진 제약사항

- 무료 등급 Gemini API의 요청 한도로 인해, 대량의 문서를 새로 인덱싱할 때는
  하루 이상 걸릴 수 있습니다.
- 현재는 암보험(무배당우리WON하는암보험) 약관 1종만 인덱싱되어 있습니다.
- 확신도(%) 표시, 다중 상품 비교, 가입설계 추천 등은 로드맵 단계로 이번
  프로토타입 범위에는 포함되지 않았습니다.
