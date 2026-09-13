"""RAG(검색증강생성) 공용 모듈.

- docs/ 안의 PDF 약관을 청크로 나누고 Gemini 임베딩으로 Chroma에 저장/로드
- 질문이 들어오면 유사도 검색으로 근거 청크를 찾고, 근거가 부족하면
  LLM이 "확인이 필요합니다"라고 답하도록 프롬프트에 명시 (할루시네이션 차단)

인덱스는 scripts/build_index.py로 미리 만들어서 chroma_db/ 에 저장해두고,
앱은 재임베딩 없이 그 폴더를 그대로 로드한다. (Streamlit Cloud는 배포마다
컨테이너가 새로 뜨기 때문에, 매번 957페이지짜리 문서를 재임베딩하면
느리고 무료 API 쿼터도 금방 소진됨)
"""
from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = "docs"
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "oy_life_policy_terms"
EMBEDDING_MODEL_META_PATH = os.path.join(PERSIST_DIR, "embedding_model.txt")

# 무료 등급 키에서 최신 모델명이 404(NotFound)로 막히는 경우가 있어
# 여러 후보를 순서대로 시도한다. (챗 모델 gemini-2.5-flash가 막혔던 것과 동일 이슈)
EMBEDDING_MODEL_CANDIDATES = [
    "models/gemini-embedding-001",
    "models/gemini-embedding-2",
    "models/gemini-embedding-2-preview",
]

# Chroma 기본 거리(L2, 작을수록 유사)의 상위 컷오프.
# 전체 1,778개 청크 인덱싱 완료 후 실제 질의로 보정한 값 (models/gemini-embedding-001 기준):
#   실제 약관 질문 거리   0.41 ~ 0.57 (암 진단비, 납입면제, 해지환급금, 고지의무, 면책기간)
#   무관한 질문 거리      0.64 ~ 0.77 (인사말, 잡담, 주식 시세 등)
# 두 그룹 사이 간격(0.57~0.64)의 중간값으로 설정.
RELEVANCE_DISTANCE_THRESHOLD = 0.60

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# 무료 등급은 임베딩 API의 분당 요청/토큰 한도가 낮아서, 큰 배치를 한 번에
# 보내면 429(RESOURCE_EXHAUSTED)가 난다. 작은 배치 + 배치 사이 대기 +
# 실패 시 지수 백오프 재시도로 나눠 보낸다.
EMBED_BATCH_SIZE = 20
EMBED_BATCH_DELAY_SECONDS = 3.0
EMBED_MAX_RETRIES = 6
EMBED_RETRY_BASE_DELAY_SECONDS = 15.0


def find_working_embedding_model(api_key: str) -> str:
    """무료 키 기준으로 실제 호출 가능한 임베딩 모델명을 찾는다.

    주의: 이 함수는 신규 인덱스를 처음 만들 때만 호출해야 한다. 한 번 컬렉션이
    특정 모델로 만들어지면, 이후 조회/재개 시에는 반드시 같은 모델을 써야
    벡터 공간이 섞이지 않는다 (get_index_embedding_model 참고).
    """
    last_error: Exception | None = None
    for model_name in EMBEDDING_MODEL_CANDIDATES:
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)
            embeddings.embed_query("연결 테스트")
            return model_name
        except Exception as e:  # noqa: BLE001 - 모델별로 다른 예외 타입이 올 수 있음
            last_error = e
            continue
    raise RuntimeError(
        f"사용 가능한 Gemini 임베딩 모델을 찾지 못했습니다. 후보: {EMBEDDING_MODEL_CANDIDATES}"
    ) from last_error


def get_index_embedding_model(api_key: str) -> str:
    """인덱싱에 쓸 임베딩 모델명을 결정한다.

    이미 chroma_db/에 기록된 모델이 있으면 그걸 그대로 재사용하고 (한 컬렉션
    안에서 서로 다른 임베딩 모델이 섞이는 것을 방지), 없으면(최초 빌드)
    사용 가능한 모델을 새로 찾아서 기록해둔다.
    """
    if os.path.exists(EMBEDDING_MODEL_META_PATH):
        with open(EMBEDDING_MODEL_META_PATH, encoding="utf-8") as f:
            saved = f.read().strip()
        if saved:
            return saved

    model_name = find_working_embedding_model(api_key)
    os.makedirs(PERSIST_DIR, exist_ok=True)
    with open(EMBEDDING_MODEL_META_PATH, "w", encoding="utf-8") as f:
        f.write(model_name)
    return model_name


def load_pdf_documents() -> list[Document]:
    """docs/ 폴더의 모든 PDF를 읽어 페이지 단위 Document 리스트로 반환한다."""
    pdf_paths = sorted(glob.glob(os.path.join(DOCS_DIR, "*.pdf")))
    documents: list[Document] = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        documents.extend(loader.load())
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def _add_in_batches(vectorstore: Chroma, chunks: list[Document], progress_cb=None) -> None:
    """청크를 작은 배치로 나눠 임베딩+저장한다.

    - 이미 저장된 id는 건너뛴다 (중간에 끊겨도 재실행 시 이어서 진행 가능).
    - 429(쿼터 초과) 발생 시 지수 백오프로 재시도한다.
    """
    ids = [f"chunk-{i:05d}" for i in range(len(chunks))]

    existing_ids: set[str] = set()
    try:
        existing = vectorstore.get(ids=ids, include=[])
        existing_ids = set(existing.get("ids", []))
    except Exception:
        pass  # 컬렉션이 비어있으면 조회 자체가 의미 없을 수 있음

    pending = [
        (cid, chunk) for cid, chunk in zip(ids, chunks) if cid not in existing_ids
    ]
    if not pending:
        return

    for batch_start in range(0, len(pending), EMBED_BATCH_SIZE):
        batch = pending[batch_start : batch_start + EMBED_BATCH_SIZE]
        batch_ids = [cid for cid, _ in batch]
        batch_texts = [chunk.page_content for _, chunk in batch]
        batch_metadatas = [chunk.metadata for _, chunk in batch]

        delay = EMBED_RETRY_BASE_DELAY_SECONDS
        for attempt in range(EMBED_MAX_RETRIES):
            try:
                vectorstore.add_texts(
                    texts=batch_texts, metadatas=batch_metadatas, ids=batch_ids
                )
                break
            except Exception as e:  # noqa: BLE001
                is_last = attempt == EMBED_MAX_RETRIES - 1
                if is_last or "429" not in str(e) and "RESOURCE_EXHAUSTED" not in str(e):
                    raise
                if progress_cb:
                    progress_cb(
                        f"  429 쿼터 초과, {delay:.0f}초 대기 후 재시도"
                        f" ({attempt + 1}/{EMBED_MAX_RETRIES})"
                    )
                time.sleep(delay)
                delay = min(delay * 2, 120.0)

        if progress_cb:
            done = min(batch_start + EMBED_BATCH_SIZE, len(pending))
            progress_cb(f"  임베딩 진행: {done}/{len(pending)}")
        time.sleep(EMBED_BATCH_DELAY_SECONDS)


def build_index(api_key: str, progress_cb=None, rebuild: bool = False) -> dict:
    """docs/ 의 PDF를 인덱싱해서 chroma_db/ 에 영속 저장한다.

    scripts/build_index.py에서 호출하는 1회성(또는 문서 변경 시) 작업이다.
    rebuild=True면 기존 인덱스를 지우고 처음부터 다시 만든다.
    기본값(False)은 이미 저장된 청크를 건너뛰고 이어서 진행한다
    (무료 API 쿼터 한도로 중간에 끊겨도 재실행하면 이어짐).
    """
    if rebuild and os.path.exists(PERSIST_DIR):
        import shutil

        shutil.rmtree(PERSIST_DIR)

    documents = load_pdf_documents()
    if not documents:
        raise RuntimeError(f"{DOCS_DIR}/ 폴더에 PDF가 없습니다.")

    chunks = split_documents(documents)
    model_name = get_index_embedding_model(api_key)
    embeddings = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    _add_in_batches(vectorstore, chunks, progress_cb=progress_cb)

    return {
        "source_files": [os.path.basename(p) for p in glob.glob(os.path.join(DOCS_DIR, "*.pdf"))],
        "page_count": len(documents),
        "chunk_count": len(chunks),
        "embedding_model": model_name,
        "vectorstore": vectorstore,
    }


def load_vectorstore(api_key: str, embedding_model: str | None = None) -> Chroma | None:
    """미리 만들어둔 chroma_db/ 를 로드한다. 없으면 None을 반환한다.

    빌드 시 기록해둔 모델명(embedding_model.txt)을 그대로 사용한다 —
    쿼리 임베딩과 문서 임베딩이 반드시 같은 모델이어야 유사도 검색이 의미 있다.
    """
    if not os.path.exists(PERSIST_DIR):
        return None
    model_name = embedding_model or get_index_embedding_model(api_key)
    embeddings = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )


@dataclass
class RetrievedChunk:
    text: str
    page: int | None
    source: str | None
    distance: float


def retrieve(vectorstore: Chroma, query: str, k: int = 4) -> list[RetrievedChunk]:
    results = vectorstore.similarity_search_with_score(query, k=k)
    chunks: list[RetrievedChunk] = []
    for doc, distance in results:
        chunks.append(
            RetrievedChunk(
                text=doc.page_content,
                page=doc.metadata.get("page"),
                source=doc.metadata.get("source"),
                distance=distance,
            )
        )
    return chunks


def has_relevant_evidence(chunks: list[RetrievedChunk]) -> bool:
    if not chunks:
        return False
    return min(c.distance for c in chunks) <= RELEVANCE_DISTANCE_THRESHOLD


def format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        page_label = f"{c.page + 1}페이지" if c.page is not None else "페이지 미상"
        parts.append(f"[근거 {i} | {page_label}]\n{c.text}")
    return "\n\n".join(parts)


def format_sources(chunks: list[RetrievedChunk]) -> str:
    pages = sorted({c.page + 1 for c in chunks if c.page is not None})
    if not pages:
        return ""
    return ", ".join(str(p) + "p" for p in pages)
