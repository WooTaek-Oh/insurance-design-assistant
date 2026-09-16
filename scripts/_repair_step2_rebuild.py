"""복구 2단계: chroma_db를 지우고, 1단계에서 저장한 재사용 가능 임베딩으로
새 컬렉션을 만든다 (API 호출 없음). 별도 프로세스로 분리한 이유: 같은
프로세스에서 같은 경로의 Chroma를 삭제 후 재생성하면 내부 캐시 때문에
"readonly database" 오류가 났다."""
import os
import pickle
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag  # noqa: E402


def load_api_key() -> str:
    import tomllib

    with open(os.path.join(".streamlit", "secrets.toml"), "rb") as f:
        return tomllib.load(f)["GOOGLE_API_KEY"]


def main():
    api_key = load_api_key()

    with open("/tmp/repair_reusable.pkl", "rb") as f:
        reusable = pickle.load(f)
    print(f"[1/3] 재사용할 항목 {len(reusable['ids'])}개 로드 완료")

    print("[2/3] chroma_db/ 초기화 중...")
    if os.path.exists(rag.PERSIST_DIR):
        shutil.rmtree(rag.PERSIST_DIR)
    os.makedirs(rag.PERSIST_DIR, exist_ok=True)
    with open(rag.EMBEDDING_MODEL_META_PATH, "w", encoding="utf-8") as f:
        f.write("models/gemini-embedding-001")

    from langchain_chroma import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    embeddings_fn = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", google_api_key=api_key
    )
    new_vs = Chroma(
        collection_name=rag.COLLECTION_NAME,
        embedding_function=embeddings_fn,
        persist_directory=rag.PERSIST_DIR,
    )

    print("[3/3] 재사용 임베딩 저장 중 (API 호출 없음)...")
    BATCH = 200
    ids, embeddings, documents, metadatas = (
        reusable["ids"],
        reusable["embeddings"],
        reusable["documents"],
        reusable["metadatas"],
    )
    for i in range(0, len(ids), BATCH):
        new_vs._collection.add(
            ids=ids[i : i + BATCH],
            embeddings=embeddings[i : i + BATCH],
            documents=documents[i : i + BATCH],
            metadatas=metadatas[i : i + BATCH],
        )
    print(f"  저장 완료: {new_vs._collection.count()}개")


if __name__ == "__main__":
    main()
