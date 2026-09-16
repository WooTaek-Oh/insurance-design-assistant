"""복구 1단계: 손상된 컬렉션에서 재사용 가능한 (id, embedding, text, metadata)를
추출해서 pickle 파일로 저장한다. chroma_db는 건드리지 않는다 (읽기 전용)."""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag  # noqa: E402


def load_api_key() -> str:
    import tomllib

    with open(os.path.join(".streamlit", "secrets.toml"), "rb") as f:
        return tomllib.load(f)["GOOGLE_API_KEY"]


def main():
    api_key = load_api_key()

    print("[1/3] 손상된 컬렉션에서 기존 임베딩 읽는 중...")
    old_vs = rag.load_vectorstore(api_key, embedding_model="models/gemini-embedding-001")
    old_data = old_vs.get(include=["embeddings", "documents", "metadatas"])
    print(f"  기존 저장된 항목: {len(old_data['ids'])}개")

    text_to_embedding = {}
    for doc, emb in zip(old_data["documents"], old_data["embeddings"]):
        if doc not in text_to_embedding:
            text_to_embedding[doc] = emb
    print(f"  고유 텍스트: {len(text_to_embedding)}개")

    print("[2/3] 전체 문서를 새 안정 id 스킴으로 다시 청킹하는 중...")
    documents = rag.load_pdf_documents()
    chunks = rag.split_documents(documents)
    canonical_ids = rag._make_stable_ids(chunks)
    print(f"  전체 청크(정답 기준): {len(chunks)}개")

    print("[3/3] 매칭 및 저장 중...")
    reusable = {"ids": [], "embeddings": [], "documents": [], "metadatas": []}
    matched_texts = set()
    for cid, chunk in zip(canonical_ids, chunks):
        text = chunk.page_content
        if text in text_to_embedding:
            reusable["ids"].append(cid)
            reusable["embeddings"].append(text_to_embedding[text])
            reusable["documents"].append(text)
            reusable["metadatas"].append(chunk.metadata)
            matched_texts.add(text)

    with open("/tmp/repair_reusable.pkl", "wb") as f:
        pickle.dump(reusable, f)

    print(f"  재사용 가능: {len(reusable['ids'])}개 / 전체 {len(chunks)}개")
    print(f"  새로 임베딩 필요: {len(chunks) - len(reusable['ids'])}개")
    print("  -> /tmp/repair_reusable.pkl 에 저장 완료")


if __name__ == "__main__":
    main()
