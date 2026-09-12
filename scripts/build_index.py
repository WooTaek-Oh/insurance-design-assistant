"""docs/ 안의 PDF 약관을 임베딩해서 chroma_db/ 에 영속 저장하는 1회성 스크립트.

로컬에서 이 스크립트를 실행해 나온 chroma_db/ 폴더를 그대로 git에 커밋한다.
Streamlit Cloud는 배포마다 컨테이너가 새로 뜨는데, 매번 957페이지짜리 약관을
재임베딩하면 느리고 무료 API 쿼터도 금방 소진되기 때문에, 인덱스는 미리
만들어서 커밋해두고 앱은 그걸 로드만 하도록 구성했다.

문서를 새로 추가하거나 교체했을 때만 다시 실행하면 된다.

실행:
    source .venv/bin/activate
    python scripts/build_index.py
"""
import os
import sys

# 프로젝트 루트를 import 경로에 추가 (scripts/ 하위에서 실행되므로)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag  # noqa: E402


def load_api_key() -> str:
    env_key = os.environ.get("GOOGLE_API_KEY")
    if env_key:
        return env_key

    secrets_path = os.path.join(".streamlit", "secrets.toml")
    if not os.path.exists(secrets_path):
        raise RuntimeError(
            "GOOGLE_API_KEY를 찾을 수 없습니다. 환경변수로 설정하거나 "
            ".streamlit/secrets.toml에 GOOGLE_API_KEY를 넣어주세요."
        )

    import tomllib

    with open(secrets_path, "rb") as f:
        secrets = tomllib.load(f)
    key = secrets.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(".streamlit/secrets.toml에 GOOGLE_API_KEY가 없습니다.")
    return key


def main():
    rebuild = "--rebuild" in sys.argv
    api_key = load_api_key()
    print(f"[1/3] {rag.DOCS_DIR}/ 에서 PDF 로딩 및 청킹 중...")
    result = rag.build_index(api_key, progress_cb=print, rebuild=rebuild)

    print(f"[2/3] 인덱싱 완료")
    print(f"  - 대상 파일: {result['source_files']}")
    print(f"  - 총 페이지 수: {result['page_count']}")
    print(f"  - 청크 수: {result['chunk_count']}")
    print(f"  - 사용된 임베딩 모델: {result['embedding_model']}")
    print(f"[3/3] 저장 위치: {rag.PERSIST_DIR}/")
    print("\n완료. chroma_db/ 폴더를 git에 커밋하세요.")


if __name__ == "__main__":
    main()
