# --- 맛집 데이터(CSV)를 AI가 검색할 수 있는 Vector DB로 변환하는 스크립트 ---

import pandas as pd
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# --- 설정 ---
# 1. Google Sheets에서 CSV로 다운로드할 파일 이름
DATA_SOURCE_FILE = "restaurants_data.csv"
# 2. 생성될 Vector DB 폴더 이름
VECTOR_STORE_SAVE_PATH = "restaurant_faiss_index"
# 3. 텍스트를 벡터로 변환할 임베딩 모델
EMBEDDING_MODEL_NAME = "jhgan/ko-sbert-nli"


def build_vector_store():
    """CSV 데이터를 읽어 FAISS Vector Store를 생성하고 로컬에 저장합니다."""
    # 0. 데이터 파일 존재 여부 확인
    if not os.path.exists(DATA_SOURCE_FILE):
        print(f"🚨 에러: '{DATA_SOURCE_FILE}'을 찾을 수 없습니다.")
        print("백엔드팀으로부터 데이터를 전달받아, 해당 이름으로 프로젝트 폴더에 저장해주세요.")
        return

    # 1. CSV 데이터 로드
    df = pd.read_csv(DATA_SOURCE_FILE)
    print(f"✅ 데이터 로딩 완료: 총 {len(df)}개의 맛집 데이터를 찾았습니다.")

    # 2. 데이터를 LangChain이 이해할 수 있는 Document 형식으로 변환
    documents = []
    for _, row in df.iterrows():
        content = (
            f"식당 이름은 '{row.get('restaurant_name', '정보 없음')}'이고, "
            f"주요 메뉴는 '{row.get('category', '정보 없음')}'입니다. "
            f"이 식당은 '{row.get('address', '정보 없음')}'에 위치해 있습니다."
        )
        metadata = row.to_dict()
        documents.append(Document(page_content=content, metadata=metadata))
    print(f"✅ {len(documents)}개의 맛집 정보를 Document 형식으로 변환했습니다.")

    # 3. 임베딩 모델 로드
    print("임베딩 모델을 로딩합니다... (처음 실행 시 시간이 걸릴 수 있습니다)")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("✅ 임베딩 모델 로딩 완료.")

    # 4. Vector Store 생성 및 저장
    print("Vector Store를 생성하고 저장합니다...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(VECTOR_STORE_SAVE_PATH)
    print(f"\n🎉 Vector Store 생성이 완료되었습니다! '{VECTOR_STORE_SAVE_PATH}' 폴더가 생성되었습니다.")


if __name__ == "__main__":
    build_vector_store()

