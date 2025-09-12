# --- 맛집 데이터(CSV)를 AI가 검색할 수 있는 Vector DB로 변환하는 스크립트 ---

import pandas as pd
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# --- 경로 설정 (새로운 폴더 구조에 맞게 수정) ---
# 현재 파일(build_db.py)의 위치는 app/ 이므로, 상위 폴더로 이동(../) 후 각 폴더로 진입
DATA_SOURCE_FILE = "../data/restaurants_data.csv"
VECTOR_STORE_SAVE_PATH = "../models/restaurant_faiss_index"
EMBEDDING_MODEL_NAME = "jhgan/ko-sbert-nli"


def build_vector_store():
    """CSV 데이터를 읽어 FAISS Vector Store를 생성하고 로컬에 저장합니다."""
    if not os.path.exists(DATA_SOURCE_FILE):
        print(f"🚨 에러: '{DATA_SOURCE_FILE}'을 찾을 수 없습니다.")
        print("백엔드팀으로부터 데이터를 전달받아, 'data/' 폴더에 해당 이름으로 저장해주세요.")
        return

    df = pd.read_csv(DATA_SOURCE_FILE)
    print(f"✅ 데이터 로딩 완료: 총 {len(df)}개의 맛집 데이터를 찾았습니다.")

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

    print("임베딩 모델을 로딩합니다... (처음 실행 시 시간이 걸릴 수 있습니다)")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("✅ 임베딩 모델 로딩 완료.")

    print("Vector Store를 생성하고 저장합니다...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(VECTOR_STORE_SAVE_PATH)
    print(f"\n🎉 Vector Store 생성이 완료되었습니다! '{VECTOR_STORE_SAVE_PATH}' 폴더가 생성되었습니다.")


if __name__ == "__main__":
    build_vector_store()
