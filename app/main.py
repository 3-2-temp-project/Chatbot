# --- RAG(Retrieval-Augmented Generation)를 사용하여 맛집을 추천하는 챗봇 API 서버 ---

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from peft import PeftModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
import os

# --- 기본 설정 ---
app = FastAPI()

# --- 경로 설정 (새로운 폴더 구조에 맞게 수정) ---
# 현재 파일(main.py)의 위치는 app/ 이므로, 상위 폴더로 이동(../) 후 각 폴더로 진입
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
FINETUNED_ADAPTER_PATH = "../models/chatbot_model"
VECTOR_STORE_PATH = "../models/restaurant_faiss_index"
EMBEDDING_MODEL_NAME = "jhgan/ko-sbert-nli"


# --- 모델 및 Vector Store 로드 ---
retriever = None
print("모델 및 Vector Store 로딩을 시작합니다...")

<<<<<<< Updated upstream
if os.path.exists(VECTOR_STORE_PATH):
=======
# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 데모 서빙
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/demo", StaticFiles(directory=STATIC_DIR, html=True), name="demo")

@app.get("/demo", include_in_schema=False)
def demo_root():
    chat_html = os.path.join(STATIC_DIR, "chat.html")
    if not os.path.exists(chat_html):
        raise HTTPException(status_code=404, detail=f"chat.html not found at {chat_html}")
    return FileResponse(chat_html)

# 채팅
@app.post("/chat", response_model=ChatResponse, summary="챗봇 응답 생성")
async def chat_with_agent(request: ChatRequest):
    if not request.query or not request.session_id:
        raise HTTPException(status_code=400, detail="session_id와 query를 모두 입력해주세요.")
>>>>>>> Stashed changes
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embedding_model, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
        print("✅ Vector Store 로딩 완료.")
    except Exception as e:
        print(f"🚨 Vector Store 로딩 실패: {e}")
else:
    print(f"🚨 경고: '{VECTOR_STORE_PATH}' 폴더를 찾을 수 없습니다.")
    print("챗봇이 정상적으로 작동하려면 'python app/build_db.py'를 먼저 실행해야 합니다.")


print("LLM 로딩 중...")
chatbot_pipeline = pipeline("text-generation", model=BASE_MODEL_ID, device_map="auto", torch_dtype=torch.float16)
chatbot_pipeline.model = PeftModel.from_pretrained(chatbot_pipeline.model, FINETUNED_ADAPTER_PATH)
print("✅ LLM 로딩 완료.")


# --- API 정의 ---
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

PROMPT_TEMPLATE = """
너는 서울과 경기 지역의 맛집 추천 전문 챗봇이야.
아래 [맛집 정보]를 바탕으로, 사용자의 [요청]에 가장 적합한 맛집을 추천해줘.
추천할 때는 식당 이름, 주소, 주요 메뉴를 명확히 언급하고, 자연스러운 대화체로 설명해줘.
만약 [맛집 정보]가 비어있거나 [요청]과 관련이 없다면, "죄송하지만 요청하신 조건에 맞는 맛집을 찾기 어렵습니다. 저는 업무추진비로 검증된 서울, 경기 지역의 식당에 대해서만 답변해 드릴 수 있어요." 라고 정중하게 답변해줘. 절대로 정보를 꾸며내면 안 돼.
[맛집 정보]
{context}
[요청]
{question}
[추천 답변]
"""

@app.post("/restaurant-chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not retriever:
        return ChatResponse(response="챗봇의 데이터베이스가 준비되지 않았습니다. 'python app/build_db.py'를 실행했는지 확인해주세요.")

    user_question = request.message
    retrieved_docs = retriever.invoke(user_question)
    context_text = "\n\n".join([f"- {doc.page_content}" for doc in retrieved_docs])
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=user_question)

    result = chatbot_pipeline(
        prompt, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9,
        pad_token_id=chatbot_pipeline.tokenizer.eos_token_id
    )[0]["generated_text"]

    answer = result.split("[추천 답변]")[-1].strip()
    return ChatResponse(response=answer)
