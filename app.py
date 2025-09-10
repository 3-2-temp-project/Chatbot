from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# FastAPI 인스턴스 생성
app = FastAPI()

# 요청/응답 데이터 모델 정의
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# 1. 학습된 모델 로드
MODEL_DIR = "./chatbot_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map="auto")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    user_input = request.message

    # 2. 입력 토큰화
    inputs = tokenizer(user_input, return_tensors="pt").to(model.device)

    # 3. 모델 추론
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9,
    )

    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return ChatResponse(response=response_text)


"""uvicorn app:app --host 0.0.0.0 --port 8000 --reload으로 실행"""