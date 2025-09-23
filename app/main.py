import sys
import os

# 이 파일이 uvicorn으로 직접 실행될 때를 대비한 경로 설정
# 예: uvicorn app.main:app --reload
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from fastapi import FastAPI, HTTPException
from app.services.chatbot_logic import get_ai_response
from app.models import ChatRequest, ChatResponse

app = FastAPI(
    title="지능형 맛집 추천 AI 챗봇 API",
    description="LangChain과 Text-to-SQL을 사용한 맛집 추천 챗봇입니다.",
    version="1.0.0",
)

@app.post("/chat", response_model=ChatResponse, summary="챗봇 응답 생성")
async def chat_with_agent(request: ChatRequest):
    """
    사용자 질문을 받아 AI 에이전트의 답변을 반환합니다.
    """
    if not request.query or not request.session_id:
        raise HTTPException(status_code=400, detail="session_id와 query를 모두 입력해주세요.")

    try:
        ai_response = get_ai_response(request.session_id, request.query)
        return ai_response
    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail=f"서버 오류 발생: {str(e)}")

@app.get("/", summary="API 상태 확인")
def read_root():
    """API가 정상 동작하는지 확인하는 기본 엔드포인트"""
    return {"message": "지능형 맛집 추천 AI 챗봇 API가 동작 중입니다."}
