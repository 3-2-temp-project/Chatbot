from fastapi import FastAPI
from fastapi.responses import StreamingResponse # 👈 StreamingResponse import 추가
from app.services.chatbot_logic import get_ai_response_stream # 👈 스트리밍 전용 함수 import
from app.models import ChatRequest

app = FastAPI(
    title="지능형 맛집 추천 AI 챗봇 API",
    description="LangChain과 Text-to-SQL을 사용한 맛집 추천 챗봇입니다.",
    version="1.0.0"
)

# 스트리밍 응답을 위한 새로운 엔드포인트
@app.post("/chat/stream", summary="챗봇 응답 실시간 생성")
async def chat_with_agent_stream(request: ChatRequest):
    """
    사용자 질문을 받아 AI 에이전트의 답변을 실시간 스트림으로 반환합니다.
    """
    # get_ai_response_stream 함수는 제너레이터(generator)여야 합니다.
    # 이 함수는 텍스트 조각(token)을 하나씩 생성하여 yield합니다.
    return StreamingResponse(
        get_ai_response_stream(request.session_id, request.query),
        media_type="text/event-stream"
    )

# 기존 엔드포인트는 테스트나 비-스트리밍용으로 남겨둘 수 있습니다.
# from app.models import ChatResponse
# from app.services.chatbot_logic import get_ai_response
# @app.post("/chat", response_model=ChatResponse, summary="챗봇 응답 생성")
# ...