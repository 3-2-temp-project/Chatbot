from fastapi import FastAPI
from .models import ChatRequest, ChatResponse
from .agent_logic import get_ai_response

# FastAPI 앱 인스턴스 생성
app = FastAPI(
    title="지능형 맛집 추천 AI 챗봇 API",
    description="LangChain과 Text-to-SQL을 사용한 맛집 추천 챗봇입니다.",
    version="1.0.0"
)

@app.get("/", summary="API 상태 확인")
def read_root():
    """API 서버가 정상적으로 실행 중인지 확인하는 기본 엔드포인트"""
    return {"status": "ok", "message": "맛집 추천 챗봇 API가 정상적으로 동작하고 있습니다."}


@app.post("/chat", response_model=ChatResponse, summary="챗봇 응답 생성")
async def chat_with_agent(request: ChatRequest):
    """
    사용자 질문을 받아 AI 에이전트의 답변을 반환합니다.

    - **session_id**: 각 사용자를 구분하기 위한 고유 ID입니다.
    - **query**: 사용자가 입력한 자연어 질문입니다.

    응답은 `type` 필드에 따라 달라집니다.
    - `type: 'text'`: 일반적인 텍스트 답변입니다.
    - `type: 'buttons'`: 사용자에게 선택지를 제공해야 할 때 사용됩니다. `options` 필드에 버튼 텍스트 목록이 포함됩니다.
    """
    response_data = get_ai_response(request.session_id, request.query)
    return ChatResponse(**response_data)