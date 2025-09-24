import sys
import os

# 프로젝트 루트(Chatbot)를 PYTHONPATH에 추가
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.services.chatbot_logic import get_ai_response, _log_event, recalc_stats
from app.models import ChatRequest, ChatResponse, EventRequest

app = FastAPI(
    title="지능형 맛집 추천 AI 챗봇 API",
    description="LangChain과 Text-to-SQL을 사용한 맛집 추천 챗봇입니다.",
    version="1.0.0",
)

# CORS (필요 시 도메인 추가)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 데모 서빙
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
print("STATIC_DIR:", STATIC_DIR, "exists:", os.path.exists(STATIC_DIR))
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
    try:
        ai_response = get_ai_response(request.session_id, request.query)
        return ai_response
    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail=f"서버 오류 발생: {str(e)}")

# 사용자 행동 이벤트 적재 (옵션: 클릭/즐겨찾기 등)
@app.post("/event", summary="사용자 이벤트 적재")
async def push_event(req: EventRequest):
    try:
        _log_event(req.event, req.session_id, req.restaurant_id, req.value)
        # 클릭/즐겨찾기는 신호가 강하므로 즉시 반영하면 체감이 좋음
        if req.event in ("click", "favorite"):
            recalc_stats()
        return {"ok": True}
    except Exception as e:
        print("[/event error]", e)
        raise HTTPException(status_code=500, detail="event log failed")

# 상태 확인
@app.get("/", summary="API 상태 확인")
def read_root():
    return {"message": "지능형 맛집 추천 AI 챗봇 API가 동작 중입니다."}
