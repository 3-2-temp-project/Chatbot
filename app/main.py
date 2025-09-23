# app/main.py
import sys
import os

# 프로젝트 루트(Chatbot)를 PYTHONPATH에 추가 (uvicorn 직접 실행 대비)
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.services.chatbot_logic import get_ai_response
from app.models import ChatRequest, ChatResponse

# =========================
# FastAPI 앱
# =========================
app = FastAPI(
    title="지능형 맛집 추천 AI 챗봇 API",
    description="LangChain과 Text-to-SQL을 사용한 맛집 추천 챗봇입니다.",
    version="1.0.0",
)

# (선택) CORS: 파일로 열린 HTML/다른 포트의 프론트에서 호출 시 필요
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

# =========================
# 정적 파일 서빙 (로컬 데모 UI)
# =========================
# 정적 폴더를 절대 경로로 고정
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
print("STATIC_DIR:", STATIC_DIR, "exists:", os.path.exists(STATIC_DIR))

# /demo 경로에 정적 폴더 마운트 (http://127.0.0.1:8000/demo/chat.html)
app.mount("/demo", StaticFiles(directory=STATIC_DIR, html=True), name="demo")

# /demo 로 접근하면 chat.html 바로 반환 (타이핑 편의)
@app.get("/demo", include_in_schema=False)
def demo_root():
    chat_html = os.path.join(STATIC_DIR, "chat.html")
    if not os.path.exists(chat_html):
        raise HTTPException(
            status_code=404,
            detail=f"chat.html not found. Place the file at: {chat_html}"
        )
    return FileResponse(chat_html)

# =========================
# 엔드포인트
# =========================
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
    """API가 정상적으로 동작하는지 확인하는 기본 엔드포인트입니다."""
    return {"message": "지능형 맛집 추천 AI 챗봇 API가 동작 중입니다."}
