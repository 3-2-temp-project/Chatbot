import sys
import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.services.chatbot_logic import get_ai_response
from app.models import ChatRequest, ChatResponse, EventRequest, db
from app.core.config import DATABASE_URI

from flask import Flask
from flask_sqlalchemy import SQLAlchemy

#  Flask + SQLAlchemy 초기화  (FastAPI와 함께 사용)
flask_app = Flask(__name__)
flask_app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URI
flask_app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(flask_app)

with flask_app.app_context():
    db.create_all()


#  FastAPI 초기화
app = FastAPI(
    title="지능형 맛집 추천 AI 챗봇 API",
    description="LangChain + Text-to-SQL 기반 맛집 추천 챗봇",
    version="1.0.0",
)


#  CORS 설정 (프론트 & 기존 백엔드 접근 가능)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",                      # 필요하면 localhost:* 로 제한 가능
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5000",
        "http://127.0.0.1:5000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#  /chat  → 챗봇 메인 API
@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):

    # 🔥 query 사용 안 함 → location + category 기반
    if not request.session_id:
        raise HTTPException(status_code=400, detail="session_id가 필요합니다.")

    try:
        ai_response = get_ai_response(
            session_id=request.session_id,
            location=request.location,
            category=request.category
        )

        return {
            "response": ai_response.get("response", ""),
            "items": ai_response.get("items", []),
        }

    except Exception as e:
        print("[Server Error]", e)
        raise HTTPException(status_code=500, detail=f"서버 오류 발생: {e}")



#  /event → 클릭/좋아요 등 로그 저장
@app.post("/event", summary="사용자 이벤트 로깅")
async def push_event(req: EventRequest):
    try:
        _log_event(req.event, req.session_id, req.restaurant_id, req.value)
        return {"ok": True}

    except Exception as e:
        print("[/event error]", e)
        raise HTTPException(status_code=500, detail="event log failed")


#  상태 확인
@app.get("/", summary="API 상태 확인")
def read_root():
    return {"message": "지능형 맛집 추천 AI 챗봇 API가 동작 중입니다."}
