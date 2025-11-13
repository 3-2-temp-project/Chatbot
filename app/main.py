# app/main.py
# 지능형 맛집 추천 AI 챗봇 API (FastAPI + Flask-SQLAlchemy 브릿지)

import os
import logging
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text  # DB ping 등에 사용

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from flask import Flask
from flask_sqlalchemy import SQLAlchemy  # noqa: F401  (실제 객체는 app.models.db 사용)

from app.services.chatbot_logic import get_ai_response
from app.services.events import log_event, recalc_stats
from app.models import ChatRequest, ChatResponse, EventRequest, db

# ─────────────────────────────────────────────────────────────────────────────
# 환경 로드 & 기본 설정
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()  # 방어적 .env 로드 (config import 실패 대비)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

# config에서 DSN 가져오되, 실패 시 환경변수로 대체
try:
    from app.core.config import DATABASE_URI  # config는 내부적으로 .env 로드
except Exception:
    DATABASE_URI = None

DATABASE_DSN = os.getenv("DATABASE_URL") or DATABASE_URI
if not DATABASE_DSN:
    raise RuntimeError("DATABASE_URL(or DATABASE_URI) not set")

# ─────────────────────────────────────────────────────────────────────────────
# Flask (SQLAlchemy) ↔ FastAPI 브릿지
# ─────────────────────────────────────────────────────────────────────────────

flask_app = Flask(__name__)
flask_app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_DSN
flask_app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(flask_app)


class FlaskAppContextMiddleware(BaseHTTPMiddleware):
    """
    모든 FastAPI 요청을 Flask app_context로 감싸서
    Flask-SQLAlchemy 세션을 안전하게 사용하도록 보장합니다.
    """

    async def dispatch(self, request: Request, call_next):
        with flask_app.app_context():
            try:
                response = await call_next(request)
                return response
            finally:
                # 요청 끝나면 세션 정리 (누수 방지)
                db.session.remove()


# 선택적 테이블 생성 (운영 안전을 위해 기본 비활성)
if os.getenv("DB_BOOTSTRAP", "false").lower() == "true":
    with flask_app.app_context():
        db.create_all()
        logging.info("[DB] create_all completed.")

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI 앱
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="지능형 맛집 추천 AI 챗봇 API",
    description="LangChain + Text-to-SQL 기반 맛집 추천 챗봇",
    version="1.0.0",
)

app.add_middleware(FlaskAppContextMiddleware)

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

# 정적 데모: /demo 함수 라우트와 충돌 피하기 위해 /static-demo 로 마운트
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static-demo", StaticFiles(directory=STATIC_DIR, html=True), name="static-demo")


@app.get("/demo", include_in_schema=False)
def demo_root():
    """
    간단한 데모 페이지 서빙.
    static/chat.html 이 존재하면 그대로 반환합니다.
    """
    chat_html = os.path.join(STATIC_DIR, "chat.html")
    if not os.path.exists(chat_html):
        raise HTTPException(status_code=404, detail=f"chat.html not found at {chat_html}")
    return FileResponse(chat_html)


# ─────────────────────────────────────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/healthz", summary="헬스 체크 (DB 포함)")
def healthz():
    """
    애플리케이션 및 DB 연결 상태 점검 엔드포인트.
    반환: {"ok": True} (정상) / 500 (비정상)
    """
    try:
        with flask_app.app_context():
            db.session.execute(text("SELECT 1"))
        return {"ok": True}
    except Exception as e:
        logging.exception("healthz failed")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Chat API
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse, summary="챗봇 응답 생성")
async def chat_with_agent(request: ChatRequest):
    """
    요청: ChatRequest(session_id, query)
    응답: ChatResponse(answer, response, items)
    - answer, response: 동일한 본문(프론트 호환을 위해 둘 다 제공)
    - items: 추천 목록 또는 옵션
    """
    if not request.query or not request.session_id:
        raise HTTPException(status_code=400, detail="session_id와 query를 모두 입력해주세요.")

    try:
        ai: dict[str, Any] = get_ai_response(request.session_id, request.query)
        answer = ai.get("answer", "")
        items = ai.get("items") or ai.get("options") or []
        return {
            "answer": answer,     # 프론트에서 data.answer 사용 가능
            "response": answer,   # 구 구현 호환 (data.response)
            "items": items,
        }
    except Exception as e:
        logging.exception("[/chat] server error")
        raise HTTPException(status_code=500, detail=f"서버 오류 발생: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# Event API
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/event", summary="사용자 이벤트 적재")
async def push_event(req: EventRequest, background_tasks: BackgroundTasks):
    """
    사용자 이벤트 적재 및 필요 시 통계 재계산을 비동기로 실행.
    - event: 'impression' | 'click' | 'favorite' ...
    """
    try:
        _log_event(req.event, req.session_id, req.restaurant_id, req.value)
        if req.event in ("click", "favorite"):
            # 재계산은 응답 지연을 막기 위해 백그라운드에서 실행
            background_tasks.add_task(_safe_recalc_stats)
        return {"ok": True}
    except Exception as e:
        logging.exception("[/event] error")
        raise HTTPException(status_code=500, detail="event log failed")


def _safe_recalc_stats():
    """백그라운드 작업에서도 Flask app_context를 확보하여 안전하게 DB 접근."""
    with flask_app.app_context():
        recalc_stats()


# ─────────────────────────────────────────────────────────────────────────────
# 기본 루트
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", summary="API 상태 확인")
def read_root():
    return {"message": "지능형 맛집 추천 AI 챗봇 API가 동작 중입니다."}
