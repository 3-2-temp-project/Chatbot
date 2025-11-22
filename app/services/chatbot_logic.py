"""
Headless Chatbot Logic (No LLM, No prompts)
→ 프론트에서 버튼/대사 제어하고
→ 서버는 location·people·category 저장 후 SQL 추천만 수행
"""

from __future__ import annotations
import os
import logging
from typing import Any, Dict, List, Tuple
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from app.core import config

# ───────────────────────────────────────────────
# 기본 설정
# ───────────────────────────────────────────────
log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ───────────────────────────────────────────────
# DB 연결
# ───────────────────────────────────────────────
DATABASE_URI = os.getenv("DATABASE_URI", getattr(config, "DATABASE_URI", None))
if not DATABASE_URI:
    raise ValueError("❌ DATABASE_URI가 없습니다.")

try:
    _ENGINE = create_engine(
        DATABASE_URI,
        pool_pre_ping=True,
        pool_recycle=1800,
        pool_size=5,
        max_overflow=10,
        future=True,
    )

    with _ENGINE.connect() as conn:
        conn.execute(text("SELECT 1"))

    log.info(f"✅ DB 연결 성공: {DATABASE_URI}")

except SQLAlchemyError as e:
    log.exception("❌ DB 연결 실패")
    raise RuntimeError(f"Database connection failed: {e}")

# ───────────────────────────────────────────────
# 세션 상태 관리
# ───────────────────────────────────────────────
user_progress: Dict[str, Dict[str, Any]] = {}

# ───────────────────────────────────────────────
# 추천 SQL 실행
# ───────────────────────────────────────────────
def _execute_recommend_sql(location: str, category: str) -> List[Tuple]:
    sql = text("""
    SELECT 
        res_id AS id,
        res_name AS name,
        category,
        address,
        lat,
        lng
    FROM restaurant_info
    WHERE address LIKE :loc
      AND category = :cat
    LIMIT 10;
""")

    with _ENGINE.begin() as conn:
        return conn.execute(sql, {
            "loc": f"%{location}%",
            "cat": category
        }).fetchall()


# ───────────────────────────────────────────────
# 메인 처리 로직
# ───────────────────────────────────────────────
def get_ai_response(session_id: str, user_query: str) -> Dict[str, Any]:
    """
    프론트가 버튼 기반으로 user_query를 보냄
    서버는 이를 순서대로 저장하고
    모든 항목이 채워지면 SQL 추천을 수행
    """

    progress = user_progress.get(session_id, {})

    # 1️⃣ 지역 선택
    if "location" not in progress:
        progress["location"] = user_query
        user_progress[session_id] = progress
        return {"type": "progress"}  # 프론트가 단계 전환

    # 2️⃣ 인원 선택
    if "people" not in progress:
        progress["people"] = user_query
        user_progress[session_id] = progress
        return {"type": "progress"}

    # 3️⃣ 카테고리 선택 → 추천 실행
    if "category" not in progress:
        progress["category"] = user_query
        user_progress[session_id] = progress

        location = progress["location"]
        category = progress["category"]

        rows = _execute_recommend_sql(location, category)

        if not rows:
            return {
                "type": "recommend",
                "items": []
            }

        # 결과 구성
        items = []
        for _id, name, category, address, lat, lng in rows:
            items.append({
                "id": _id,
                "name": name,
                "address": address,
                "lat": float(lat),
                "lng": float(lng),
                "category": category,
                "score": None,   # rating 없음 → None
            })

        return {
            "type": "recommend",
            "items": items
        }

    # 이후 단계라면 초기화 안내 가능
    return {
        "type": "done",
    }
