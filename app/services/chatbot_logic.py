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
    print("────────────────────────────────────────────")
    print("🚀 SQL 실행 시작")
    print("📌 location =", repr(location))
    print("📌 category =", repr(category))
    print("📌 LIKE =", repr(f"%{location}%"))
    print("🔎 실제 SQL 조건: address LIKE '%{}%' AND category='{}'".format(location, category))

    sql = text("""
    SELECT 
        res_id AS id,
        res_name AS name,
        category,
        address,
        lat,
        lng
    FROM restaurant_info
    WHERE TRIM(address) LIKE :loc
      AND TRIM(category) = :cat
    LIMIT 10;
    """)

    with _ENGINE.begin() as conn:
        rows = conn.execute(sql, {
            "loc": f"%{location}%",
            "cat": category
        }).fetchall()

    print("📌 SQL 결과 개수 =", len(rows))
    for row in rows:
        print("➡️ 결과 Row =", row)

    print("────────────────────────────────────────────")
    return rows

# ───────────────────────────────────────────────
# 메인 처리 로직
# ───────────────────────────────────────────────
def get_ai_response(session_id: str, location: str | None, category: str | None):

    # 둘 다 있어야 실제 쿼리 실행
    if location and category:
        rows = _execute_recommend_sql(location, category)

        return {
            "response": "추천 결과",
            "items": [
                {
                    "id": r[0],
                    "name": r[1],
                    "category": r[2],
                    "address": r[3],
                    "lat": float(r[4]),
                    "lng": float(r[5]),
                }
                for r in rows
            ],
        }

    # 위치만 있는 경우 → 다음 버튼 출력
    if location and not category:
        return { "response": "카테고리를 선택해주세요", "items": [] }

    # 첫 진입
    return { "response": "위치를 선택해주세요", "items": [] }
