"""
챗봇 메인 orchestrator — 각 모듈을 조립하여 응답 생성
"""

import logging
from typing import Any, Dict
from app.services.slots import extract_slots
from app.services.sql_builder import (
    generate_sql_from_llm,
    is_safe_sql,
    is_schema_safe,
    fallback_sql_from_slots,
)
from app.services.db_query import execute_sql
from app.services.events import log_event, recalc_stats
from app.core.config import STATS_IMMEDIATE

log = logging.getLogger(__name__)
user_progress: Dict[str, Dict[str, Any]] = {}


def get_ai_response(session_id: str, user_query: str) -> Dict[str, Any]:
    log.info(f"[Chat] session={session_id}, query={user_query}")
    progress = user_progress.get(session_id, {})

    # 1) 슬롯 추출
    before = progress.copy()
    progress = extract_slots(user_query, progress)
    user_progress[session_id] = progress

    log.info(f"[Before] {before}")
    log.info(f"[After Slot] {progress}")

    # 슬롯 채우는 단계별 질문
    if "location" not in progress:
        return {"type": "text", "answer": "어디 근처에서 찾을까요?"}

    if "people" not in progress:
        return {"type": "text", "answer": f"{progress['location']} 근처에서 몇 명이 식사하나요?"}

    if "category" not in progress:
        return {"type": "buttons", "answer": "어떤 종류의 음식을 원하시나요?", "options": ["한식", "중식", "일식", "양식"]}

    if "purpose" not in progress:
        return {"type": "buttons", "answer": "방문의 목적은 무엇인가요?", "options": ["회식", "오찬", "접대"]}

    # 모든 슬롯 채워졌으면 추천 시작
    sql = generate_sql_from_llm(progress)

    if not sql or not is_safe_sql(sql) or not is_schema_safe(sql):
        sql = fallback_sql_from_slots(progress)

    raw = execute_sql(sql)

    # 이벤트 기록
    for r in raw:
        rid = r[0]
        log_event("impression", session_id, rid)

    if STATS_IMMEDIATE:
        recalc_stats()

    if not raw:
        return {"type": "text", "answer": "조건에 맞는 식당을 찾지 못했습니다."}

    # 결과 구성
    items = []
    for row in raw:
        _id, name, category, location, rating = row[:5]
        items.append({
            "id": _id,
            "name": name,
            "category": category,
            "location": location,
            "rating": rating,
        })

    return {
        "type": "text",
        "answer": "추천 결과입니다:",
        "items": items,
        "options": items,
    }
