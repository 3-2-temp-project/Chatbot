"""
챗봇 메인 orchestrator — 각 모듈을 조립하여 응답 생성
"""

import logging
from typing import Any, Dict

from app.services.slots import parse_initial_query, check_triggers
from app.services.sql_builder import (
    generate_sql_from_llm,
    is_safe_sql,
    is_schema_safe,
    fallback_sql_from_slots,
)
from app.services.db_query import (
    execute_sql,
    render_results,
    names_to_ids,
    fetch_geo_map_by_names,
)
from app.services.events import log_event

log = logging.getLogger(__name__)
user_progress: Dict[str, Dict[str, Any]] = {}


def get_ai_response(session_id: str, user_query: str) -> Dict[str, Any]:
    """대화 흐름 전체를 제어하는 메인 함수"""

    progress = user_progress.get(session_id, {})
    log.info(f"[Chat] session={session_id}, query={user_query}")
    log.info(f"[Before] progress={progress}")

    # -----------------------------------------------------------
    # 1) 트리거 감지 및 슬롯 업데이트
    # -----------------------------------------------------------
    check_triggers(user_query, progress)
    parse_initial_query(user_query, progress)
    user_progress[session_id] = progress

    # -----------------------------------------------------------
    # 2) 슬롯이 모두 채워졌는지 확인
    # -----------------------------------------------------------
    if "location" not in progress:
        return {"type": "text", "answer": "원하시는 지역을 알려주세요."}

    if "people" not in progress:
        loc = progress["location"]
        return {"type": "text", "answer": f"{loc} 근처 인원 수를 알려주세요."}

    if "category" not in progress:
        return {
            "type": "buttons",
            "answer": "어떤 종류의 음식을 원하시나요?",
            "options": ["한식", "중식", "일식", "양식"],
        }

    if "purpose" not in progress:
        return {
            "type": "buttons",
            "answer": "방문의 목적은 무엇인가요?",
            "options": ["오찬", "회식", "접대"],
        }

    # -----------------------------------------------------------
    # 3) SQL 생성 (LLM → 안전성 필터 → fallback)
    # -----------------------------------------------------------
    sql = generate_sql_from_llm(progress)
    log.info(f"[SQL LLM] {sql}")

    if not is_safe_sql(sql) or not is_schema_safe(sql):
        sql = fallback_sql_from_slots(progress)
        log.warning("[SQL Fallback] 안전하지 않아 fallback SQL로 전환")

    # -----------------------------------------------------------
    # 4) SQL 실행 및 결과 조회
    # -----------------------------------------------------------
    rows = execute_sql(sql)
    names = [r[1] for r in rows]

    # 사용자 행동 기록
    for rid in names_to_ids(names):
        log_event("impression", session_id, rid)

    if not rows:
        return {"type": "text", "answer": "조건에 맞는 식당을 찾지 못했어요."}

    rendered = render_results(rows)
    geo_map = fetch_geo_map_by_names(names)

    # -----------------------------------------------------------
    # 5) 최종 응답 구성
    # -----------------------------------------------------------
    items = []
    for r in rows:
        _id, name, category, location, rating = r[:5]
        geo = geo_map.get(name, {})

        items.append({
            "id": _id,
            "name": name,
            "category": category,
            "address": geo.get("address", location),
            "lat": geo.get("lat"),
            "lng": geo.get("lng"),
            "score": rating,
        })

    return {
        "type": "text",
        "answer": "추천 결과입니다:\n" + rendered,
        "items": items,
        "options": items,
    }
