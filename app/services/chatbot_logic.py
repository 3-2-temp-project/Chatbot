from __future__ import annotations

import os
import re
import logging
from typing import Any, Dict, List, Tuple

from sqlalchemy import create_engine, text
from langchain_community.utilities import SQLDatabase
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import create_sql_query_chain

from app.core import config
import app.services.prompts as prompts  # 이모지/문구 설정 연동

# ─────────────────────────────────────────────────────────────
# 전역 설정
# ─────────────────────────────────────────────────────────────
log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_ENGINE = create_engine(config.DATABASE_URI, pool_pre_ping=True, future=True)
db = SQLDatabase.from_uri(config.DATABASE_URI)

llm = HuggingFacePipeline.from_model_id(
    model_id=config.LLM_MODEL_ID,
    task=config.LLM_TASK,
    pipeline_kwargs={
        "max_new_tokens": config.LLM_MAX_NEW_TOKENS,
        "temperature": config.LLM_TEMPERATURE,
        "do_sample": True if config.LLM_TEMPERATURE > 0 else False,
        "return_full_text": False,
    },
    device=config.DEVICE,
)

user_progress: Dict[str, Dict[str, Any]] = {}
STATS_IMMEDIATE = os.getenv("STATS_IMMEDIATE", "false").lower() == "true"
_ALLOWED_TABLES = {"restaurants"}

# ─────────────────────────────────────────────────────────────
# 슬롯 추출
# ─────────────────────────────────────────────────────────────
def parse_initial_query(query: str, progress: Dict[str, Any]) -> Dict[str, Any]:
    if "location" not in progress:
        loc = re.search(r"(\S+역|\S+동|\S+시|\S+구)", query)
        if loc:
            progress["location"] = loc.group(0)
        return progress

    if "people" not in progress:
        people = re.search(r"(\d+)\s*명", query)
        if people:
            progress["people"] = int(people.group(1))
        return progress

    if "category" not in progress:
        cat = re.search(r"(한식|일식|중식|양식)", query)
        if cat:
            progress["category"] = cat.group(0)
        return progress

    if "purpose" not in progress:
        purp = re.search(r"(오찬|접대|회식)", query)
        if purp:
            progress["purpose"] = purp.group(0)
        return progress

    return progress

# ─────────────────────────────────────────────────────────────
# 이벤트 기록 & 통계 갱신
# ─────────────────────────────────────────────────────────────
def _log_event(event: str, session_id: str, restaurant_id: int | None = None, value: str | None = None) -> None:
    try:
        with _ENGINE.begin() as conn:
            conn.execute(
                text("INSERT INTO user_events (session_id, restaurant_id, event, value) VALUES (:sid, :rid, :event, :val)"),
                {"sid": session_id, "rid": restaurant_id, "event": event, "val": value},
            )
    except Exception as e:
        log.exception("[event log error] %s", e)


def recalc_stats() -> None:
    with _ENGINE.begin() as conn:
        conn.exec_driver_sql("""
        INSERT INTO restaurant_stats(restaurant_id, impressions_7d, impressions_30d, clicks_7d, clicks_30d)
        SELECT r.id,
               SUM(CASE WHEN ue.event='impression' AND ue.created_at >= NOW() - INTERVAL '7 days' THEN 1 ELSE 0 END),
               SUM(CASE WHEN ue.event='impression' AND ue.created_at >= NOW() - INTERVAL '30 days' THEN 1 ELSE 0 END),
               SUM(CASE WHEN ue.event='click' AND ue.created_at >= NOW() - INTERVAL '7 days' THEN 1 ELSE 0 END),
               SUM(CASE WHEN ue.event='click' AND ue.created_at >= NOW() - INTERVAL '30 days' THEN 1 ELSE 0 END)
        FROM restaurants r
        LEFT JOIN user_events ue ON ue.restaurant_id = r.id
        GROUP BY r.id
        ON CONFLICT (restaurant_id) DO UPDATE
        SET impressions_7d = EXCLUDED.impressions_7d,
            impressions_30d = EXCLUDED.impressions_30d,
            clicks_7d = EXCLUDED.clicks_7d,
            clicks_30d = EXCLUDED.clicks_30d;
        """)

# ─────────────────────────────────────────────────────────────
# SQL 유틸
# ─────────────────────────────────────────────────────────────
def _is_safe_sql(sql: str) -> bool:
    s = sql.strip().lower()
    if not s.startswith("select"):
        return False
    forbidden = ("insert", "update", "delete", "drop", "alter", "create", "truncate")
    if any(k in s for k in forbidden):
        return False
    return True


def _is_schema_safe(sql: str) -> bool:
    s = sql.lower()
    if any(word in s for word in [" join ", " with ", " group by ", " union "]):
        return False
    tables = re.findall(r"\bfrom\s+([a-zA-Z_][\w]*)", s)
    return all(t in _ALLOWED_TABLES for t in tables)


def _normalize_sql(sql: str) -> str:
    if re.search(r"\blimit\b", sql, flags=re.I) is None:
        sql = sql.rstrip(";") + " LIMIT 10;"
    return sql


def _execute_sql(sql: str) -> List[Tuple]:
    with _ENGINE.begin() as conn:
        return list(conn.execute(text(sql)).fetchall())


def _render_results(rows: List[Tuple]) -> str:
    if not rows:
        return ""
    items = []
    for row in rows:
        _id, name, category, location, rating = row[:5]
        items.append(f"- **{name}** — {location} · {category} · 평점 {rating}")
    return "\n".join(items)


def _esc(v: str) -> str:
    return str(v).replace("'", "''")[:50] if v else ""


def _fetch_geo_map_by_names(names: List[str]) -> Dict[str, Dict[str, Any]]:
    if not names:
        return {}
    placeholders = ", ".join([f":n{i}" for i in range(len(names))])
    q = text(f"""
        SELECT res_name, address, lat, lng
        FROM restaurant_info
        WHERE res_name IN ({placeholders})
    """)
    params = {f"n{i}": n for i, n in enumerate(names)}
    geo: Dict[str, Dict[str, Any]] = {}
    with _ENGINE.begin() as conn:
        for rn, addr, lat, lng in conn.execute(q, params).fetchall():
            geo[str(rn)] = {"address": addr, "lat": float(lat), "lng": float(lng)}
    return geo


def _names_to_ids(names: List[str]) -> List[int]:
    if not names:
        return []
    placeholders = ", ".join([f":n{i}" for i in range(len(names))])
    q = text(f"SELECT id, name FROM restaurants WHERE name IN ({placeholders})")
    params = {f"n{i}": n for i, n in enumerate(names)}
    with _ENGINE.begin() as conn:
        rows = conn.execute(q, params).fetchall()
    name_to_id = {str(name): int(rid) for rid, name in rows}
    return [name_to_id[n] for n in names if n in name_to_id]


def _check_triggers(user_query: str, progress: Dict[str, Any]) -> None:
    if "오늘 점심" in user_query:
        progress["_today_lunch"] = True


def _maybe_today_lunch(progress: Dict[str, Any]) -> bool:
    return bool(progress.get("_today_lunch"))

# ─────────────────────────────────────────────────────────────
# 메인 로직
# ─────────────────────────────────────────────────────────────
def get_ai_response(session_id: str, user_query: str) -> Dict[str, Any]:
    log.info(f"\n=== [Chat Start] session={session_id} user_query='{user_query}' ===")
    progress = user_progress.get(session_id, {})
    log.info(f"[Chat] progress(before)={progress}")

    _check_triggers(user_query, progress)
    soft = parse_initial_query(user_query, progress)
    log.info(f"[Chat] soft(extracted)={soft}")
    log.info(f"[Chat] progress(after)={progress}")

    # 단계별 대화 흐름
    if "location" not in progress:
        progress["last_question"] = "location"
        user_progress[session_id] = progress
        return {"type": "text", "answer": "원하시는 지역을 알려주세요.", "options": None}

    if "people" not in progress:
        progress["last_question"] = "people"
        user_progress[session_id] = progress
        return {"type": "text", "answer": f"{progress['location']} 근처에서 식사하실 인원은 몇 명인가요?", "options": None}

    if "category" not in progress:
        progress["last_question"] = "category"
        user_progress[session_id] = progress
        return {"type": "buttons", "answer": "어떤 종류의 음식을 원하시나요?", "options": ["한식", "중식", "일식", "양식"]}

    if "purpose" not in progress:
        progress["last_question"] = "purpose"
        user_progress[session_id] = progress
        return {"type": "buttons", "answer": "방문의 목적은 무엇인가요?", "options": ["오찬", "회식", "접대"]}

    # 모든 슬롯 채워짐 → 추천 수행
    try:
        question = (
            f"{progress.get('location')} 근처에서 {progress.get('people')}명이 먹기 좋은 "
            f"{progress.get('category')} 식당을 추천해줘. 목적은 '{progress.get('purpose')}'. "
            "단일 SELECT 문만 생성하고 ';'로 끝내. "
            "오직 'restaurants' 테이블만 사용. "
            "컬럼: id,name,category,location,rating. "
            "location은 LIKE '%강남%' 같은 부분일치만 허용. LIMIT 10 이하."
        )

        chain = create_sql_query_chain(llm, db)
        sql = chain.invoke({"question": question})
        sql = sql.strip().strip("```").replace("sql", "").strip()
        sql = _normalize_sql(sql)
        log.info(f"[SQL Generated] {sql}")  # ✅ SQL 로그 추가

        if not _is_safe_sql(sql) or not _is_schema_safe(sql):
            sql = _fallback_sql_from_slots(progress)
            log.info("[Fallback SQL used]")

        raw = _execute_sql(sql)
        names = _extract_names_from_raw(raw)

        for rid in _names_to_ids(names):
            _log_event("impression", session_id, rid, None)

        if STATS_IMMEDIATE:
            recalc_stats()

        if not raw:
            answer = f"{progress['location']} 근처에서 조건에 맞는 맛집을 찾지 못했어요."
            items = []
        else:
            rendered = _render_results(raw)
            geo_map = _fetch_geo_map_by_names(names)
            answer = "다음 추천 결과입니다:\n" + rendered
            items = []
            for r in raw:
                _id, name, category, location, rating = r[:5]
                geo = geo_map.get(str(name))
                items.append({
                    "id": int(_id),
                    "name": str(name),
                    "address": (geo.get("address") if geo else location),
                    "lat": (geo.get("lat") if geo else None),
                    "lng": (geo.get("lng") if geo else None),
                    "category": category,
                    "score": rating,
                })

        progress.pop("last_question", None)
        user_progress[session_id] = progress
        return {"type": "text", "answer": answer, "items": items, "options": items}

    except Exception as e:
        log.exception("[Agent Error] %s", e)
        return {"type": "text", "answer": "서버 내부 오류가 발생했습니다.", "options": None}


def _fallback_sql_from_slots(progress: Dict[str, Any]) -> str:
    loc = _esc(progress.get("location", ""))
    cat = _esc(progress.get("category", ""))
    purp = progress.get("purpose", "")

    where = []
    if loc:
        where.append(f"r.location LIKE '%{loc}%'")
    if cat:
        where.append(f"r.category = '{cat}'")
    if "회식" in purp:
        where.append("r.has_private_room = TRUE")

    where_sql = " AND ".join(where) if where else "1=1"
    return (
        "SELECT r.id, r.name, r.category, r.location, r.rating "
        "FROM restaurants r "
        "LEFT JOIN restaurant_stats s ON s.restaurant_id = r.id "
        f"WHERE {where_sql} "
        "ORDER BY COALESCE(s.popularity_score, r.rating/5.0) DESC "
        "LIMIT 10;"
    )


def _extract_names_from_raw(rows: List[Tuple]) -> List[str]:
    return [str(row[1]) for row in rows if len(row) > 1]
