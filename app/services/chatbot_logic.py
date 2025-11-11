"""
Chatbot logic for restaurant recommendation (LangChain 0.2.x + Flask/PostgreSQL).
"""

from __future__ import annotations
import os
import re
import logging
from typing import Any, Dict, List, Tuple
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_huggingface import HuggingFacePipeline

from app.core import config
import app.services.prompts as prompts  # prompts.py 안에 USE_EMOJI = True 추가 필요

# ───────────────────────────────────────────────
# 기본 설정
# ───────────────────────────────────────────────
log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ───────────────────────────────────────────────
# Database 연결 (SQLAlchemy + LangChain 호환)
# ───────────────────────────────────────────────
DATABASE_URI = os.getenv("DATABASE_URI", getattr(config, "DATABASE_URI", None))

if not DATABASE_URI:
    raise ValueError("❌ DATABASE_URI가 설정되어 있지 않습니다. .env 또는 config.py를 확인하세요.")

try:
    _ENGINE = create_engine(
        DATABASE_URI,
        pool_pre_ping=True,      # DB 연결 사전검사
        pool_recycle=1800,       # 30분마다 재연결
        pool_size=5,
        max_overflow=10,
        future=True,
    )

    # LangChain용 SQLDatabase 객체 (LLM이 DB 스키마를 인식할 수 있도록)
    db = SQLDatabase.from_uri(DATABASE_URI)

    # 연결 테스트
    with _ENGINE.connect() as conn:
        conn.execute(text("SELECT 1"))
    log.info(f"✅ Database 연결 성공: {DATABASE_URI}")

except SQLAlchemyError as e:
    log.exception("❌ Database 연결 실패: %s", e)
    raise RuntimeError(f"Database connection failed: {e}")

# ───────────────────────────────────────────────
# HuggingFace 기반 LLM 파이프라인
# ───────────────────────────────────────────────
llm = HuggingFacePipeline.from_model_id(
    model_id=config.LLM_MODEL_ID,
    task=config.LLM_TASK,
    pipeline_kwargs={
        "max_new_tokens": config.LLM_MAX_NEW_TOKENS,
        "temperature": config.LLM_TEMPERATURE,
        "do_sample": config.LLM_TEMPERATURE > 0,
        "return_full_text": False,
    },
    device=config.DEVICE,  # 0=CUDA, -1=CPU
)

# ───────────────────────────────────────────────
# 세션 상태 / 설정
# ───────────────────────────────────────────────
user_progress: Dict[str, Dict[str, Any]] = {}
STATS_IMMEDIATE = os.getenv("STATS_IMMEDIATE", "false").lower() == "true"
_ALLOWED_TABLES = {"restaurant_info"}  # 실제 DB 테이블 이름


# ───────────────────────────────────────────────
# 슬롯 추출 (location → people → category → purpose)
# ───────────────────────────────────────────────
def parse_initial_query(query: str, progress: Dict[str, Any]) -> Dict[str, Any]:
    """문장에서 슬롯(location, people, category, purpose)을 단계별로 추출"""
    # 위치
    if "location" not in progress:
        loc = re.search(r"(\S+역|\S+동|\S+시|\S+구)", query)
        if loc:
            progress["location"] = loc.group(0)
        return progress

    # 인원수
    if "people" not in progress:
        people = re.search(r"(\d+)\s*명", query)
        if people:
            progress["people"] = int(people.group(1))
        return progress

    # 음식 종류
    if "category" not in progress:
        cat = re.search(r"(한식|일식|중식|양식)", query)
        if cat:
            progress["category"] = cat.group(0)
        return progress

    # 목적
    if "purpose" not in progress:
        purp = re.search(r"(오찬|접대|회식)", query)
        if purp:
            progress["purpose"] = purp.group(0)
        return progress

    return progress


# ───────────────────────────────────────────────
# 이벤트 기록 & 통계 갱신
# ───────────────────────────────────────────────
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
    """restaurant_stats 갱신"""
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

        conn.exec_driver_sql("""
        UPDATE restaurant_stats
        SET spend_cnt = COALESCE((SELECT COUNT(*) FROM biz_spend b WHERE b.restaurant_id = restaurant_stats.restaurant_id), 0),
            last_spend_date = (SELECT MAX(spend_date) FROM biz_spend b WHERE b.restaurant_id = restaurant_stats.restaurant_id);
        """)

        conn.exec_driver_sql("""
        UPDATE restaurant_stats
        SET popularity_score =
              0.55 * (SELECT COALESCE(rating,0)/5.0 FROM restaurants r WHERE r.id = restaurant_stats.restaurant_id)
            + 0.25 * LN(1 + COALESCE(clicks_30d,0))
            + 0.15 * LN(1 + COALESCE(spend_cnt,0))
            + 0.05 * CASE WHEN COALESCE(last_spend_date,'1970-01-01') >= NOW() - INTERVAL '90 days' THEN 1 ELSE 0 END;
        """)


# ───────────────────────────────────────────────
# SQL 유틸 함수 (그대로 유지)
# ───────────────────────────────────────────────
def _is_safe_sql(sql: str) -> bool:
    s = sql.strip().lower()
    if not s.startswith("select"):
        return False
    forbidden = ("insert", "update", "delete", "drop", "alter", "create", "truncate")
    return not any(k in s for k in forbidden)


def _is_schema_safe(sql: str) -> bool:
    s = sql.lower()
    if " join " in s or " with " in s or ("select" in s[1:] and "(" in s and ")" in s):
        return False
    tables = re.findall(r"\bfrom\s+([a-zA-Z_][\w]*)", s) + re.findall(r"\bjoin\s+([a-zA-Z_][\w]*)", s)
    return all(t in _ALLOWED_TABLES for t in tables)


def _normalize_sql(sql: str) -> str:
    if re.search(r"\blimit\b", sql, flags=re.I) is None:
        sql = sql.rstrip(";") + " LIMIT 10;"
    return sql


def _execute_sql(sql: str) -> List[Tuple]:
    with _ENGINE.begin() as conn:
        return list(conn.execute(text(sql)).fetchall())

def _render_results(rows: List[Tuple]) -> str:
    """사람 친화 문자열 렌더링."""
    if not rows:
        return ""
    items = []
    for row in rows:
        _id, name, category, location, rating = row[:5]
        items.append(f"- **{name}** — {location} · {category} · 평점 {rating}")
    return "\n".join(items)

def _esc(v: str) -> str:
    """간단 이스케이프."""
    return str(v).replace("'", "''")[:50] if v else ""

# ─────────────────────────────────────────────────────────────────────────────
# 지도 좌표 보강 / name→id 매핑 (restaurant_info 사용)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_geo_map_by_names(names: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    restaurant_info.res_name 기준으로 address/lat/lng 조회
    예: {"새벽집 강남": {"address":"...", "lat": 37.49, "lng":127.02}, ...}
    """
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
    """식당 이름 목록 → restaurants.id 목록 (IN 한 번에)."""
    if not names:
        return []
    placeholders = ", ".join([f":n{i}" for i in range(len(names))])
    q = text(f"SELECT id, name FROM restaurants WHERE name IN ({placeholders})")
    params = {f"n{i}": n for i, n in enumerate(names)}
    with _ENGINE.begin() as conn:
        rows = conn.execute(q, params).fetchall()
    name_to_id = {str(name): int(rid) for rid, name in rows}
    return [name_to_id[n] for n in names if n in name_to_id]

# ─────────────────────────────────────────────────────────────────────────────
# 트리거/슬롯 유틸
# ─────────────────────────────────────────────────────────────────────────────

def _check_triggers(user_query: str, progress: Dict[str, Any]) -> None:
    if "오늘 점심" in user_query:
        progress["_today_lunch"] = True

def _maybe_today_lunch(progress: Dict[str, Any]) -> bool:
    return bool(progress.get("_today_lunch"))

# ─────────────────────────────────────────────────────────────────────────────
# 메인 로직
# ─────────────────────────────────────────────────────────────────────────────

def get_ai_response(session_id: str, user_query: str) -> Dict[str, Any]:
    """
    메인 함수:
      1) 슬롯 수집(지역→인원→카테고리→목적)
      2) LLM으로 단일 SELECT SQL 생성 (가드레일)
      3) 결과 + 지도용 lat/lng 보강
      4) 노출 이벤트 적재(+선택적 통계 갱신)
    """
    progress = user_progress.get(session_id, {})

    # 초기화
    if "처음으로" in user_query or "다시 시작" in user_query:
        user_progress.pop(session_id, None)
        return {
            "type": "text",
            "answer": "대화를 처음부터 다시 시작할게요"
                      + (" 🙂" if prompts.USE_EMOJI else "")
                      + "\n먼저 원하시는 지역을 알려주세요.",
            "options": None,
        }

    # 트리거/슬롯 갱신
    _check_triggers(user_query, progress)
    soft = parse_initial_query(user_query, {})
    for k in ("location", "people", "category", "purpose"):
        if k in soft:
            progress[k] = soft[k]

    # 직전 질문 응답 반영
    last_q = progress.get("last_question")
    if last_q and last_q not in soft:
        progress[last_q] = user_query.strip()
        progress.pop("last_question", None)

    user_progress[session_id] = progress

    # 단계별 질문
    if "location" not in progress:
        progress["last_question"] = "location"
        user_progress[session_id] = progress
        return {
            "type": "text",
            "answer": "안녕하세요! 공맛집입니다"
                      + (" 🍽️" if prompts.USE_EMOJI else "")
                      + "\n원하시는 지역을 알려주세요. (예: 강남역, 수원시, 홍대입구 등)",
            "options": None,
        }

    if "people" not in progress:
        progress["last_question"] = "people"
        user_progress[session_id] = progress
        return {
            "type": "text",
            "answer": f"{progress['location']} 근처에서 식사하실 인원은 몇 명인가요?",
            "options": None,
        }

    if "category" not in progress:
        progress["last_question"] = "category"
        user_progress[session_id] = progress
        return {
            "type": "buttons",
            "answer": "어떤 종류의 음식을 원하시나요?",
            "options": ["한식", "중식", "일식", "양식"],
        }

    if "purpose" not in progress:
        progress["last_question"] = "purpose"
        user_progress[session_id] = progress
        return {
            "type": "buttons",
            "answer": "방문의 목적은 무엇인가요?",
            "options": ["오찬", "회식", "접대"],
        }

    # 모든 슬롯 채워짐 → 추천 수행
    try:
        # LLM 프롬프트: restaurants 뷰만 사용, 단일 SELECT, 조인/CTE 금지
        question = (
            f"{progress.get('location')} 근처에서 {progress.get('people')}명이 먹기 좋은 "
            f"{progress.get('category')} 식당을 추천해줘. 목적은 '{progress.get('purpose')}'. "
            "반드시 단일 SELECT 문만 생성하고 ';'로 끝내. "
            "오직 'restaurants' 테이블만 사용하고 JOIN/서브쿼리/CTE 금지. "
            "사용 가능한 컬럼: id,name,category,location,rating,has_private_room,recommended_for. "
            "id, name, category, location, rating 컬럼만 선택해. "
            "location은 LIKE 부분일치만 허용(예: '%강남%'). LIMIT는 10 이하."
        )

        # ──────────────── 로그 및 LLM 호출 ────────────────
        log.info(f"[INPUT] user_query={user_query}")
        log.info(f"[PROGRESS] {progress}")
        log.info(f"[LLM QUESTION] {question}")

        chain = create_sql_query_chain(llm, db)
        sql = chain.invoke({"question": question})
        log.info(f"[LLM SQL OUTPUT RAW] {sql}")

        # 출력 정제 및 폴백
        if not sql or "select" not in sql.lower():
            log.warning("[LLM] SQL 생성 실패 → 폴백 SQL 사용")
            sql = _fallback_sql_from_slots(progress)

        sql = sql.strip().strip("```").replace("sql", "").strip()
        sql = _normalize_sql(sql)
        log.info(f"[FINAL SQL] {sql}")

        # SQL 실행
        raw = _execute_sql(sql)
        log.info(f"[SQL RESULT] {raw}")

        # ──────────────── 결과 구성 ────────────────
        if not raw:
            final_answer = (
                ("죄송합니다 😢" if prompts.USE_EMOJI else "죄송합니다.")
                + f" {progress['location']} 근처에서 조건에 맞는 맛집을 찾지 못했어요.\n"
                + f"- 조건: 인원 {progress.get('people')}명, 카테고리 {progress.get('category')}, 목적 {progress.get('purpose')}\n"
                + "검색 범위를 넓히거나 다른 카테고리로 다시 시도해볼까요?"
            )
            items: List[Dict[str, Any]] = []
        else:
            rendered = _render_results(raw)
            final_answer = (
                "다음 추천을 준비했어요!"
                + (" 👇" if prompts.USE_EMOJI else "")
                + "\n"
                + f"{rendered}\n\n지도로 위치를 함께 보시려면 각 항목의 **지도로 이동** 버튼을 눌러주세요."
            )

            names = _extract_names_from_raw(raw)
            geo_map = _fetch_geo_map_by_names(names)

            items = []
            for r in raw:
                _id, name, category, location, rating = r[:5]
                geo = geo_map.get(str(name))
                items.append({
                    "id": int(_id),
                    "name": str(name),
                    "address": (geo.get("address") if geo else None) or str(location),
                    "lat": (geo.get("lat") if geo else None),
                    "lng": (geo.get("lng") if geo else None),
                    "category": str(category) if category is not None else None,
                    "price": None,
                    "score": float(rating) if rating is not None else None,
                })

            # 이벤트 로깅
            for rid in _names_to_ids(names):
                _log_event("impression", session_id, rid, None)

            if STATS_IMMEDIATE:
                recalc_stats()

        progress.pop("last_question", None)
        user_progress[session_id] = progress

        if not final_answer:
            log.warning("[WARN] final_answer is empty")

        return {"type": "text", "answer": final_answer, "items": items, "options": items}

    except Exception as e:
        log.exception("[Agent Error] %s", e)
        user_progress[session_id] = progress
        return {"type": "text", "answer": "서버 내부 오류가 발생했습니다.", "options": None}



# ─────────────────────────────────────────────────────────────────────────────
# LLM 실패 시 폴백 SQL (JOIN 허용)
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_sql_from_slots(progress: Dict[str, Any]) -> str:
    """슬롯 기반 수동 SQL (restaurants + restaurant_stats LEFT JOIN)"""
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
    """결과 튜플에서 식당명만 추출."""
    return [str(row[1]) for row in rows if len(row) > 1]