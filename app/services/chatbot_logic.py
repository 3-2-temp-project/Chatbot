import os
import re
from sqlalchemy import create_engine, text
from langchain_community.utilities import SQLDatabase
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import create_sql_query_chain

from app.core import config
import app.services.prompts as prompts


# =========================
# DB & LLM 준비
# =========================
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


def setup_database():
    """PostgreSQL용 스키마 보장 + 샘플 데이터 시드."""
    engine = create_engine(config.DATABASE_URI)

    with engine.begin() as conn:
        # --- 기본 스키마 ---
        conn.exec_driver_sql("""
        CREATE TABLE IF NOT EXISTS restaurants (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,        -- 한식/일식/중식/양식
            location TEXT NOT NULL,        -- 강남역/판교역/정자동/...
            rating REAL,                   -- 0~5
            has_private_room BOOLEAN,      -- true/false
            recommended_for TEXT           -- 오찬/접대/회식 등
        );
        """)

        conn.exec_driver_sql("""
        CREATE TABLE IF NOT EXISTS biz_spend (
            id SERIAL PRIMARY KEY,
            restaurant_id INTEGER REFERENCES restaurants(id),
            agency TEXT,
            spend_date DATE,
            amount_krw INTEGER,
            purpose TEXT
        );
        """)

        # --- 행동 로그 / 집계 스키마 ---
        conn.exec_driver_sql("""
        CREATE TABLE IF NOT EXISTS user_events (
            id SERIAL PRIMARY KEY,
            session_id TEXT,
            restaurant_id INTEGER,
            event TEXT,                -- 'impression','click','favorite' 등
            value TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        conn.exec_driver_sql("""
        CREATE TABLE IF NOT EXISTS restaurant_stats (
            restaurant_id INTEGER PRIMARY KEY,
            clicks_7d INTEGER DEFAULT 0,
            clicks_30d INTEGER DEFAULT 0,
            impressions_7d INTEGER DEFAULT 0,
            impressions_30d INTEGER DEFAULT 0,
            favorites_30d INTEGER DEFAULT 0,
            spend_cnt INTEGER DEFAULT 0,
            last_spend_date DATE,
            popularity_score REAL DEFAULT 0
        );
        """)

        # --- 샘플 시드 ---
        cnt = conn.execute(text("SELECT COUNT(*) FROM restaurants")).scalar()
        if not cnt:
            restaurants = [
                ("새벽집 강남",     "한식", "강남역", 4.4, True, "회식"),
                ("진가와 강남",     "일식", "강남역", 4.6, True, "접대"),
                ("마라공방 강남",   "중식", "강남역", 4.5, True, "회식"),
                ("청키면가 강남",   "중식", "강남역", 4.2, False, "오찬"),
                ("우래옥 강남",     "한식", "강남역", 4.5, True, "접대"),
                ("진대감 삼성",     "한식", "삼성역", 4.6, True, "회식"),
                ("봉피양 삼성",     "한식", "삼성역", 4.5, True, "접대"),
                ("팔선생 삼성",     "중식", "삼성역", 4.3, False, "오찬"),
                ("취향중식 판교",   "중식", "판교역", 4.5, True, "회식"),
                ("스시야 판교",     "일식", "판교역", 4.5, True, "접대"),
                ("봉피양 판교",     "한식", "판교역", 4.4, False, "오찬"),
                ("오스테리아 오르조","양식","판교역", 4.7, False, "오찬"),
                ("초마 홍대",       "중식", "홍대입구", 4.4, False, "오찬"),
                ("뜨락 홍대",       "한식", "홍대입구", 4.2, False, "오찬"),
                ("정자 스시",       "일식", "정자동", 4.6, True, "접대"),
                ("정자동 마라탕",   "중식", "정자동", 4.3, False, "오찬"),
                ("인계 라멘",       "일식", "수원 인계동", 4.2, False, "오찬"),
                ("인계 양꼬치",     "중식", "수원 인계동", 4.5, False, "회식"),
                ("라페 라멘",       "일식", "일산 라페스타", 4.1, False, "오찬"),
                ("라페 차이",       "중식", "일산 라페스타", 4.2, False, "오찬"),
            ]
            for row in restaurants:
                conn.execute(
                    text("INSERT INTO restaurants (name, category, location, rating, has_private_room, recommended_for) "
                         "VALUES (:name, :category, :location, :rating, :has_private_room, :recommended_for)"),
                    {
                        "name": row[0],
                        "category": row[1],
                        "location": row[2],
                        "rating": row[3],
                        "has_private_room": row[4],
                        "recommended_for": row[5],
                    },
                )

            # 업추비 더미 데이터
            name_id = dict((v, k) for (k, v) in conn.execute(text("SELECT id, name FROM restaurants")))
            spend = [
                (name_id["마라공방 강남"], "서울시 경제정책과", "2025-08-12", 180000, "업무협의 오찬"),
                (name_id["마라공방 강남"], "서울시 경제정책과", "2025-09-03", 240000, "간담회"),
                (name_id["진가와 강남"],   "서울시 투자유치과", "2025-08-28", 350000, "외빈 접대"),
                (name_id["새벽집 강남"],   "서울시 총무과",     "2025-09-02", 220000, "부서 회식"),
                (name_id["정자 스시"],     "성남시 기업지원과", "2025-08-21", 270000, "외부 미팅"),
                (name_id["취향중식 판교"], "성남시 일자리과",   "2025-09-05", 190000, "업무협의"),
            ]
            for r_id, agency, d, amount, purpose in spend:
                conn.execute(
                    text("INSERT INTO biz_spend (restaurant_id, agency, spend_date, amount_krw, purpose) "
                         "VALUES (:rid, :agency, :d, :amount, :purpose)"),
                    {"rid": r_id, "agency": agency, "d": d, "amount": amount, "purpose": purpose},
                )



setup_database()

# 세션 진행상태(간단 메모리)
user_progress = {}


# =========================
# 유틸: 질의 파싱
# =========================
def parse_initial_query(query: str, progress: dict) -> dict:
    loc = re.search(r"(\S+역|\S+동|\S+시|\S+구)", query)
    if loc:
        progress["location"] = loc.group(0)
    cat = re.search(r"(한식|일식|중식|양식)", query)
    if cat:
        progress["category"] = cat.group(0)
    purp = re.search(r"(오찬|접대|회식)", query)
    if purp:
        progress["purpose"] = purp.group(0)
    return progress


# =========================
# 이벤트 로깅 & 집계
# =========================
def _log_event(event: str, session_id: str, restaurant_id: int | None = None, value: str | None = None):
    try:
        engine = create_engine(config.DATABASE_URI)
        with engine.begin() as conn:
            conn.execute(
                text("INSERT INTO user_events (session_id, restaurant_id, event, value) "
                     "VALUES (:session_id, :restaurant_id, :event, :value)"),
                {
                    "session_id": session_id,
                    "restaurant_id": restaurant_id,
                    "event": event,
                    "value": value,
                },
            )
    except Exception as e:
        print("[event log error]", e)


def recalc_stats():
    engine = create_engine(config.DATABASE_URI)
    with engine.begin() as conn:
        # 1. 임시 집계 결과를 계산
        conn.exec_driver_sql("""
        INSERT INTO restaurant_stats (restaurant_id, impressions_7d, impressions_30d, clicks_7d, clicks_30d)
        SELECT r.id,
               SUM(CASE WHEN ue.event='impression' AND ue.created_at >= CURRENT_DATE - INTERVAL '7 days' THEN 1 ELSE 0 END),
               SUM(CASE WHEN ue.event='impression' AND ue.created_at >= CURRENT_DATE - INTERVAL '30 days' THEN 1 ELSE 0 END),
               SUM(CASE WHEN ue.event='click' AND ue.created_at >= CURRENT_DATE - INTERVAL '7 days' THEN 1 ELSE 0 END),
               SUM(CASE WHEN ue.event='click' AND ue.created_at >= CURRENT_DATE - INTERVAL '30 days' THEN 1 ELSE 0 END)
        FROM restaurants r
        LEFT JOIN user_events ue ON ue.restaurant_id = r.id
        GROUP BY r.id
        ON CONFLICT (restaurant_id)
        DO UPDATE SET
            impressions_7d = EXCLUDED.impressions_7d,
            impressions_30d = EXCLUDED.impressions_30d,
            clicks_7d = EXCLUDED.clicks_7d,
            clicks_30d = EXCLUDED.clicks_30d;
        """)

        # 2. 나머지 업데이트 쿼리도 PostgreSQL 문법으로 변경 필요
        conn.exec_driver_sql("""
        UPDATE restaurant_stats
        SET spend_cnt = COALESCE((
              SELECT COUNT(*) FROM biz_spend b WHERE b.restaurant_id = restaurant_stats.restaurant_id
            ), 0),
            last_spend_date = (
              SELECT MAX(spend_date) FROM biz_spend b WHERE b.restaurant_id = restaurant_stats.restaurant_id
            );
        """)

        conn.exec_driver_sql("""
        UPDATE restaurant_stats
        SET popularity_score =
              0.55 * (SELECT COALESCE(rating,0)/5.0 FROM restaurants r WHERE r.id = restaurant_stats.restaurant_id)
            + 0.25 * LOG(1 + COALESCE(clicks_30d,0))
            + 0.15 * LOG(1 + COALESCE(spend_cnt,0))
            + 0.05 * CASE
                WHEN COALESCE(last_spend_date, DATE '1970-01-01') >= CURRENT_DATE - INTERVAL '90 days' THEN 1
                ELSE 0
              END;
        """)


# =========================
# SQL 가드레일 & 폴백
# =========================
_ALLOWED_TABLES = {"restaurants"}

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
    if " join " in s or " with " in s or ("select" in s[1:] and "(" in s and ")" in s):
        return False
    tables = re.findall(r"\bfrom\s+([a-zA-Z_][\w]*)", s) + re.findall(r"\bjoin\s+([a-zA-Z_][\w]*)", s)
    return all(t in _ALLOWED_TABLES for t in tables)

def _normalize_sql(sql: str) -> str:
    if re.search(r"\blimit\b", sql, flags=re.I) is None:
        sql = sql.rstrip(";") + " LIMIT 10;"
    return sql

def _execute_sql(sql: str) -> str:
    return db.run(sql)

def _render_results(raw: str) -> str:
    if not raw or raw.strip() in ("[]",):
        return ""
    lines = [l for l in raw.strip().splitlines() if l.strip()]
    if len(lines) >= 2 and "|" in lines[0]:
        header = [h.strip() for h in lines[0].split("|")]
        items = []
        for row in lines[1:]:
            cols = [c.strip() for c in row.split("|")]
            rec = dict(zip(header, cols))
            name = rec.get("name") or "알 수 없음"
            loc = rec.get("location") or ""
            cat = rec.get("category") or ""
            rating = rec.get("rating") or ""
            items.append(f"- **{name}** — {loc} · {cat} · 평점 {rating}")
        return "\n".join(items)
    return raw

def _esc(v: str) -> str:
    return str(v).replace("'", "''")[:50] if v else ""

def _fallback_sql_from_slots(progress: dict) -> str:
    loc = _esc(progress.get("location", ""))
    cat = _esc(progress.get("category", ""))
    purp = progress.get("purpose", "")
    where = []
    if loc:
        where.append(f"r.location LIKE '%{loc}%'")
    if cat:
        where.append(f"r.category = '{cat}'")
    if "회식" in purp:
        where.append("r.has_private_room = true")  # ✅ boolean 비교 수정
    where_sql = " AND ".join(where) if where else "1=1"
    return (
        "SELECT r.id, r.name, r.category, r.location, r.rating "
        "FROM restaurants r "
        "LEFT JOIN restaurant_stats s ON s.restaurant_id = r.id "
        f"WHERE {where_sql} "
        "ORDER BY COALESCE(s.popularity_score, r.rating/5.0) DESC "
        "LIMIT 10;"
    )


def _extract_names_from_raw(raw: str) -> list[str]:
    lines = [l for l in raw.strip().splitlines() if l.strip()]
    names = []
    if len(lines) >= 2 and "|" in lines[0]:
        header = [h.strip() for h in lines[0].split("|")]
        idx = {h: i for i, h in enumerate(header)}
        if "name" in idx:
            for row in lines[1:]:
                cols = [c.strip() for c in row.split("|")]
                if len(cols) > idx["name"]:
                    names.append(cols[idx["name"]])
    return names

def _names_to_ids(names: list[str]) -> list[int]:
    if not names: return []
    engine = create_engine(config.DATABASE_URI)
    with engine.begin() as conn:
        out = []
        for n in names:
            row = conn.execute(text("SELECT id FROM restaurants WHERE name = :n"), {"n": n}).fetchone()
            if row:
                out.append(int(row[0]))
        return out

def _check_triggers(user_query: str, progress: dict) -> None:
    if "오늘 점심" in user_query:
        progress["_today_lunch"] = True

def _maybe_today_lunch(progress: dict) -> bool:
    return progress.get("_today_lunch", False)


# =========================
# 메인 로직
# =========================
def get_ai_response(session_id: str, user_query: str):
    progress = user_progress.get(session_id, {})

    # 리셋
    if "처음으로" in user_query or "다시 시작" in user_query:
        user_progress.pop(session_id, None)
        return {"type": "text", "answer": prompts.RESET_MESSAGE, "options": None}

    # 트리거
    _check_triggers(user_query, progress)

    # 이번 턴에서 새로 언급된 슬롯만 소프트 업데이트
    soft = parse_initial_query(user_query, {})
    for k in ("location", "category", "purpose"):
        if k in soft:
            progress[k] = soft[k]

    # 직전 질문의 답 처리
    last_q = progress.get("last_question")
    if last_q and last_q not in soft:
        progress[last_q] = user_query.strip()
        progress.pop("last_question", None)

    user_progress[session_id] = progress

    # 슬롯 수집
    if "location" not in progress:
        progress["last_question"] = "location"
        user_progress[session_id] = progress
        return {"type": "text", "answer": prompts.GREETING_MESSAGE, "options": None}

    if "category" not in progress:
        progress["last_question"] = "category"
        user_progress[session_id] = progress
        return {
            "type": "buttons",
            "answer": prompts.ASK_CATEGORY.format(location=progress["location"]),
            "options": prompts.CATEGORY_OPTIONS,
        }

    if "purpose" not in progress:
        progress["last_question"] = "purpose"
        user_progress[session_id] = progress
        return {"type": "buttons", "answer": prompts.ASK_PURPOSE, "options": prompts.PURPOSE_OPTIONS}

    # '오늘 점심' 간단 추천
    if _maybe_today_lunch(progress):
        sql = f"""
        SELECT name, category, location, rating
        FROM restaurants
        WHERE location LIKE '%{_esc(progress['location'])}%'
          AND category = '{_esc(progress['category'])}'
        ORDER BY rating DESC
        LIMIT 1;
        """
        raw = _execute_sql(sql)

        # impression 로깅 + 즉시 재계산
        names = _extract_names_from_raw(raw)
        for rid in _names_to_ids(names):
            _log_event("impression", session_id, rid, None)
        recalc_stats()

        if raw.strip() in ("[]", ""):
            answer = prompts.NO_RESULT_MESSAGE.format(
                location=progress["location"], category=progress["category"]
            )
        else:
            lines = [l for l in raw.splitlines() if l.strip()]
            if len(lines) >= 2 and "|" in lines[0]:
                cols = [c.strip() for c in lines[1].split("|")]
                name, cat, loc, rating = cols[:4]
                answer = prompts.TODAY_LUNCH_PROMPT.format(
                    name=name, category=cat, location=loc, rating=rating
                ) + prompts.ASK_SIMILAR_RESTAURANT
            else:
                answer = _render_results(raw)

        progress.pop("last_question", None)
        progress.pop("_today_lunch", None)
        user_progress[session_id] = progress
        return {"type": "text", "answer": answer, "options": None}

    # 안전한 Text-to-SQL 생성 + 실행
    try:
        question = (
            f"{progress.get('location')} 근처 {progress.get('category')} 식당을 추천해줘. "
            f"목적은 '{progress.get('purpose')}'. 상위 평점 위주로. "
            "반드시 단일 SELECT 문만 생성하고 ';'로 끝내. "
            "오직 'restaurants' 테이블만 사용하고 JOIN/서브쿼리/CTE 금지. "
            "사용 가능한 컬럼: id,name,category,location,rating,has_private_room,recommended_for. "
            "id, name, category, location, rating 컬럼만 선택해. "
            "location은 LIKE 부분일치만 허용(예: '%강남%'). LIMIT는 10 이하."
        )
        chain = create_sql_query_chain(llm, db)
        sql = chain.invoke({"question": question})
        sql = sql.strip().strip("```").replace("sql", "").strip()
        sql = _normalize_sql(sql)

        if not _is_safe_sql(sql) or not _is_schema_safe(sql):
            sql = _fallback_sql_from_slots(progress)

        raw = _execute_sql(sql)

        # impression 로깅 + 즉시 재계산
        names = _extract_names_from_raw(raw)
        for rid in _names_to_ids(names):
            _log_event("impression", session_id, rid, None)
        recalc_stats()

        if not raw or raw.strip() in ("[]",):
            final_answer = prompts.NO_RESULT_MESSAGE.format(
                location=progress.get("location"),
                category=progress.get("category"),
            )
        else:
            rendered = _render_results(raw)
            final_answer = rendered + prompts.ASK_SIMILAR_RESTAURANT

        progress.pop("last_question", None)
        user_progress[session_id] = progress
        return {"type": "text", "answer": final_answer, "options": None}

    except Exception as e:
        print(f"[Agent Error] {e}")
        user_progress[session_id] = progress
        return {"type": "text", "answer": prompts.ERROR_MESSAGE, "options": None}
