import os
import re
import torch
from sqlalchemy import create_engine, text
from langchain_community.utilities import SQLDatabase
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import create_sql_query_chain

from app.core import config
import app.services.prompts as prompts


db = SQLDatabase.from_uri(config.DATABASE_URI)

USE_GPU = torch.cuda.is_available()

llm = HuggingFacePipeline.from_model_id(
    model_id=config.LLM_MODEL_ID,
    task=config.LLM_TASK,
    pipeline_kwargs={
        "max_new_tokens": config.LLM_MAX_NEW_TOKENS,
        "temperature": config.LLM_TEMPERATURE,
        "do_sample": True if config.LLM_TEMPERATURE > 0 else False,
        "return_full_text": False,
    },
    device=0 if USE_GPU else -1,
)


def setup_database():
    os.makedirs(os.path.join(config.BASE_DIR, "data"), exist_ok=True)
    engine = create_engine(config.DATABASE_URI)

    with engine.begin() as conn:
        conn.exec_driver_sql("""
        CREATE TABLE IF NOT EXISTS restaurants (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            location TEXT NOT NULL,
            rating REAL,
            has_private_room BOOLEAN,
            recommended_for TEXT
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

        conn.exec_driver_sql("""
        CREATE TABLE IF NOT EXISTS user_events (
            id SERIAL PRIMARY KEY,
            session_id TEXT,
            restaurant_id INTEGER,
            event TEXT,
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

        cnt = conn.execute(text("SELECT COUNT(*) FROM restaurants")).scalar()
        if not cnt:
            restaurants = [
                ("새벽집 강남", "한식", "강남역", 4.4, True, "회식"),
                ("진가와 강남", "일식", "강남역", 4.6, True, "접대"),
                ("마라공방 강남", "중식", "강남역", 4.5, True, "회식"),
                ("청키면가 강남", "중식", "강남역", 4.2, False, "오찬"),
                ("우래옥 강남", "한식", "강남역", 4.5, True, "접대"),
                ("진대감 삼성", "한식", "삼성역", 4.6, True, "회식"),
                ("봉피양 삼성", "한식", "삼성역", 4.5, True, "접대"),
                ("팔선생 삼성", "중식", "삼성역", 4.3, False, "오찬"),
                ("취향중식 판교", "중식", "판교역", 4.5, True, "회식"),
                ("스시야 판교", "일식", "판교역", 4.5, True, "접대"),
                ("봉피양 판교", "한식", "판교역", 4.4, False, "오찬"),
                ("오스테리아 오르조", "양식", "판교역", 4.7, False, "오찬"),
                ("초마 홍대", "중식", "홍대입구", 4.4, False, "오찬"),
                ("뜨락 홍대", "한식", "홍대입구", 4.2, False, "오찬"),
                ("정자 스시", "일식", "정자동", 4.6, True, "접대"),
                ("정자동 마라탕", "중식", "정자동", 4.3, False, "오찬"),
                ("인계 라멘", "일식", "수원 인계동", 4.2, False, "오찬"),
                ("인계 양꼬치", "중식", "수원 인계동", 4.5, False, "회식"),
                ("라페 라멘", "일식", "일산 라페스타", 4.1, False, "오찬"),
                ("라페 차이", "중식", "일산 라페스타", 4.2, False, "오찬"),
            ]
            for row in restaurants:
                conn.execute(
                    text("INSERT INTO restaurants (name, category, location, rating, has_private_room, recommended_for) "
                         "VALUES (:name, :category, :location, :rating, :has_private_room, :recommended_for)"),
                    {
                        "name": row[0], "category": row[1], "location": row[2],
                        "rating": row[3], "has_private_room": row[4], "recommended_for": row[5],
                    }
                )

            name_id = dict((v, k) for (k, v) in conn.execute(text("SELECT id, name FROM restaurants")))

            spend = [
                (name_id["마라공방 강남"], "서울시 경제정책과", "2025-08-12", 180000, "업무협의 오찬"),
                (name_id["마라공방 강남"], "서울시 경제정책과", "2025-09-03", 240000, "간담회"),
                (name_id["진가와 강남"], "서울시 투자유치과", "2025-08-28", 350000, "외빈 접대"),
                (name_id["새벽집 강남"], "서울시 총무과", "2025-09-02", 220000, "부서 회식"),
                (name_id["정자 스시"], "성남시 기업지원과", "2025-08-21", 270000, "외부 미팅"),
                (name_id["취향중식 판교"], "성남시 일자리과", "2025-09-05", 190000, "업무협의"),
            ]
            for r_id, agency, d, amount, purpose in spend:
                conn.execute(
                    text("INSERT INTO biz_spend (restaurant_id, agency, spend_date, amount_krw, purpose) "
                         "VALUES (:r_id, :agency, :d, :amount, :purpose)"),
                    {"r_id": r_id, "agency": agency, "d": d, "amount": amount, "purpose": purpose}
                )


setup_database()
user_progress = {}


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


def _log_event(event: str, session_id: str, restaurant_id: int | None = None, value: str | None = None):
    try:
        engine = create_engine(config.DATABASE_URI)
        with engine.begin() as conn:
            conn.execute(
                text("INSERT INTO user_events (session_id, restaurant_id, event, value) "
                     "VALUES (:sid, :rid, :event, :val)"),
                {"sid": session_id, "rid": restaurant_id, "event": event, "val": value}
            )
    except Exception as e:
        print("[event log error]", e)


def recalc_stats():
    engine = create_engine(config.DATABASE_URI)
    with engine.begin() as conn:
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
            + 0.25 * LOG(1 + COALESCE(clicks_30d,0))
            + 0.15 * LOG(1 + COALESCE(spend_cnt,0))
            + 0.05 * CASE WHEN COALESCE(last_spend_date,'1970-01-01') >= NOW() - INTERVAL '90 days' THEN 1 ELSE 0 END;
        """)


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

def _execute_sql(sql: str):
    engine = create_engine(config.DATABASE_URI)
    with engine.begin() as conn:
        result = conn.execute(text(sql))
        return result.fetchall()

def _render_results(rows: list[tuple]) -> str:
    if not rows:
        return ""
    items = []
    for row in rows:
        _id, name, category, location, rating = row[:5]
        items.append(f"- **{name}** — {location} · {category} · 평점 {rating}")
    return "\n".join(items)

def _esc(v: str) -> str:
    return str(v).replace("'", "''")[:50] if v else ""

def _fallback_sql_from_slots(progress: dict) -> str:
    loc = _esc(progress.get("location", "")); cat = _esc(progress.get("category", "")); purp = progress.get("purpose", "")
    where = []
    if loc: where.append(f"r.location LIKE '%{loc}%'")
    if cat: where.append(f"r.category = '{cat}'")
    if "회식" in purp: where.append("r.has_private_room = TRUE")
    where_sql = " AND ".join(where) if where else "1=1"
    return (
        "SELECT r.id, r.name, r.category, r.location, r.rating "
        "FROM restaurants r "
        "LEFT JOIN restaurant_stats s ON s.restaurant_id = r.id "
        f"WHERE {where_sql} "
        "ORDER BY COALESCE(s.popularity_score, r.rating/5.0) DESC "
        "LIMIT 10;"
    )


def _extract_names_from_raw(rows: list[tuple]) -> list[str]:
    if not rows:
        return []
    return [row[1] for row in rows if len(row) > 1]


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


def get_ai_response(session_id: str, user_query: str):
    progress = user_progress.get(session_id, {})

    if "처음으로" in user_query or "다시 시작" in user_query:
        user_progress.pop(session_id, None)
        return {"type": "text", "answer": prompts.RESET_MESSAGE, "options": None}

    _check_triggers(user_query, progress)

    soft = parse_initial_query(user_query, {})
    for k in ("location", "category", "purpose"):
        if k in soft:
            progress[k] = soft[k]

    last_q = progress.get("last_question")
    if last_q and last_q not in soft:
        progress[last_q] = user_query.strip()
        progress.pop("last_question", None)

    user_progress[session_id] = progress

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
        names = [row[0] for row in raw]
        for rid in _names_to_ids(names):
            _log_event("impression", session_id, rid, None)
        recalc_stats()

        if not raw:
            answer = prompts.NO_RESULT_MESSAGE.format(
                location=progress["location"], category=progress["category"]
            )
        else:
            name, cat, loc, rating = raw[0][:4]
            answer = prompts.TODAY_LUNCH_PROMPT.format(
                name=name, category=cat, location=loc, rating=rating
            ) + prompts.ASK_SIMILAR_RESTAURANT

        progress.pop("last_question", None)
        progress.pop("_today_lunch", None)
        user_progress[session_id] = progress
        return {"type": "text", "answer": answer, "options": None}

    try:
        question = (
            f"{progress.get('location')} 근처 {progress.get('category')} 식당을 추천해줘. "
            f"목적은 '{progress.get('purpose')}'. "
            "만약 사용자가 '근처', '주변', '가까운', '지도' 등의 단어를 언급했다면 "
            "PostGIS의 ST_Distance 함수를 사용해 위치 기반 정렬 쿼리를 생성해. "
            "예를 들어, ST_Distance(location, ST_GeomFromText('POINT(127.0276 37.4979)', 4326)) "
            "같은 식으로 표현하고, 가까운 순서대로 정렬해 LIMIT 10 이하로. "
            "오직 restaurants 테이블만 사용하고, "
            "id,name,category,location,rating 컬럼만 선택해. "
            "JOIN/서브쿼리/CTE 금지. 단일 SELECT 문으로 끝내."
        )
        chain = create_sql_query_chain(llm, db)
        sql = chain.invoke({"question": question})
        sql = sql.strip().strip("```").replace("sql", "").strip()
        sql = _normalize_sql(sql)

        if not _is_safe_sql(sql) or not _is_schema_safe(sql):
            sql = _fallback_sql_from_slots(progress)

        raw = _execute_sql(sql)
        names = _extract_names_from_raw(raw)
        for rid in _names_to_ids(names):
            _log_event("impression", session_id, rid, None)
        recalc_stats()

        if not raw:
            final_answer = prompts.NO_RESULT_MESSAGE.format(
                location=progress.get("location"),
                category=progress.get("category"),
            )
        else:
            rendered = _render_results(raw)
            summary_prompt = (
                "다음 맛집 목록을 사용자에게 자연스럽게 말하듯 요약해줘:\n" + rendered
            )
            natural_summary = llm.invoke(summary_prompt)
            final_answer = natural_summary + "\n\n" + prompts.ASK_SIMILAR_RESTAURANT

        progress.pop("last_question", None)
        user_progress[session_id] = progress
        return {"type": "text", "answer": final_answer, "options": None}

    except Exception as e:
        print(f"[Agent Error] {e}")
        user_progress[session_id] = progress
        return {"type": "text", "answer": prompts.ERROR_MESSAGE, "options": None}
