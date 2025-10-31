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
    device=0,
)


def setup_database():
    """
    PostgreSQL 초기화 함수
    - DB 스키마 생성 (restaurants, biz_spend, user_events, restaurant_stats)
    - restaurants 테이블이 비어있으면 샘플 데이터와 업추비 더미 데이터를 추가
    """
    os.makedirs(os.path.join(config.BASE_DIR, "data"), exist_ok=True)
    engine = create_engine(config.DATABASE_URI)

    with engine.begin() as conn:
        # --- 기본 스키마 ---
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

        # --- 샘플 데이터 시드 ---
        cnt = conn.execute(text("SELECT COUNT(*) FROM restaurants")).scalar()
        if not cnt:
            # 예시 맛집 삽입
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

            # 업추비 데이터 더미
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

# 세션 진행상태(간단 메모리)
user_progress = {}


# =========================
# 유틸: 질의 파싱
# =========================

def parse_initial_query(query: str, progress: dict) -> dict:
    """
    사용자가 입력한 문장에서 슬롯(location, people, category, purpose)을 추출
    단, 단계가 건너뛰지 않도록 '현재 진행 중인 단계'에 따라 제한적으로 추출
    """
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



# =========================
# 이벤트 로깅 & 집계
# =========================
def _log_event(event: str, session_id: str, restaurant_id: int | None = None, value: str | None = None):
    """
    user_events 테이블에 사용자 행동(노출, 클릭, 즐겨찾기 등)을 기록
    """
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
    """
    restaurant_stats 테이블 업데이트
    - 최근 7일/30일 노출/클릭 수 집계
    - 업추비 집계 (횟수, 마지막 사용일)
    - popularity_score(인기도 점수) 갱신
    """
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



# =========================
# SQL 가드레일 & 폴백
# =========================
_ALLOWED_TABLES = {"restaurants"}

def _is_safe_sql(sql: str) -> bool:
    """SQL이 SELECT 문으로 시작하고 위험한 키워드가 없는지 검사"""
    s = sql.strip().lower()
    if not s.startswith("select"):
        return False
    forbidden = ("insert", "update", "delete", "drop", "alter", "create", "truncate")
    if any(k in s for k in forbidden):
        return False
    return True

def _is_schema_safe(sql: str) -> bool:
    """허용된 테이블만 사용하는지 검사 (JOIN, CTE, 서브쿼리 금지)"""
    s = sql.lower()
    if " join " in s or " with " in s or ("select" in s[1:] and "(" in s and ")" in s):
        return False
    tables = re.findall(r"\bfrom\s+([a-zA-Z_][\w]*)", s) + re.findall(r"\bjoin\s+([a-zA-Z_][\w]*)", s)
    return all(t in _ALLOWED_TABLES for t in tables)

def _normalize_sql(sql: str) -> str:
    """LIMIT 절이 없으면 LIMIT 10 추가"""
    if re.search(r"\blimit\b", sql, flags=re.I) is None:
        sql = sql.rstrip(";") + " LIMIT 10;"
    return sql

def _execute_sql(sql: str):
    """Postgres 쿼리 실행 후 튜플 리스트 반환"""
    engine = create_engine(config.DATABASE_URI)
    with engine.begin() as conn:
        result = conn.execute(text(sql))
        return result.fetchall()

def _render_results(rows: list[tuple]) -> str:
    """쿼리 결과를 사람이 읽기 좋은 문자열로 변환"""
    if not rows:
        return ""
    items = []
    for row in rows:
        _id, name, category, location, rating = row[:5]
        items.append(f"- **{name}** — {location} · {category} · 평점 {rating}")
    return "\n".join(items)

def _esc(v: str) -> str:
    """SQL 인젝션 방지를 위해 작은 따옴표 이스케이프"""
    return str(v).replace("'", "''")[:50] if v else ""

def _fallback_sql_from_slots(progress: dict) -> str:
    """
    LLM이 안전하지 않은 SQL을 생성했을 경우,
    직접 location/category/purpose 정보를 이용해 SQL 생성 (백업용)
    """
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
    """
    SQL 실행 결과 (list of tuples)에서 식당 이름만 추출
    예: [(1, '새벽집 강남', '한식', '강남역', 4.4), ...] → ['새벽집 강남', ...]
    """
    if not rows:
        return []
    return [row[1] for row in rows if len(row) > 1]


def _names_to_ids(names: list[str]) -> list[int]:
    """식당 이름 목록을 받아 DB에서 id 리스트 반환"""
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
    """사용자 질의에 '오늘 점심'이 포함되면 플래그 저장"""
    if "오늘 점심" in user_query:
        progress["_today_lunch"] = True

def _maybe_today_lunch(progress: dict) -> bool:
    """'오늘 점심' 트리거 여부 확인"""
    return progress.get("_today_lunch", False)


# =========================
# 메인 로직
# =========================
def get_ai_response(session_id: str, user_query: str):
    """
    메인 함수: 사용자의 질의를 받아 AI 응답을 생성
    1️⃣ 지역 → 2️⃣ 인원수 → 3️⃣ 음식 종류 → 4️⃣ 목적 순서로 대화 진행
    5️⃣ 모든 슬롯이 채워지면 SQL 생성 → 결과 반환
    """
    progress = user_progress.get(session_id, {})

    # (1) "처음으로" 입력 → 세션 초기화
    if "처음으로" in user_query or "다시 시작" in user_query:
        user_progress.pop(session_id, None)
        return {
            "type": "text",
            "answer": "대화를 처음부터 다시 시작할게요 🙂\n먼저 원하시는 지역을 알려주세요.",
            "options": None
        }

    # (2) 트리거 확인 (오늘 점심 등)
    _check_triggers(user_query, progress)

    # (3) 새 슬롯 추출
    soft = parse_initial_query(user_query, {})
    for k in ("location", "people", "category", "purpose"):
        if k in soft:
            progress[k] = soft[k]

    # (4) 직전 질문에 대한 응답 처리
    last_q = progress.get("last_question")
    if last_q and last_q not in soft:
        progress[last_q] = user_query.strip()
        progress.pop("last_question", None)

    user_progress[session_id] = progress

    # (5) 단계별 슬롯 질문 로직 --------------------------------

    # 1️⃣ 지역
    if "location" not in progress:
        progress["last_question"] = "location"
        user_progress[session_id] = progress
        return {
            "type": "text",
            "answer": "안녕하세요! 공맛집입니다 🍽️\n먼저 원하시는 지역을 알려주세요. (예: 강남역, 수원시, 홍대입구 등)",
            "options": None
        }

    # 2️⃣ 인원수
    if "people" not in progress:
        progress["last_question"] = "people"
        user_progress[session_id] = progress
        return {
            "type": "text",
            "answer": f"{progress['location']} 근처에서 식사하실 인원은 몇 명인가요?",
            "options": None
        }

    # 3️⃣ 음식 종류
    if "category" not in progress:
        progress["last_question"] = "category"
        user_progress[session_id] = progress
        return {
            "type": "buttons",
            "answer": "어떤 종류의 음식을 원하시나요?",
            "options": ["한식", "중식", "일식", "양식"]
        }

    # 4️⃣ 목적
    if "purpose" not in progress:
        progress["last_question"] = "purpose"
        user_progress[session_id] = progress
        return {
            "type": "buttons",
            "answer": "방문의 목적은 무엇인가요?",
            "options": ["오찬", "회식", "접대"]
        }

    # ✅ 모든 슬롯이 채워졌을 때 추천 수행
    try:
        question = (
            f"{progress.get('location')} 근처에서 {progress.get('people')}명이 먹기 좋은 "
            f"{progress.get('category')} 식당을 추천해줘. 목적은 '{progress.get('purpose')}'. "
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

        # 결과가 없을 때
        if not raw:
            final_answer = f"죄송합니다 😢 {progress['location']} 근처에서 조건에 맞는 맛집을 찾지 못했어요.\n다른 조건으로 다시 시도해볼까요?"
            items = []
        else:
            rendered = _render_results(raw)
            final_answer = f"다음 맛집들을 추천드릴게요! 👇\n{rendered}"

            # ✅ items 리스트로 지도에 넘길 데이터 구성
            items = [
                {
                    "name": row[0],
                    "category": row[1],
                    "location": row[2],
                    "rating": row[3]
                }
                for row in raw
            ]

        progress.pop("last_question", None)
        user_progress[session_id] = progress

        # ✅ items 함께 반환
        return {
            "type": "text",
            "answer": final_answer,
            "items": items,
            "options": None
        }

    except Exception as e:
        print(f"[Agent Error] {e}")
        user_progress[session_id] = progress
        return {"type": "text", "answer": "서버 내부 오류가 발생했습니다.", "options": None}
