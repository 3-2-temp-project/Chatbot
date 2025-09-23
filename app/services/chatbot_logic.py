import os
import re
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_huggingface import HuggingFacePipeline

# 경로 명시 (패키지 임포트)
from app.core import config
import app.services.prompts as prompts

# SQL만 생성하는 체인 (LangChain 0.2+)
from langchain.chains import create_sql_query_chain

# --- 1) DB & LLM 초기화 ---
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
    device=-1,  # CPU
)


def setup_database():
    """PoC용 SQLite DB가 없으면 생성하고 샘플 5건 입력"""
    if os.path.exists("./data/sample_db.sqlite"):
        return
    os.makedirs("./data", exist_ok=True)
    engine = create_engine(config.DATABASE_URI)
    connection = engine.raw_connection()
    cursor = connection.cursor()
    cursor.execute(
        """
        CREATE TABLE restaurants (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            location TEXT NOT NULL,
            rating REAL,
            has_private_room BOOLEAN,
            recommended_for TEXT
        );
    """
    )
    sample_data = [
        ("우래옥", "한식", "강남역", 4.5, 1, "접대"),
        ("스시효", "일식", "강남역", 4.8, 1, "접대"),
        ("진대감", "한식", "삼성역", 4.6, 1, "회식"),
        ("오스테리아 오르조", "양식", "판교역", 4.7, 0, "오찬"),
        ("봉피양", "한식", "판교역", 4.4, 0, "오찬"),
    ]
    cursor.executemany(
        "INSERT INTO restaurants VALUES (NULL, ?, ?, ?, ?, ?, ?)", sample_data
    )
    connection.commit()
    connection.close()


setup_database()

# 간단 메모리(세션별 진행상황)
chat_histories = {}
user_progress = {}

# --- 2) 자유질문 파서 ---
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


# --- 3) SQL 가드레일 ---
def _is_safe_sql(sql: str) -> bool:
    s = sql.strip().lower()
    if not s.startswith("select"):
        return False
    forbidden = ("insert", "update", "delete", "drop", "alter", "create", "truncate")
    if any(k in s for k in forbidden):
        return False
    return True


def _normalize_sql(sql: str) -> str:
    # LIMIT 없으면 10으로 강제
    if re.search(r"\blimit\b", sql, flags=re.I) is None:
        sql = sql.rstrip(";") + " LIMIT 10;"
    return sql


def _execute_sql(sql: str) -> str:
    return db.run(sql)


def _render_results(raw: str) -> str:
    """
    SQLDatabase.run 은 텍스트 테이블/리스트 문자열을 반환.
    간단 파싱해서 카드 형태로 정리.
    """
    if not raw or raw.strip() in ("[]",):
        return ""
    lines = [l for l in raw.strip().splitlines() if l.strip()]
    if len(lines) <= 1:
        return raw
    header = [h.strip() for h in lines[0].split("|")]
    items = []
    for row in lines[1:]:
        cols = [c.strip() for c in row.split("|")]
        rec = dict(zip(header, cols))
        name = rec.get("name") or rec.get("NAME") or "알 수 없음"
        loc = rec.get("location") or ""
        cat = rec.get("category") or ""
        rating = rec.get("rating") or ""
        items.append(f"- **{name}** — {loc} · {cat} · 평점 {rating}")
    return "\n".join(items)


# --- 4) 유틸 트리거 ---
def _check_triggers(user_query: str, progress: dict) -> None:
    if "오늘 점심" in user_query:
        progress["_today_lunch"] = True


def _maybe_today_lunch(progress: dict) -> bool:
    return progress.get("_today_lunch", False)


# --- 5) 메인 로직 ---
def get_ai_response(session_id: str, user_query: str):
    progress = user_progress.get(session_id, {})
    _check_triggers(user_query, progress)

    # 리셋
    if "처음으로" in user_query or "다시 시작" in user_query:
        user_progress.pop(session_id, None)
        chat_histories.pop(session_id, None)
        return {"type": "text", "answer": prompts.RESET_MESSAGE, "options": None}

    # 첫 턴 자동 파싱
    if not progress:
        progress = parse_initial_query(user_query, progress)

    # 이전 질문에 대한 응답 수집
    last_q = progress.get("last_question")
    if last_q:
        progress[last_q] = user_query
        progress.pop("last_question", None)

    # 슬롯 채우기
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
        return {
            "type": "buttons",
            "answer": prompts.ASK_PURPOSE,
            "options": prompts.PURPOSE_OPTIONS,
        }

    # 오늘 점심 — 매우 단순 랜덤/상위 1개 추천 (평점 순)
    if _maybe_today_lunch(progress):
        sql = f"""
        SELECT name, category, location, rating
        FROM restaurants
        WHERE location LIKE '%{progress['location'][:10]}%'
          AND category = '{progress['category']}'
        ORDER BY rating DESC
        LIMIT 1;
        """
        raw = _execute_sql(sql)
        if raw.strip() in ("[]", ""):
            answer = prompts.NO_RESULT_MESSAGE.format(
                location=progress["location"], category=progress["category"]
            )
        else:
            lines = [l for l in raw.splitlines() if l.strip()]
            if len(lines) >= 2:
                cols = [c.strip() for c in lines[1].split("|")]
                # header: name|category|location|rating
                name, cat, loc, rating = cols[:4]
                answer = prompts.TODAY_LUNCH_PROMPT.format(
                    name=name, category=cat, location=loc, rating=rating
                ) + prompts.ASK_SIMILAR_RESTAURANT
            else:
                answer = raw
        user_progress.pop(session_id, None)
        return {"type": "text", "answer": answer, "options": None}

    # --- 안전한 Text-to-SQL (2-스텝) ---
    try:
        user_question = (
            f"{progress.get('location')} 근처 {progress.get('category')} 식당을 추천해줘. "
            f"목적은 '{progress.get('purpose')}'. 상위 평점 위주로."
            " SQL만 생성해. SELECT만. LIMIT 10 이하."
            " 사용 가능한 테이블과 컬럼: restaurants(id,name,category,location,rating,has_private_room,recommended_for)."
            " location은 부분일치 LIKE 사용 가능(예: '%강남%')."
        )

        # 1) SQL 생성
        chain = create_sql_query_chain(llm, db)
        sql = chain.invoke({"question": user_question})
        sql = sql.strip().strip("```").replace("sql", "").strip()
        sql = _normalize_sql(sql)

        if not _is_safe_sql(sql):
            raise ValueError(f"Unsafe SQL: {sql}")

        # 2) 실행
        raw = _execute_sql(sql)

        if not raw or raw.strip() in ("[]",):
            final_answer = prompts.NO_RESULT_MESSAGE.format(
                location=progress.get("location"),
                category=progress.get("category"),
            )
        else:
            rendered = _render_results(raw)
            final_answer = rendered + prompts.ASK_SIMILAR_RESTAURANT

        # 세션 종료(원샷)
        user_progress.pop(session_id, None)
        return {"type": "text", "answer": final_answer, "options": None}

    except Exception as e:
        print(f"Error occurred: {e}")
        user_progress.pop(session_id, None)
        return {"type": "text", "answer": prompts.ERROR_MESSAGE, "options": None}
