"""
LLM 기반 SQL 생성 / SQL 안전성 검사 / Fallback SQL 모듈
"""

import re
import concurrent.futures
from typing import Dict, Any

from sqlalchemy import create_engine, text
from langchain_community.utilities import SQLDatabase
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import create_sql_query_chain

from app.core import config

# DB 연결
_ENGINE = create_engine(config.DATABASE_URI, pool_pre_ping=True, future=True)
db = SQLDatabase.from_uri(config.DATABASE_URI)

# LLM 초기화
llm = HuggingFacePipeline.from_model_id(
    model_id=config.LLM_MODEL_ID,
    task=config.LLM_TASK,
    pipeline_kwargs={
        "max_new_tokens": config.LLM_MAX_NEW_TOKENS,
        "temperature": config.LLM_TEMPERATURE,
        "do_sample": config.LLM_TEMPERATURE > 0,
        "return_full_text": False,
    },
    device=config.DEVICE,
)

_ALLOWED_TABLES = {"restaurants"}


# -------------------------------
# LLM SQL 생성 + 5초 timeout
# -------------------------------
def generate_sql_from_llm(progress: Dict[str, Any]) -> str | None:
    """LLM에게 SQL 생성 요청 (5초 제한)"""

    question = (
        f"{progress['location']} 근처에서 {progress['people']}명이 먹기 좋은 "
        f"{progress['category']} 식당을 추천해줘. 목적은 '{progress['purpose']}'. "
        "단일 SELECT 문만 생성하고 ';'로 끝내. "
        "오직 restaurants 테이블만 사용. "
        "컬럼: id,name,category,location,rating. "
        "location은 LIKE '%강남%' 형태의 부분일치만 허용. LIMIT 10 이하."
    )

    chain = create_sql_query_chain(llm, db)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as exe:
        future = exe.submit(chain.invoke, {"question": question})
        try:
            sql = future.result(timeout=5)
        except concurrent.futures.TimeoutError:
            return None

    sql = sql.strip().strip("```").replace("sql", "").strip()
    return normalize_sql(sql)


# -------------------------------
# SQL 안전성 검사
# -------------------------------
def normalize_sql(sql: str) -> str:
    if "limit" not in sql.lower():
        sql = sql.rstrip(";") + " LIMIT 10;"
    return sql


def is_safe_sql(sql: str) -> bool:
    s = sql.lower()
    if not s.startswith("select"):
        return False

    forbidden = ["insert", "update", "delete", "drop", "alter", "create", "truncate"]
    return not any(f in s for f in forbidden)


def is_schema_safe(sql: str) -> bool:
    s = sql.lower()
    # 비허용 JOIN, GROUP BY 등 차단
    if any(x in s for x in [" join ", " union ", " group by", " with "]):
        return False

    tables = re.findall(r"from\s+([a-zA-Z_][\w]*)", s)
    return all(t in _ALLOWED_TABLES for t in tables)


# -------------------------------
# Fallback SQL
# -------------------------------
def fallback_sql_from_slots(progress: Dict[str, Any]) -> str:
    loc = progress.get("location", "")
    cat = progress.get("category", "")
    purp = progress.get("purpose", "")

    where = []
    if loc:
        where.append(f"r.location LIKE '%{loc}%'")
    if cat:
        where.append(f"r.category = '{cat}'")
    if "회식" in purp:
        where.append("r.has_private_room = TRUE")

    where_sql = " AND ".join(where) if where else "1=1"

    return f"""
        SELECT r.id, r.name, r.category, r.location, r.rating
        FROM restaurants r
        LEFT JOIN restaurant_stats s ON s.restaurant_id = r.id
        WHERE {where_sql}
        ORDER BY COALESCE(s.popularity_score, r.rating) DESC
        LIMIT 10;
    """
