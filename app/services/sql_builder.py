"""
LLM 기반 SQL 생성 / SQL 안전성 검사 / Fallback SQL 모듈
"""

import re
from typing import Dict, Any
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import create_sql_query_chain
from app.core import config

# -----------------------------------------------------------
# DB + LLM 초기화
# -----------------------------------------------------------
_ENGINE = create_engine(config.DATABASE_URI, pool_pre_ping=True, future=True)
db = SQLDatabase.from_uri(config.DATABASE_URI)

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


# -----------------------------------------------------------
# SQL 생성
# -----------------------------------------------------------
def generate_sql_from_llm(progress: Dict[str, Any]) -> str:
    """
    LLM에게 자연어 기반 SQL 생성을 요청한다.
    """

    question = (
        f"{progress['location']} 근처에서 {progress['people']}명이 먹기 좋은 "
        f"{progress['category']} 식당을 추천해줘. 목적은 '{progress['purpose']}'. "
        "단일 SELECT 문만 생성하고 ';'로 끝내. "
        "오직 restaurants 테이블만 사용. "
        "컬럼: id,name,category,location,rating. "
        "location은 LIKE '%강남%' 같은 부분일치만 허용. LIMIT 10 이하."
    )

    chain = create_sql_query_chain(llm, db)
    sql = chain.invoke({"question": question})

    sql = sql.strip().strip("```").replace("sql", "").strip()
    return normalize_sql(sql)


# -----------------------------------------------------------
# SQL 안전성 검사
# -----------------------------------------------------------
def normalize_sql(sql: str) -> str:
    """LIMIT 없으면 강제로 LIMIT 10 추가"""
    if "limit" not in sql.lower():
        sql = sql.rstrip(";") + " LIMIT 10;"
    return sql


def is_safe_sql(sql: str) -> bool:
    """SELECT만 허용하고 시스템 변경 SQL 차단"""
    s = sql.lower()
    if not s.startswith("select"):
        return False

    forbidden = ["insert", "update", "delete", "drop", "alter", "create", "truncate"]
    return not any(f in s for f in forbidden)


def is_schema_safe(sql: str) -> bool:
    """허용되지 않은 조인/집계/서브쿼리 차단"""
    s = sql.lower()
    if any(x in s for x in [" join ", " union ", " group by ", " with "]):
        return False

    tables = re.findall(r"from\s+([a-zA-Z_][\w]*)", s)
    return all(t in _ALLOWED_TABLES for t in tables)


# -----------------------------------------------------------
# Fallback SQL
# -----------------------------------------------------------
def fallback_sql_from_slots(progress: Dict[str, Any]) -> str:
    """LLM SQL이 안전하지 않을 때 사용하는 대체 SQL"""

    def esc(v: str) -> str:
        return str(v).replace("'", "''")[:50]

    loc = esc(progress.get("location", ""))
    cat = esc(progress.get("category", ""))
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
        "ORDER BY COALESCE(s.popularity_score, r.rating / 5.0) DESC "
        "LIMIT 10;"
    )
