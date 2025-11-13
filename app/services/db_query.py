"""
DB 조회 / 위치정보 조회 / ID 변환 / 결과 포맷 모듈
"""

from typing import List, Tuple, Dict, Any
from sqlalchemy import create_engine, text
from app.core import config

_ENGINE = create_engine(config.DATABASE_URI, pool_pre_ping=True, future=True)


# -----------------------------------------------------------
# SQL 실행
# -----------------------------------------------------------
def execute_sql(sql: str) -> List[Tuple]:
    with _ENGINE.begin() as conn:
        return list(conn.execute(text(sql)).fetchall())


# -----------------------------------------------------------
# 결과 문자열 생성
# -----------------------------------------------------------
def render_results(rows: List[Tuple]) -> str:
    if not rows:
        return ""

    lines = []
    for row in rows:
        _id, name, category, location, rating = row[:5]
        lines.append(f"- **{name}** — {location} · {category} · 평점 {rating}")
    return "\n".join(lines)


# -----------------------------------------------------------
# 이름 → ID 변환
# -----------------------------------------------------------
def names_to_ids(names: List[str]) -> List[int]:
    if not names:
        return []

    placeholders = ", ".join([f":n{i}" for i in range(len(names))])
    q = text(f"SELECT id, name FROM restaurants WHERE name IN ({placeholders})")
    params = {f"n{i}": n for i, n in enumerate(names)}

    with _ENGINE.begin() as conn:
        rows = conn.execute(q, params).fetchall()

    name_to_id = {name: rid for rid, name in rows}
    return [name_to_id[n] for n in names if n in name_to_id]


# -----------------------------------------------------------
# 좌표/주소 조회
# -----------------------------------------------------------
def fetch_geo_map_by_names(names: List[str]) -> Dict[str, Dict[str, Any]]:
    if not names:
        return {}

    placeholders = ", ".join([f":n{i}" for i in range(len(names))])
    q = text(f"""
        SELECT res_name, address, lat, lng
        FROM restaurant_info
        WHERE res_name IN ({placeholders})
    """)

    params = {f"n{i}": n for i, n in enumerate(names)}

    result = {}
    with _ENGINE.begin() as conn:
        for rn, addr, lat, lng in conn.execute(q, params).fetchall():
            result[str(rn)] = {
                "address": addr,
                "lat": float(lat),
                "lng": float(lng)
            }
    return result
