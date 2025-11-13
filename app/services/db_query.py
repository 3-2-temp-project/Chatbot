"""
DB 조회 / 위치정보 조회 / ID 변환 / 결과 포맷 모듈
"""

from sqlalchemy import text
from app.services.sql_builder import _ENGINE

def execute_sql(sql: str):
    """DB에서 SELECT 실행"""

    try:
        with _ENGINE.begin() as conn:
            result = conn.execute(text(sql)).fetchall()
            return list(result)
    except Exception as e:
        print(f"[DB ERROR] SQL 실행 실패: {e}")
        return []
