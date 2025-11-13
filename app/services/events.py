"""
user_events 기록 / restaurant_stats 갱신 모듈
"""

from sqlalchemy import create_engine, text
from app.core import config

_ENGINE = create_engine(config.DATABASE_URI, pool_pre_ping=True, future=True)


def log_event(event: str, session_id: str, restaurant_id: int, value=None) -> None:
    """사용자 행동 기록"""
    try:
        with _ENGINE.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO user_events (session_id, restaurant_id, event, value)
                    VALUES (:sid, :rid, :event, :val)
                """),
                {"sid": session_id, "rid": restaurant_id, "event": event, "val": value},
            )
    except Exception as e:
        print("[event log error]", e)


def recalc_stats() -> None:
    """식당 통계 즉시 갱신"""
    with _ENGINE.begin() as conn:
        conn.exec_driver_sql("""
        INSERT INTO restaurant_stats (restaurant_id, impressions_7d, impressions_30d,
                                     clicks_7d, clicks_30d)
        SELECT r.id,
               SUM(CASE WHEN ue.event = 'impression' AND ue.created_at >= NOW() - INTERVAL '7 days' THEN 1 ELSE 0 END),
               SUM(CASE WHEN ue.event = 'impression' AND ue.created_at >= NOW() - INTERVAL '30 days' THEN 1 ELSE 0 END),
               SUM(CASE WHEN ue.event = 'click'      AND ue.created_at >= NOW() - INTERVAL '7 days' THEN 1 ELSE 0 END),
               SUM(CASE WHEN ue.event = 'click'      AND ue.created_at >= NOW() - INTERVAL '30 days' THEN 1 ELSE 0 END)
        FROM restaurants r
        LEFT JOIN user_events ue ON ue.restaurant_id = r.id
        GROUP BY r.id
        ON CONFLICT (restaurant_id) DO UPDATE
        SET impressions_7d = EXCLUDED.impressions_7d,
            impressions_30d = EXCLUDED.impressions_30d,
            clicks_7d = EXCLUDED.clicks_7d,
            clicks_30d = EXCLUDED.clicks_30d;
        """)
