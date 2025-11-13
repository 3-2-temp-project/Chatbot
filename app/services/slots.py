"""
슬롯 추출 및 트리거 처리 모듈
"""

import re
from typing import Dict, Any


def parse_initial_query(query: str, progress: Dict[str, Any]) -> Dict[str, Any]:
    """
    사용자의 자연어 입력에서 location, people, category, purpose를 순차적으로 추출한다.
    """

    # 1) 지역
    if "location" not in progress:
        loc = re.search(r"(\S+역|\S+동|\S+시|\S+구)", query)
        if loc:
            progress["location"] = loc.group(0)
        return progress

    # 2) 인원
    if "people" not in progress:
        match = re.search(r"(\d+)\s*명", query)
        if match:
            progress["people"] = int(match.group(1))
        return progress

    # 3) 음식 카테고리
    if "category" not in progress:
        cat = re.search(r"(한식|일식|중식|양식)", query)
        if cat:
            progress["category"] = cat.group(0)
        return progress

    # 4) 목적
    if "purpose" not in progress:
        purp = re.search(r"(오찬|접대|회식)", query)
        if purp:
            progress["purpose"] = purp.group(0)
        return progress

    return progress


def check_triggers(query: str, progress: Dict[str, Any]) -> None:
    """특수 상황(예: '오늘 점심')을 감지하여 플래그를 설정한다."""
    if "오늘 점심" in query:
        progress["_today_lunch"] = True


def is_today_lunch(progress: Dict[str, Any]) -> bool:
    """오늘 점심 추천 트리거 여부"""
    return progress.get("_today_lunch", False)
