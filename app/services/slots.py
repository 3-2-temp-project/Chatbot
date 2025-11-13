"""
슬롯 추출 및 트리거 처리 모듈
"""

import re
from typing import Dict, Any

# 강화된 정규식 패턴 세트
LOCATION_PATTERN = re.compile(r"([가-힣A-Za-z0-9]+(역|동|구|시))")
PEOPLE_PATTERN = re.compile(r"(\d+)\s*(명|사람)")
CATEGORY_PATTERN = re.compile(r"(한식|중식|일식|양식|고기|카페|분식)")
PURPOSE_PATTERN = re.compile(r"(회식|오찬|접대|데이트|모임)")

def extract_slots(query: str, progress: Dict[str, Any]) -> Dict[str, Any]:
    """사용자 입력에서 슬롯을 단계적으로 추출"""

    # 이미 해당 슬롯이 있다면 건너뛰고 다음 것으로
    if "location" not in progress:
        m = LOCATION_PATTERN.search(query)
        if m:
            progress["location"] = m.group(1)
        return progress

    if "people" not in progress:
        m = PEOPLE_PATTERN.search(query)
        if m:
            progress["people"] = int(m.group(1))
        return progress

    if "category" not in progress:
        m = CATEGORY_PATTERN.search(query)
        if m:
            progress["category"] = m.group(1)
        return progress

    if "purpose" not in progress:
        m = PURPOSE_PATTERN.search(query)
        if m:
            progress["purpose"] = m.group(1)
        return progress

    return progress
