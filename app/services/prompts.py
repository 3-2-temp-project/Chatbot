USE_EMOJI = True
EMOJI_SORRY = " 😥" if USE_EMOJI else ""

# --- 인사 ---
GREETING_MESSAGE = (
    "안녕하세요. 업무추진비 기반 AI 맛집 추천 서비스입니다. "
    "지역을 입력해 주시면 주변 맛집을 찾아드릴게요! (예: 강남역)"
)

# --- 질문 ---
ASK_CATEGORY = "{location} 주변에서 어떤 종류의 음식을 원하시나요?"
ASK_PEOPLE = "몇 명이 식사하시나요?"

# --- 선택 옵션 ---
CATEGORY_OPTIONS = ["한식", "중식", "일식", "양식", "카페"]

# --- 안내 문구 ---
TODAY_LUNCH_PROMPT = (
    "{location} 근처 {category} 추천 맛집으로 {name}을(를) 추천드려요! "
    "평점은 {rating}점입니다."
)

ASK_SIMILAR_RESTAURANT = "\n\n비슷한 맛집도 더 찾아드릴까요?"
SIMILAR_RESTAURANT_PROMPT = (
    "다른 추천 맛집으로는 {name}({category})이 있어요. "
    "{location}에 위치해 있습니다."
)

# --- 오류 문구 ---
NO_RESULT_MESSAGE = (
    "죄송합니다. {location} 근처에서 조건에 맞는 {category} 맛집을 찾지 못했어요."
    + EMOJI_SORRY
    + " 다른 조건으로 다시 시도해 보시겠어요?"
)

ERROR_MESSAGE = (
    "답변 생성 중 오류가 발생했습니다. ‘처음으로’라고 입력해 다시 시작할 수 있어요."
)

RESET_MESSAGE = (
    "네, 처음부터 다시 시작할게요! 어느 지역의 맛집을 찾아드릴까요?"
)

# --- 안전 포맷 ---
def fmt(template: str, **kwargs) -> str:
    class _D(dict):
        def __missing__(self, k): return "?"
    return template.format_map(_D(kwargs))
