# prompt.py

USE_EMOJI = True
EMOJI_SORRY = " 😥" if USE_EMOJI else ""

# --- 인사 및 질문 문구 ---
GREETING_MESSAGE = (
    "안녕하세요. 업무추진비 맛집 추천 AI입니다. ‘오늘 점심’ 추천을 원하시면 말씀해 주세요. "
    "특정 맛집을 찾으시면 지역을 알려주세요. (예: 강남역)"
)
ASK_CATEGORY = "{location} 근처에서 찾으시는 음식 종류가 있으신가요?"
ASK_PURPOSE = "어떤 목적으로 식사 장소를 찾으시나요?"

# --- 버튼 옵션 ---
CATEGORY_OPTIONS = ["한식", "일식", "중식", "양식"]
PURPOSE_OPTIONS  = ["간단한 실무자 오찬", "외부 손님 접대", "부서 회식"]

# --- 기능별 문구 ---
TODAY_LUNCH_PROMPT = (
    "오늘 점심 메뉴로 {name}({category}) 어떠신가요? "
    "{location}에 위치해 있고, 평점은 {rating}점이에요!"
)
ASK_SIMILAR_RESTAURANT = "\n\n비슷한 다른 맛집도 추천해 드릴까요?"
SIMILAR_RESTAURANT_PROMPT = (
    "비슷한 스타일의 다른 맛집으로 {name}({category})을 추천드려요. "
    "이곳은 {location}에 있어요."
)

# --- 안내 및 오류 문구 ---
NO_SIMILAR_RESTAURANT = "죄송하지만, 더 비슷한 맛집을 찾지 못했어요." + EMOJI_SORRY
NO_RESULT_MESSAGE = (
    "죄송합니다. {location} 근처에서 조건에 맞는 {category} 맛집을 찾지 못했어요."
    + EMOJI_SORRY
    + " 다른 조건으로 다시 시도해 보시겠어요?"
)
ERROR_MESSAGE = "죄송합니다. 답변을 생성하는 중 오류가 발생했어요. ‘처음으로’라고 입력해 다시 시작해 주세요."
RESET_MESSAGE = "네, 처음부터 다시 시작할게요. 어느 지역의 맛집을 찾아드릴까요?"

# --- 안전 포맷 유틸: 누락 키는 '?'로 대체해 예외 방지 ---
def fmt(template: str, **kwargs) -> str:
    class _D(dict):
        def __missing__(self, k): return "?"
    return template.format_map(_D(kwargs))