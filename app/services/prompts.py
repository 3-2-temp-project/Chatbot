# --- 인사 및 질문 문구 ---
GREETING_MESSAGE = (
    "안녕하세요. 업무추진비 맛집 추천 AI입니다. '오늘 점심' 추천을 원하시면 말씀해주세요! "
    "특정 맛집을 찾으시면 지역을 알려주세요. (예: 강남역)"
)
ASK_CATEGORY = "{location} 근처에서 찾으시는 음식 종류가 있으신가요?"
ASK_PURPOSE = "어떤 목적으로 식사 장소를 찾으시나요?"

# --- 버튼 옵션 ---
CATEGORY_OPTIONS = ["한식", "일식", "중식", "양식"]
PURPOSE_OPTIONS = ["간단한 실무자 오찬", "외부 손님 접대", "부서 회식"]

# --- 기능별 문구 ---
TODAY_LUNCH_PROMPT = (
    "오늘 점심 메뉴로 {name}({category}) 어떠신가요? "
    "{location}에 위치해있고, 평점도 {rating}점으로 좋은 곳이에요!"
)
ASK_SIMILAR_RESTAURANT = "\n\n혹시 이와 비슷한 다른 맛집도 추천해드릴까요?"
SIMILAR_RESTAURANT_PROMPT = (
    "네, 비슷한 스타일의 다른 맛집으로 {name}({category})을 추천드려요. "
    "이 곳은 {location}에 위치해있습니다."
)
NO_SIMILAR_RESTAURANT = "죄송하지만, 더 이상 비슷한 맛집을 찾지 못했어요. 😥"

# --- 안내 및 오류 문구 ---
NO_RESULT_MESSAGE = (
    "죄송합니다. {location} 근처에는 조건에 맞는 {category} 맛집을 찾지 못했어요. 😥 "
    "다른 조건으로 다시 시도해 보시겠어요?"
)
ERROR_MESSAGE = "죄송합니다. 답변을 생성하는 중에 오류가 발생했어요. '처음으로' 라고 입력하여 다시 시작해주세요."
RESET_MESSAGE = "네, 처음부터 다시 시작하겠습니다. 어느 지역의 맛집을 찾아드릴까요?"
