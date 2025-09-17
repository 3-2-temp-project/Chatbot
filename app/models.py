from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    """클라이언트(프론트엔드)가 보내는 요청 형식"""
    session_id: str  # 사용자별 대화 맥락 유지를 위한 세션 ID
    query: str       # 사용자가 입력한 질문

class ChatResponse(BaseModel):
    """서버가 클라이언트에게 보내는 응답 형식"""
    type: str  # 답변 종류: 'text' (일반 텍스트), 'buttons' (버튼 제시)
    answer: str
    options: Optional[List[str]] = None # 버튼 목록 (type이 'buttons'일 때 사용)