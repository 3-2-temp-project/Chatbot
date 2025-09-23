from pydantic import BaseModel
from typing import Optional, List

# 요청 본문을 위한 Pydantic 모델
class ChatRequest(BaseModel):
    session_id: str
    query: str

# 응답 본문을 위한 Pydantic 모델
class ChatResponse(BaseModel):
    type: str
    answer: str
    options: Optional[List[str]] = None
