from pydantic import BaseModel
from typing import Optional, List

# /chat 요청/응답
class ChatRequest(BaseModel):
    session_id: str
    query: str

class ChatResponse(BaseModel):
    type: str
    answer: str
    options: Optional[List[str]] = None

# /event (사용자 행동 로그 적재)
class EventRequest(BaseModel):
    session_id: str
    event: str                   # 'impression' | 'click' | 'favorite' ...
    restaurant_id: Optional[int] = None
    value: Optional[str] = None
