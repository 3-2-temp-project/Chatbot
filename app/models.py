from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    """클라이언트가 서버로 보내는 요청 데이터의 구조"""
    session_id: str
    query: str

class ChatResponse(BaseModel):
    """서버가 클라이언트로 보내는 응답 데이터의 구조"""
    type: str
    answer: str
    options: Optional[List[str]] = None