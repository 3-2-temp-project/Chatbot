from flask_sqlalchemy import SQLAlchemy
from pydantic import BaseModel
from typing import Optional, List
from passlib.context import CryptContext

db = SQLAlchemy()
pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ===== SQLAlchemy ORM =====
class RestaurantInfo(db.Model):
    __tablename__ = 'restaurant_info'
    res_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    res_name = db.Column(db.String(128), nullable=False)
    address = db.Column(db.String(255), nullable=False)
    lat = db.Column(db.Float, nullable=False)
    lng = db.Column(db.Float, nullable=False)
    res_phone = db.Column(db.String(32), nullable=True)
    category = db.Column(db.String(64), nullable=True)
    price = db.Column(db.Integer, nullable=True)
    score = db.Column(db.Float, nullable=True)

class User(db.Model):
    __tablename__ = 'users'
    user_num = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.String(64), unique=True, nullable=False)
    user_name = db.Column(db.String(64), nullable=False)
    user_nickname = db.Column(db.String(64), nullable=False)
    password = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(254), unique=True, nullable=False)
    date = db.Column(db.TIMESTAMP, nullable=True)

    # === 비밀번호 해시 처리 (필수) ===
    def set_password(self, plain: str) -> None:
        """평문 비밀번호를 해시로 변환하여 저장"""
        self.password = pwd.hash(plain)

    def check_password(self, plain: str) -> bool:
        """입력 평문이 저장된 해시와 일치하는지 검증"""
        return pwd.verify(plain, self.password)

# ===== FastAPI Request/Response 모델 =====
class ChatRequest(BaseModel):
    session_id: str
    query: str

class ChatResponse(BaseModel):
    answer: str = ""
    response: str
    items: list | None = None

class EventRequest(BaseModel):
    session_id: str
    event: str                   # 'impression' | 'click' | 'favorite' ...
    restaurant_id: Optional[int] = None
    value: Optional[str] = None