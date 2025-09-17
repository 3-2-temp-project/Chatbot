import os
import re
import sqlite3
from sqlalchemy import create_engine

# LangChain 라이브러리
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_huggingface import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory

# 분리된 설정 및 프롬프트 파일 import
from app.core import config
from app import prompts

# --- 1. 초기 설정 ---
db = SQLDatabase.from_uri(config.DATABASE_URI)
llm = HuggingFacePipeline.from_model_id(
    model_id=config.LLM_MODEL_ID,
    task=config.LLM_TASK,
    pipeline_kwargs={"max_new_tokens": 100, "temperature": 0.1},
    device=-1,
)

def setup_database():
    if os.path.exists("./data/sample_db.sqlite"): return
    os.makedirs("./data", exist_ok=True)
    engine = create_engine(config.DATABASE_URI)
    connection = engine.raw_connection()
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE restaurants (
            id INTEGER PRIMARY KEY, name TEXT NOT NULL, category TEXT NOT NULL,
            location TEXT NOT NULL, rating REAL, has_private_room BOOLEAN,
            recommended_for TEXT
        );
    """)
    sample_data = [
        ('우래옥', '한식', '강남역', 4.5, 1, '접대'), ('스시효', '일식', '강남역', 4.8, 1, '접대'),
        ('진대감', '한식', '삼성역', 4.6, 1, '회식'), ('고향집', '한식', '삼성역', 4.2, 0, '오찬'),
        ('오스테리아 오르조', '양식', '판교역', 4.7, 0, '오찬'), ('봉피양', '한식', '판교역', 4.4, 0, '오찬')
    ]
    cursor.executemany("INSERT INTO restaurants VALUES (NULL, ?, ?, ?, ?, ?, ?)", sample_data)
    connection.commit()
    connection.close()
setup_database()

chat_histories = {}
user_progress = {}

# --- 2. 신규 기능 로직 (⭐️ 추가됨) ---
def get_random_recommendation():
    """DB에서 랜덤으로 맛집 하나를 추천하는 함수"""
    conn = sqlite3.connect("./data/sample_db.sqlite")
    cursor = conn.cursor()
    cursor.execute("SELECT name, category, location, rating FROM restaurants ORDER BY RANDOM() LIMIT 1")
    result = cursor.fetchone()
    conn.close()
    if result:
        return prompts.TODAY_LUNCH_PROMPT.format(name=result[0], category=result[1], location=result[2], rating=result[3])
    return "죄송합니다, 추천할 맛집을 찾지 못했어요."

def get_similar_recommendation(last_recommendation):
    """이전 추천과 비슷한 카테고리의 다른 맛집을 추천하는 함수"""
    conn = sqlite3.connect("./data/sample_db.sqlite")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name, category, location FROM restaurants WHERE category = ? AND name != ? ORDER BY RANDOM() LIMIT 1",
        (last_recommendation['category'], last_recommendation['name'])
    )
    result = cursor.fetchone()
    conn.close()
    if result:
        return prompts.SIMILAR_RESTAURANT_PROMPT.format(name=result[0], category=result[1], location=result[2])
    return prompts.NO_SIMILAR_RESTAURANT

# --- 3. 메인 대화 로직 ---
def parse_initial_query(query, progress):
    # ... (이전과 동일)
    return progress

def get_ai_response(session_id: str, user_query: str):
    memory = chat_histories.get(session_id, ConversationBufferMemory(memory_key="chat_history"))
    progress = user_progress.get(session_id, {})

    # '처음으로' 명령어
    if "처음으로" in user_query or "다시 시작" in user_query:
        user_progress.pop(session_id, None)
        chat_histories.pop(session_id, None)
        return {"type": "text", "answer": prompts.RESET_MESSAGE, "options": None}

    # '오늘 점심' 기능 처리 (⭐️ 추가됨)
    if "오늘 점심" in user_query or "맛집 추천" == user_query.strip():
        recommendation_text = get_random_recommendation()
        return {"type": "text", "answer": recommendation_text, "options": None}

    # '비슷한 맛집' 기능 처리 (⭐️ 추가됨)
    if ("비슷한" in user_query and "맛집" in user_query) or "다른 곳" in user_query:
        if "last_recommendation" in progress:
            recommendation_text = get_similar_recommendation(progress['last_recommendation'])
            return {"type": "text", "answer": recommendation_text, "options": None}
        else:
            return {"type": "text", "answer": "죄송합니다, 먼저 맛집 추천을 받은 후에 비슷한 곳을 찾아볼 수 있어요.", "options": None}

    # ... (이하 순차적 질문 로직은 이전과 거의 동일)
    is_first_turn = not progress and not any(kw in user_query for kw in ["오늘 점심", "비슷한", "다른 곳"])
    if is_first_turn:
        progress = parse_initial_query(user_query, progress)

    # ... (질문 순서 로직)

    # 4. 에이전트 실행
    try:
        # ... (에이전트 실행 로직 동일)
        result = agent_executor.invoke(full_query)
        final_answer = result['output']

        if "I don't know" in final_answer or "couldn't find" in final_answer:
            final_answer = prompts.NO_RESULT_MESSAGE.format(...)
        else:
            # 추천 성공 시, 유사 맛집 추천 질문 및 업적 카운트 추가 (⭐️ 추가됨)
            progress['recommendation_count'] = progress.get('recommendation_count', 0) + 1
            if progress['recommendation_count'] == 3:
                final_answer += "\n\n" + "벌써 3번째 추천이네요! '맛집 탐험가' 칭호를 드립니다! 🏅"
            
            # 유사 맛집 추천을 위한 정보 저장
            # (실제로는 AI가 추천한 맛집의 정보를 정확히 파싱해야 함)
            progress['last_recommendation'] = {'name': '우래옥', 'category': '한식'} # 설명을 위한 하드코딩
            final_answer += prompts.ASK_SIMILAR_RESTAURANT
        
        user_progress[session_id] = progress # 유사 맛집 추천을 위해 progress 유지
        chat_histories[session_id] = memory
        return {"type": "text", "answer": final_answer, "options": None}
    except Exception as e:
        # ... (오류 처리)
        return {"type": "text", "answer": prompts.ERROR_MESSAGE, "options": None}