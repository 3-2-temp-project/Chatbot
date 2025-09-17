import os
from langchain_community.utilities import SQLDatabase
from langchain_huggingface import HuggingFacePipeline
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.memory import ConversationBufferMemory
from sqlalchemy import create_engine
import re

# --- 1. 초기 설정: 환경 변수 및 모델, DB 로드 ---

# 실제 DB 연결 시 이 부분을 교체합니다. (To-be 1번 과제)
# 예: "postgresql://user:password@host:port/database"
DB_URI = "sqlite:///./data/sample_db.sqlite"

# 임시 DB 및 테이블 생성 (실제 환경에서는 이미 DB가 구축되어 있으므로 이 부분은 필요 없음)
def setup_database():
    if os.path.exists("./data/sample_db.sqlite"):
        return
    os.makedirs("./data", exist_ok=True)
    engine = create_engine(DB_URI)
    connection = engine.raw_connection()
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE restaurants (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            location TEXT NOT NULL,
            rating REAL
        );
    """)
    cursor.execute("INSERT INTO restaurants (name, category, location, rating) VALUES ('우래옥', '한식', '강남역', 4.5);")
    cursor.execute("INSERT INTO restaurants (name, category, location, rating) VALUES ('스시효', '일식', '강남역', 4.8);")
    cursor.execute("INSERT INTO restaurants (name, category, location, rating) VALUES ('진대감', '한식', '삼성역', 4.6);")
    cursor.execute("INSERT INTO restaurants (name, category, location, rating) VALUES ('오스테리아 오르조', '양식', '판교역', 4.7);")
    cursor.execute("INSERT INTO restaurants (name, category, location, rating) VALUES ('봉피양', '한식', '판교역', 4.4);")
    connection.commit()
    connection.close()

setup_database()

# LangChain DB 연결
db = SQLDatabase.from_uri(DB_URI)

# LLM 모델 로드 (Qwen)
# 경량 모델이므로 CPU에서도 어느 정도 실행 가능합니다.
llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen1.5-1.8B-Chat",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 512,
        "top_k": 50,
        "temperature": 0.1,
    },
    device=-1, # GPU 사용 가능 시 'cuda', 없을 시 'cpu' 또는 'auto'
)

# 대화 기록 저장을 위한 메모리
# 세션별로 대화 기록을 관리하기 위해 딕셔너리 사용
chat_histories = {}

# --- 2. 하이브리드 대화 로직 및 에이전트 실행 함수 ---

def get_ai_response(session_id: str, user_query: str):
    """
    사용자의 질문을 받아 AI의 답변을 생성하는 메인 함수
    - 정보가 부족하면 되묻거나 버튼을 제시
    - 정보가 충분하면 Text-to-SQL 에이전트 실행
    """
    # 세션 ID에 맞는 대화 기록 가져오기
    memory = chat_histories.get(session_id, ConversationBufferMemory(memory_key="chat_history"))

    # 1단계: 필수 정보(지역, 음식 종류) 추출
    location_match = re.search(r'(\S+역|\S+동|\S+시)', user_query)
    category_match = re.search(r'(한식|일식|중식|양식)', user_query)

    location = location_match.group(0) if location_match else None
    category = category_match.group(0) if category_match else None

    # 2단계: 정보 확인 및 되묻기 / 버튼 제시 (하이브리드 모델)
    if not location:
        return {
            "type": "buttons",
            "answer": "어느 지역의 맛집을 찾아드릴까요? 🤔",
            "options": ["강남역", "판교역", "삼성역"]
        }
    
    if not category:
        return {
            "type": "buttons",
            "answer": f"{location} 근처 어떤 종류의 음식을 원하세요? 🍽️",
            "options": ["한식", "일식", "양식", "중식"]
        }

    # 3단계: 모든 정보 수집 완료 -> Text-to-SQL 에이전트 실행
    try:
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        agent_executor = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            agent_type="openai-tools",
            memory=memory,
            handle_parsing_errors=True, # 파싱 오류 처리
        )
        
        prompt = f"""
        Based on the following user request, find restaurants from the database.
        User request: "{user_query}"
        The final answer should be a friendly sentence recommending one or two restaurants.
        Do not just list the query result.
        Example: "Of course! For {category} near {location}, I recommend [Restaurant Name]. It has a great rating."
        """

        result = agent_executor.invoke(prompt)
        
        # 대화 기록 업데이트
        chat_histories[session_id] = memory

        return {
            "type": "text",
            "answer": result['output']
        }

    except Exception as e:
        print(f"Error occurred: {e}")
        return {
            "type": "text",
            "answer": "죄송합니다. 답변을 생성하는 중에 오류가 발생했어요. 잠시 후 다시 시도해주세요. 😥"
        }