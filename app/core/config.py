import os
from dotenv import load_dotenv

load_dotenv()

# ===== 절대 경로 고정 =====
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB_PATH = os.path.join(BASE_DIR, "data", "sample_db.sqlite")
DATABASE_URI = f"sqlite:///{DB_PATH.replace('\\', '/')}"

# ===== LLM 설정 =====
# 예: Qwen/Qwen2.5-1.5B-Instruct (로컬 HF 파이프라인)
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
LLM_TASK = "text-generation"
LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "256"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# 기타(필요 시)
API_KEY = os.getenv("API_KEY")
# HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
