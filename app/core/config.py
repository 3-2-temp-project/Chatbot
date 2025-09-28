import os
from dotenv import load_dotenv

load_dotenv()

# ===== DB 설정 =====
DATABASE_URI = os.getenv("DATABASE_URL")

# ===== LLM 설정 =====
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
LLM_TASK = "text-generation"
LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "256"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

API_KEY = os.getenv("API_KEY")
