import os
from dotenv import load_dotenv

load_dotenv()

# =========================
# Database
# =========================
DATABASE_URI = "sqlite:///./data/sample_db.sqlite"

# =========================
# LLM (Hugging Face Hub)
# - from_model_id 에는 HF Hub 모델 ID가 필요합니다.
#   예) "Qwen/Qwen2.5-1.5B-Instruct" 또는 "google/gemma-2-2b-it"
# =========================
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
LLM_TASK = "text-generation"

LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "256"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# Optional keys
API_KEY = os.getenv("API_KEY")
# HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
