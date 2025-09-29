import os
from pathlib import Path
from dotenv import load_dotenv
import torch  # GPU 감지용

# === BASE_DIR 설정 ===
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# === .env 로드 ===
load_dotenv(BASE_DIR / ".env")

# ===== DB 설정 =====
DATABASE_URI = os.getenv("DATABASE_URL")

# ===== Redis =====
REDIS_URL = os.getenv("REDIS_URL")

# ===== GeoCoding & 캐싱 설정 =====
GEOCODE_TTL_SECONDS = int(os.getenv("GEOCODE_TTL_SECONDS", "2592000"))
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))

# ===== LLM 설정 =====
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
LLM_TASK = "text-generation"
LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "256"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# ===== 디바이스 자동 감지 =====
if torch.cuda.is_available():
    DEVICE = 0   # 첫 번째 GPU
    print("[INFO] GPU 감지됨 → CUDA device=0")
else:
    DEVICE = -1  # CPU
    print("[INFO] GPU 없음 → CPU 사용")