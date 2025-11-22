# Chatbot
수원대학교 2025-2학기 시스템보안프로젝트 AI_Chatbot

# 라이브러리 설치
pip install -r requirements.txt

# 서버 실행 
uvicorn app.main:app --reload

# 실행 확인 
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [xxxxx] using statreload
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.

# API 테스트
# 웹 브라우저를 열고 http://127.0.0.1:8000/docs      접속해서 챗봇 API를 테스트해보세요!