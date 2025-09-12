# chat.py
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # 학습시 사용한 원본 모델
FINETUNED_PATH = "./chatbot_model"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")

# LoRA weight 불러오기
model = PeftModel.from_pretrained(base_model, FINETUNED_PATH)

chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

print("=== Chatbot 테스트 시작 ===")
print("종료하려면 'exit' 입력\n")

while True:
    q = input("사용자: ")
    if q.lower().strip() == "exit":
        print("종료합니다.")
        break

    # 모델에게 입력 전달 (Q: ~ 형식 유지)
    prompt = f"Q: {q}\nA:"
    resp = chatbot(
        prompt,
        max_new_tokens=256,   # 답변 최대 길이 늘림
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=chatbot.tokenizer.eos_token_id
    )[0]["generated_text"]

    # "Q: ~ A:" 이후 답변만 추출
    if "A:" in resp:
        answer = resp.split("A:")[-1].strip().split("\n")[0]  # 첫 줄만 가져오기
    else:
        answer = resp.strip().split("\n")[0]

    print("챗봇:", answer)
