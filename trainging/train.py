from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig

# 1. 데이터 로드
dataset = load_dataset("json", data_files={"train":"train.jsonl", "eval":"eval.jsonl"})

# 2. formatting_func 정의 (SFTTrainer 전용)
def formatting_func(example):
    q = example["instruction"]
    a = example["response"]
    return f"Q: {q}\nA: {a}"

# 3. 모델 선택
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token   # ✅ 패딩 설정

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True)
)

# 4. LoRA 설정
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=16,
    lora_dropout=0.1
)

# 5. TrainingArguments
training_args = TrainingArguments(
    output_dir="./chatbot_model",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    save_steps=10,
    logging_steps=1,
    logging_first_step=True,
    save_total_limit=2,
    fp16=True,
    bf16=False,
    report_to="tensorboard",  
    logging_dir="./logs",
    eval_steps=10,
    dataloader_pin_memory=False
)

# 6. Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["eval"],
    peft_config=peft_config,
    args=training_args,
    formatting_func=formatting_func
)

# 7. 학습 시작
trainer.train()

# 8. 모델 저장
trainer.model.save_pretrained("./chatbot_model")
tokenizer.save_pretrained("./chatbot_model")
