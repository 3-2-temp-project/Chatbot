# train_resume.py
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig

# 데이터 로드
dataset = load_dataset("json", data_files={"train": "train.jsonl", "eval": "eval.jsonl"})

def formatting_func(example):
    return [f"Q: {example['instruction']}\nA: {example['response']}"]

# 모델/토크나이저 로드
checkpoint_dir = "./chatbot_model/checkpoint-3"
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, device_map="auto")

# LoRA 설정 (이전과 동일해야 함)
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=16,
    lora_dropout=0.1
)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./chatbot_model",
    num_train_epochs=1,  # 이어서 추가할 에포크
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    save_steps=10,
    logging_steps=2,
    fp16=True,
    report_to="tensorboard",
    logging_dir="./logs",
    save_total_limit=2,
)

# Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["eval"],
    peft_config=peft_config,
    args=training_args,
    formatting_func=formatting_func
)

# 이어서 학습 시작
trainer.train(resume_from_checkpoint=checkpoint_dir)

# 저장
trainer.model.save_pretrained("./chatbot_model")
tokenizer.save_pretrained("./chatbot_model")
