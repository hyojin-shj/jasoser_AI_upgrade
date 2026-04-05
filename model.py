import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import mlflow
import mlflow.pytorch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset

# 🔧 설정
BASE_MODEL = "gpt2"
MODEL_DIR = Path("models/slm_resume_specialist")
LOGS_DIR = Path("logs/slm_training")
REGISTERED_MODEL_NAME = "InHouseResumeSLM"

def is_model_trained():
    """모델이 이미 학습되어 저장되어 있는지 확인"""
    return MODEL_DIR.exists() and (MODEL_DIR / "pytorch_model.bin").exists() or (MODEL_DIR / "model.safetensors").exists()

def prepare_slm_dataset(data_path: str):
    """JSON 데이터를 HuggingFace Dataset으로 변환"""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    texts = []
    labels = []
    for item in data:
        text = f"질문: {item.get('question', '')} 답변: {item.get('answer1', '')} {item.get('answer2', '')}"
        scrap_score = min(item.get("스크랩수", 0) / 50.0, 1.0)
        if text.strip():
            texts.append(text)
            labels.append(scrap_score)
            
    df = pd.DataFrame({"text": texts, "label": labels})
    return Dataset.from_pandas(df)

def train_slm():
    """자체 SLM Fine-tuning & MLflow 모델 레지스트리 등록"""
    data_path = "data/linkareer_it_cover_letters.json"
    if not os.path.exists(data_path):
        return False, "데이터 파일이 없습니다."

    dataset = prepare_slm_dataset(data_path)
    dataset = dataset.train_test_split(test_size=0.1)
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)
    model.config.pad_token_id = model.config.eos_token_id

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

    tokenized_datasets = dataset.map(tokenize, batched=True)
    
    mlflow.set_experiment("In-House-SLM-Specialist")

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        num_train_epochs=1,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir=str(LOGS_DIR),
        use_cpu=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    with mlflow.start_run() as run:
        trainer.train()
        
        # 모델 로컬 저장
        trainer.save_model()
        tokenizer.save_pretrained(MODEL_DIR)
        
        # MLflow 모델 로깅 및 공식 등록(Registration)
        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME
        )
        
        return True, f"학습 및 레지스트리 등록 완료! (Model ID: {run.info.run_id})"

def predict_slm_score(text: str):
    """저장된 모델(또는 레지스트리 모델)을 로드하여 예측"""
    if not is_model_trained():
        return 75.0
        
    try:
        # 우선 로컬 디렉토리에서 로드 (빠름)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
            raw_score = outputs.logits.item()
            score = min(max(raw_score * 100, 0), 100)
            
        return round(score, 1)
    except Exception as e:
        # 레지스트리 모델로 시도하는 등의 폴백 로직 추가 가능
        return 70.0
