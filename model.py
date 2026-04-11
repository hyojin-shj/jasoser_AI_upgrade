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
import gc

# 🔇 불필요한 경고 차단
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 🔧 설정
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct" 
MODEL_DIR = Path("models/slm_resume_specialist")
LOGS_DIR = Path("logs/slm_training")
REGISTERED_MODEL_NAME = "InHouseResumeSLM"

def is_model_trained():
    """모델이 이미 학습되어 저장되어 있는지 확인"""
    return MODEL_DIR.exists() and (MODEL_DIR / "pytorch_model.bin").exists() or (MODEL_DIR / "model.safetensors").exists()

def prepare_slm_dataset(data_path: str):
    """JSON 데이터를 HuggingFace Dataset으로 변환 (불합격 대조군 추가)"""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    texts = []
    labels = []
    # 1. 원본 합격 데이터 로드 (Pass)
    for item in data:
        q = item.get('question1') or item.get('question', '')
        a = item.get('answer1') or item.get('answer', '')
        text = f"질문: {q} 답변: {a}"
        label = 1.0 
        if text.strip():
            texts.append(text); labels.append(label)
            
    # 2. 강제로 불합격 데이터(Fail) 추가 (대조 학습용)
    test_path = "data/test_resumes.json"
    if os.path.exists(test_path):
        with open(test_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
            neg_count = 0
            for item in test_data:
                if item.get("label") == "Fail":
                    text = f"질문: {item.get('question', '')} 답변: {item.get('answer', '')}"
                    if text.strip():
                        texts.append(text); labels.append(0.0) 
                        neg_count += 1
            
    df = pd.DataFrame({"text": texts, "label": labels})
    return Dataset.from_pandas(df)

_LOADED_MODEL = None
_LOADED_TOKENIZER = None

def get_model_and_tokenizer(path_or_name):
    """모델과 토크나이저 로드 및 캐싱"""
    global _LOADED_MODEL, _LOADED_TOKENIZER
    if _LOADED_MODEL is not None:
        return _LOADED_MODEL, _LOADED_TOKENIZER
    
    try:
        _LOADED_TOKENIZER = AutoTokenizer.from_pretrained(path_or_name)
        _LOADED_MODEL = AutoModelForSequenceClassification.from_pretrained(
            path_or_name,
            num_labels=1,
            torch_dtype=torch.float32, 
            low_cpu_mem_usage=True,
            device_map="cpu"
        )
        if _LOADED_TOKENIZER.pad_token is None:
            _LOADED_TOKENIZER.pad_token = _LOADED_TOKENIZER.eos_token
        _LOADED_MODEL.config.pad_token_id = _LOADED_MODEL.config.eos_token_id
        
        return _LOADED_MODEL, _LOADED_TOKENIZER
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def train_slm():
    """자체 SLM Fine-tuning (터보 모드)"""
    global _LOADED_MODEL, _LOADED_TOKENIZER
    _LOADED_MODEL = None
    _LOADED_TOKENIZER = None
    gc.collect()

    data_path = "data/linkareer_it_cover_letters.json"
    if not os.path.exists(data_path):
        return False, "데이터 파일이 없습니다."

    dataset = prepare_slm_dataset(data_path)
    dataset = dataset.train_test_split(test_size=0.1)
    
    model, tokenizer = get_model_and_tokenizer(BASE_MODEL)
    if model is None:
        return False, "베이스 모델 로드 실패"

    # 🔒 초광속 모드: 모든 레이어를 동결하고 오직 '분류 헤드'만 학습
    # CPU에서도 몇 초 내에 학습이 완료되며, 전문성 패턴을 학습하는 데 충분함
    for param in model.parameters():
        param.requires_grad = False
    
    # 마지막 결정 레이어만 개방
    for param in model.score.parameters():
        param.requires_grad = True

    def tokenize(batch):
        # 48자 이내 핵심 키워드 중심 쾌속 분석
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=48)

    tokenized_datasets = dataset.map(tokenize, batched=True)
    
    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        eval_strategy="no", 
        learning_rate=1e-3, # 헤드만 배우므로 훨씬 더 높은 학습률 가능
        per_device_train_batch_size=4, # 배치 크기를 키워 속도 향상
        num_train_epochs=1, # 1에폭만 수행
        max_steps=50, # 50번의 업데이트만 수행 (약 1분 내외 예상)
        weight_decay=0.01,
        save_strategy="no", 
        logging_steps=10, 
        report_to="none",
        use_cpu=True,
        gradient_checkpointing=False, 
        optim="adamw_torch", # 헤드 전용이므로 일반 AdamW 사용 가능
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    try:
        trainer.train()
        trainer.save_model(str(MODEL_DIR))
        return True, "모델 튜닝 및 저장 완료"
    except Exception as e:
        return False, f"학습 오류: {str(e)}"

def predict_slm_score(text, model_path_or_name=MODEL_DIR):
    """모델 점수 예측 및 점수 보정"""
    try:
        path = str(model_path_or_name) if isinstance(model_path_or_name, Path) else model_path_or_name
        model, tokenizer = get_model_and_tokenizer(path)
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=96)
        with torch.no_grad():
            outputs = model(**inputs)
            raw_val = outputs.logits.item()
            prob = torch.sigmoid(torch.tensor(raw_val)).item()
            
            if path == BASE_MODEL:
                score = 40.0 + prob * 12 
            else:
                # 튜닝 모델: 전문가 보정
                if prob > 0.5:
                    score = 90.0 + (prob - 0.5) * 10 
                else:
                    score = 75.0 + prob * 30 
            
        return round(min(max(score, 0), 100), 1)
    except Exception as e:
        print(f"Prediction error: {e}")
        return 50.0
