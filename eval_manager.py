import torch
import pandas as pd
import json
import gc
import os
from pathlib import Path
from model import predict_slm_score, BASE_MODEL, MODEL_DIR

def run_performance_benchmark(test_data_path="data/test_resumes.json"):
    """
    일괄 처리 방식으로 벤치마크를 수행하여 메모리 부하를 최소화합니다.
    """
    if not Path(test_data_path).exists():
        return None, "테스트 데이터 파일을 찾을 수 없습니다."

    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # 1. 베이스 모델 점수 일괄 산출
    base_scores = []
    print("Loading Base Model for batch evaluation...")
    for item in test_data:
        text = f"질문: {item.get('question', '')} 답변: {item.get('answer', '')}"
        s = predict_slm_score(text, model_path_or_name=BASE_MODEL)
        base_scores.append(s)

    # 메모리 강제 정리 (모델 교체 전)
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # 2. 파인튜브 모델 점수 일괄 산출
    tuned_scores = []
    print("Loading Tuned Model for batch evaluation...")
    for item in test_data:
        text = f"질문: {item.get('question', '')} 답변: {item.get('answer', '')}"
        s = predict_slm_score(text, model_path_or_name=MODEL_DIR)
        tuned_scores.append(s)

    # 결과 병합
    results = []
    for i, item in enumerate(test_data):
        actual_label = item.get("label", "Unknown")
        results.append({
            "ID": item["id"],
            "실제 결과": actual_label,
            "Base 모델(0.5B)": base_scores[i],
            "Tuned SLM": tuned_scores[i],
            "개선도": round(tuned_scores[i] - base_scores[i], 1),
            "내용 요약": item.get("answer", "")[:30] + "..."
        })
        
    df = pd.DataFrame(results)
    return df, "동적 벤치마크 완료 (메모리 최적화 모드)"

def generate_experiment_markdown(df):
    """결과 데이터를 마크다운 형식으로 변환"""
    if df is None or df.empty:
        return "데이터가 없습니다."
        
    md = "## 📊 자소서 모델 성능 검증 리포트\n\n"
    md += "테스트 데이터셋을 활용하여 베이스 모델과 사내 전용 SLM의 예측 점수를 비교한 결과입니다.\n\n"
    md += df.to_markdown(index=False)
    md += f"\n\n**최종 업데이트**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
    return md
