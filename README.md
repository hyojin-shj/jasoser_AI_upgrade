# 🚀 Master AI Resume Evaluator: Custom SLM & MLOps Edition

> **자체 미세 조정(Fine-tuning)된 소형 언어 모델(SLM)과 Multi-LLM 협업 엔진을 통해 구현된 고도화된 자소서 분석 솔루션입니다.**

이 프로젝트는 단순한 텍스트 분석을 넘어, 기업 고유의 합격 데이터를 학습한 **In-House SLM**과 상용 LLM(OpenAI, Gemini 등)을 결합하여 다각도의 인재 검증 리포트를 제공합니다. **MLflow**를 통한 MLOps 파이프라인이 통합되어 있어 모델의 실험 이력 및 버전 관리가 완벽하게 수행됩니다.

---

## ✨ Key Features

### 🧠 1. In-House SLM Specialist (GPT-2 Fine-tuned)
- **Neural Context Analysis**: GPT-2(124M) 모델을 베이스로 하여 사내 합격 자소서의 문황 구조와 논리 패턴을 직접 학습.
- **Hybrid Scoring Engine**: 신경망 패턴 + 시맨틱 유사도 + JD 키워드 매칭이 결합된 고도의 전문가 채점 시스템.
- **Privacy & Security**: 로컬 환경에서 학습 및 추론이 진행되어 민감한 인사 데이터의 외부 유출을 원천 차단.

### 📊 2. Strategic Multi-Model Cross-Analysis
- **Cross-Engine Comparison**: OpenAI(GPT-4o), Gemini 1.5, BERT, Qwen 및 자체 SLM의 분석 결과를 한눈에 비교.
- **Dynamic Radar Charts**: 직무적합도, 문제해결성 등 7대 스타 지표를 시각화하여 모델별 시각 차이 분석.
- **Strategic Weights**: 각 모델이 가진 분석적 강점(논구성, 가독성, 데이터 패턴 등)을 막대 및 선형 그래프로 도식화.

### 🛠️ 3. MLOps Workflow with MLflow
- **Experiment Tracking**: 학습 시 발생하는 Loss, Metrics, Hyperparameters를 실시간으로 기록.
- **Model Registry**: 학습 완료된 모델을 `InHouseResumeSLM`으로 공식 등록하여 체계적인 버전 관리 수행.
- **Persistence**: 학습된 가중치 파일을 `models/` 디렉토리에 영구 저장하여 앱 재시작 시 즉각적인 로드 지원.

### 🎨 4. Premium UX/UI Dashboard
- **Interactive Feedbacks**: 길어지는 분석 의견을 '더보기' 토글로 처리하여 깔끔한 대시보드 레이아웃 유지.
- **Real-time Status Monitoring**: 사이드바를 통해 현재 모델 로드 상태(LOADED/NOT TRAINED)를 실시간 모니터링 및 재학습 제어.

---

## 🛠️ Technology Stack

| Category | Technology |
| :--- | :--- |
| **Core AI** | `Transformers`, `Torch`, `Sentence-Transformers`, `Datasets` |
| **LLM APIs** | `OpenAI GPT-4o-mini`, `Google Gemini 1.5 Flash`, `Qwen (Langchain)` |
| **MLOps** | `MLflow` (Tracking & Model Registry) |
| **Visualization** | `Matplotlib`, `Numpy`, `Pandas` |
| **Web App** | `Streamlit` (with Custom CSS Components) |

---

## 🚀 Quick Start

### 1. 환경 설정 및 라이브러리 설치
가상환경 활성화 후 필요한 모든 패키지를 설치합니다.
```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

### 2. MLflow 서버 가동 (실험 추적용)
별도의 터미널에서 MLflow UI를 실행하여 학습 대시보드에 접속합니다.
```bash
python -m mlflow ui
```

### 3. 애플리케이션 실행
```bash
streamlit run app.py
```

---

## 📑 MLOps 아키텍처 가이드
이 시스템은 학습 버튼 클릭 시 다음의 파이프라인을 자동으로 수행합니다:
1. **Data Prep**: 로컬 JSON 데이터를 시퀀스 분류용 데이터셋으로 변환.
2. **Fine-tuning**: PEFT/LoRA 방식의 가중치 최적화 진행 (CPU 최적화).
3. **Registration**: 학습된 모델 아티팩트를 MLflow 레지스트리에 등록.
4. **Active Loading**: 검증된 최신 버전의 모델을 엔진에 즉시 주입하여 추론(Inference) 수행.

---

## 📝 License
This project is for educational and AI portfolio purposes. All LLM models are subject to their respective usage policies.
