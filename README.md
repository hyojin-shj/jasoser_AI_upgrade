# 🚀 AI Resume Coach: 맞춤형 자기소개서 평가 및 개선 시스템
"Small Language Model(SLM)을 활용한 데이터 기반 취업 컨설팅 및 이력서 최적화 서비스"

<img width="1920" height="1080" alt="1" src="https://github.com/user-attachments/assets/549d904c-c234-4f60-950a-04f7d1be33fc" />


![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

---

## 🔗 서비스 구성 (Deployment & Repo)

*   **시현 영상:** [👉 [프로젝트 소개 영상 바로가기](https://youtu.be/t5xqTE-M338)]
*   <img width="1920" height="1080" alt="7" src="https://github.com/user-attachments/assets/04e0e6e3-1d7e-4f14-ba1c-2dfabbc184c2" />

---

## 📅 프로젝트 개요
*   **프로젝트 기간:** 2026.03.23 ~ 2026.04.13
*   **한 줄 소개:**  특정 도메인(자소서)에 최적화된 SLM 및 멀티 LLM모델의 AI 이력서 코칭 비교 분석 플랫폼

---

## 💡 주요 기능 (Key Features)

### 1️⃣ 실시간 AI 자소서 평가 및 피드백
*   **다각도 분석:** 가독성, 논리성, 직무 적합성 등 7가지 지표를 바탕으로 점수화 및 상세 피드백 제공.
*   **멀티 모델 비교:** GPT-4와 같은 거대 모델과 자체 튜닝한 SLM(Qwen 1.5B 기반)의 평가 결과를 실시간으로 비교 분석.
*   <img width="1920" height="1080" alt="12" src="https://github.com/user-attachments/assets/88334227-0c65-4bc2-9466-903fc08544a7" />
<img width="1920" height="1080" alt="자소서AI" src="https://github.com/user-attachments/assets/16fe7752-98c2-435b-a004-8cca3385ff10" />




### 2️⃣ 데이터 기반 SLM 최적화 (MLOps)
*   **MLflow 연동:** 모델 학습 과정(Loss, Metrics)을 실시간으로 로깅하고 시각화하여 최적의 가중치 관리.
*   **경량 모델 활용:** Qwen 2.5 1.5B 모델을 LoRA/QLoRA 기법으로 파인튜닝하여 저비용·고효율의 성능 확보.

<img width="1920" height="1080" alt="9" src="https://github.com/user-attachments/assets/d73399e3-7784-473f-b4d9-e5e31f7edf06" />


### 3️⃣ 성능 벤치마킹 대시보드
*   **Ground Truth 비교:** 실제 전문가의 평가 데이터와 AI의 평가 결과 간의 유사도를 측정하여 모델의 신뢰성 검증.
*   **시각화 리포트:** Plotly와 Matplotlib을 활용하여 모델별 성능 분포를 한눈에 파악할 수 있는 차트 제공.

<img width="1920" height="1080" alt="14" src="https://github.com/user-attachments/assets/af80c0a9-6c2a-4cbb-a561-a8b41b975f11" />

---

## 📊 데이터 분석 및 연구 (Research & Insights)

1.  **데이터 선정:** 실제 합격 자소서 300건을 활용하여 평가 기준(Ground Truth) 수립.
2.  **인사이트:** 
    *   **비용 효율성:** SLM이 거대 모델 대비 10% 미만의 비용으로 유사한 도메인 성능을 낼 수 있음을 확인.
    *   **정교한 피드백:** 단순 요약을 넘어, 문장 단위의 구체적인 개선안 제시 가능.

<img width="1920" height="1080" alt="16" src="https://github.com/user-attachments/assets/5853e5d3-7ad3-4f20-8915-5c9ca667cb53" />

---

## 🛠️ 기술 스택 (Tech Stack)

| 구분 | 기술 | 활용 내용 |
| :--- | :--- | :--- |
| **Frontend** | **Streamlit** | 대시보드 시각화 및 사용자 인터페이스(UI) 구성 |
| **Backend** | **Python** | 자소서 평가 로직 구현 및 모델 추론 엔진 구축 |
| **AI / ML** | **Qwen / PyTorch** | SLM 파인튜닝 (LoRA), HuggingFace Transformers 활용 |
| **MLOps** | **MLflow** | 실험 관리, 하이퍼파라미터 트래킹 및 모델 레지스트리 |
| **Analysis** | **Plotly / Pandas** | 평가 지표 시각화 및 데이터 전처리 |

---

## 📂 프로젝트 구조 (Directory Structure)
```text
📂 jasoser_AI_upgrade
 ┣ 📜 app.py               # 메인 Streamlit 대시보드
 ┣ 📜 engine.py            # AI 추론 엔진 및 모델 로딩 로직
 ┣ 📜 eval_manager.py      # 성능 평가 및 비교 매니저
 ┣ 📜 model.py             # 모델 구조 정의 및 가중치 관리
 ┣ 📂 data/                # 학습 및 테스트 데이터셋 (.json)
 ┗ 📜 requirements.txt     # 프로젝트 의존성 관리
```<img width="1920" height="1080" alt="1" src="https://github.com/user-attachments/assets/ecbe47b0-be29-470d-b3d8-92b0a1e45b49" />

## 🚀 트러블 슈팅 (Troubleshooting)

### 🔧 이슈 1: 로컬 PC 환경에서의 모델 튜닝 한계 및 모델 전환
*   **문제 상황:** Llama 계열 모델을 활용하여 파인튜닝을 시도했으나, CPU 중심의 로컬 환경에서 VRAM/MEM 부족으로 인해 시스템이 중단되는 현상 발생.
*   **해결 방법:** 모델 사이즈를 대폭 줄인 **Qwen 1.5B/2.5B와 같은 초경량 SLM(Small Language Model)으로 전환**. 로컬 환경에서도 안정적인 학습과 추론이 가능한 하드웨어 최적화 모델 선택.
  
### 🔧 이슈 2: 파인튜닝 후 평가 기준의 과도한 엄격화 (Accuracy vs Strictness)
*   **문제 상황:** 도메인 데이터로 학습된 튜닝 모델이 일반 모델보다 훨씬 깐깐한 기준으로 자소서를 평가함에 따라, 합격권 자소서임에도 점수가 과도하게 낮게 산출되는 문제 확인.
*   **해결 방법:** 평가 프롬프트의 'Ground Truth' 기준을 재설정하고, 모델이 정성적 평가와 정량적 성능 사이에서 균형을 잡을 수 있도록 **평가 로직 가중치를 재조정**함.

### 🔧 이슈 3: 무거운 모델 로딩으로 인한 사용자 경험(UX) 저하
*   **문제 상황:** 페이지를 새로고침하거나 분석 탭으로 이동할 때마다 대형 가중치 파일을 반복해서 로드하여 앱 반응 속도가 매우 느려짐.
*   **해결 방법:** Streamlit의 **캐싱 리소스(`@st.cache_resource`) 기능**을 도입하여 초기 1회 로드 후 메모리에 모델을 상주시킴으로써 로딩 속도를 획기적으로 향상.


### 🔧 이슈 4: 실험 반복에 따른 API 비용 및 리소스 낭비
*   **문제 상황:** 다수의 실험과 테스트를 진행하면서 토큰 소모량이 급증하고 처리 시간이 늘어나는 문제 발생.
*   **해결 방법:** 불필요한 페르소나 설명 등의 컨텍스트를 제거하고 핵심 정보 위주로 프롬프트를 압축하여 **토큰 소모량을 최소화**함. 이를 통해 초당 처리(TPS) 효율을 높이고 운영 비용 절감.
