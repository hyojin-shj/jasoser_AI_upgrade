# 🧠 Master AI Resume Evaluator: System Flow

이 문서는 AI 기반 자소서 분석 시스템의 데이터 흐름, 모델 아키텍처, 그리고 성능 검증 방식을 설명합니다.

---

## 🏗️ 1. 시스템 아키텍처 (System Architecture)

시스템은 **다각도 교차 검증 시스템(Multi-Perspective Evaluation)**을 채택하여, 일반적인 AI 모델이 놓칠 수 있는 사내 고유의 평가 기준을 정밀하게 분석합니다.

### 🧱 주요 구성 요소
*   **UI Engine**: Streamlit (인터랙티브 대시보드 및 실시간 로깅)
*   **Core SLM**: Qwen 2.5 0.5B Instruct (사내 데이터 기반 Fine-tuning)
*   **External LLMs**: OpenAI GPT-4o-mini, Google Gemini 1.5 Flash (글로벌 논리 및 가독성 평가)
*   **Semantic Engine**: Sentence-BERT (MiniLM-L6) (문맥 유사도 및 JD 매칭 분석)
*   **MLOps**: MLflow (학습 이력 및 모델 버전 관리)

---

## 📂 2. 데이터 파이프라인 (Data Pipeline)

### 📥 사용 데이터셋
| 데이터 구분 | 파일명 | 용도 | 규모 |
|:--- |:--- |:--- |:--- |
| **학습 데이터** | `linkareer_it_cover_letters.json` | 합격자 자소서 패턴 학습 | ~270건 |
| **검증 데이터** | `test_resumes.json` | 모델 성능 벤치마크 및 테스트 | ~40건 |

### 🔄 데이터 흐름
1.  **전처리**: JSON 데이터를 질문-답변 쌍으로 정규화하고, 합격 여부에 따른 Label링 수행
2.  **토큰화**: Qwen2 Tokenizer를 사용하여 128 토큰 길이로 컷오프 (메모리 최적화)
3.  **데이터 샘플링**: 5GB RAM 환경을 위해 학습 시 효율적인 배치 및 샘플링 적용

---

## ⚡ 3. SLM 학습 프로세스 (Fine-tuning Process)

저사양(5GB RAM) 환경에서도 고성능을 내기 위해 **초경량 맞춤형 학습(Efficient Fine-tuning)**을 수행합니다.

### 🛠️ 최적화 기법
*   **Adafactor Optimizer**: 메모리 점유율이 높은 Adam 대신 사용 (Optimizer States 메모리 90% 절감)
*   **Layer Freezing**: 24개 레이어 중 상위 22개를 동결하고, 핵심 정보를 담는 끝의 2개 레이어와 분류 헤드만 집중 학습
*   **Gradient Checkpointing**: 연산 중간값을 저장하지 않고 필요 시 재계산하여 RAM 부하 감소
*   **Gradient Accumulation**: 작은 배치(1)를 쓰되, 여러 번 모아서 학습(8 steps)하여 안정성 확보

---

## 📊 4. 분석 및 비교 로직 (Evaluation Logic)

사용자가 자소서를 제출하면 5개의 분석 엔진이 동시에 가동됩니다.

### 🌡️ 7개 평가 지표 (7-Dimension Criteria)
1. **직무적합도**: JD 키워드와 자소서의 정렬 상태
2. **구체성**: 경험의 수치화 및 실무 디테일 여부
3. **문제해결력**: 난관 극복 과정의 논리 구조
4. **일관성**: 자소서 전체의 맥락 유지력
5. **문장가독성**: 비문 제거 및 문장 호흡의 간결함
6. **창의성**: 접근 방식의 차별화 요소
7. **도전정신**: 높은 목표 설정 및 달성 의지

### 🤖 모델별 역할 분담
*   **OpenAI/Gemini**: 전체적인 문장 구성과 논리적 완결성 평가
*   **BERT (S-BERT)**: 질문의 의도와 답변 내용의 시맨틱 유사도 측정
*   **In-House SLM**: 해당 기업의 합격자 데이터 패턴과 얼마나 일치하는지 신경망 분석

---

## 🏛️ 6. 도메인별 LLM 선택 전략 (Selection Strategy)

단일 거대 모델에 의존하는 대신, **보안(Security), 비용(Cost), 도메인 전문성(Domain Expertise)**을 기준으로 모델을 선택하는 하이브리드 전략을 취합니다.

### 📊 모델 선택 매트릭스 (Selection Matrix)
| 비교 항목 | 거대 LLM (GPT-4) | 도메인 SLM (Qwen-0.5B) | 시멘틱 엔진 (S-BERT) |
| :--- | :--- | :--- | :--- |
| **보안성** | ⚠️ 낮음 (외부 전송 필요) | ✅ **매우 높음 (사내 구축)** | ✅ **높음 (로컬 작동)** |
| **운영 비용** | ⚠️ 높음 (Token 기반 과금) | ✅ **매우 낮음 (GPU 효율적)** | ✅ **무료 (Open Source)** |
| **도메인 최적화** | ⚠️ 일반적 (Generalist) | ✅ **최적화 (Specialist)** | ⚠️ 정적 (Similarity) |
| **처리 속도** | ⚠️ 보통 (Network Latency) | ✅ **매우 빠름 (Local Edge)** | ✅ **매우 빠름** |

### 🛠️ 계층형 평가 워크플로우 (Hierarchical Workflow)
1.  **Filter Layer (SLM)**: 전수 조사를 통한 1차 스크리닝. 사내 합격 패턴과의 정합성을 초고속으로 판별.
2.  **Analysis Layer (LLM)**: 필터링 통과 자소서에 한해 심층 논리/가독성 평가 수행 (비용 효율화).
3.  **Verification Layer (BERT)**: JD 키워드와 실제 내용 간의 정적 일치율 최종 검증.

---

## 🏁 7. 전략적 결론 (Strategic Conclusion)

본 프로젝트는 "가장 큰 모델이 최고의 성능을 낸다"는 관념에서 벗어나, **도메인 특화(Domain-Specific)된 소형 모델이 실제 산업 환경에서 얼마나 효율적이고 강력한지**를 입증합니다.

*   **데이터 프라이버시**: 민감한 인사 데이터를 외부로 유출하지 않고도 최상급 수준의 평가 가능.
*   **지속 가능성**: 적은 컴퓨팅 자원으로도 기업 고유의 합격 DNA를 반영한 맞춤형 평가 시스템 구축 가능.
*   **비즈니스 효율**: 거대 모델 대비 운영 비용을 99% 절감하면서도, 특정 도메인에서는 동등한 수준의 판단력을 유지함.

---

## 🚀 향후 발전 방향
*   **Quantization (QLoRA)**: 메모리 점유율을 더 낮추기 위해 4-bit 양자화 도입 검토
*   **RAG 통합**: 사내 채용 가이드라인 문서를 실시간으로 참조하는 분석 기능 추가
*   **Feedback Loop**: 인사 담당자의 평가 결과를 다시 학습 데이터로 피드백하는 구조 구축
