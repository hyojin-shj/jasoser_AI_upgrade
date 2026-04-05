import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from engine import HREvaluator, CRITERIA
from model import train_slm, is_model_trained, REGISTERED_MODEL_NAME, MODEL_DIR

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(layout="wide", page_title="Master AI Resume Evaluator", page_icon="🧠")

# --- 커스텀 CSS ---
st.markdown("""
<style>
    .report-card { background-color: #f8f9fa; border-radius: 10px; padding: 20px; border-left: 5px solid #6c5ce7; margin-bottom: 20px; transition: all 0.3s; }
    .report-card:hover { box-shadow: 0 10px 20px rgba(0,0,0,0.1); }
    .metric-container { text-align: center; background: linear-gradient(135deg, #6c5ce7, #a29bfe); color: white; border-radius: 10px; padding: 12px; margin-bottom: 15px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
    .status-badge { padding: 4px 12px; border-radius: 12px; font-size: 0.75rem; font-weight: bold; margin-bottom: 10px; display: inline-block; }
    .status-online { background-color: #2ecc71; color: white; }
    .status-offline { background-color: #95a5a6; color: white; }
</style>
""", unsafe_allow_html=True)

if "qa_list" not in st.session_state:
    st.session_state.qa_list = [{"question": "", "answer": ""}]

evaluator = HREvaluator()

# --- 사이드바 ---
with st.sidebar:
    st.title("📂 시스템 컨트롤")
    company = st.text_input("회사명", value="현대자동차")
    job = st.text_input("지원 직무", value="AI 엔지니어")
    description = st.text_area("직무 설명(JD)")
    
    st.divider()
    st.subheader("🤖 전용 SLM 전문가 상태")
    
    if is_model_trained():
        st.markdown("<div class='status-badge status-online'>● LOADED (학습 완료)</div>", unsafe_allow_html=True)
        st.success(f"모델 레지스트리에 '{REGISTERED_MODEL_NAME}' 버전이 로드되었습니다.")
        btn_label = "🆕 모델 재학습 (Re-train)"
    else:
        st.markdown("<div class='status-badge status-offline'>● NOT TRAINED (학습 필요)</div>", unsafe_allow_html=True)
        st.warning("데이터 학습이 필요합니다.")
        btn_label = "🏗️ SLM Fine-tuning 시작"
        
    if st.button(btn_label):
        with st.status("SLM 정밀 튜닝 및 레지스트리 등록 중...", expanded=True) as status:
            st.write("1. 데이터셋 및 사전 학습 모델 로드...")
            time.sleep(0.5)
            st.write("2. Fine-tuning 가중치 최적화 (MLflow 로깅)...")
            success, msg = train_slm()
            if success:
                st.write(f"3. {msg}")
                status.update(label="학습 완료!", state="complete", expanded=False)
                time.sleep(1); st.rerun()
            else:
                st.error(f"❌ 실패: {msg}")
                status.update(label="오류 발생", state="error")
    
    st.divider()
    if st.button("🔄 설정 초기화"):
        st.session_state.clear(); st.rerun()

# --- 메인 영역 ---
st.title("🚀 Custom SLM 기반 고도화 자소서 분석")

with st.container(border=True):
    st.subheader("📝 분석 대상 자기소개서")
    for i, qa in enumerate(st.session_state.qa_list):
        col1, col2 = st.columns([0.9, 0.1])
        with col1:
             qa["question"] = st.text_input(f"문항 {i+1}", value=qa["question"], key=f"q_{i}")
             qa["answer"] = st.text_area(f"내용 {i+1}", value=qa["answer"], key=f"a_{i}", height=150)
        with col2:
            st.write(""); st.write("")
            if len(st.session_state.qa_list) > 1 and st.button("❌", key=f"del_{i}"):
                st.session_state.qa_list.pop(i); st.rerun()
    if st.button("➕ 문항 추가", use_container_width=True):
        st.session_state.qa_list.append({"question": "", "answer": ""}); st.rerun()

st.divider()

if st.button("🧐 전 방위 교차 분석 실행", type="primary", use_container_width=True):
    if not company or not job:
         st.warning("정보를 입력해주세요.")
    elif not is_model_trained():
         st.error("자체 모델이 학습되지 않았습니다. 사이드바에서 학습을 먼저 진행해 주세요.")
    else:
        with st.spinner("AI 전문가 그룹이 협업 분석 중입니다..."):
            data = {"company": company, "job": job, "description": description, "qa_list": st.session_state.qa_list}
            results = evaluator.evaluate_all_models(data)
            
            # --- 1. 통합 평가 대시보드 (Radar Charts) ---
            st.header("🏆 통합 평가 대시보드")
            radar_cols = st.columns(len(results))
            
            def create_radar(model_name, scores_dict, color='#6c5ce7'):
                labels = list(scores_dict.keys()); stats = list(scores_dict.values())
                angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
                stats += stats[:1]; angles += angles[:1]
                fig, ax = plt.subplots(figsize=(3,3), subplot_kw=dict(polar=True))
                ax.fill(angles, stats, color=color, alpha=0.15)
                ax.plot(angles, stats, color=color, linewidth=1.5, marker='o', markersize=3)
                ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=7)
                ax.set_title(model_name, size=10, weight='bold', pad=15)
                return fig

            for i, res in enumerate(results):
                with radar_cols[i]:
                    st.markdown(f"<div class='metric-container'><h4>{res['total_score']}점</h4><p style='font-size:0.75rem;'>{res['model']}</p></div>", unsafe_allow_html=True)
                    st.pyplot(create_radar(res['model'], res['scores']))

            st.divider()

            # --- 2. 모델별 전략적 비교 (Bar & Line Charts) ---
            st.subheader("⚖️ 모델별 전략적 분석 (Weights & Performance Simulation)")
            c1, c2 = st.columns(2)
            
            with c1:
                st.write("**[평가 가중치/강점 영역]**")
                emp_list = [{"Model": r["model"], "Criteria": k, "Weight": v} for r in results for k, v in r["emphasis"].items()]
                df_emp = pd.DataFrame(emp_list).pivot(index="Criteria", columns="Model", values="Weight")
                st.bar_chart(df_emp)
            
            with c2:
                st.write("**[운영 효율성 지표]**")
                spec_list = [{"Model": r["model"], "Criteria": k, "Value": v} for r in results for k, v in r["specs"].items()]
                df_spec = pd.DataFrame(spec_list).pivot(index="Criteria", columns="Model", values="Value")
                st.line_chart(df_spec)

            st.divider()

            # --- 3. 상세 분석 리포트 ---
            st.subheader("📑 모델별 상세 인사이트 리포트")
            grid_cols = st.columns(3)
            for i, res in enumerate(results):
                with grid_cols[i % 3]:
                    color = "#ff4757" if "SLM" in res['model'] else "#6c5ce7"
                    with st.container(border=True):
                        st.markdown(f"<h4 style='color: {color}; margin-top:0;'>📍 {res['model']} Analyst</h4>", unsafe_allow_html=True)
                        
                        # 피드백 내용 (토글 방식)
                        feedback = res['feedback']
                        if len(feedback) > 120:
                            summary = feedback[:120] + "..."
                            st.write(summary)
                            with st.expander("📝 전체 분석 의견 보기"):
                                st.write(feedback)
                        else:
                            st.write(feedback)
                            
                        st.divider()
                        st.caption(f"💰 효율: {res['specs']['비용']} | 🛠️ 전문성: {res['specs']['학습난이도']} | 🔒 보안: {res['specs']['보안']}")
            
            # --- 4. 요약 가이드 ---
            st.header("🏁 분석 결과 요약 가이드")
            with st.expander("💡 분석 결과 종합 가이드", expanded=True):
                st.info(f"""
                - **종합 의견**: 현재 자소서는 **{results[0]['model']}** 모델 기준 {results[0]['total_score']}점으로 평가되었습니다.
                - **점수 차이 발생 이유**: 
                    - **BERT**는 기존 데이터와의 유사성을 높게 평가하여 보수적인 점수를 주는 경향이 있습니다.
                    - **OpenAI**는 전체적인 논리 구조가 탄탄할 때 높은 점수를 부여합니다.
                    - **Qwen**은 직무 키워드 및 기술적 구체성이 드러날 때 가점을 줍니다.
                    - **Gemini**는 문장의 가독성과 창의적 표현력을 중시합니다.
                """)
            
            with st.expander("🧠 In-House SLM 엔진 심층 가이드", expanded=True):
                st.success(f"""
                - **자체 분석 의견**: 기업 전용 모델(SLM)은 이 글이 과거 합격자 데이터 패턴과 {results[-1]['total_score']}% 일치한다고 판단했습니다.
                - **MLOps 관리 상태**: 
                    - **Version Control**: 현재 **{REGISTERED_MODEL_NAME}** 모델이 공식 레지스트리에 저장되어 버전 관리 중입니다.
                    - **Deep Learning Detail**: GPT-2 베이스의 124M 파라미터를 미세 조정하여, 외부 AI가 놓칠 수 있는 사내 고유의 채용 기준을 분석합니다.
                """)
