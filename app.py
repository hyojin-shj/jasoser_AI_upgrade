import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from engine import HREvaluator, CRITERIA

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(layout="wide", page_title="Premium AI 자소서 분석기", page_icon="🔥")

# --- 커스텀 CSS ---
st.markdown("""
<style>
    .report-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #6c5ce7;
        height: 250px;
        overflow-y: auto;
        margin-bottom: 20px;
    }
    .metric-container {
        text-align: center;
        background: linear-gradient(135deg, #6c5ce7, #a29bfe);
        color: white;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

if "qa_list" not in st.session_state:
    st.session_state.qa_list = [{"question": "", "answer": ""}]

evaluator = HREvaluator()

# --- 사이드바 ---
with st.sidebar:
    st.title("📂 분석 설정")
    company = st.text_input("회사명", placeholder="예: 삼성전자")
    job = st.text_input("지원 직무", placeholder="예: 경영지원")
    preferences = st.text_area("우대 사항")
    description = st.text_area("직무 설명(JD)")
    st.divider()
    st.button("🔄 설정 초기화", on_click=lambda: st.session_state.clear())

# --- 메인 영역 ---
st.title("🎯 AI 자소서 멀티 모델 비교 분석")

# 입력 섹션
with st.container(border=True):
    st.subheader("📝 자기소개서 입력")
    for i, qa in enumerate(st.session_state.qa_list):
        col1, col2 = st.columns([0.9, 0.1])
        with col1:
             qa["question"] = st.text_input(f"질문 {i+1}", value=qa["question"], key=f"q_{i}", placeholder="질문을 입력하세요.")
             qa["answer"] = st.text_area(f"답변 {i+1}", value=qa["answer"], key=f"a_{i}", height=150, placeholder="답변을 입력하세요.")
        with col2:
            st.write("") # 간격
            st.write("")
            if len(st.session_state.qa_list) > 1:
                if st.button("❌", key=f"del_{i}"):
                    st.session_state.qa_list.pop(i); st.rerun()
    
    if st.button("➕ 문항 추가", use_container_width=True):
        st.session_state.qa_list.append({"question": "", "answer": ""}); st.rerun()

st.divider()

# 분석 버튼
if st.button("🚀 멀티 모델 정밀 분석 시작", type="primary", use_container_width=True):
    if not company or not job:
        st.warning("회사명과 직무 정보를 입력해주세요.")
    else:
        with st.spinner("4개의 AI 모델이 협력하여 분석 중입니다..."):
            data = {"company": company, "job": job, "preferences": preferences, "description": description, "qa_list": st.session_state.qa_list}
            results = evaluator.evaluate_all_models(data)
            
            # --- 1. 대시보드 상단: 총점 및 스타차트 ---
            st.header("🏆 수치 기반 통합 리포트")
            
            radar_cols = st.columns(4)
            def create_radar_chart(model_name, scores_dict):
                labels = CRITERIA
                stats = [scores_dict[l] for l in labels]
                angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
                stats += stats[:1]; angles += angles[:1]
                
                fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
                ax.fill(angles, stats, color='#6c5ce7', alpha=0.2)
                ax.plot(angles, stats, color='#6c5ce7', linewidth=2, marker='o', markersize=4)
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(labels, fontsize=9)
                ax.set_title(model_name, size=14, pad=20, weight='bold')
                return fig

            for i, res in enumerate(results):
                with radar_cols[i]:
                    st.markdown(f"""<div class='metric-container'><h3>{res['total_score']}점</h3><p>{res['model']}</p></div>""", unsafe_allow_html=True)
                    st.pyplot(create_radar_chart(res['model'], res['scores']))

            st.divider()

            # --- 2. 모델별 분석 가중치 비교 (통합 시각화) ---
            st.subheader("⚖️ 모델별 평가 가중치 비교 (Emphasis Analysis)")
            st.write("각 모델이 어떤 기준에 가중치를 두어 점수를 부여했는지 비교합니다.")
            
            emp_list = []
            for res in results:
                for k, v in res["emphasis"].items():
                    emp_list.append({"Model": res["model"], "Criteria": k, "Weight": v})
            
            df_emp = pd.DataFrame(emp_list)
            
            # 피벗하여 그룹형 막대 그래프 생성
            df_pivot = df_emp.pivot(index="Criteria", columns="Model", values="Weight")
            st.bar_chart(df_pivot)

            st.divider()

            # --- 3. 모델별 상세 피드백 (균형 잡힌 레이아웃) ---
            st.subheader("📑 모델별 상세 분석 리포트")
            
            # 2x2 그리드로 배치
            grid_cols = st.columns(2)
            for i, res in enumerate(results):
                with grid_cols[i % 2]:
                    st.markdown(f"""
                    <div class='report-card'>
                        <h4 style='color: #6c5ce7;'>📍 {res['model']} Analyst</h4>
                        <p style='font-size: 0.95rem; line-height: 1.6;'>{res['feedback']}</p>
                        <hr>
                        <small><b>핵심 강조:</b> {list(res['emphasis'].keys())[0]}</small>
                    </div>
                    """, unsafe_allow_html=True)

            # --- 4. 최종 인사이트 ---
            with st.expander("💡 분석 결과 종합 가이드 (Total Insight)", expanded=True):
                st.info(f"""
                - **종합 의견**: 현재 자소서는 **{results[0]['model']}** 모델 기준 {results[0]['total_score']}점으로 평가되었습니다.
                - **점수 차이 발생 이유**: 
                    - **BERT**는 기존 데이터와의 유사성을 높게 평가하여 보수적인 점수를 주는 경향이 있습니다.
                    - **OpenAI**는 전체적인 논리 구조가 탄탄할 때 높은 점수를 부여합니다.
                    - **Qwen**은 직무 키워드 및 기술적 구체성이 드러날 때 가점을 줍니다.
                    - **Gemini**는 문장의 가독성과 창의적 표현력을 중시합니다.
                """)
