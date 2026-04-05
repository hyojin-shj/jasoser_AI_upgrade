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
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .spec-box {
        background-color: #e3f2fd;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        font-size: 0.85rem;
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
    if st.button("🔄 설정 초기화"):
        st.session_state.clear(); st.rerun()

# --- 메인 영역 ---
st.title("🎯 AI 자소서 멀티 모델 비교 분석")

with st.container(border=True):
    st.subheader("📝 자기소개서 입력")
    for i, qa in enumerate(st.session_state.qa_list):
        col1, col2 = st.columns([0.9, 0.1])
        with col1:
             qa["question"] = st.text_input(f"질문 {i+1}", value=qa["question"], key=f"q_{i}")
             qa["answer"] = st.text_area(f"답변 {i+1}", value=qa["answer"], key=f"a_{i}", height=150)
        with col2:
            st.write(""); st.write("")
            if len(st.session_state.qa_list) > 1:
                if st.button("❌", key=f"del_{i}"):
                    st.session_state.qa_list.pop(i); st.rerun()
    if st.button("➕ 문항 추가", use_container_width=True):
        st.session_state.qa_list.append({"question": "", "answer": ""}); st.rerun()

st.divider()

if st.button("🚀 멀티 모델 정밀 분석 시작", type="primary", use_container_width=True):
    if not company or not job:
        st.warning("정보를 입력해주세요.")
    else:
        with st.spinner("AI 엔진이 다각도로 분석 중입니다..."):
            data = {"company": company, "job": job, "preferences": preferences, "description": description, "qa_list": st.session_state.qa_list}
            results = evaluator.evaluate_all_models(data)
            
            # --- 1. 대시보드 상단 ---
            st.header("🏆 수치 기반 통합 리포트")
            radar_cols = st.columns(4)
            
            def create_radar(model_name, scores_dict, color='#6c5ce7'):
                labels = list(scores_dict.keys())
                stats = list(scores_dict.values())
                angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
                stats += stats[:1]; angles += angles[:1]
                fig, ax = plt.subplots(figsize=(4,4), subplot_kw=dict(polar=True))
                ax.fill(angles, stats, color=color, alpha=0.2)
                ax.plot(angles, stats, color=color, linewidth=2, marker='o')
                ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=8)
                ax.set_title(model_name, size=14, weight='bold', pad=20)
                return fig

            for i, res in enumerate(results):
                with radar_cols[i]:
                    st.markdown(f"<div class='metric-container'><h3>{res['total_score']}점</h3><p>{res['model']}</p></div>", unsafe_allow_html=True)
                    st.pyplot(create_radar(res['model'], res['scores']))

            st.divider()

            # --- 2. 가중치 및 스펙 비교 ---
            st.subheader("⚖️ 모델별 전략적 비교 (Weights & Specs)")
            
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.write("**[평가 가중치 비교]** 모델별 중점 항목")
                emp_list = [{"Model": r["model"], "Criteria": k, "Weight": v} for r in results for k, v in r["emphasis"].items()]
                df_emp = pd.DataFrame(emp_list).pivot(index="Criteria", columns="Model", values="Weight")
                st.bar_chart(df_pivot if "df_pivot" in locals() else df_emp)

            with col_right:
                st.write("**[운영 효율성 시뮬레이션]** 비용, 난이도, 보안")
                spec_list = [{"Model": r["model"], "Criteria": k, "Value": v} for r in results for k, v in r["specs"].items()]
                df_spec = pd.DataFrame(spec_list).pivot(index="Criteria", columns="Model", values="Value")
                st.line_chart(df_spec) # 선형 차트로 효율성 비교

            st.divider()

            # --- 3. 상세 리포트 ---
            st.subheader("📑 모델별 상세 분석 리포트")
            grid_cols = st.columns(2)
            for i, res in enumerate(results):
                with grid_cols[i % 2]:
                    st.markdown(f"""
                    <div class='report-card'>
                        <h4 style='color: #6c5ce7;'>📍 {res['model']} Analyst</h4>
                        <p style='font-size: 0.92rem;'>{res['feedback']}</p>
                        <hr>
                        <div style='display: flex; justify-content: space-between;'>
                            <span>💰 비용: {res['specs']['비용']}</span>
                            <span>🛠️ 난이도: {res['specs']['학습난이도']}</span>
                            <span>🔒 보안: {res['specs']['보안']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # --- 4. 요약 ---
            with st.expander("💡 분석 결과 종합 가이드", expanded=True):
                st.info(f"""
                - **종합 의견**: 현재 자소서는 **{results[0]['model']}** 모델 기준 {results[0]['total_score']}점으로 평가되었습니다.
                - **점수 차이 발생 이유**: 
                    - **BERT**는 기존 데이터와의 유사성을 높게 평가하여 보수적인 점수를 주는 경향이 있습니다.
                    - **OpenAI**는 전체적인 논리 구조가 탄탄할 때 높은 점수를 부여합니다.
                    - **Qwen**은 직무 키워드 및 기술적 구체성이 드러날 때 가점을 줍니다.
                    - **Gemini**는 문장의 가독성과 창의적 표현력을 중시합니다.
                """)
