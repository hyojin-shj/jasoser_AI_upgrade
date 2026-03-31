import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from engine import HREvaluator

st.set_page_config(layout="wide")

st.title("🔥 AI 자소서 평가 시스템 (Pro)")

evaluator = HREvaluator()

# 🔹 기본 정보
company = st.text_input("회사명")
job = st.text_input("지원 직무")

description = st.text_area("직무 설명")
preferences = st.text_area("우대 사항")

st.divider()

# 🔹 상태 유지
if "qa_list" not in st.session_state:
    st.session_state.qa_list = [{"question": "", "answer": ""}]

def add_qa():
    st.session_state.qa_list.append({"question": "", "answer": ""})

def remove_qa(index):
    st.session_state.qa_list.pop(index)

# 🔹 버튼
col1, col2 = st.columns(2)
with col1:
    st.button("➕ 문항 추가", on_click=add_qa)
with col2:
    if len(st.session_state.qa_list) > 1:
        st.button("➖ 마지막 삭제", on_click=lambda: remove_qa(-1))

# 🔹 입력 UI
for i, qa in enumerate(st.session_state.qa_list):
    st.subheader(f"문항 {i+1}")

    qa["question"] = st.text_area(f"질문 {i+1}", key=f"q{i}")
    qa["answer"] = st.text_area(f"답변 {i+1}", key=f"a{i}")

    st.divider()

# 🔹 평가
if st.button("🚀 평가하기"):

    data = {
        "company": company,
        "job": job,
        "description": description,
        "preferences": preferences,
        "qa_list": st.session_state.qa_list
    }

    result = evaluator.evaluate_all(data)

    st.subheader("📊 전체 결과")
    st.metric("평균 점수", result["average"])

    if len(result["details"]) == 0:
        st.warning("입력된 문항이 없습니다.")
        st.stop()

    df = pd.DataFrame(result["details"])

    # 🔥 모델별 비교
    st.subheader("📈 모델별 점수 비교")
    st.bar_chart(df[["bert", "keyword", "gpt", "final"]])

    # 🔥 레이더 차트
    st.subheader("🎯 평가 구조")

    labels = ["BERT", "Keyword", "GPT", "Final"]
    values = df.iloc[0][["bert", "keyword", "gpt", "final"]].tolist()

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)

    angles = [n / float(len(labels)) * 2 * 3.14159 for n in range(len(labels))]
    values += values[:1]
    angles += angles[:1]

    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.3)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    st.pyplot(fig)

    # 🔥 문항별 상세
    for i, row in df.iterrows():
        st.subheader(f"문항 {i+1} 상세")

        st.write(f"✅ BERT: {row['bert']}")
        st.write(f"✅ Keyword: {row['keyword']}")
        st.write(f"✅ GPT: {row['gpt']}")
        st.write(f"🔥 Final: {row['final']}")

        st.write("📌 피드백")
        st.write(row["feedback"])