import os
from typing import Dict
from dotenv import load_dotenv

import streamlit as st
from sentence_transformers import SentenceTransformer, util
from langchain_openai import ChatOpenAI

load_dotenv()


# 🔥 모델 캐싱 (속도 핵심)
@st.cache_resource
def load_embedding():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3
    )


# 🔥 GPT 캐싱 (클래스 밖으로 빼야 함)
@st.cache_data(show_spinner=False)
def gpt_eval_cached(question, answer, context):
    llm = load_llm()

    prompt = f"""
    당신은 HR 전문가입니다.

    질문: {question}
    답변: {answer}
    직무 정보: {context}

    아래 형식으로 평가:
    점수: (숫자만)
    이유:
    """

    result = llm.invoke(prompt).content

    try:
        score = int(result.split("점수:")[1].split("\n")[0].strip())
    except:
        score = 70

    return score, result


class HREvaluator:
    def __init__(self):
        self.embedding_model = load_embedding()

    # 🔹 BERT 유사도
    def bert_similarity(self, text1, text2):
        emb1 = self.embedding_model.encode(text1, convert_to_tensor=True)
        emb2 = self.embedding_model.encode(text2, convert_to_tensor=True)
        return util.cos_sim(emb1, emb2).item() * 100

    # 🔹 키워드 점수
    def keyword_score(self, context, answer):
        keywords = context.split()
        if len(keywords) == 0:
            return 0
        hit = sum(1 for k in keywords if k in answer)
        return min((hit / len(keywords)) * 100, 100)

    # 🔹 문항 평가
    def evaluate_qa(self, question, answer, context):
        bert = self.bert_similarity(question + context, answer)
        keyword = self.keyword_score(context, answer)
        gpt_score, gpt_text = gpt_eval_cached(question, answer, context)

        final = 0.5 * gpt_score + 0.3 * bert + 0.2 * keyword

        return {
            "bert": round(bert, 2),
            "keyword": round(keyword, 2),
            "gpt": gpt_score,
            "final": round(final, 2),
            "feedback": gpt_text
        }

    # 🔹 전체 평가
    def evaluate_all(self, data: Dict):
        context = f"""
        회사: {data['company']}
        직무: {data['job']}
        직무 설명: {data['description']}
        우대사항: {data['preferences']}
        """

        results = []

        for qa in data["qa_list"]:
            if qa["question"] and qa["answer"]:
                res = self.evaluate_qa(
                    qa["question"],
                    qa["answer"],
                    context
                )
                results.append(res)

        if len(results) == 0:
            return {"average": 0, "details": []}

        avg = sum(r["final"] for r in results) / len(results)

        return {
            "average": round(avg, 2),
            "details": results
        }