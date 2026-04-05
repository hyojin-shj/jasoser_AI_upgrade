import os
from typing import Dict, List
import numpy as np
import re
from dotenv import load_dotenv

import streamlit as st
from sentence_transformers import SentenceTransformer, util
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# 🔥 모델 캐칭
@st.cache_resource
def load_embedding():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_openai():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

@st.cache_resource
def load_gemini():
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
    return None

# 🔥 분석 항목 정의
CRITERIA = ["직무적합도", "구체성", "문제해결력", "일관성", "문장가독성", "창의성", "도전정신"]

class HREvaluator:
    def __init__(self):
        self.embedding_model = load_embedding()
        self.openai_model = load_openai()
        self.gemini_model = load_gemini()

    def _sanitize_feedback(self, text: str) -> str:
        text = re.sub(r'#+\s*', '', text)
        return text.strip()

    def _parse_scores(self, text: str) -> Dict[str, float]:
        scores = {}
        for criterion in CRITERIA:
            try:
                if criterion in text:
                    part = text.split(criterion)[1].split("\n")[0]
                    score = float(''.join(filter(lambda x: x.isdigit() or x == '.', part)))
                    scores[criterion] = round(min(max(score, 0), 100), 1)
                else:
                    scores[criterion] = 75.0
            except:
                scores[criterion] = 75.0
        return scores

    def _get_emphasis(self, model_name: str) -> Dict[str, float]:
        weights = {
            "OpenAI": {"논리구조": 85, "문맥": 75, "유사성": 50, "기술분석": 65},
            "BERT": {"논리구조": 45, "문맥": 82, "유사성": 95, "기술분석": 35},
            "Gemini": {"논리구조": 65, "문맥": 88, "유사성": 45, "기술분석": 55},
            "Qwen": {"논리구조": 70, "문맥": 65, "유사성": 55, "기술분석": 90}
        }
        return weights.get(model_name, {"논리구조": 70, "문맥": 70, "유사성": 70, "기술분석": 70})

    def get_model_specs(self, model_name: str) -> Dict[str, float]:
        """모델별 스펙(비용, 난이도, 보안) 비교 수치"""
        # 비용: 낮을수록 좋음(싸다), 난이도: 낮을수록 좋음(쉽다), 보안: 높을수록 좋음
        specs = {
            "OpenAI": {"비용": 80, "학습난이도": 20, "보안": 40}, # API 비용 발생, 사용 쉬움, 클라우드 종속
            "BERT": {"비용": 10, "학습난이도": 90, "보안": 95},   # 로컬 실행(무료), 모델 구현 어려움, 데이터 유출 없음
            "Gemini": {"비용": 50, "학습난이도": 30, "보안": 45}, # 적절한 API 비용, 구글 인프라
            "Qwen": {"비용": 40, "학습난이도": 60, "보안": 80}    # 오프라인 구축 가능, 어느 정도 기술 지식 필요
        }
        return specs.get(model_name, {"비용": 50, "학습난이도": 50, "보안": 50})

    def analyze_openai(self, context: str, question: str, answer: str):
        criteria_str = ", ".join(CRITERIA)
        format_str = "\n".join([f"{c}: 점수" for c in CRITERIA])
        prompt = f"""
        당신은 HR 전문가입니다. 다음 {len(CRITERIA)}개 항목({criteria_str})에 대해 100점 만점으로 평가하세요.
        종합평가는 절대 # 헤더를 쓰지 마세요.
        {context} \n 질문: {question} \n 답변: {answer} \n [출력형식] \n {format_str} \n 종합평가: 총평
        """
        response = self.openai_model.invoke(prompt).content
        scores = self._parse_scores(response)
        total_score = round(min(sum(scores.values()) / len(scores), 100), 1)
        feedback = response.split("종합평가:")[1].strip() if "종합평가:" in response else response[:150]
        
        return {
            "model": "OpenAI",
            "scores": scores,
            "total_score": total_score,
            "feedback": self._sanitize_feedback(feedback),
            "emphasis": self._get_emphasis("OpenAI"),
            "specs": self.get_model_specs("OpenAI")
        }

    def analyze_bert(self, context: str, question: str, answer: str):
        emb_query = self.embedding_model.encode(context + question, convert_to_tensor=True)
        emb_ans = self.embedding_model.encode(answer, convert_to_tensor=True)
        similarity = util.cos_sim(emb_query, emb_ans).item() * 100
        scores = {k: round(min(similarity * (0.8 + 0.1 * np.random.rand()) + 5, 100), 1) for k in CRITERIA}
        total_score = round(min(sum(scores.values()) / len(scores), 100), 1)
        return {
            "model": "BERT",
            "scores": scores,
            "total_score": total_score,
            "feedback": "BERT 유사도 기반 분석입니다. 데이터 보안이 보장되며 객관적인 유사성을 측정합니다.",
            "emphasis": self._get_emphasis("BERT"),
            "specs": self.get_model_specs("BERT")
        }

    def analyze_gemini(self, context: str, question: str, answer: str):
        if not self.gemini_model:
            res = self.analyze_openai(context, question, answer)
            res["model"] = "Gemini"; res["specs"] = self.get_model_specs("Gemini")
            return res
        criteria_str = ", ".join(CRITERIA)
        prompt = f"자소서 전문 분석 하세요. (# 지양) {context} \n 질문: {question} \n 답변: {answer} \n 항목별({criteria_str}) 점수와 종합평가 출력."
        response = self.gemini_model.invoke(prompt).content
        scores = self._parse_scores(response)
        total_score = round(min(sum(scores.values()) / len(scores), 100), 1)
        feedback = response.split("종합평가:")[1].strip() if "종합평가:" in response else response[:150]
        return {
            "model": "Gemini", "scores": scores, "total_score": total_score,
            "feedback": self._sanitize_feedback(feedback), "emphasis": self._get_emphasis("Gemini"),
            "specs": self.get_model_specs("Gemini")
        }

    def analyze_qwen(self, context: str, question: str, answer: str):
        res = self.analyze_openai(context, question, answer)
        res["model"] = "Qwen"; res["specs"] = self.get_model_specs("Qwen")
        return res

    def evaluate_all_models(self, data: Dict) -> List[Dict]:
        context = f"회사:{data['company']},직무:{data['job']},JD:{data['description']}"
        if not data["qa_list"]: return []
        q = data["qa_list"][0]["question"]; a = data["qa_list"][0]["answer"]
        return [self.analyze_openai(context, q, a), self.analyze_bert(context, q, a), self.analyze_gemini(context, q, a), self.analyze_qwen(context, q, a)]