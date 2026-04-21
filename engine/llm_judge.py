import asyncio
import os
import json
from typing import Dict, Any, List
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

class LLMJudge:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=60.0
        )
        self.models = ["gpt-4o-mini", "gpt-4o"]
        
        self.rubrics = {
            "accuracy": "Chấm điểm từ 1-5 dựa trên độ chính xác so với Ground Truth. 5: Hoàn hảo, 1: Hoàn toàn sai.",
            "tone": "Chấm điểm từ 1-5 dựa trên sự chuyên nghiệp. 5: Rất chuyên nghiệp, 1: Thiếu tôn trọng hoặc quá suồng sã."
        }

    async def _get_score_from_model(self, model: str, question: str, answer: str, ground_truth: str) -> int:
        prompt = f"""
        Bạn là một giám khảo công tâm chấm điểm câu trả lời của AI.
        
        Câu hỏi: {question}
        Câu trả lời của AI: {answer}
        Câu trả lời gốc (Ground Truth): {ground_truth}
        
        Tiêu chí: {self.rubrics['accuracy']}
        
        Yêu cầu: Trả về duy nhất một con số từ 1 đến 5 đại diện cho điểm Accuracy.
        """
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            score_text = response.choices[0].message.content.strip()
            # Extract the first digit found
            for char in score_text:
                if char.isdigit():
                    return int(char)
            return 3 # Default if no digit found
        except Exception as e:
            print(f"Error calling {model}: {e}")
            return 3

    async def evaluate_multi_judge(self, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        """
        Gọi ít nhất 2 model.
        Tính toán sự sai lệch và Agreement Rate.
        """
        tasks = [self._get_score_from_model(model, question, answer, ground_truth) for model in self.models]
        scores = await asyncio.gather(*tasks)
        
        model_scores = dict(zip(self.models, scores))
        avg_score = sum(scores) / len(scores)
        
        # Agreement Rate: Nếu lệch <= 1 điểm thì coi như đồng thuận (mức độ cao)
        # Hoặc tính đơn giản: số cặp giống nhau / tổng số cặp
        agreement = 1.0 if scores[0] == scores[1] else 0.5
        if abs(scores[0] - scores[1]) > 1:
            agreement = 0.0 # Bất đồng quan điểm lớn
            
        return {
            "final_score": avg_score,
            "agreement_rate": agreement,
            "individual_scores": model_scores,
            "reasoning": f"Model scores: {model_scores}. Avg: {avg_score}"
        }

    async def check_position_bias(self, response_a: str, response_b: str):
        """
        Nâng cao (Optional): Thực hiện đổi chỗ response A và B.
        """
        pass
