import asyncio
import time
from typing import List, Dict


class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge):
        """
        agent     : Agent thực hiện RAG, phải có method async query(question) -> Dict
        evaluator : ExpertEvaluator, tính RAGAS metrics (faithfulness, hit_rate, mrr...)
        judge     : MultiModelJudge, chấm điểm câu trả lời theo thang 0–5
        """
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge

    async def run_single_test(self, test_case: Dict) -> Dict:
        start_time = time.perf_counter()

        # 1. Gọi Agent — phải trả về {"answer": str, "metadata": {"retrieved_ids": [...]}}
        response = await self.agent.query(test_case["question"])
        latency = time.perf_counter() - start_time

        # 2. Chạy Multi-Judge cho Generation stage
        judge_result = await self.judge.evaluate_multi_judge(
            test_case["question"],
            response["answer"],
            test_case["expected_answer"],
        )

        # 3. Chạy RAGAS / ExpertEvaluator cho Retrieval stage
        ragas_result = await self.evaluator.score(test_case, response)

        # 4. Thu thập retrieved_ids từ metadata của agent
        retrieved_ids = response.get("metadata", {}).get("retrieved_ids", [])

        return {
            "question": test_case["question"],
            "agent_response": response["answer"],
            "expected_answer": test_case["expected_answer"],
            "expected_retrieval_ids": test_case.get("expected_retrieval_ids", []),
            "retrieved_ids": retrieved_ids,
            "latency": latency,
            "ragas": ragas_result,       # {"faithfulness": ..., "relevancy": ..., "retrieval": {"hit_rate": ..., "mrr": ...}}
            "judge": judge_result,       # {"final_score": ..., "agreement_rate": ..., "reasoning": ...}
            "status": "fail" if judge_result["final_score"] < 3 else "pass",
        }

    async def run_all(self, dataset: List[Dict], batch_size: int = 2) -> List[Dict]:
        """
        Chạy song song bằng asyncio.gather với giới hạn batch_size để không bị Rate Limit.
        """
        results = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            tasks = [self.run_single_test(case) for case in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            await asyncio.sleep(1)  # Khoảng nghỉ nhỏ để tránh Rate Limit của LLM Judge
        return results
