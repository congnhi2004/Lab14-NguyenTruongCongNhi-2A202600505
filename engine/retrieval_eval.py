from typing import List, Dict

class RetrievalEvaluator:
    def __init__(self):
        pass

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:
        """
        Tính toán xem ít nhất 1 trong expected_ids có nằm trong top_k của retrieved_ids không.
        """
        if not expected_ids:
            return 0.0
        top_retrieved = retrieved_ids[:top_k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        Tính Mean Reciprocal Rank.
        Tìm vị trí đầu tiên của một expected_id trong retrieved_ids.
        MRR = 1 / position (vị trí 1-indexed). Nếu không thấy thì là 0.
        """
        if not expected_ids:
            return 0.0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    async def score(self, test_case: Dict, response: Dict) -> Dict:
        """
        Tính toán các chỉ số Retrieval (Hit Rate, MRR) cho một test case.
        """
        expected_ids = test_case.get("expected_retrieval_ids", [])
        retrieved_ids = response.get("metadata", {}).get("retrieved_ids", [])

        hit_rate = self.calculate_hit_rate(expected_ids, retrieved_ids)
        mrr = self.calculate_mrr(expected_ids, retrieved_ids)

        # Trả về format chuẩn mà BenchmarkRunner (main.py) yêu cầu
        return {
            "faithfulness": 1.0,  # Placeholder (Ragas metric thường cần LLM-Judge)
            "relevancy": 1.0,     # Placeholder
            "retrieval": {
                "hit_rate": hit_rate,
                "mrr": mrr
            }
        }

    async def evaluate_batch(self, dataset: List[Dict], retrieved_results: List[List[str]]) -> Dict:
        """
        Chạy eval cho toàn bộ bộ dữ liệu.
        Dataset cần có trường 'expected_retrieval_ids'.
        """
        total_hit_rate = 0.0
        total_mrr = 0.0
        n = len(dataset)
        
        if n == 0:
            return {"avg_hit_rate": 0.0, "avg_mrr": 0.0}

        for i in range(n):
            expected = dataset[i].get("expected_retrieval_ids", [])
            retrieved = retrieved_results[i]
            total_hit_rate += self.calculate_hit_rate(expected, retrieved)
            total_mrr += self.calculate_mrr(expected, retrieved)

        return {
            "avg_hit_rate": total_hit_rate / n,
            "avg_mrr": total_mrr / n
        }
