import asyncio
from typing import List, Dict

class SupportAgentV2:
    """
    Agent phiên bản 2: Optimized RAG simulation.
    Cải tiến logic retrieval và format câu trả lời chuyên nghiệp hơn.
    """
    def __init__(self):
        self.name = "Agent-V2-Optimized"

    async def query(self, question: str) -> Dict:
        # Giả lập độ trễ (V2 nhanh hơn hoặc xử lý tốt hơn)
        await asyncio.sleep(0.3) 
        
        # Giả lập retrieval (V2 luôn lấy đúng ID nếu từ khóa phù hợp)
        retrieved_ids = ["doc_evaluation_001"]
        
        return {
            "answer": f"Chào bạn! Dựa trên phân tích kỹ lưỡng từ tài liệu hệ thống về AI Evaluation, tôi xin trả lời: [Sự cải tiến về nội dung cho câu hỏi '{question}']. Hy vọng thông tin này hữu ích!",
            "metadata": {
                "model": "gpt-4o",
                "retrieved_ids": retrieved_ids
            }
        }
