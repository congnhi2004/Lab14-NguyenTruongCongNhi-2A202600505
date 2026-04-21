import asyncio
from typing import List, Dict

class SupportAgentV1:
    """
    Agent phiên bản 1: Basic RAG simulation.
    Chỉ tìm kiếm đơn giản và trả lời.
    """
    def __init__(self):
        self.name = "Agent-V1-Base"

    async def query(self, question: str) -> Dict:
        # Giả lập độ trễ
        await asyncio.sleep(0.2) 
        
        # Giả lập retrieval (V1 đôi khi lấy sai ID)
        retrieved_ids = ["doc_evaluation_001"] if "AI" in question else ["unknown_doc"]
        
        return {
            "answer": f"Dựa trên tài liệu hệ thống, tôi xin trả lời câu hỏi '{question}' theo cách cơ bản nhất.",
            "metadata": {
                "model": "gpt-4o-mini",
                "retrieved_ids": retrieved_ids
            }
        }
