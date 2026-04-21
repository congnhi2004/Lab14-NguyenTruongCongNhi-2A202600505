import json
from typing import Dict

from dotenv import find_dotenv, load_dotenv

# Load .env vào os.environ TRƯỚC khi LangChain đọc API key
load_dotenv(find_dotenv())

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from agent.tools import create_retrieval_tool

# System prompt hướng dẫn agent suy nghĩ + gọi tool nhiều lần nếu cần
SYSTEM_PROMPT = """Bạn là một AI hỗ trợ thông minh. Nhiệm vụ của bạn là trả lời câu hỏi của người dùng \
dựa trên tài liệu được lưu trong knowledge base.

Quy trình bắt buộc:
1. PHÂN TÍCH câu hỏi để xác định từ khoá, chủ đề chính, và các góc nhìn liên quan.
2. TÌM KIẾM bằng tool `retrieve_documents`. Đừng chỉ truyền nguyên văn câu hỏi — hãy tối ưu query:
   - Rút gọn thành từ khoá chủ chốt.
   - Thử nhiều cách diễn đạt khác nhau (tiếng Việt, tiếng Anh, v.v.).
   - Nếu lần đầu không đủ thông tin, gọi lại với query khác.
3. TỔNG HỢP kết quả tìm được để đưa ra câu trả lời rõ ràng, chính xác.
4. Chỉ trả lời dựa trên nội dung tìm được. Nếu không tìm thấy thông tin liên quan sau nhiều lần thử, \
hãy thành thật nói không có đủ dữ liệu.
"""


class LangGraphAgent:
    def __init__(self, model_name: str = "gpt-4o-mini", collection_name: str = "lab14"):
        self.name = "LangGraph-ReAct-Agent"
        self.model_name = model_name
        self.collection_name = collection_name

        self.llm = ChatOpenAI(model=model_name, temperature=0.0)
        self.retrieval_tool = create_retrieval_tool(collection_name=collection_name)
        self.tools = [self.retrieval_tool]

        # create_react_agent hỗ trợ prompt để inject system message
        self.agent_executor = create_react_agent(
            self.llm,
            self.tools,
            prompt=SYSTEM_PROMPT,
        )

    async def query(self, question: str) -> Dict:
        """
        Thực thi ReAct loop:
        - Agent suy nghĩ → gọi retrieve_documents (có thể nhiều lần) → tổng hợp câu trả lời.
        Trả về dict chuẩn cho BenchmarkRunner:
          {"answer": str, "metadata": {"model": str, "retrieved_ids": list[str]}}
        """
        messages = [HumanMessage(content=question)]
        response = await self.agent_executor.ainvoke({"messages": messages})

        # Câu trả lời cuối cùng nằm ở message cuối (luôn là AIMessage)
        final_answer = response["messages"][-1].content

        # Thu thập tất cả doc_id được tool trả về qua các lần gọi
        retrieved_ids: set[str] = set()
        for msg in response["messages"]:
            if isinstance(msg, ToolMessage) and msg.name == "retrieve_documents":
                try:
                    tool_output = json.loads(msg.content)
                    # tool_output là list[dict] hoặc dict (nếu empty)
                    if isinstance(tool_output, list):
                        for item in tool_output:
                            doc_id = item.get("doc_id", "unknown")
                            if doc_id != "unknown":
                                retrieved_ids.add(doc_id)
                except (json.JSONDecodeError, AttributeError):
                    pass

        return {
            "answer": final_answer,
            "metadata": {
                "model": self.model_name,
                "retrieved_ids": list(retrieved_ids),
            },
        }
