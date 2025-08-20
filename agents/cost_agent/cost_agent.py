# agents/cost_agent/cost_agent.py
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessageChunk
from typing import Any, Literal, AsyncIterable
import pandas as pd, os

memory = MemorySaver()

class ResponseFormat(BaseModel):
    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str

class CostAgent:
    SYSTEM_INSTRUCTION = 'You are a cost estimator assistant that tells users the cost of treatment for a disease.'
    RESPONSE_FORMAT_INSTRUCTION = 'Trả lời bằng tiếng Việt và chọn "completed".'
    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']

    def __init__(self, mcp_tools: list[Any]):
        self.model = ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_GENAI_MODEL", "gemini-pro"))
        self.mcp_tools = mcp_tools
        self.df = pd.read_csv('data/cost_data.csv')  # 📌 đọc từ file CSV

    def estimate_cost(self, query: str) -> str:
        for _, row in self.df.iterrows():
            if row['bệnh'] in query.lower():
                return f"Chi phí điều trị cho {row['bệnh']} là {row['chi_phí']} VNĐ."
        return "Xin lỗi, tôi không tìm thấy chi phí cho bệnh này."

    async def ainvoke(self, query: str, session_id: str) -> dict[str, Any]:
        message = self.estimate_cost(query)
        return {
            "is_task_complete": True,
            "require_user_input": False,
            "content": message,
            "status": "completed"
        }

    async def stream(self, query: str, session_id: str) -> AsyncIterable[Any]:
        yield {
            "is_task_complete": False,
            "require_user_input": False,
            "content": "Đang tìm kiếm chi phí điều trị..."
        }
        yield await self.ainvoke(query, session_id)
