# agents/booking_agent/booking_agent.py
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessageChunk
from typing import Any, Literal, AsyncIterable
import os

memory = MemorySaver()

class ResponseFormat(BaseModel):
    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str

class BookingAgent:
    SYSTEM_INSTRUCTION = 'You are a booking assistant that helps patients book medical appointments.'
    RESPONSE_FORMAT_INSTRUCTION = 'Respond in Vietnamese with status "completed".'
    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']

    def __init__(self, mcp_tools: list[Any]):
        model_name = os.getenv("GOOGLE_GENAI_MODEL", "gemini-pro")
        self.model = ChatGoogleGenerativeAI(model=model_name)
        self.mcp_tools = mcp_tools

    async def ainvoke(self, query: str, session_id: str) -> dict[str, Any]:
        booking_agent = create_react_agent(
            self.model,
            tools=self.mcp_tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=(self.RESPONSE_FORMAT_INSTRUCTION, ResponseFormat),
        )
        config = {"configurable": {"thread_id": session_id}}
        langgraph_input = {"messages": [("user", query)]}
        await booking_agent.ainvoke(langgraph_input, config)
        return self._get_response(config, booking_agent)

    def _get_response(self, config, agent) -> dict:
        state = agent.get_state(config)
        result = state.values.get('structured_response')
        return {
            "is_task_complete": True,
            "require_user_input": False,
            "content": result.message if result else "Không có phản hồi",
            "status": result.status if result else "error"
        }

    async def stream(self, query: str, session_id: str) -> AsyncIterable[Any]:
        booking_agent = create_react_agent(
            self.model,
            tools=self.mcp_tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=(self.RESPONSE_FORMAT_INSTRUCTION, ResponseFormat),
        )
        config = {"configurable": {"thread_id": session_id}}
        langgraph_input = {"messages": [("user", query)]}

        async for chunk in booking_agent.astream_events(langgraph_input, config, version="v1"):
            content = None
            if chunk.get("event") == "on_chat_model_stream":
                chunk_data = chunk["data"].get("chunk")
                if isinstance(chunk_data, AIMessageChunk) and chunk_data.content:
                    content = chunk_data.content
            if content:
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": content,
                }
        yield self._get_response(config, booking_agent)
