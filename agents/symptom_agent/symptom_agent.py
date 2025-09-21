# Import các thư viện cần thiết

import os
from dotenv import load_dotenv


# Load .env trước khi đọc biến môi trường
load_dotenv(override=True)


# Debug
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    import sys
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(h)

logger.debug("GOOGLE_API_KEY present: %s", bool(os.getenv("GOOGLE_API_KEY")))
logger.debug("GOOGLE_GENAI_USE_VERTEXAI: %s", os.getenv("GOOGLE_GENAI_USE_VERTEXAI"))

from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, field_validator

from agents.symptom_agent.utils import safe_parse_response

from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain_nvidia_ai_endpoints import ChatNVIDIA


#from langchain_ollama import ChatOllama
from typing import Literal, Any, AsyncIterable

from langchain_core.messages import AIMessage

# Import tool tìm triệu chứng từ CSV
from tools.symptoms_tool import search_symptoms
from langchain_mcp_adapters.client import MultiServerMCPClient
import json

# Khởi tạo memory để lưu state của agent
memory = MemorySaver()

# Định nghĩa format phản hồi chuẩn
class ResponseFormat(BaseModel):
    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str
    data: dict | None = None  # thêm field data để đồng bộ với tool

    @field_validator("data", mode="before")
    def parse_data(cls, v: Any):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                return None
        return v

# Định nghĩa Symptom Agent
class SymptomAgent:
    # Quan trọng phải build SYSTEM_INSTRUCTION thật chuẩn

    SYSTEM_INSTRUCTION = (
    """
    Bạn là trợ lý y tế. 
    ⚠️ Quan trọng: Luôn trả về duy nhất một JSON hợp lệ theo schema sau (không thêm văn bản ngoài JSON):

    {
    "status": "completed",
    "message": "<câu trả lời>",
    "data": {
        "diseases": [
        {"name": "<tên bệnh>", "note": "<mô tả ngắn gọn>"}
        ],
        "explanation": "<giải thích>",
        "advice": "<lời khuyên>"
    }
    }

    RÀNG BUỘC:
    - "data" PHẢI là object JSON, KHÔNG được bọc trong chuỗi.
    - KHÔNG in thêm giải thích, chữ thừa, markdown hoặc ```json.
    - Nếu không có bệnh, trả lời:
    {
        "status": "completed",
        "message": "Hiện chưa xác định được bệnh. Bạn nên đi khám bác sĩ để kiểm tra kỹ hơn.",
        "data": {
        "diseases": [],
        "explanation": "",
        "advice": "Bạn nên đi khám bác sĩ để được tư vấn chi tiết."
        }
    }
    """
    )

    RESPONSE_FORMAT_INSTRUCTION = 'Select status as "completed" and write the answer in Vietnamese.'
    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']

    def __init__(self, mcp_tools: list[Any]):
        # Debug: thông báo đăng ký tool
        #print("[DEBUG] Đang khởi tạo SymptomAgent...")
        
        # Đọc model từ biến môi trường, ví dụ GOOGLE_GENAI_MODEL = "LLM gì đó cái thứ nhất" hoặc "LLM khác gì đó cái thứ 2 đọc gọi nếu cái thứ 1 không tồn tại"
        model_name = os.getenv("GOOGLE_GENAI_MODEL", "gemini-pro")
        self.model = ChatGoogleGenerativeAI(model=model_name)
        
        
        # Bọc tool CSV để log khi chạy
        ''''
        def debug_tool_wrapper(query: str) -> str:
            print(f"[DEBUG] Tool search_symptoms_csv được gọi với query: '{query}'")
            result = search_symptoms(query)
            print(f"[DEBUG] Kết quả từ CSV: '{result}'")
            return result

        debug_tool_wrapper.name = "search_symptoms_csv"
        debug_tool_wrapper.description = "Tìm thông tin triệu chứng từ file CSV"
        '''
        # Đăng ký tool search_symptoms vào danh sách tool
        self.mcp_tools = mcp_tools + [search_symptoms]

    # Phương thức ainvoke(): khởi chạy Agent và nhận kết quả không-stream
    # Trong ainvoke():
    # Phương thức ainvoke(): khởi chạy Agent và nhận kết quả không-stream
    async def ainvoke(self, query: str, session_id: str) -> dict[str, Any]:
        model_name = os.getenv("GOOGLE_GENAI_MODEL", "gemini-pro")
        logger.debug(f"[ainvoke] Dùng model: {model_name}, session_id={session_id}, query={query[:200]}")

        model = ChatGoogleGenerativeAI(model=model_name)

        runnable = create_react_agent(
            model,
            tools=self.mcp_tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=(self.RESPONSE_FORMAT_INSTRUCTION, ResponseFormat),
        )

        config = {'configurable': {'thread_id': session_id}}
        langgraph_input = {'messages': [('user', query)]}
        logger.debug(f"[ainvoke] LangGraph input: {langgraph_input}")

        try:
            result = await runnable.ainvoke(langgraph_input, config)
            logger.debug(f"[ainvoke] Kết quả từ runnable.ainvoke: {result}")
            
            response = self._get_agent_response_from_state(config, runnable, result)
            logger.debug(f"[ainvoke] Response sau khi parse state: {response}")

            # ✅ đảm bảo luôn có final response hợp lệ
            if not response:
                logger.warning("[ainvoke] Không lấy được response từ agent, trả về error mặc định")
                response = {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': "Không thể lấy kết quả từ agent.",
                    'status': 'error',
                    'data': None,
                }
            else:
                response["is_task_complete"] = True  # ép đánh dấu hoàn tất

            logger.debug(f"[ainvoke] Final response: {response}")
            return response

        except Exception as e:
            logger.exception("[ainvoke] Lỗi khi chạy runnable.ainvoke")
            return {
                'is_task_complete': True,
                'require_user_input': False,
                'content': f"Lỗi khi xử lý: {str(e)}",
                'status': 'error',
                'data': None,
            }

    # Hàm _get_agent_response_from_state(): Trích xuất kết quả cuối từ Agent
    def _get_agent_response_from_state(self, config, runnable, result=None) -> dict:
        logger.debug(f"[_get_agent_response_from_state] result input: {result}")
        current_state = runnable.get_state(config)
        logger.debug(f"[_get_agent_response_from_state] current_state: {current_state.values}")

        structured_response = current_state.values.get("structured_response")
        logger.debug("[_get_agent_response_from_state] structured_response=%s", structured_response)
        
        if structured_response:
            try:
                raw = structured_response.model_dump()
                logger.debug("[_get_agent_response_from_state] structured_response.model_dump=%s", raw)
            except Exception as e:
                logger.warning("[_get_agent_response_from_state] model_dump failed: %s", e)
                raw = structured_response

            parsed = safe_parse_response(raw)
            logger.debug("[_get_agent_response_from_state] Parsed response=%s", parsed)

            # 🚨 fallback parse từ result.content khi data=None
            if parsed.get("data") is None and isinstance(result, AIMessage):
                try:
                    raw_content = result.content
                    logger.debug("[_get_agent_response_from_state][fallback] Raw AIMessage.content=%s", raw_content)
                    if isinstance(raw_content, str):
                        if raw_content.startswith("```json"):
                            raw_content = raw_content.strip("```json\n").strip("```")
                        parsed_fallback = json.loads(raw_content)
                        logger.debug("[_get_agent_response_from_state][fallback] Parsed fallback JSON=%s", parsed_fallback)
                        if "data" in parsed_fallback:
                            parsed["data"] = parsed_fallback["data"]
                            logger.warning("[_get_agent_response_from_state][fallback] Data recovered from AIMessage.content")
                except Exception as e:
                    logger.error("[_get_agent_response_from_state][fallback] Parse error: %s", e)

            return {
                'is_task_complete': True,
                'require_user_input': False,
                'content': parsed.get("message"),
                'status': parsed.get("status", "completed"),
                'data': parsed.get("data"),
            }

        elif result:
            logger.debug(f"[_get_agent_response_from_state] result object fallback: {result}")
            return {
                'is_task_complete': True,
                'require_user_input': False,
                'content': getattr(result, "message", "Không có nội dung."),
                'status': getattr(result, "status", "completed"),
                'data': getattr(result, "data", None)
            }
        else:
            logger.warning("[_get_agent_response_from_state] Không tìm thấy structured_response và result rỗng")
            return {
                'is_task_complete': True,
                'require_user_input': False,
                'content': "Không thể lấy được kết quả từ agent.",
                'status': 'error',
                'data': None
            }

    # Hàm stream trong trường hợp xài symptom_search như một tool được định nghĩa trong symptoms_tools.py
    async def stream(self, query: str, session_id: str):
        logger.debug("stream start session=%s", session_id)
        model_name = os.getenv("GOOGLE_GENAI_MODEL", "gemini-pro")
        model = ChatGoogleGenerativeAI(model=model_name)
        runnable = create_react_agent(
            model,
            tools=self.mcp_tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=(self.RESPONSE_FORMAT_INSTRUCTION, ResponseFormat),
        )

        config = {"configurable": {"thread_id": session_id}}
        langgraph_input = {"messages": [("user", query)]}

        final_response = None
        try:
            async for chunk in runnable.astream_events(langgraph_input, config, version="v1"):
                # chunk.data có thể None; guard lại
                if chunk is None:
                    continue

                if getattr(chunk, "data", None) and "structured_response" in chunk.data:
                    resp = safe_parse_response(chunk.data["structured_response"])
                    yield {
                        "is_task_complete": False,
                        "require_user_input": resp.get("status") == "input_required",
                        "content": resp.get("message"),
                        "status": resp.get("status", "completed"),
                        "data": resp.get("data"),
                    }
            # nếu loop kết thúc bình thường, lấy final state
            final_response = self._get_agent_response_from_state(config, runnable)
            if not final_response:
                final_response = {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": "Không thể lấy kết quả từ agent.",
                    "status": "error",
                    "data": None,
                }
            final_response["is_task_complete"] = True
        except Exception as e:
            logger.exception("Exception while streaming for session=%s", session_id)
            # Trường hợp lỗi: trả final chunk báo lỗi (không đóng socket đột ngột)
            final_response = {
                "is_task_complete": True,
                "require_user_input": False,
                "content": f"Agent error during streaming: {str(e)}",
                "status": "error",
                "data": None,
            }
        # luôn yield final response (một chunk cuối)
        yield final_response
