import os
from dotenv import load_dotenv
import logging
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Literal, Any, AsyncIterable
from tools.cost_tool import cost_tool_rag



# Load .env trước khi đọc biến môi trường
load_dotenv(override=True)

# Debug API key
print("API key: %s", os.getenv("GOOGLE_API_KEY"))
print("Use Vertex: %s", os.getenv("GOOGLE_GENAI_USE_VERTEXAI"))


# Khởi tạo memory để lưu state của agent
memory = MemorySaver()

# Định nghĩa format phản hồi chuẩn
class ResponseFormat(BaseModel):
    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str
    data: dict | None = None  # thêm field data để đồng bộ với tool

# Định nghĩa Cost Agent
class CostAgent:
    # Quan trọng phải build SYSTEM_INSTRUCTION thật chuẩn thật chi tiết

    SYSTEM_INSTRUCTION = (
    '''
    Bạn là trợ lý y tế chuyên về chi phí khám chữa bệnh, nhiệm vụ của bạn là viết lại câu trả lời từ key "message" hoặc "data.synthesized_answer" trong JSON đầu vào sao cho ngắn gọn, mạch lạc, đúng cấu trúc. Đầu vào là output từ symptom_agent, chứa triệu chứng, pdf_results, và synthesized_answer.

    QUY TẮC:
    1. **Trích xuất danh sách bệnh**:
       - CHỈ được lấy bệnh từ "data.pdf_results" hoặc "data.synthesized_answer" từ symptoms_tool.py.
       - Không được tự suy đoán bệnh ngoài nguồn này.
       - Nếu trong "data.pdf_results" có snippet chứa tên bệnh (ví dụ: "xơ gan", "tăng áp lực tĩnh mạch cửa"), bạn phải ưu tiên trích xuất và liệt kê.
       - Viết thành mục **"Các bệnh có thể liên quan"**, đánh số thứ tự (1, 2, 3...).
       - Giữ nguyên tên bệnh, mô tả ngắn gọn (1–2 câu) dựa trên snippet.

    2. **Đề xuất chuyên khoa và gói dịch vụ**:
       - Sử dụng tool cost_tool_rag để ánh xạ bệnh sang chuyên khoa, gói dịch vụ và chi phí.
       - Viết thành mục **"Đề xuất chuyên khoa và gói dịch vụ"**.
       - Liệt kê chuyên khoa, sau đó là các gói dịch vụ kèm chi phí (min-max, đơn vị VND).
       - Ưu tiên gói có relevance_score cao nhất.

    3. **Lời khuyên về chi phí**:
       - Viết thành mục **"Lời khuyên về chi phí"**.
       - Dựa trên gói dịch vụ đề xuất, đưa ra lời khuyên tiết kiệm chi phí, ví dụ: chọn gói cơ bản nếu triệu chứng nhẹ, hoặc nâng cao nếu cần xét nghiệm sâu.

    4. **Văn phong**:
       - Súc tích, khoa học, tránh lặp lại.
       - Luôn có đủ 3 mục: 1. Các bệnh có thể liên quan, 2. Đề xuất chuyên khoa và gói dịch vụ, 3. Lời khuyên về chi phí.

    5. **Trường hợp không có dữ liệu bệnh**:
       - Nếu "data.pdf_results" và "data.synthesized_answer" đều trống, trả lời:
         “Hiện chưa xác định được bệnh. Bạn nên đi khám bác sĩ để kiểm tra kỹ hơn và ước lượng chi phí.”

    6. **Xử lý JSON và văn bản thô**:
       - Nếu input là JSON string, parse nó trước khi xử lý.
       - Nếu không thể, trả lời như mục 5.
    '''
    )

    RESPONSE_FORMAT_INSTRUCTION = 'Select status as "completed" and write the answer in Vietnamese.'
    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']

    def __init__(self, mcp_tools: list[Any]):
        # Debug: thông báo đăng ký tool
        #logger.debug("Khởi tạo CostAgent với %d MCP tools", len(mcp_tools))
        
        # Đọc model từ biến môi trường
        model_name = os.getenv("GOOGLE_GENAI_MODEL", "gemini-pro")
        self.model = ChatGoogleGenerativeAI(model=model_name)
        
        # Đăng ký tool cost_tool_rag vào danh sách tool
        self.mcp_tools = mcp_tools + [cost_tool_rag]
        #logger.debug("Đã đăng ký tool cost_tool_rag")

    # Phương thức ainvoke(): khởi chạy Agent và nhận kết quả không-stream
    async def ainvoke(self, query: str, session_id: str) -> dict[str, Any]:
        
        #logger.debug("Gọi ainvoke với query: %s, session_id: %s", query, session_id)
        # Khởi tạo model mới mỗi request
        model_name = os.getenv("GOOGLE_GENAI_MODEL", "gemini-pro")
        model = ChatGoogleGenerativeAI(model=model_name)

        cost_agent_runnable = create_react_agent(
            model,
            tools=self.mcp_tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=(self.RESPONSE_FORMAT_INSTRUCTION, ResponseFormat),
        )

        config = {'configurable': {'thread_id': session_id}}
        langgraph_input = {'messages': [('user', query)]}
        #logger.debug("Đầu vào LangGraph: %s", langgraph_input)

        try:
            await cost_agent_runnable.ainvoke(langgraph_input, config)
            response = self._get_agent_response_from_state(config, cost_agent_runnable)
            #logger.debug("Kết quả từ agent: %s", response)
            return response
        except Exception as e:
            #logger.error("Lỗi trong ainvoke: %s", str(e))
            return {
                'is_task_complete': True,
                'require_user_input': False,
                'content': f'Lỗi khi xử lý: {str(e)}',
                'status': 'error'
            }

    # Hàm _get_agent_response_from_state(): Trích xuất kết quả cuối từ Agent
    def _get_agent_response_from_state(self, config, agent_runnable) -> dict:
        # Logger Debug
        #logger.debug("Trích xuất phản hồi từ state với config: %s", config)
        current_state = agent_runnable.get_state(config)
        structured_response = current_state.values.get('structured_response')

        if structured_response:
            #logger.debug("Structured response: %s", structured_response)
            # Trả về response chuẩn
            return {
                'is_task_complete': True,
                'require_user_input': False,
                'content': structured_response.message,
                'status': structured_response.status,
                'data': getattr(structured_response, "data", None)  # giữ data JSON nếu có 
            }
        else:
            #logger.warning("Không tìm thấy structured_response")
            # Nếu không có structured_response thì trả về lỗi
            return {
                'is_task_complete': True,
                'require_user_input': False,
                'content': 'Không thể lấy được kết quả từ agent.',
                'status': 'error'
            }

    # Hàm stream(): Phản hồi theo thời gian thực
    async def stream(self, query: str, session_id: str) -> AsyncIterable[Any]:
        # logger debug
        #logger.debug("Gọi stream với query: %s, session_id: %s", query, session_id)
        # Khởi tạo model mới mỗi request
        model_name = os.getenv("GOOGLE_GENAI_MODEL", "gemini-pro")
        model = ChatGoogleGenerativeAI(model=model_name)

        cost_agent_runnable = create_react_agent(
            model,
            tools=self.mcp_tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=(self.RESPONSE_FORMAT_INSTRUCTION, ResponseFormat),
        )

        config = {'configurable': {'thread_id': session_id}}
        langgraph_input = {'messages': [('user', query)]}
        #logger.debug("Đầu vào LangGraph: %s", langgraph_input)

        try:
            async for chunk in cost_agent_runnable.astream_events(langgraph_input, config, version='v1'):
                event_name = chunk.get('event')
                data = chunk.get('data', {})
                content_to_yield = None

                if event_name == 'on_tool_start':
                    #logger.debug("Bắt đầu gọi tool: %s", data.get('name', 'một tool'))
                    content_to_yield = None
                elif event_name == 'on_chat_model_stream':
                    message_chunk = data.get('chunk')
                    if isinstance(message_chunk, AIMessageChunk) and message_chunk.content:
                        content_to_yield = message_chunk.content
                        #logger.debug("Stream chunk: %s", content_to_yield)

                if content_to_yield:
                    yield {
                        'is_task_complete': False,
                        'require_user_input': False,
                        'content': content_to_yield,
                    }

            final_response = self._get_agent_response_from_state(config, cost_agent_runnable)
            #logger.debug("Phản hồi cuối cùng từ stream: %s", final_response)
            yield final_response
        except Exception as e:
            #logger.error("Lỗi trong stream: %s", str(e))
            yield {
                'is_task_complete': True,
                'require_user_input': False,
                'content': f'Lỗi khi xử lý: {str(e)}',
                'status': 'error'
            }