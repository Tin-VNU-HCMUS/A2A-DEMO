# Import các thư viện cần thiết

import os
from dotenv import load_dotenv


# Load .env trước khi đọc biến môi trường
load_dotenv(override=True)


# Debug
print("API key:", os.getenv("GOOGLE_API_KEY"))
print("Use Vertex:", os.getenv("GOOGLE_GENAI_USE_VERTEXAI"))


from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel



from langchain_google_genai import ChatGoogleGenerativeAI

#from langchain_ollama import ChatOllama
from typing import Literal, Any, AsyncIterable


# Import tool tìm triệu chứng từ CSV
from tools.symptoms_tool import search_symptoms
from langchain_mcp_adapters.client import MultiServerMCPClient

# Khởi tạo memory để lưu state của agent
memory = MemorySaver()

# Định nghĩa format phản hồi chuẩn
class ResponseFormat(BaseModel):
    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str
    data: dict | None = None  # thêm field data để đồng bộ với tool


# Định nghĩa Symptom Agent
class SymptomAgent:
    # Quan trọng phải build SYSTEM_INSTRUCTION thật chuẩn

    SYSTEM_INSTRUCTION = (
        '''
        Bạn là trợ lý y tế, nhiệm vụ của bạn là viết lại câu trả lời từ key "message" trong JSON sao cho gãy gọn, mạch lạc, đúng cấu trúc.

        QUY TẮC:
        1. Nếu trong "message" có liệt kê các bệnh:
        - Viết lại thành mục **"Các bệnh có thể liên quan"**, đánh số thứ tự (1, 2, 3...).
        - Giữ nguyên tên bệnh, không thay đổi, không thêm bệnh mới.
        - Mỗi bệnh mô tả ngắn gọn (1–2 câu), rõ ràng, dễ hiểu, không lặp ý.

        2. Nếu "message" có phần giải thích (thường bắt đầu bằng "Tại sao các triệu chứng..."):
        - Viết lại thành mục **"Tại sao các triệu chứng này có thể liên quan?"**
        - Giải thích rõ ràng dựa trên triệu chứng mà người dùng đưa ra và các bệnh đã liệt kê.
        - Không nhắc lại nguyên văn danh sách bệnh, mà chỉ phân tích sự liên quan triệu chứng ↔ bệnh.
        - Tóm gọn nguyên nhân liên hệ giữa triệu chứng và bệnh, không nhắc lại các đoạn thừa.
        - Có thể tự bịa thêm phần này để hoàn chỉnh

        3. Nếu "message" có phần khuyến cáo:
        - Viết lại thành mục **"Lời khuyên"**.
        - Luôn nhấn mạnh đây chỉ là gợi ý, KHÔNG phải chẩn đoán y tế chính thức.
        - Khuyên người dùng nên đi khám bác sĩ, không đưa ra phác đồ điều trị chi tiết.
        - Ở phần cuối cùng: Đưa ra lời khuyên (nên đi khám bác sĩ, xét nghiệm, theo dõi) chỉ cần đưa ra lời khuyên, không hiển thị hay nói cách khác là bỏ qua các mục sau: Tại sao các triệu chứng này có thể liên quan?, Lời khuyên
        
        4. Văn phong: 
        - Súc tích, mạch lạc, tránh lặp lại.
        - Sử dụng gạch đầu dòng hoặc đánh số khi cần, trình bày khoa học.
        - Không nói vòng vo, không để sót các mục 1–3.

        5. Nếu "message" không có dữ liệu bệnh (trống hoặc lỗi):
        - Trả lời chung chung: “Hiện chưa xác định được bệnh. Bạn nên đi khám bác sĩ để kiểm tra kỹ hơn.”
        '''
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
    async def ainvoke(self, query: str, session_id: str) -> dict[str, Any]:
        # Khởi tạo model mới mỗi request
        model_name = os.getenv("GOOGLE_GENAI_MODEL", "gemini-pro")
        model = ChatGoogleGenerativeAI(model=model_name)

        symptom_agent_runnable = create_react_agent(
            model,
            tools=self.mcp_tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=(self.RESPONSE_FORMAT_INSTRUCTION, ResponseFormat),
        )

        config = {'configurable': {'thread_id': session_id}}
        langgraph_input = {'messages': [('user', query)]}

        await symptom_agent_runnable.ainvoke(langgraph_input, config)
        return self._get_agent_response_from_state(config, symptom_agent_runnable)


    # Hàm _get_agent_response_from_state(): Trích xuất kết quả cuối từ Agent
    def _get_agent_response_from_state(self, config, agent_runnable) -> dict:
        current_state = agent_runnable.get_state(config)
        structured_response = current_state.values.get('structured_response')

        if structured_response:
            # Trả về response chuẩn
            return {
                'is_task_complete': True,
                'require_user_input': False,
                'content': structured_response.message,
                'status': structured_response.status,
                'data': getattr(structured_response, "data", None)  # giữ data JSON nếu có 
            }
        else:
            # Nếu không có structured_response thì trả về lỗi
            return {
                'is_task_complete': True,
                'require_user_input': False,
                'content': 'Không thể lấy được kết quả từ agent.',
                'status': 'error'
            }
        

    # Hàm stream trong trường hợp xài symptom_search như một tool được định nghĩa trong symptoms_tools.py
    # Hàm stream(): Phản hồi theo thời gian thực
    async def stream(self, query: str, session_id: str) -> AsyncIterable[Any]:
        # Khởi tạo model mới mỗi request
        model_name = os.getenv("GOOGLE_GENAI_MODEL", "gemini-pro")
        model = ChatGoogleGenerativeAI(model=model_name)

        symptom_agent_runnable = create_react_agent(
            model,
            tools=self.mcp_tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=(self.RESPONSE_FORMAT_INSTRUCTION, ResponseFormat),
        )

        config = {'configurable': {'thread_id': session_id}}
        langgraph_input = {'messages': [('user', query)]}

    
        # Lặp qua các chunk event
        async for chunk in symptom_agent_runnable.astream_events(langgraph_input, config, version='v1'):
            event_name = chunk.get('event')
            data = chunk.get('data', {})
            content_to_yield = None

            # Nếu đang gọi tool thì thông báo tool nào đang được dùng
            if event_name == 'on_tool_start':
                content_to_yield = None
                #content_to_yield = f"Đang sử dụng tool: {data.get('name', 'một tool')}..."

            # Nếu LLM stream response thì yield ra từng phần
            elif event_name == 'on_chat_model_stream':
                message_chunk = data.get('chunk')
                if isinstance(message_chunk, AIMessageChunk) and message_chunk.content:
                    content_to_yield = message_chunk.content
        

            # Nếu có content thì yield ra
            if content_to_yield:
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': content_to_yield, # Không gửi lại text đã stream
                }


        # Sau khi stream xong, lấy kết quả cuối cùng
        final_response = self._get_agent_response_from_state(config, symptom_agent_runnable)
        yield final_response
