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
    Bạn là trợ lý y tế, nhiệm vụ của bạn là viết lại câu trả lời từ key "message" hoặc "data.synthesized_answer" trong JSON sao cho ngắn gọn, mạch lạc, đúng cấu trúc.

    QUY TẮC:
    1. **Trích xuất danh sách bệnh**:
        - CHỈ được lấy bệnh từ "data.pdf_results" hoặc "data.synthesized_answer".
        - Không được tự suy đoán bệnh ngoài nguồn này.
        - Nếu trong "data.pdf_results" có snippet chứa tên bệnh (ví dụ: "xơ gan", "bệnh gan do rượu"), bạn phải ưu tiên trích xuất và liệt kê.
        - Giữ nguyên tên bệnh, mô tả ngắn gọn (1–2 câu) dựa trên snippet hoặc synthesized_answer.


    2. **Giải thích triệu chứng**:
        - Viết thành mục **"Tại sao các triệu chứng này có thể liên quan?"**.
        - Dựa trên "data.pdf_results" để phân tích sự liên quan.
        - Không thêm bệnh mới, chỉ phân tích sự liên quan triệu chứng ↔ bệnh đã liệt kê.


    3. **Lời khuyên**:
       - Viết thành mục **"Lời khuyên"**.
       - Tạo lời khuyên theo triệu chứng và các bệnh có liên quan đã được liệt kê ở trên mục 1 (Trích xuất danh sách bệnh)”


    4. **Văn phong**:
       - Súc tích, khoa học, tránh lặp lại.
       - Luôn có đủ 3 mục: 1. Bệnh bạn có thể mắc phải là, 2. Tại sao các triệu chứng trên lại liên quan tới bệnh này, 3. Lời khuyên.


    5. **Trường hợp không có dữ liệu bệnh**:
       - Nếu "data.pdf_results" và "data.synthesized_answer" đều trống, trả lời:
         “Hiện chưa xác định được bệnh. Bạn nên đi khám bác sĩ để kiểm tra kỹ hơn.”

    6. **Xử lý JSON và văn bản thô**:
       - Nếu "message" chỉ có văn bản thô, cố gắng trích bệnh từ "data.pdf_results".
       - Nếu không thể, trả lời như mục 5.
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
