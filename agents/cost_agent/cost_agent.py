
import os
from dotenv import load_dotenv

# Load .env trước khi đọc biến môi trường
load_dotenv(override=True)

# Debug API key
#print("API key: %s", os.getenv("GOOGLE_API_KEY"))
#print("Use Vertex: %s", os.getenv("GOOGLE_GENAI_USE_VERTEXAI"))


import logging
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Literal, Any, AsyncIterable
from tools.cost_tool import cost_tool_rag
from langchain_core.messages import HumanMessage
#from langchain_nvidia_ai_endpoints import ChatNVIDIA
import json

# Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Khởi tạo memory để lưu state của agent
memory = MemorySaver()

# Định nghĩa format phản hồi chuẩn
class ResponseFormat(BaseModel):
    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str | None = None
    data: dict | None = None  # thêm field data để đồng bộ với tool

# Định nghĩa Cost Agent
class CostAgent:
    # Quan trọng phải build SYSTEM_INSTRUCTION thật chuẩn thật chi tiết

    SYSTEM_INSTRUCTION = (
    '''
    Bạn là trợ lý y tế chuyên về **chi phí khám chữa bệnh**. 
    **Nguồn dữ liệu chính và duy nhất để trích xuất danh sách bệnh** trong ngữ cảnh này là phần **Final response_parts** do HostAgent gửi (ví dụ: HostAgent.send_message -> Final response_parts: [...]) và goi_kham_vip_full.json (các gói khám với giá, items).

    MỤC TIÊU:
    - Dựa vào Final response_parts, viết lại và cấu trúc lại thông tin sao cho ngắn gọn, mạch lạc, đúng cấu trúc, phục vụ việc đề xuất chuyên khoa, gói dịch vụ và chi phí.
    - Xử lý giá linh hoạt (all, male/female)
    
    QUY TẮC CHI TIẾT (bắt buộc tuân thủ)
    1) NHẬN DẠNG ĐẦU VÀO:
    - Đầu vào có thể là:
        a) Một object JSON chứa khóa "final_response_parts" hoặc "final_response_parts" được truyền trực tiếp; 
        b) Hoặc một mảng/chuỗi text mà trong đó chứa các mục giống như Final response_parts.
    - Bắt buộc **parse** Final response_parts (một mảng các đoạn văn). KHÔNG tự suy đoán bệnh ngoài những gì có trong Final response_parts. 
    - Nếu Final response_parts không có hoặc rỗng, HOẶC không chứa cấu trúc rõ ràng (như list bệnh), fallback sang "data.pdf_results" hoặc "data.synthesized_answer" hoặc user query. Nếu tất cả đều rỗng thì xử lý theo phần "Không có dữ liệu" bên dưới.

    2) TRÍCH XUẤT "CÁC BỆNH CÓ THỂ MẮC":
    - Tìm trong Final response_parts phần chứa tiêu đề như "Các bệnh có thể mắc", "Các bệnh", "2. **Các bệnh có thể mắc**", bullet list, hoặc các dòng liệt kê (dấu * hoặc -).
    - Trích chính xác các **tên bệnh** được liệt kê (ví dụ: "Giãn tĩnh mạch thực quản và dạ dày", "Xuất huyết tiêu hóa", "Bệnh gan mãn tính (xơ gan)").
    - Với mỗi bệnh, giữ nguyên **tên** như trong Final response_parts và **ghi thêm 1–2 câu mô tả ngắn** dựa trên snippet hoặc đoạn văn liên quan trong cùng final_response_parts (không thêm suy đoán y học mới).
    - Nếu Final response_parts chứa **Nhiệm định chính** (ví dụ "Nhận định chính: ...") lấy đoạn ngắn đó để làm mô tả cho bệnh liên quan.
    - **Fallback nếu không khớp pattern**: Quét plain text trong final_response_parts hoặc synthesized_answer/user query để tìm tên bệnh tiềm năng (chứa từ như "bệnh", "liên quan tới", hoặc match với disease list qua tool). Sử dụng cost_tool_rag để assist extraction nếu cần.

    2.5) XỬ LÝ COST-ONLY QUERY (KHÔNG QUA SYMPTOM AGENT):
    - Nếu final_response_parts giống hoặc chứa trực tiếp user query (không có cấu trúc list bệnh từ SymptomAgent), treat as cost-only.
    - Extract bệnh từ user query/synthesized_answer: Tìm tên bệnh explicit (ví dụ: "Xuất huyết tiêu hóa" trong "gói khám bệnh liên quan tới Xuất huyết tiêu hóa").
    - Gọi cost_tool_rag với input chứa synthesized_answer = user query, và extracted_symptoms = [query] để tool tự normalize và match bệnh.

    3) ĐỀ XUẤT CHUYÊN KHOA & GÓI DỊCH VỤ:
    - Luôn gọi tool `cost_tool_rag` (đã đăng ký) để:
        1. Ánh xạ từng bệnh sang chuyên khoa tương ứng.
        2. Lấy các gói dịch vụ và chi phí (min-max) kèm `relevance_score`.
    - Chỉ sử dụng kết quả trả về từ `cost_tool_rag` (không tự ý sửa giá).
    - Sắp xếp gói theo `relevance_score` giảm dần và **ưu tiên hiển thị top 3** cho mỗi chuyên khoa (nếu có).
    - Hiển thị định dạng:
        "Chuyên khoa: <Tên chuyên khoa>
        - <Tên gói 1> — <min>-<max> VND — relevance_score: <0.00>
        - <Tên gói 2> — ..."

    4) LỜI KHUYÊN VỀ CHI PHÍ (cần có luôn):
    - Dựa trên mức relevance của gói cao nhất:
        - Nếu gói có relevance cao (>= 0.7): gợi ý chọn gói đề xuất; nếu triệu chứng mô tả nhẹ thì gợi ý gói cơ bản, nếu nặng thì gợi ý gói nâng cao.
        - Nếu relevance thấp (< 0.7) hoặc không có gói: nói rõ "Không tìm thấy gói khám phù hợp với bệnh này dựa trên dữ liệu hiện có."
    - Nếu không tìm thấy gói phù hợp (No match), trả lời kèm hướng dẫn: 
        "Không có gói khám phù hợp với bệnh 'X' (hoặc không đủ tương đồng). Vui lòng nhập lại **đúng tên bệnh** như đã được dự đoán ở phía trên (ví dụ: 'Xuất huyết tiêu hóa') để tôi dò gói và giá chính xác."

    5) XỬ LÝ TRƯỜNG HỢP NGƯỜI DÙNG HỎI GÓI NGAY TỪ ĐẦU (Test case 1):
    - Nếu user query trực tiếp chứa tên bệnh (ví dụ "Khám gan bao nhiêu tiền?", "Gói khám xuất huyết tiêu hóa?"), bạn phải:
        1. Thử trích bệnh từ câu user (dùng embedding/fuzzy bằng cost_tool_rag).
        2. Gọi cost_tool_rag với đầu vào chứa: {"session_id":..., "data": {"synthesized_answer": "<user query>", "extracted_symptoms": [], "pdf_results": []}}
        3. Trả kết quả gói & giá tương tự cách ở mục 3.

    6) XỬ LÝ TRƯỜNG HỢP NGƯỜI DÙNG ĐÃ QUA SYMPTOM_AGENT (Test case 2):
    - Khi Final response_parts đã có danh sách bệnh:
        1. Dùng danh sách đó làm nguồn chính để gọi cost_tool_rag.
        2. Nếu user hỏi "cho tôi gói khám của bệnh này", hiểu "bệnh này" là **toàn bộ danh sách** trong mục 2 của Final response_parts — hiển thị gói cho từng bệnh trong danh sách (ưu tiên top gói mỗi chuyên khoa).

    7) ĐỊNH DẠNG PHẢN HỒI (bắt buộc):
    - Phải có đủ 3 mục (theo thứ tự):
        1. **Các bệnh có thể liên quan:** (liệt kê số thứ tự + tên + mô tả 1–2 câu)
        2. **Đề xuất chuyên khoa và gói dịch vụ:** (chuyên khoa -> danh sách gói + giá min-max VND + relevance_score)
        3. **Lời khuyên về chi phí:** (1–3 câu ngắn gọn)
    - Trả lời bằng **Tiếng Việt**, ngôn ngữ súc tích, chuyên nghiệp, không dài dòng.

    8) XỬ LÝ DỮ LIỆU & LƯU LỊCH SỬ:
    - Nếu input là JSON string, parse nó. Nếu parse thất bại, fallback kiểm tra xem chuỗi có pattern "Các bệnh có thể mắc" và cố trích.
    - Sau khi trả kết quả, gọi tool hoặc service để **lưu lịch sử** (history) gồm: session_id, final_response_parts (nguồn), user query (nếu có), và kết quả gói đã đề xuất.

    9) TRƯỜNG HỢP KHÔNG CÓ DỮ LIỆU:
    - Nếu không extract được bất kỳ bệnh nào sau fallback, trả về chính xác:
        “Hiện chưa xác định được bệnh. Bạn nên đi khám bác sĩ để kiểm tra kỹ hơn và ước lượng chi phí.”

    10) HÀNH VI KHI KHÔNG CHẮC CHẮN:
    - Không tự thêm bệnh mới.
    - Không đưa ra khuyến nghị điều trị.
    - Nếu cần thêm thông tin từ user (ví dụ: "Bạn muốn gói cơ bản hay nâng cao?"), yêu cầu ngắn gọn và trực tiếp, nhưng **chỉ khi thật cần**.

    NOTE kỹ thuật:
    - Luôn truyền `session_id` (nếu có) vào call tới `cost_tool_rag` để tool có thể lưu history.
    - Sử dụng threshold similarity mặc định (ví dụ 0.7) do hệ thống cài đặt; nếu score < threshold, báo "No match" như mục 4.

    KẾT:
    - Mục tiêu là biến Final response_parts của HostAgent thành 1 kết quả chi phí/gói khám rõ ràng, dễ đọc, có thể dùng cho UI demo. 
    - Luôn ưu tiên dữ liệu gốc từ HostAgent (Final response_parts) — mọi hành vi khác là fallback có kiểm soát.
    '''
)

    RESPONSE_FORMAT_INSTRUCTION = 'Select status as "completed" and write the answer in Vietnamese.'
    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']

    def __init__(self, mcp_tools: list[Any]):
        model_name = os.getenv("GOOGLE_GENAI_MODEL", "gemini-pro")
        self.model = ChatGoogleGenerativeAI(model=model_name)
        self.mcp_tools = mcp_tools + [cost_tool_rag]

# --- ainvoke: log rõ hơn, normalize kết quả trước khi trả ---
    async def ainvoke(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        session_id = input_dict.get("session_id", "default_session")
        query = input_dict.get("query", "")
        final_response_parts = input_dict.get("final_response_parts", [])

        # Trong stream (và tương tự trong ainvoke):
        user_content = json.dumps({  # Chuyển dict thành string để content hợp lệ
            "query": query,
            "final_response_parts": final_response_parts,
        })  # Hoặc chỉ dùng query làm content, và final_parts vào additional_kwargs nếu cần

        langgraph_input = {
            "messages": [
                HumanMessage(
                    content=user_content,  # Bây giờ là string
                    additional_kwargs={"session_id": session_id}  # Dữ liệu bổ sung nếu cần
                )
            ]
        }

        runnable = create_react_agent(
            self.model,
            tools=self.mcp_tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=(self.RESPONSE_FORMAT_INSTRUCTION, ResponseFormat),
        )

        config = {"configurable": {"thread_id": session_id}}

        try:
            result = await runnable.ainvoke(langgraph_input, config)
            response = self._get_agent_response_from_state(config, runnable, result)

            if not response:
                response = {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": "Không thể lấy kết quả từ agent.",
                    "status": "error",
                    "data": None,
                }
            else:
                # bảo đảm content luôn là string
                response["content"] = str(response.get("content") or "")
                response["is_task_complete"] = True
            return response

        except Exception as e:
            logger.exception("Error in ainvoke")
            return {
                "is_task_complete": True,
                "require_user_input": False,
                "content": f"Lỗi khi xử lý: {str(e)}",
                "status": "error",
                "data": None,
            }
        
# --- stream: xử lý defensive và normalize mọi chunk trước khi yield ---
    async def stream(self, message: HumanMessage) -> AsyncIterable[dict]:
        session_id = message.additional_kwargs.get("session_id", "default_session")
        final_response_parts = message.additional_kwargs.get("final_response_parts", [])
        query = str(message.content or "")

        # Trong stream (và tương tự trong ainvoke):
        user_content = json.dumps({  # Chuyển dict thành string để content hợp lệ
            "query": query,
            "final_response_parts": final_response_parts,
        })  # Hoặc chỉ dùng query làm content, và final_parts vào additional_kwargs nếu cần

        langgraph_input = {
            "messages": [
                HumanMessage(
                    content=user_content,  # Bây giờ là string
                    additional_kwargs={"session_id": session_id}  # Dữ liệu bổ sung nếu cần
                )
            ]
        }

        runnable = create_react_agent(
            self.model,
            tools=self.mcp_tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=(self.RESPONSE_FORMAT_INSTRUCTION, ResponseFormat),
        )

        config = {"configurable": {"thread_id": session_id}}
        has_yielded = False

        try:
            async for chunk in runnable.astream_events(langgraph_input, config, version="v1"):
                # normalize chunk -> data (hỗ trợ dict hoặc object có .data)
                if isinstance(chunk, dict):
                    data = chunk.get("data") or {}
                else:
                    data = getattr(chunk, "data", {}) or {}

                logger.debug(f"[CostAgent] received chunk keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")

                # structured_response có thể nằm trong data (dict) hoặc attribute
                structured = None
                if isinstance(data, dict) and "structured_response" in data:
                    structured = data["structured_response"]
                else:
                    structured = getattr(data, "structured_response", None) or getattr(chunk, "structured_response", None)

                if not structured:
                    # không phải structured, bỏ qua
                    continue

                # `structured` có thể là pydantic model, dict hoặc SimpleNamespace
                # lấy các trường an toàn
                status = getattr(structured, "status", None) or (structured.get("status") if isinstance(structured, dict) else None)
                message_text = getattr(structured, "message", None) or (structured.get("message") if isinstance(structured, dict) else None) or ""
                data_field = getattr(structured, "data", None) or (structured.get("data") if isinstance(structured, dict) else None)

                has_yielded = True
                yield {
                    "is_task_complete": False,
                    "require_user_input": status == "input_required",
                    "content": str(message_text),
                    "status": status or "input_required",
                    "data": data_field,
                }

        except Exception as e:
            logger.exception("Exception while streaming from runnable")
            # Trả 1 event lỗi thay vì để crash
            yield {
                "is_task_complete": True,
                "require_user_input": False,
                "content": f"Lỗi nội bộ khi xử lý: {e}",
                "status": "error",
                "data": None,
            }

        # final_response từ state
        final_response = self._get_agent_response_from_state(config, runnable)
        if not final_response:
            final_response = {
                "is_task_complete": True,
                "require_user_input": False,
                "content": "Không thể lấy kết quả từ agent.",
                "status": "error",
                "data": None,
            }
        else:
            final_response["is_task_complete"] = True
            final_response["content"] = str(final_response.get("content") or "")

        yield final_response


    def _get_agent_response_from_state(self, config: dict, runnable: Any, result: Any = None) -> dict:
        try:
            # Lấy state từ checkpointer
            # state = runnable.checkpointer.get(config["configurable"]["thread_id"])
            state = runnable.checkpointer.get(config)
            if state:
                # messages thường nằm trong state['channel_values']['messages']
                messages = state.get('channel_values', {}).get('messages', [])
                if messages:
                    last_message = messages[-1]
                    if isinstance(last_message, AIMessage):
                        return {
                            "content": last_message.content,
                            "is_task_complete": True,
                            "require_user_input": False,
                            "status": "completed",
                            "data": getattr(last_message, "additional_kwargs", {}).get("data")
                        }
            # Fallback nếu không có
            return {
                "content": "",
                "is_task_complete": True,
                "require_user_input": False,
                "status": "error",
                "data": None
            }
        except Exception as e:
            logger.error(f"Error getting response from state: {e}")
            return {
                "content": "",
                "is_task_complete": True,
                "require_user_input": False,
                "status": "error",
                "data": None
            }