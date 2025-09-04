import asyncio
import uuid
import gradio as gr
from host_agent.routing_agent import get_initialized_routing_agent_sync
from google.adk.tools import ToolContext

# Khởi tạo Agent định tuyến (host agent)
routing_agent = get_initialized_routing_agent_sync([
    "http://localhost:10001",  # SymptomAgent
    "http://localhost:10002",  # CostAgent
    "http://localhost:10003",  # BookingAgent
])

import logging
import sys

# Cấu hình logging phải chạy tại entrypoint trước khi import các module khác
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG để in tất cả
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,  # buộc ghi đè handlers cũ (Python 3.8+)
)

# Tạo logger dùng chung
logger = logging.getLogger("HostAgent")
logger.setLevel(logging.DEBUG)


# Hàm lấy phản hồi từ Agent
async def get_response_from_agent(message, history, session_state):
    # Lấy hoặc tạo session_id, user_id từ state
    if not session_state.get("session_id"):
        session_state["session_id"] = str(uuid.uuid4())
        session_state["user_id"] = str(uuid.uuid4())
    tool_context = ToolContext(
        session_id=session_state["session_id"],
        user_id=session_state["user_id"]
    )
    try:
        # Gọi host agent để tự động phân tích và delegate task
        response = await routing_agent.generate_content_async(
            contents=[{"role": "user", "parts": [{"text": message}]}],
            tool_context=tool_context
        )
        # Extract text từ response (dựa trên cấu trúc Google ADK)
        if response and response.candidates:
            return response.candidates[0].content.parts[0].text, session_state
        return "Không nhận được phản hồi hợp lệ. Vui lòng thử lại.", session_state
    except Exception as e:
        return f"Lỗi: {str(e)}. Vui lòng thử lại hoặc cung cấp thêm chi tiết.", session_state

# Hàm async chính
async def main():
    print("Hệ thống Chatbot Y tế đã sẵn sàng...")

    with gr.Blocks() as demo:
        session_state = gr.State({})
        gr.Image(
            value="https://cdn-icons-png.flaticon.com/512/2965/2965567.png",
            label="AI Bác sĩ hỗ trợ",
            show_label=True,
            width=150
        )
        gr.ChatInterface(
            fn=get_response_from_agent,
            title="Trợ lý Y Tế Ảo - Bệnh viện A",
            description="Bạn có thể hỏi về triệu chứng bệnh, gói khám, đặt lịch hẹn bác sĩ.",
            additional_inputs=[session_state]
        )

    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=8083)

# Chạy app
if __name__ == "__main__":
    asyncio.run(main())