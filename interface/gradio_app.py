# gradio_demo.py

import gradio as gr
import asyncio

from agents.symptom_agent.symptom_agent import SymptomAgent
from tools.symptoms_tool import search_symptoms

# MUST be at very top of entrypoint before importing routing_agent or others
import logging, sys

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,   # buộc ghi đè handlers cũ
)

# đảm bảo root logger bật DEBUG
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# thêm explicit handler cho HostAgent logger (ghi đè nếu cần)
host_logger = logging.getLogger("HostAgent")
if not host_logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    host_logger.addHandler(ch)
host_logger.setLevel(logging.DEBUG)

# optional: also make sure RemoteAgentConnections logger exists
rac_logger = logging.getLogger("RemoteAgentConnections")
rac_logger.setLevel(logging.DEBUG)



# import HostAgent và helper
from host_agent.routing_agent import get_initialized_routing_agent_sync

# Danh sách địa chỉ remote agents (ví dụ)
remote_agent_addresses = [
    "http://localhost:10001",  # SymptomAgent
    "http://localhost:10002",  # CostAgent
    # "http://localhost:8003",  # BookingAgent (nếu có)
]

# Thay vì gọi trực tiếp HostAgent()
# agent = HostAgent(remote_agent_addresses=remote_agent_addresses, http_client=http_client)

# Gọi helper để tạo agent đồng bộ
agent = get_initialized_routing_agent_sync(remote_agent_addresses)



# Hàm async lấy phản hồi từ agent (dùng stream hoặc ainvoke)
async def get_agent_response_async(message: str) -> str:
    chunks = []
    async for chunk in agent.stream(message, session_id="demo-session"):
        content = chunk.get("content")
        if content:
            chunks.append(content)
    return "".join(chunks)

# Hàm đồng bộ để dùng với Gradio
def get_response(message: str) -> str:
    return asyncio.run(get_agent_response_async(message))

# Giao diện Gradio giống ChatGPT
with gr.Blocks(css="""
.chatbot {
    height: 80vh !important;
}
""") as demo:
    gr.Markdown("<h1 style='text-align:center'> Chatbot Tư Vấn Y Tế</h1>")

    chatbot = gr.Chatbot(type="messages", elem_classes="chatbot", height=600)
    msg = gr.Textbox(placeholder="Nhập câu hỏi về triệu chứng của bạn...", scale=9)
    clear = gr.Button("Xóa hội thoại", scale=1)

    # Tin nhắn mở đầu của bot
    chatbot.value = [
        {"role": "assistant", "content": "Xin chào, tôi là trợ lý y tế. Bạn cần hỗ trợ tư vấn gì hôm nay?"}
    ]

    def respond(user_message, history):
        history.append({"role": "user", "content": user_message})
        bot_response = get_response(user_message)
        history.append({"role": "assistant", "content": bot_response})
        return "", history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: [{"role": "assistant", "content": "Xin chào, tôi là trợ lý y tế. Bạn cần hỗ trợ tư vấn gì hôm nay?"}], None, chatbot)

if __name__ == "__main__":
    # Chạy Gradio
    demo.launch(
        server_name="127.0.0.1",   # Bind trực tiếp vào loopback
        server_port=8080,
        share=False
    )
    print("\n Mở trình duyệt và truy cập: http://127.0.0.1:8080\n")
