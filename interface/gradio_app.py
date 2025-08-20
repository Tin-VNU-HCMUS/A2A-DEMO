# gradio_demo.py

import gradio as gr
import asyncio

from agents.symptom_agent.symptom_agent import SymptomAgent
from tools.symptoms_tool import search_symptoms

# Tạo agent (giả định bạn có danh sách tool từ MCP hoặc định nghĩa thủ công)
agent = SymptomAgent(mcp_tools=[search_symptoms])

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
