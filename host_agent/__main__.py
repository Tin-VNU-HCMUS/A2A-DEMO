import asyncio
import gradio as gr
from host_agent.routing_agent import get_initialized_routing_agent_sync
from google.adk.tools import ToolContext

# Khởi tạo Agent định tuyến (host agent)
routing_agent = get_initialized_routing_agent_sync([
    "http://localhost:10001",  # SymptomAgent
    "http://localhost:10002",  # CostAgent
    "http://localhost:10003",  # BookingAgent
])

# Hàm lấy phản hồi từ Agent
async def get_response_from_agent(message, history):
    tool_context = ToolContext(session_id="demo-session", user_id="user123")
    result = await routing_agent.tools["send_message"](
        agent_name = "Symptom Agent",  #  Có thể gán động tuỳ theo intent
        task = message,
        tool_context = tool_context,
    )
    return result

# Hàm async chính
async def main():
    print(" Hệ thống Chatbot Y tế đã sẵn sàng...")

    with gr.Blocks() as demo:
        gr.Image(value = "https://cdn-icons-png.flaticon.com/512/2965/2965567.png", 
                 label = "AI Bác sĩ hỗ trợ", show_label=True, width=150)
        
        gr.ChatInterface(
            fn=get_response_from_agent,
            title ="Trợ lý Y Tế Ảo - Bệnh viện A",
            description ="Bạn có thể hỏi về triệu chứng bệnh, gói khám, đặt lịch hẹn bác sĩ.",
        )

    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=8083)

# Chạy app
if __name__ == "__main__":
    asyncio.run(main())


