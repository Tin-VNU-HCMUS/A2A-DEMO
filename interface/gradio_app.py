import gradio as gr
import asyncio
import logging
import sys
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

from host_agent.routing_agent import get_initialized_routing_agent_sync  # Sửa ở routing_agent.py

# Nếu có Google ADK HostAgent, dùng để bọc LlmAgent khi cần
try:
    from google.adk.agents import HostAgent as GoogleHostAgent
except Exception:
    GoogleHostAgent = None

# Logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

host_logger = logging.getLogger("HostAgent")
if not host_logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    host_logger.addHandler(ch)
host_logger.setLevel(logging.DEBUG)

rac_logger = logging.getLogger("RemoteAgentConnections")
rac_logger.setLevel(logging.DEBUG)

# Danh sách địa chỉ remote agents
remote_agent_addresses = [
    "http://localhost:10001",  # SymptomAgent
    "http://localhost:10002",  # CostAgent
]

# Khởi tạo agent bằng helper (sync)
agent = get_initialized_routing_agent_sync(remote_agent_addresses)  # Bây giờ trả về HostAgent instance với ainvoke

'''
# --- Đảm bảo agent có ainvoke với fallback cải thiện ---
def ensure_async_ainvoke(agent_obj):
    """
    Đảm bảo agent_obj có method async 'ainvoke'.
    1) Nếu GoogleHostAgent khả dụng -> bọc bằng GoogleHostAgent
    2) Nếu agent đã có ainvoke -> return
    3) Nếu không, thử monkey-patch async wrapper gọi invoke/run/generate sync trong executor.
    4) Logging chi tiết và raise nếu không fallback được.
    """
    if hasattr(agent_obj, "ainvoke") and asyncio.iscoroutinefunction(getattr(agent_obj, "ainvoke")):
        logging.debug("Agent đã có phương thức ainvoke async.")
        return agent_obj

    # Nếu có lớp GoogleHostAgent, bọc vào (ưu tiên)
    if GoogleHostAgent is not None:
        try:
            if not isinstance(agent_obj, GoogleHostAgent):
                logging.debug("Bọc agent bằng GoogleHostAgent để hỗ trợ ainvoke.")
                return GoogleHostAgent(agent=agent_obj)
            else:
                return agent_obj
        except Exception as e:
            logging.warning(f"Không bọc được bằng GoogleHostAgent: {e}. Chuyển sang tạo async wrapper.")

    # Fallback: monkey-patch một async ainvoke chạy invoke/run/generate trong executor
    possible_methods = ["invoke", "run", "generate"]  # Thử các method có thể
    invoke_fn = None
    for meth in possible_methods:
        if hasattr(agent_obj, meth):
            invoke_fn = getattr(agent_obj, meth)
            logging.debug(f"Found fallback method '{meth}' for ainvoke.")
            break

    if not invoke_fn:
        raise AttributeError("Agent không có method 'invoke', 'run' hoặc 'generate' để fallback.")

    async def _ainvoke(self, message, session_id=None, **kwargs):
        loop = asyncio.get_event_loop()
        # Gọi sync method trong executor để không block
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, lambda: invoke_fn(message, session_id=session_id, **kwargs))

    # Gắn vào class hoặc instance
    try:
        cls = agent_obj.__class__
        if not hasattr(cls, "ainvoke"):
            setattr(cls, "ainvoke", _ainvoke)
            logging.debug("Đã gắn phương thức ainvoke (monkey-patch) lên class của agent.")
    except Exception:
        import types
        agent_obj.ainvoke = types.MethodType(_ainvoke, agent_obj)
        logging.debug("Đã gắn phương thức ainvoke (monkey-patch) lên instance của agent.")

    return agent_obj
'''

# Hàm async lấy phản hồi từ HostAgent / agent đã hỗ trợ ainvoke
async def get_agent_response_async(message: str) -> str:
    try:
        # gọi async ainvoke (GoogleHostAgent hoặc wrapper)
        result = await agent.ainvoke(message, session_id="demo-session")
        if isinstance(result, dict):
            return result.get("content") or result.get("text") or str(result)
        return str(result)
    except Exception as e:
        logging.exception("Exception khi gọi agent.ainvoke")
        return f"[Error] Exception khi gọi HostAgent: {e}. Query: {message}"

# Hàm đồng bộ an toàn: chạy coroutine ngay cả khi event loop đã chạy
def get_response(message: str) -> str:
    """
    Trả về kết quả đồng bộ cho Gradio. 
    Nếu asyncio.run có thể chạy -> dùng nó. Nếu loop đang chạy (RuntimeError), khởi 1 thread mới
    và chạy loop riêng để thực thi coroutine.
    """
    try:
        return asyncio.run(get_agent_response_async(message))
    except RuntimeError:
        # Môi trường đã có event loop (uvicorn, jupyter, ...). Chạy coroutine trong thread riêng.
        q = queue.Queue()

        def _runner():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                res = loop.run_until_complete(get_agent_response_async(message))
                q.put(res)
            except Exception as e:
                q.put(f"[Error-thread] {e}")
            finally:
                try:
                    loop.close()
                except Exception:
                    pass

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        t.join()
        return q.get()

# Giao diện Gradio
with gr.Blocks(css="""
.chatbot {
    height: 80vh !important;
}
""") as demo:
    gr.Markdown("<h1 style='text-align:center'> Chatbot Tư Vấn Y Tế</h1>")
    chatbot = gr.Chatbot(type="messages", elem_classes="chatbot", height=600)
    msg = gr.Textbox(placeholder="Nhập câu hỏi về triệu chứng của bạn...", scale=9)
    clear = gr.Button("Xóa hội thoại", scale=1)

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