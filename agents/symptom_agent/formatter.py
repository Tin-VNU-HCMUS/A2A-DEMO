from typing import Any, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
import os

def format_response_with_llm(message: str, data: Dict[str, Any] | None) -> str:
    """
    Dùng Google Gemini LLM để biên tập câu trả lời cuối cùng cho user.
    """
    # Khởi tạo mô hình với API key và model
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # Tên mô hình đúng
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    prompt = f"""
Bạn là bác sĩ AI. Dựa trên thông tin structured response sau, hãy viết lại câu trả lời rõ ràng, gọn gàng với format:

1. Nhận định chính (từ message).
2. Các bệnh có thể mắc (danh sách từ data.diseases).
3. Giải thích (từ data.explanation).
4. Lời khuyên (từ data.advice).

Chỉ trả lời bằng tiếng Việt, ngắn gọn, không lặp ý, không in ra thông tin kỹ thuật (cost, booking, error) và ĐẶC BIỆT KHÔNG BỊA THÔNG TIN MÀ CHỈ ĐƯỢC PHÉP LẤY TRONG structured response để dùng cho câu trả lời.

Message: {message}

Data: {data}
"""

    # Gọi mô hình để tạo phản hồi
    response = model.invoke(prompt)

    # Trả về văn bản đã xử lý hoặc message gốc nếu không có phản hồi
    return response.content.strip() if response and response.content else message