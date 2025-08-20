from langchain_core.tools import ToolDefinition

def estimate_cost(service_code: str) -> str:
    """
    Nhận mã dịch vụ/thủ tục khám (hoặc câu hỏi như 'gói khám tổng quát'),
    trả về chi phí ước tính và hướng dẫn BHYT nếu có.
    """
    mapping = {
        "gói khám tổng quát": 500000,
        "xét nghiệm máu": 300000,
        "siêu âm ổ bụng": 700000,
    }
    cost = mapping.get(service_code.lower())
    if cost:
        return f"Chi phí khoảng ~{cost} VNĐ. BHYT có thể hỗ trợ theo đúng mệnh giá."
    return "Mình không có thông tin về dịch vụ đó. Bạn vui lòng nói rõ hơn."

