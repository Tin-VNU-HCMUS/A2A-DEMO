# tools/booking_tool.py
from langchain_core.tools import ToolDefinition

def suggest_appointment(dept: str, preferred_date: str = None) -> str:
    """
    Gợi ý lịch khám và bác sĩ phù hợp cho chuyên khoa (dept)
    và ngày mà người dùng mong muốn (nếu có).
    """
    doctors = {
        "nhi": ["BS. Hà (Sáng), BS. Linh (Chiều)"],
        "ngoại": ["BS. Minh (Thứ 2, 4, 6)", "BS. An (Thứ 3, 5)"],
    }
    schedule = doctors.get(dept.lower())
    if schedule:
        return (
            f"Chuyên khoa {dept} hiện có lịch khám với: {', '.join(schedule)}. "
            + (f"Bạn muốn đặt vào ngày {preferred_date} không?" if preferred_date else "")
        )
    return "Chuyên khoa không tìm thấy. Vui lòng kiểm tra lại."

