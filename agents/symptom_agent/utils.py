# agents/symptom_agent/utils.py
import json
import logging

logger = logging.getLogger(__name__)

def safe_parse_response(raw):
    """
    Ép kết quả từ LLM thành dict chuẩn.
    - Nếu 'data' bị trả về dạng string JSON => parse lại bằng json.loads
    - Nếu parse thất bại thì giữ nguyên và log warning
    """
    try:
        if isinstance(raw, dict):
            # Nếu data là string, parse lại
            if "data" in raw and isinstance(raw["data"], str):
                try:
                    raw["data"] = json.loads(raw["data"])
                except Exception as e:
                    logger.warning(f"Không parse được data string: {e}")
            return raw

        elif isinstance(raw, str):
            parsed = json.loads(raw)
            return safe_parse_response(parsed)

        else:
            logger.warning(f"safe_parse_response nhận type lạ: {type(raw)}")
            return {"status": "error", "message": "Invalid format", "data": None}

    except Exception as e:
        logger.error(f"Lỗi safe_parse_response: {e}")
        return {"status": "error", "message": str(e), "data": None}
