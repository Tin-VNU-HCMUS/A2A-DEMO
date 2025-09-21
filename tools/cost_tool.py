# cost_tools.py
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import asyncio
import logging
from functools import lru_cache
from langchain_core.tools import tool
from sentence_transformers import SentenceTransformer, util
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from rapidfuzz import fuzz

# Config
SIM_THRESHOLD = 0.55
BASE_DIR = Path(__file__).resolve().parent.parent / "data"
HISTORY_DIR = BASE_DIR / "history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler('cost_tool_rag.log', encoding='utf-8')])
logger = logging.getLogger(__name__)

# Gemini / Google client init
def init_gemini_client():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("GOOGLE_API_KEY chưa thiết lập. Sử dụng chế độ fallback (LLM sẽ không hoạt động).")
        return None
    model_name = os.getenv("GOOGLE_GENAI_MODEL", "gemini-pro")
    return ChatGoogleGenerativeAI(model=model_name, api_key=api_key, temperature=0.2, max_output_tokens=512)

_gemini_client = init_gemini_client()

def call_llm(prompt: str) -> str:
    """Call Gemini (synchronous wrapper). Nếu client không thiết lập -> raise RuntimeError."""
    if not _gemini_client:
        logger.warning("Gemini client không khả dụng, fallback sang fuzzy matching.")
        return ""  # hoặc trả về kết quả từ fuzzy matching
    try:
        logger.debug("Gọi LLM với prompt: %s", prompt[:200])
        response = _gemini_client.invoke([HumanMessage(content=prompt)])
        # response có thể khác tuỳ lib; cố gắng truy xuất .content hoặc str(response)
        text = getattr(response, "content", None) or str(response)
        text = text.strip()
        logger.debug("LLM trả về: %s", text[:400])
        return text
    except Exception as e:
        logger.exception("Lỗi gọi LLM: %s", e)
        raise RuntimeError(str(e))

# Load embedding model
logger.debug("Load embedding model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Load DB files
try:
    with open(BASE_DIR / "disease_specialty.json", "r", encoding="utf-8") as f:
        disease_specialty = json.load(f)
    with open(BASE_DIR / "specialty_package.json", "r", encoding="utf-8") as f:
        specialty_package = json.load(f)
    with open(BASE_DIR / "package_cost.json", "r", encoding="utf-8") as f:
        package_cost = json.load(f)
except FileNotFoundError as e:
    logger.error("Không tìm thấy file dữ liệu trong data/: %s", e)
    raise

# Helpers & indexes
def preprocess_disease_variants(disease_specialty: List[Dict]) -> Dict[str, str]:
    """Hợp nhất biến thể thành tên chuẩn -> trả về dict {disease_name: specialty}"""
    disease_map = {}
    for item in disease_specialty:
        disease = item["disease"].strip()
        key = disease.lower().split(" (")[0].split(" do ")[0].strip()
        if key not in disease_map or len(disease) < len(disease_map[key]["disease"]):
            disease_map[key] = item
    return {v["disease"]: v["specialty"] for v in disease_map.values()}

disease_to_specialty = preprocess_disease_variants(disease_specialty)
disease_list = list(disease_to_specialty.keys())
specialty_to_packages = {s["specialty"]: s["packages"] for s in specialty_package}
package_cost_map = {c["id"]: c for c in package_cost}
logger.debug("Index ready: %d bệnh, %d chuyên khoa, %d gói", len(disease_list), len(specialty_to_packages), len(package_cost_map))

@lru_cache(maxsize=4096)
def cached_embedding(text: str, to_tensor: bool = False):
    return model.encode(text, convert_to_tensor=to_tensor)

def save_history(session_id: str, entry: dict):
    """Lưu history theo session_id (append vào file JSON list)."""
    path = HISTORY_DIR / f"history_{session_id}.json"
    try:
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
        else:
            existing = []
        existing.append(entry)
        path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.debug("Lưu history cho session %s", session_id)
    except Exception as e:
        logger.exception("Lỗi lưu history: %s", e)

def normalize_disease_rag(disease: str, disease_list: List[str]) -> Dict:
    """Chuẩn hoá tên bệnh -> trả về dict {"name":..., "matched":bool, "score":float}"""
    disease_clean = disease.strip().lower()
    if not disease_clean:
        return {"name": "", "matched": False, "score": 0.0}
    # tính embedding similarity
    try:
        emb_all = model.encode([disease_clean] + disease_list, convert_to_tensor=True)
        sims = util.cos_sim(emb_all[0], emb_all[1:])[0].cpu().numpy()
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        best_name = disease_list[best_idx]
        logger.debug("Best match '%s' score=%.4f for input '%s'", best_name, best_score, disease)
        if best_score >= SIM_THRESHOLD:
            return {"name": best_name, "matched": True, "score": best_score}
        # else fallback LLM attempt
        if _gemini_client:
            prompt = f"Chuẩn hóa tên bệnh '{disease}' dựa trên danh sách: {disease_list}. Trả về tên bệnh trong danh sách nếu có, hoặc trả về 'UNKNOWN'."
            resp = call_llm(prompt)
            resp_clean = resp.strip().capitalize()
            if resp_clean in disease_list:
                return {"name": resp_clean, "matched": True, "score": best_score}
        # fuzzy fallback
        for cand in disease_list:
            if fuzz.partial_ratio(disease_clean, cand.lower()) > 85:
                return {"name": cand, "matched": True, "score": 0.5}
        return {"name": disease.strip().capitalize(), "matched": False, "score": best_score}
    except Exception as e:
        logger.exception("normalize_disease_rag lỗi: %s", e)
        return {"name": disease.strip().capitalize(), "matched": False, "score": 0.0}

def build_vector_store(pdf_results: List[Dict]) -> FAISS:
    documents = [Document(page_content=item.get("snippet",""), metadata={"page_id": item.get("page_id")}) for item in pdf_results]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return FAISS.from_documents(documents, embeddings)

def extract_diseases_fallback(data: dict, disease_list: List[str]) -> List[str]:
    diseases = []
    pdf_results = data.get("pdf_results", [])
    for item in pdf_results:
        snippet = item.get("snippet", "").lower()
        for disease in disease_list:
            if fuzz.partial_ratio(disease.lower(), snippet) > 80 and disease not in diseases:
                diseases.append(disease)
    if not diseases and "synthesized_answer" in data:
        text = data["synthesized_answer"].lower()
        for disease in disease_list:
            if fuzz.partial_ratio(disease.lower(), text) > 80 and disease not in diseases:
                diseases.append(disease)
    return list(set(diseases))

def extract_diseases_rag(data: dict, disease_list: List[str]) -> List[str]:
    pdf_results = data.get("pdf_results", [])
    diseases = []
    try:
        if pdf_results:
            vector_store = build_vector_store(pdf_results)
            query = " ".join(data.get("extracted_symptoms", [])) + " " + data.get("synthesized_answer", "")
            retrieved_docs = vector_store.similarity_search(query, k=5)
            context = "\n".join([doc.page_content for doc in retrieved_docs])
            if _gemini_client:
                llm_prompt = f"Dựa trên ngữ cảnh sau, liệt kê tên bệnh (tách bằng dấu phẩy) tồn tại trong danh sách: {disease_list}\n\nContext:\n{context}"
                llm_response = call_llm(llm_prompt)
                # LLM có thể trả về các tên, split bằng dấu phẩy/ xuống dòng
                candidates = [t.strip().capitalize() for t in llm_response.replace("\n", ",").split(",") if t.strip()]
                for c in candidates:
                    if c in disease_list and c not in diseases:
                        diseases.append(c)
        # Fallback: nếu không có kết quả từ RAG, dùng fuzzy trên snippet/synthesized_answer
        if not diseases:
            diseases = extract_diseases_fallback(data, disease_list)
    except Exception as e:
        logger.exception("extract_diseases_rag lỗi: %s", e)
        diseases = extract_diseases_fallback(data, disease_list)
    return diseases

def map_disease_to_specialty(disease: str) -> Optional[str]:
    # direct mapping
    sp = disease_to_specialty.get(disease)
    if sp:
        return sp
    # fuzzy fallback
    for key, spkgs in specialty_to_packages.items():
        if fuzz.partial_ratio(disease.lower(), key.lower()) > 85:
            return key
    # LLM fallback
    if _gemini_client:
        prompt = f"Bệnh '{disease}' thường thuộc chuyên khoa nào? Trả về 1 từ (vd: Nội tiêu hóa, Gan mật, Tim mạch) trong danh sách: {list(specialty_to_packages.keys())}"
        try:
            resp = call_llm(prompt).strip()
            # chọn nếu đúng list
            if resp in specialty_to_packages:
                return resp
        except Exception:
            logger.warning("LLM không thể ánh xạ chuyên khoa cho %s", disease)
    return None

def compute_relevance(description: str, disease: str, symptoms: List[str]) -> float:
    try:
        query = f"{disease} {' '.join(symptoms)}"
        emb = model.encode([query, description], convert_to_tensor=True)
        score = float(util.cos_sim(emb[0], emb[1]).cpu().numpy()[0][0])
    except Exception:
        score = 0.0
    # LLM blended score if possible
    if _gemini_client:
        try:
            llm_prompt = f"Trả về điểm từ 0 đến 1 cho mức độ phù hợp của gói mô tả: '{description}' với bệnh '{disease}' và triệu chứng {symptoms}."
            llm_response = call_llm(llm_prompt)
            llm_score = float(llm_response.strip().split()[0])
            return 0.7 * score + 0.3 * llm_score
        except Exception:
            pass
    return score

def enrich_packages_for_specialty_rag(specialty: str, disease: str, symptoms: List[str]) -> List[Dict]:
    pkgs = specialty_to_packages.get(specialty, [])
    enriched = []
    for p in pkgs:
        cost = package_cost_map.get(p["id"], {})
        rel = compute_relevance(p.get("description",""), disease, symptoms)
        enriched.append({
            "id": p["id"],
            "name": p.get("name"),
            "description": p.get("description"),
            "cost_min": cost.get("min"),
            "cost_max": cost.get("max"),
            "currency": cost.get("currency"),
            "relevance_score": rel
        })
    return sorted(enriched, key=lambda x: x["relevance_score"], reverse=True)

# ==== LangChain Tool ====
@tool
async def cost_tool_rag(agent_output: Dict) -> Dict:
    """
    Input agent_output: dict gồm:
      {
        "session_id": str,
        "user_query": str,
        "final_response_parts": list[str]
      }
    """
    logger.debug("cost_tool_rag called")

    try:
        # --- Validate input ---
        if not isinstance(agent_output, dict):
            raise ValueError("agent_output phải là dict với user_query và final_response_parts")

        session_id = agent_output.get("session_id", "unknown")
        user_query = agent_output.get("user_query", "")
        response_parts = agent_output.get("final_response_parts", [])

        if not isinstance(response_parts, list):
            raise ValueError("final_response_parts phải là list[str]")

        # --- Build synthesized_answer từ response_parts + user_query ---
        synthesized_answer = user_query.strip() + "\n\n" + "\n".join(response_parts)

        data = {
            "synthesized_answer": synthesized_answer,
            "extracted_symptoms": []  # nếu bạn đã có module extract_symptoms thì gắn vào đây
        }

        # --- Extract diseases ---
        diseases = extract_diseases_rag(data, disease_list)

        if not diseases and data.get("synthesized_answer"):
            text = data["synthesized_answer"]
            for d in disease_list:
                if fuzz.partial_ratio(d.lower(), text.lower()) > 80:
                    diseases.append(d)

        diseases = list(dict.fromkeys(diseases))  # dedupe

        # --- Normalize diseases ---
        normalized = [normalize_disease_rag(d, disease_list) for d in diseases]

        # --- Map to specialties ---
        specialties = []
        for n in normalized:
            sp = map_disease_to_specialty(n["name"])
            if sp and sp not in specialties:
                specialties.append(sp)

        # --- Enrich packages ---
        specialty_packages = []
        for sp in specialties:
            pkgs = enrich_packages_for_specialty_rag(
                sp, normalized[0]["name"] if normalized else "", data["extracted_symptoms"]
            )
            specialty_packages.append({"specialty": sp, "packages": pkgs})

        # --- Build result ---
        result = {
            "status": "completed",
            "message": "Xử lý thành công",
            "data": {
                "input_query": user_query,
                "input_diseases": normalized,
                "specialties": specialty_packages,
                "sim_threshold": SIM_THRESHOLD
            }
        }

        # --- Save history ---
        save_history(session_id, {
            "type": "cost_tool_rag",
            "input": agent_output,
            "result": result
        })

        logger.debug("cost_tool_rag completed")
        return result

    except Exception as e:
        logger.exception("Error in cost_tool_rag: %s", e)
        return {"status": "error", "message": str(e), "data": None}
