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
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from rapidfuzz import fuzz



# Cấu hình logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('cost_tool_rag.log')
    ]
)
logger = logging.getLogger(__name__)



# Khởi tạo client Gemini
def init_gemini_client():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY không được thiết lập trong biến môi trường")
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=api_key,
        temperature=0.2,
        max_output_tokens=512
    )

gemini_client = init_gemini_client()

def call_llm(prompt: str) -> str:
    logger.debug("Gọi Gemini API với prompt: %s", prompt)
    try:
        response = gemini_client.invoke([HumanMessage(content=prompt)])
        logger.debug("Kết quả từ Gemini API: %s", response.content.strip())
        return response.content.strip()
    except Exception as e:
        logger.error("Lỗi khi gọi Gemini API: %s", str(e))
        raise RuntimeError(f"Lỗi khi gọi Gemini API: {str(e)}")

# Load model embedding
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
logger.debug("Đã load model embedding: paraphrase-multilingual-MiniLM-L12-v2")

# Load DB files
base = Path("/mnt/data")
try:
    logger.debug("Đang đọc file disease_specialty.json")
    with open(base / "disease_specialty.json", "r", encoding="utf-8") as f:
        disease_specialty = json.load(f)
    logger.debug("Đang đọc file specialty_package.json")
    with open(base / "specialty_package.json", "r", encoding="utf-8") as f:
        specialty_package = json.load(f)
    logger.debug("Đang đọc file package_cost.json")
    with open(base / "package_cost.json", "r", encoding="utf-8") as f:
        package_cost = json.load(f)
except FileNotFoundError as e:
    logger.error("Không tìm thấy file dữ liệu: %s", str(e))
    raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {e}")

# Preprocess disease variants
def preprocess_disease_variants(disease_specialty: List[Dict]) -> Dict:
    # Logger debug
    logger.debug("Hợp nhất các biến thể bệnh")
    """Hợp nhất các biến thể bệnh thành một tên chuẩn."""
    disease_map = {}
    for item in disease_specialty:
        disease = item["disease"].lower()
        base_disease = disease.split(" (")[0].split(" do ")[0].strip()
        if base_disease not in disease_map or len(disease) < len(disease_map[base_disease]["disease"]):
            disease_map[base_disease] = item
    return {v["disease"]: v["specialty"] for v in disease_map.values()}

# Build index
disease_to_specialty = preprocess_disease_variants(disease_specialty)
specialty_to_packages = {s["specialty"]: s["packages"] for s in specialty_package}
package_cost_map = {c["id"]: c for c in package_cost}
disease_list = list(disease_to_specialty.keys())
logger.debug("Đã tạo index: %d bệnh, %d chuyên khoa, %d gói", len(disease_list), len(specialty_to_packages), len(package_cost_map))


# Helpers
@lru_cache(maxsize=1000)
def cached_embedding(text: str) -> np.ndarray:
    """Cache kết quả embedding để tối ưu hiệu suất."""
    logger.debug("Tính embedding cho: %s", text[:50])
    return model.encode(text)

def normalize_disease_rag(disease: str, disease_list: List[str]) -> str:
    logger.debug("Chuẩn hóa bệnh: %s", disease)
    disease = disease.strip().lower()
    disease_embeddings = model.encode([disease] + disease_list, convert_to_tensor=True)
    similarities = util.cos_sim(disease_embeddings[0], disease_embeddings[1:])
    top_indices = np.argsort(similarities[0])[-10:]
    filtered_disease_list = [disease_list[i] for i in top_indices]
    logger.debug("Top 10 bệnh tương đồng: %s", filtered_disease_list)
    if similarities[0][top_indices[-1]] > 0.8:
        logger.debug("Khớp embedding: %s", filtered_disease_list[-1])
        return filtered_disease_list[-1]
    try:
        llm_prompt = f"Chuẩn hóa tên bệnh '{disease}' dựa trên danh sách: {filtered_disease_list}"
        llm_response = call_llm(llm_prompt)
        logger.debug("Kết quả chuẩn hóa từ LLM: %s", llm_response)
        return llm_response.strip().capitalize()
    except RuntimeError:
        logger.warning("Fallback: trả về bệnh gốc: %s", disease)
        return disease.capitalize()

def build_vector_store(pdf_results: List[Dict]) -> FAISS:
    """Tạo vector store từ pdf_results."""
    documents = [Document(page_content=item["snippet"], metadata={"page_id": item["page_id"]}) for item in pdf_results]
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    return FAISS.from_documents(documents, embeddings)


def extract_diseases_fallback(data: dict, disease_list: List[str]) -> List[str]:
    logger.debug("Fallback: Tìm kiếm bệnh bằng fuzzy matching")
    diseases = []
    pdf_results = data.get("pdf_results", [])
    for item in pdf_results:
        snippet = item.get("snippet", "").lower()
        for disease in disease_list:
            if fuzz.partial_ratio(disease.lower(), snippet) > 80 and disease not in diseases:
                diseases.append(disease)
                logger.debug("Tìm thấy bệnh trong snippet: %s", disease)
    if not diseases and "synthesized_answer" in data:
        text = data["synthesized_answer"].lower()
        for disease in disease_list:
            if fuzz.partial_ratio(disease.lower(), text) > 80 and disease not in diseases:
                diseases.append(disease)
                logger.debug("Tìm thấy bệnh trong synthesized_answer: %s", disease)
    logger.debug("Kết quả fallback: %s", diseases)
    return list(set(diseases))


def extract_diseases_rag(data: dict, disease_list: List[str]) -> List[str]:
    logger.debug("Trích xuất bệnh bằng RAG từ data: %s", data)
    pdf_results = data.get("pdf_results", [])
    diseases = []
    try:
        vector_store = build_vector_store(pdf_results)
        query = " ".join(data.get("extracted_symptoms", [])) + " " + data.get("synthesized_answer", "")
        logger.debug("Query cho FAISS: %s", query)
        retrieved_docs = vector_store.similarity_search(query, k=5)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        logger.debug("Ngữ cảnh từ FAISS: %s", context[:200])
        llm_prompt = f"Dựa trên triệu chứng và ngữ cảnh sau, liệt kê các bệnh liên quan từ danh sách {disease_list}:\n{context}"
        llm_response = call_llm(llm_prompt)
        extracted_diseases = [d.strip().capitalize() for d in llm_response.split(",") if d.strip() in disease_list]
        diseases.extend(extracted_diseases)
        logger.debug("Bệnh trích xuất từ LLM: %s", extracted_diseases)
    except RuntimeError:
        logger.warning("Lỗi RAG, chuyển sang fallback")
        diseases.extend(extract_diseases_fallback(data, disease_list))
    return list(set(diseases))


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
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from rapidfuzz import fuzz

# Cấu hình logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('cost_tool_rag.log')
    ]
)
logger = logging.getLogger(__name__)

# Khởi tạo client Gemini
def init_gemini_client():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY không được thiết lập")
        raise ValueError("GOOGLE_API_KEY không được thiết lập trong biến môi trường")
    logger.debug("Khởi tạo Gemini client với model gemini-1.5-pro")
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=api_key,
        temperature=0.2,
        max_output_tokens=512
    )

gemini_client = init_gemini_client()

def call_llm(prompt: str) -> str:
    logger.debug("Gọi Gemini API với prompt: %s", prompt)
    try:
        response = gemini_client.invoke([HumanMessage(content=prompt)])
        logger.debug("Kết quả từ Gemini API: %s", response.content.strip())
        return response.content.strip()
    except Exception as e:
        logger.error("Lỗi khi gọi Gemini API: %s", str(e))
        raise RuntimeError(f"Lỗi khi gọi Gemini API: {str(e)}")

# Load model embedding
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
logger.debug("Đã load model embedding: paraphrase-multilingual-MiniLM-L12-v2")

# Load DB files
base = Path("/mnt/data")
try:
    logger.debug("Đang đọc file disease_specialty.json")
    with open(base / "disease_specialty.json", "r", encoding="utf-8") as f:
        disease_specialty = json.load(f)
    logger.debug("Đang đọc file specialty_package.json")
    with open(base / "specialty_package.json", "r", encoding="utf-8") as f:
        specialty_package = json.load(f)
    logger.debug("Đang đọc file package_cost.json")
    with open(base / "package_cost.json", "r", encoding="utf-8") as f:
        package_cost = json.load(f)
except FileNotFoundError as e:
    logger.error("Không tìm thấy file dữ liệu: %s", str(e))
    raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {e}")

# Preprocess disease variants
def preprocess_disease_variants(disease_specialty: List[Dict]) -> Dict:
    logger.debug("Hợp nhất các biến thể bệnh")
    disease_map = {}
    for item in disease_specialty:
        disease = item["disease"].lower()
        base_disease = disease.split(" (")[0].split(" do ")[0].strip()
        if base_disease not in disease_map or len(disease) < len(disease_map[base_disease]["disease"]):
            disease_map[base_disease] = item
    return {v["disease"]: v["specialty"] for v in disease_map.values()}

disease_to_specialty = preprocess_disease_variants(disease_specialty)
specialty_to_packages = {s["specialty"]: s["packages"] for s in specialty_package}
package_cost_map = {c["id"]: c for c in package_cost}
disease_list = list(disease_to_specialty.keys())
logger.debug("Đã tạo index: %d bệnh, %d chuyên khoa, %d gói", len(disease_list), len(specialty_to_packages), len(package_cost_map))

# Helpers
@lru_cache(maxsize=1000)
def cached_embedding(text: str) -> np.ndarray:
    logger.debug("Tính embedding cho: %s", text[:50])
    return model.encode(text)

def normalize_disease_rag(disease: str, disease_list: List[str]) -> str:
    logger.debug("Chuẩn hóa bệnh: %s", disease)
    disease = disease.strip().lower()
    disease_embeddings = model.encode([disease] + disease_list, convert_to_tensor=True)
    similarities = util.cos_sim(disease_embeddings[0], disease_embeddings[1:])
    top_indices = np.argsort(similarities[0])[-10:]
    filtered_disease_list = [disease_list[i] for i in top_indices]
    logger.debug("Top 10 bệnh tương đồng: %s", filtered_disease_list)
    if similarities[0][top_indices[-1]] > 0.8:
        logger.debug("Khớp embedding: %s", filtered_disease_list[-1])
        return filtered_disease_list[-1]
    try:
        llm_prompt = f"Chuẩn hóa tên bệnh '{disease}' dựa trên danh sách: {filtered_disease_list}"
        llm_response = call_llm(llm_prompt)
        logger.debug("Kết quả chuẩn hóa từ LLM: %s", llm_response)
        return llm_response.strip().capitalize()
    except RuntimeError:
        logger.warning("Fallback: trả về bệnh gốc: %s", disease)
        return disease.capitalize()

def build_vector_store(pdf_results: List[Dict]) -> FAISS:
    logger.debug("Tạo vector store với %d pdf_results", len(pdf_results))
    documents = [Document(page_content=item["snippet"], metadata={"page_id": item["page_id"]}) for item in pdf_results]
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    return FAISS.from_documents(documents, embeddings)

def extract_diseases_fallback(data: dict, disease_list: List[str]) -> List[str]:
    logger.debug("Fallback: Tìm kiếm bệnh bằng fuzzy matching")
    diseases = []
    pdf_results = data.get("pdf_results", [])
    for item in pdf_results:
        snippet = item.get("snippet", "").lower()
        for disease in disease_list:
            if fuzz.partial_ratio(disease.lower(), snippet) > 80 and disease not in diseases:
                diseases.append(disease)
                logger.debug("Tìm thấy bệnh trong snippet: %s", disease)
    if not diseases and "synthesized_answer" in data:
        text = data["synthesized_answer"].lower()
        for disease in disease_list:
            if fuzz.partial_ratio(disease.lower(), text) > 80 and disease not in diseases:
                diseases.append(disease)
                logger.debug("Tìm thấy bệnh trong synthesized_answer: %s", disease)
    logger.debug("Kết quả fallback: %s", diseases)
    return list(set(diseases))

def extract_diseases_rag(data: dict, disease_list: List[str]) -> List[str]:
    logger.debug("Trích xuất bệnh bằng RAG từ data: %s", data)
    pdf_results = data.get("pdf_results", [])
    diseases = []
    try:
        vector_store = build_vector_store(pdf_results)
        query = " ".join(data.get("extracted_symptoms", [])) + " " + data.get("synthesized_answer", "")
        logger.debug("Query cho FAISS: %s", query)
        retrieved_docs = vector_store.similarity_search(query, k=5)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        logger.debug("Ngữ cảnh từ FAISS: %s", context[:200])
        llm_prompt = f"Dựa trên triệu chứng và ngữ cảnh sau, liệt kê các bệnh liên quan từ danh sách {disease_list}:\n{context}"
        llm_response = call_llm(llm_prompt)
        extracted_diseases = [d.strip().capitalize() for d in llm_response.split(",") if d.strip() in disease_list]
        diseases.extend(extracted_diseases)
        logger.debug("Bệnh trích xuất từ LLM: %s", extracted_diseases)
    except RuntimeError:
        logger.warning("Lỗi RAG, chuyển sang fallback")
        diseases.extend(extract_diseases_fallback(data, disease_list))
    return list(set(diseases))

def map_disease_to_specialty(disease: str, disease_to_specialty: Dict, symptoms: List[str]) -> Optional[str]:
    logger.debug("Ánh xạ bệnh %s sang chuyên khoa", disease)
    specialty = disease_to_specialty.get(disease)
    if specialty:
        logger.debug("Tìm thấy chuyên khoa: %s", specialty)
        return specialty
    try:
        llm_prompt = f"Bệnh '{disease}' với triệu chứng {symptoms} thuộc chuyên khoa nào? Trả về một trong: {list(specialty_to_packages.keys())}"
        specialty = call_llm(llm_prompt).strip()
        logger.debug("Chuyên khoa từ LLM: %s", specialty)
        return specialty if specialty in specialty_to_packages else None
    except RuntimeError:
        logger.warning("Không tìm thấy chuyên khoa cho bệnh: %s", disease)
        return None

def compute_relevance(description: str, disease: str, symptoms: List[str]) -> float:
    logger.debug("Tính relevance_score cho bệnh %s, triệu chứng %s", disease, symptoms)
    query = f"{disease} {' '.join(symptoms)}"
    embeddings = model.encode([query, description], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1])[0][0].item()
    logger.debug("Embedding score: %f", score)
    try:
        llm_prompt = f"Đánh giá mức độ phù hợp của gói dịch vụ '{description}' với bệnh '{disease}' và triệu chứng {symptoms}. Trả về điểm từ 0 đến 1."
        llm_score = float(call_llm(llm_prompt))
        logger.debug("LLM score: %f", llm_score)
        return 0.7 * score + 0.3 * llm_score
    except RuntimeError:
        logger.warning("Fallback: chỉ dùng embedding score")
        return score

def enrich_packages_for_specialty_rag(specialty: str, disease: str, symptoms: List[str]) -> List[Dict]:
    logger.debug("Lấy gói dịch vụ cho chuyên khoa: %s", specialty)
    pkgs = specialty_to_packages.get(specialty, [])
    enriched = []
    for p in pkgs:
        cost = package_cost_map.get(p["id"], {})
        relevance_score = compute_relevance(p["description"], disease, symptoms)
        enriched.append({
            "id": p["id"],
            "name": p["name"],
            "description": p["description"],
            "cost_min": cost.get("min"),
            "cost_max": cost.get("max"),
            "currency": cost.get("currency"),
            "relevance_score": relevance_score
        })
        logger.debug("Gói %s, relevance_score: %f", p["name"], relevance_score)
    return sorted(enriched, key=lambda x: x["relevance_score"], reverse=True)

# ==== Cost Tool as LangChain Tool ====
@tool
async def cost_tool_rag(agent_output: dict) -> Dict:
    """
    Công cụ phân tích bệnh, chuyên khoa, gói dịch vụ và chi phí dựa trên đầu ra từ symptom_agent.

    Mô tả tổng quan
    ---------------
    Công cụ này nhận đầu ra từ `search_symptoms` hoặc một agent tương tự, phân tích để trích xuất bệnh,
    ánh xạ sang chuyên khoa, và đề xuất các gói dịch vụ kèm chi phí. Sử dụng RAG (Retrieval-Augmented Generation)
    để chuẩn hóa tên bệnh, trích xuất bệnh từ ngữ cảnh, và ưu tiên gói dịch vụ phù hợp. Quy trình gồm:
      1. Trích xuất bệnh từ `pdf_results` và `synthesized_answer` bằng FAISS và Google Gemini API.
      2. Chuẩn hóa tên bệnh bằng embedding và Gemini API fallback.
      3. Ánh xạ bệnh sang chuyên khoa, sử dụng RAG nếu không tìm thấy trực tiếp.
      4. Lấy danh sách gói dịch vụ cho từng chuyên khoa, xếp hạng theo mức độ liên quan.
      5. Trả về kết quả dạng JSON với danh sách bệnh, chuyên khoa, gói dịch vụ và chi phí.

    Tham số
    --------
    agent_output : dict
        Đầu ra từ `search_symptoms` hoặc tương tự, chứa các trường:
        - status: str (trạng thái xử lý)
        - message: str (thông điệp tóm tắt)
        - data: dict (chứa extracted_symptoms, pdf_results, synthesized_answer)

    Giá trị trả về
    -------------
    Trả về một dict với các trường:
      - status: str ('completed' | 'error')
      - message: str (thông điệp tóm tắt)
      - data: dict, chứa:
          * input_diseases: List[str] (danh sách bệnh trích xuất)
          * specialties: List[dict] (danh sách chuyên khoa và gói dịch vụ kèm chi phí)

    Yêu cầu
    -------
      - File dữ liệu: disease_specialty.json, specialty_package.json, package_cost.json
      - Thư viện: sentence-transformers, langchain, faiss-cpu, langchain-google-genai
      - Biến môi trường: GOOGLE_API_KEY cho Gemini API
      - Nếu Gemini API không khả dụng, công cụ sẽ fallback về tìm kiếm từ khóa đơn giản.

    Xử lý lỗi
    ----------
      - Nếu file dữ liệu không tồn tại, trả về status='error'.
      - Nếu Gemini API không khả dụng, sử dụng tìm kiếm từ khóa cơ bản.
      - Xử lý lỗi bất đồng bộ và đảm bảo hiệu suất với caching.

    Ví dụ
    -----
    >>> agent_output = {
    ...     "status": "completed",
    ...     "message": "Các bệnh có thể liên quan: Xơ gan, Tăng áp lực tĩnh mạch cửa",
    ...     "data": {
    ...         "extracted_symptoms": ["vàng da", "mệt mỏi"],
    ...         "pdf_results": [
    ...             {"page_id": 21, "snippet": "CHƯƠNG 166 ... Tăng áp lực tĩnh mạch cửa ..."},
    ...             {"page_id": 13, "snippet": "CHƯƠNG 165 ... Xơ gan và bệnh gan do rượu ..."}
    ...         ],
    ...         "synthesized_answer": "Các bệnh có thể liên quan: Xơ gan, Tăng áp lực tĩnh mạch cửa"
    ...     }
    ... }
    >>> result = await cost_tool_rag(agent_output)
    >>> print(result)
    """
    logger.debug("Gọi cost_tool_rag với đầu vào: %s", agent_output)
    try:
        data = agent_output.get("data", {})
        logger.debug("Data từ agent_output: %s", data)
        diseases = extract_diseases_rag(data, disease_list)
        logger.debug("Bệnh trích xuất: %s", diseases)
        diseases = [normalize_disease_rag(d, disease_list) for d in diseases]
        logger.debug("Bệnh sau chuẩn hóa: %s", diseases)

        specialties = []
        symptoms = data.get("extracted_symptoms", [])
        for d in diseases:
            sp = map_disease_to_specialty(d, disease_to_specialty, symptoms)
            if sp and sp not in specialties:
                specialties.append(sp)
        logger.debug("Chuyên khoa: %s", specialties)

        specialty_packages = []
        for sp in specialties:
            enriched_pkgs = enrich_packages_for_specialty_rag(sp, diseases[0] if diseases else "", symptoms)
            specialty_packages.append({"specialty": sp, "packages": enriched_pkgs})
        logger.debug("Gói dịch vụ: %s", specialty_packages)

        result = {
            "status": "completed",
            "message": "Xử lý thành công",
            "data": {
                "input_diseases": diseases,
                "specialties": specialty_packages
            }
        }
        logger.debug("Kết quả cost_tool_rag: %s", result)
        return result
    except Exception as e:
        logger.error("Lỗi trong cost_tool_rag: %s", str(e))
        return {
            "status": "error",
            "message": f"Lỗi khi xử lý: {str(e)}",
            "data": None
        }
