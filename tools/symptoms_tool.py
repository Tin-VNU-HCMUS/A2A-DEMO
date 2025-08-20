#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
search_symptoms.py - Công cụ tìm triệu chứng theo pipeline RAG + Gemini (Phiên bản cải thiện)

Cải thiện chính:
- Chuẩn hóa đơn giản: Chỉ dùng ViTokenizer để chuẩn hóa câu hỏi trong JSON.
- Quản lý lỗi và phụ thuộc tốt hơn: Kiểm tra phụ thuộc, khởi tạo mô hình rõ ràng.
- Hiệu suất: Lưu trữ embedding vào file, sử dụng IndexHNSW cho FAISS.
- Tìm kiếm: Kết hợp semantic và fuzzy search với trọng số, ngưỡng động.
- Trích xuất triệu chứng: Mở rộng regex, thêm few-shot prompt cho LLM.
- Đầu ra: Trả về JSON với extracted_symptoms và validation.
- Bảo trì: Sử dụng config dict, docstring chi tiết.
"""

import os
import re
import json
import logging
from typing import List, Dict, Any
from rapidfuzz import process, fuzz
import pandas as pd
from sentence_transformers import util, SentenceTransformer
import faiss
import pickle
from pyvi import ViTokenizer
from cleantext import clean

# ======= BẬT LOGGING CHI TIẾT =======
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ====== CONFIG ======
CONFIG = {
    "json_path": "data/dataset_test.json",
    "embed_cache_path": "data/embeddings_cache.pkl",
    "faiss_index_path": "data/faiss_index.index",
    "top_k": 5,
    "fuzzy_threshold": 60,
    "semantic_weight": 0.7,
    "fuzzy_weight": 0.3,
    "merge_threshold": 0.5,
    "embed_model_name": "bkai-foundation-models/vietnamese-bi-encoder",
    "embed_fallback": "all-MiniLM-L6-v2",
    "gemini_model": os.getenv("GOOGLE_GENAI_MODEL", "gemini-1.5-flash"),
}

# ====== Stopwords ======
try:
    import nltk
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    STOPWORDS_BASE = set(stopwords.words("vietnamese"))
except:
    STOPWORDS_BASE = set()

STOPWORDS = STOPWORDS_BASE.union({
    "là", "và", "có", "bị", "ở", "trong", "khi", "các", "những",
    "tôi", "bạn", "thì", "một", "này", "đang", "được", "với", "cho",
    "của", "tại", "trên", "dưới", "gì", "nào", "ai", "đó", "đây",
    "kia", "mà", "như", "lại", "còn", "đã", "chỉ", "mỗi", "để",
    "từ", "ra", "vào", "lên", "xuống", "nếu", "vì", "bởi", "do",
    "nên", "thế", "nhưng", "hay", "hoặc", "chưa", "rằng", "nữa",
    "luôn", "vẫn", "đều", "rất", "quá", "hết", "cùng", "theo",
    "về", "bằng", "ngoài", "giữa", "trước", "sau", "kể", "từng",
    "chẳng", "chứ", "mới", "đi", "làm", "nói", "nghe", "thấy",
    "bên", "nơi", "chỗ", "hơn", "ít", "nhiều", "vài", "tất", "cả",
    "mấy", "ai", "cái", "con", "người", "việc", "nào", "đâu",
    "thôi", "đấy", "ấy", "vậy", "thế", "nào", "bao", "giờ", "khiến",
    "bệnh", "triệu chứng", "cảm thấy", "thường", "thỉnh thoảng", "đột ngột",
    "mạnh", "yếu", "nhẹ", "nặng", "liên tục", "gián đoạn", "tăng", "giảm"
})

# ====== Kiểm tra phụ thuộc và API key ======
if not os.getenv("GOOGLE_API_KEY"):
    logger.error("GOOGLE_API_KEY not set")
    HAS_LLM = False
else:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        HAS_LLM = True
    except ImportError:
        logger.warning("langchain_google_genai not installed. LLM disabled.")
        HAS_LLM = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDER = True
except ImportError:
    logger.warning("SentenceTransformer not available. Falling back to fuzzy search only.")
    HAS_EMBEDDER = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    logger.warning("FAISS not available. Semantic search disabled.")
    HAS_FAISS = False

# ====== GLOBAL CACHE ======
_df: pd.DataFrame = None
_corpus: List[str] = []
_corpus_norm: List[str] = []
_corpus_for_embedding: List[str] = []
_meta: List[Dict[str, Any]] = []
_index: Any = None
_embed_model: Any = None
_gemini_model: Any = None


# ====== STEP 1: Load JSON ======
def _load_json() -> pd.DataFrame:
    logger.debug("[LOAD_JSON] Loading JSON dataset...")
    global _df
    if _df is None:
        path = CONFIG["json_path"]
        if not os.path.exists(path):
            logger.error(f"JSON not found at {path}")
            _df = pd.DataFrame(columns=["question", "answer", "context", "question_norm"])
            return _df
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                logger.error("JSON must be a list of dictionaries")
                return pd.DataFrame(columns=["question", "answer", "context", "question_norm"])
            normalized_data = []
            for d in data:
                if not isinstance(d, dict) or "question" not in d:
                    logger.warning(f"Invalid entry in JSON: {d}")
                    continue
                q = str(d.get("question", ""))
                qn = d.get("question_norm", "")
                if not qn:
                    try:
                        qn = ViTokenizer.tokenize(q.lower())
                    except Exception:
                        qn = q.lower()
                normalized_data.append({
                    "question": q,
                    "question_norm": qn,
                    "answer": str(d.get("answer", "")),
                    "context": d.get("context", [])
                })
            _df = pd.DataFrame(normalized_data)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON at {path}: {e} (line {e.lineno}, col {e.colno})")
            _df = pd.DataFrame(columns=["question", "answer", "context", "question_norm"])
        except Exception as e:
            logger.error(f"Error loading JSON: {e}")
            _df = pd.DataFrame(columns=["question", "answer", "context", "question_norm"])
    logger.debug(f"[LOAD_JSON] Loaded {len(_df)} records from JSON")
    return _df

# ---------- STEP 2: Init Embeddings with Caching ----------

def _init_embedding_index():
    logger.debug("[EMBEDDING] Initializing embedding index...")
    global _corpus, _meta, _index, _embed_model, _corpus_norm, _corpus_for_embedding
    json_path = CONFIG["json_path"]
    cache_path = CONFIG["embed_cache_path"]
    faiss_path = CONFIG["faiss_index_path"]

    if not os.path.exists(json_path):
        _load_json()
        logger.info("No JSON found; skipping embedding init.")
        return

    last_modified = os.path.getmtime(json_path)

    # ---- Load cache nếu hợp lệ
    if os.path.exists(cache_path) and os.path.exists(faiss_path):
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("last_modified") == last_modified:
                _corpus = cached["corpus"]
                _corpus_norm = cached["corpus_norm"]
                _corpus_for_embedding = cached["corpus_for_embedding"]
                _meta = cached["meta"]
                _index = faiss.read_index(faiss_path)
                try:
                    _index.hnsw.efSearch = 64
                except Exception:
                    pass
                logger.info("Loaded corpus + FAISS index from cache.")
                return
        except Exception as e:
            logger.warning(f"Failed to load cache, rebuilding: {e}")

    # ---- Rebuild
    df = _load_json()
    _corpus, _meta, _corpus_norm = [], [], []
    for i, row in df.iterrows():
        # Lưu text gốc để debug
        context_str = "; ".join(row.get("context", []))
        raw_text = f"Question: {row.get('question','')} Context: {context_str}"
        _corpus.append(raw_text)

        # Chuẩn hóa question + context
        qn = row.get("question_norm", "").replace("_", " ").strip()
        ctx = " ".join(row.get("context", []))
        try:
            ctx_norm = ViTokenizer.tokenize(ctx.lower()) if ctx else ""
            ctx_norm = ctx_norm.replace("_", " ")
        except Exception:
            ctx_norm = ctx.lower() if ctx else ""
        combined = re.sub(r"\s+", " ", f"{qn} {ctx_norm}".strip())
        _corpus_norm.append(combined)

        _meta.append({
            "idx": int(i),
            "question": row.get("question", ""),
            "question_norm": qn,
            "answer": row.get("answer", ""),
            "context": row.get("context", [])
        })

    # Corpus dùng cho semantic search = normalized
    _corpus_for_embedding = _corpus_norm.copy()

    if HAS_EMBEDDER and HAS_FAISS and _embed_model:
        logger.info("[EMBEDDING] Encoding normalized corpus for FAISS index...")
        corpus_emb = _embed_model.encode(_corpus_for_embedding, convert_to_numpy=True, show_progress_bar=True)
        faiss.normalize_L2(corpus_emb)
        dim = corpus_emb.shape[1]
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efConstruction = 64
        index.add(corpus_emb)
        _index = index

        # cache lại
        with open(cache_path, "wb") as f:
            pickle.dump({
                "last_modified": last_modified,
                "corpus": _corpus,
                "corpus_norm": _corpus_norm,
                "corpus_for_embedding": _corpus_for_embedding,
                "meta": _meta
            }, f)
        faiss.write_index(_index, faiss_path)
        logger.info("[EMBEDDING] Rebuilt and cached FAISS index.")



# ====== STEP 3: Khởi tạo mô hình ======
def _init_models():
    global _embed_model, _gemini_model
    if HAS_EMBEDDER and _embed_model is None:
        try:
            _embed_model = SentenceTransformer(CONFIG["embed_model_name"])
        except Exception as e:
            logger.warning(f"Failed to load {CONFIG['embed_model_name']}: {e}. Falling back to {CONFIG['embed_fallback']}")
            _embed_model = SentenceTransformer(CONFIG["embed_fallback"])
    if HAS_LLM and _gemini_model is None:
        try:
            _gemini_model = ChatGoogleGenerativeAI(model=CONFIG["gemini_model"], google_api_key=os.getenv("GOOGLE_API_KEY"))
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini: {e}")
            _gemini_model = None

# ====== STEP 4: Normalize Text ======
def _normalize_text(text: str) -> str:
    logger.debug(f"[NORMALIZE] Raw text: {text}")
    try:
        text = ViTokenizer.tokenize(text)
        words = text.split()
    except Exception as e:
        logger.warning(f"ViTokenizer failed: {e}, falling back to basic split")
        words = text.split()
    words = [w for w in words if w not in STOPWORDS]
    text = " ".join(words).strip()
    text = re.sub(r"\s+", " ", text).strip()
    logger.debug(f"[NORMALIZE] Normalized text: {text}")
    return text

# ====== STEP 5: Extract Symptoms ======
def _extract_symptoms(query: str) -> List[str]:
    logger.debug(f"[SYMPTOM_EXTRACT] Input query: {query}")
    # 1) Thử LLM (nếu có)
    if HAS_LLM and _gemini_model:
        try:
            prompt = f"""
            Bạn là trợ lý y khoa. Liệt kê các triệu chứng chính trong câu dưới đây.
            Trả về một JSON array (ví dụ: ["đau đầu", "sốt cao"]) duy nhất, KHÔNG giải thích thêm.
            Câu: "{query}"
            """
            resp = _gemini_model.invoke(prompt)
            text = getattr(resp, "content", "") or str(resp or "")
            text = text.strip()
            if text:
                # 1a. thử load nguyên văn
                try:
                    symptoms = json.loads(text)
                except Exception:
                    # 1b. thử tìm JSON array bên trong (ví dụ LLM trả kèm giải thích)
                    m = re.search(r'(\[.*?\])', text, flags=re.S)
                    if m:
                        try:
                            symptoms = json.loads(m.group(1))
                        except Exception:
                            symptoms = None
                    else:
                        # 1c. fallback parse từng dòng (list bullet)
                        lines = []
                        for ln in text.splitlines():
                            ln2 = ln.strip().lstrip("-•*0123456789. \t")
                            if len(ln2) > 0 and len(ln2) < 100:
                                lines.append(ln2)
                        symptoms = lines if lines else None

                if isinstance(symptoms, list) and symptoms:
                    # normalize each symptom and return
                    return [_normalize_text(str(s)) for s in symptoms if str(s).strip()]
        except Exception as e:
            logger.warning(f"LLM symptom extraction failed: {e}")

    # 2) Fallback: regex mở rộng (nắm các triệu chứng y khoa hay gặp)
    patterns = [
        r"(đau [^.,;]+)", r"(ngứa [^.,;]+)", r"(sốt [^.,;]+)", r"(phát ban [^.,;]+)",
        r"(khó thở[^.,;]*)", r"(nôn|ói mửa|buồn nôn[^.,;]*)", r"(vàng da[^.,;]*)",
        r"(tiêu chảy|phân lỏng[^.,;]*)", r"(mệt mỏi|suy nhược[^.,;]*)", r"(ho [^.,;]+)",
        r"(chảy máu[^.,;]+)", r"(sưng [^.,;]+)", r"(mờ mắt[^.,;]*)", r"(rụng tóc[^.,;]*)",
        r"(khó nuốt[^.,;]*)", r"(cổ trướng[^.,;]*)", r"(lách to[^.,;]*)", r"(lách[^.,;]*)",
        r"(bệnh lý não[^.,;]*)", r"(chán ăn[^.,;]*)", r"(khó chịu[^.,;]*)"
    ]
    found = []
    for pat in patterns:
        matches = re.findall(pat, query, flags=re.IGNORECASE)
        found.extend(matches)
    # loại trùng, chuẩn hóa
    normalized = list(dict.fromkeys([_normalize_text(f) for f in found if f and f.strip()]))
    logger.debug(f"[SYMPTOM_EXTRACT] Extracted symptoms: {normalized}")
    return normalized




# ---------- STEP 6: Semantic Search and Fuzzy Search ----------
def _semantic_search(query: str, k: int = CONFIG["top_k"]):
    logger.debug(f"[SEMANTIC] Searching for (normalized): {query}")
    if (_index is None) or (_embed_model is None):
        return []

    try:
        # Encode query đã normalize
        q_emb = _embed_model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(q_emb)

        if HAS_FAISS:
            D, I = _index.search(q_emb, k)
            hits = []
            if len(I) > 0:
                for dist, idx in zip(D[0].tolist(), I[0].tolist()):
                    if idx == -1:
                        continue
                    score = max(0.0, 1.0 - 0.5 * float(dist))  # cosine ≈ 1 - 0.5*L2^2
                    hits.append({
                        "score": score,
                        "meta": _meta[idx],
                        "text": _corpus_norm[idx],   # trả normalized corpus text
                        "corpus_id": idx
                    })
            logger.debug(f"[SEMANTIC] Hits: {[(h['meta']['question'], round(h['score'], 3)) for h in hits]}")
            return hits
        else:
            return []
    except Exception as e:
        logger.warning(f"Semantic search failed: {e}")
        return []


def _fuzzy_search(query: str, threshold: int = CONFIG["fuzzy_threshold"], top_n: int = CONFIG["top_k"]):
    logger.debug(f"[FUZZY] Searching for: {query}")
    global _corpus_norm, _corpus, _meta
    # ensure corpus_norm ready
    if not _corpus_norm:
        # build from _meta if needed:
        df = _load_json()
        _corpus_norm = [ViTokenizer.tokenize(str(r.get("question","")).lower()).replace("_", " ") for _, r in df.iterrows()]

    # normalize incoming query similarly (use ViTokenizer)
    try:
        q_norm = ViTokenizer.tokenize(query.lower()).replace("_", " ")
    except Exception:
        q_norm = query.lower()

    hits = process.extract(q_norm, _corpus_norm, scorer=fuzz.token_sort_ratio, score_cutoff=threshold, limit=top_n)

    results = []
    for match_text, score, idx in hits:
        results.append({
            "score": float(score) / 100.0,
            "meta": _meta[idx],
            "text": _corpus[idx],
            "corpus_id": idx,
            "matched_text": match_text
        })
    logger.debug(f"[FUZZY] Hits: {[(r['meta']['question'], round(r['score'], 3)) for r in results]}")
    return results


# ---------- STEP 7: Merge ----------
def _merge_results(sem_hits, fuzzy_hits, threshold: float = CONFIG["merge_threshold"]):

    logger.debug(f"[MERGE] Starting merge. Semantic: {len(sem_hits)}, Fuzzy: {len(fuzzy_hits)}")
    merged = {}
    for hit in sem_hits:
        idx = hit["meta"].get("idx")
        score = float(hit.get("score", 0.0)) * CONFIG["semantic_weight"]
        if idx not in merged or score > merged[idx]["similarity"]:
            merged[idx] = {
                "idx": idx,
                "question": hit["meta"]["question"],
                "answer": hit["meta"]["answer"],
                "context": hit["meta"]["context"],
                "similarity": round(score, 3),
                "matched_text": hit.get("text", "")
            }
    for hit in fuzzy_hits:
        idx = hit["meta"].get("idx")
        score = float(hit.get("score", 0.0)) * CONFIG["fuzzy_weight"]
        if idx in merged:
            merged[idx]["similarity"] = max(merged[idx]["similarity"], score)
        else:
            merged[idx] = {
                "idx": idx,
                "question": hit["meta"]["question"],
                "answer": hit["meta"]["answer"],
                "context": hit["meta"]["context"],
                "similarity": round(score, 3),
                "matched_text": hit.get("matched_text", hit.get("text", ""))
            }
    results = sorted(merged.values(), key=lambda x: x["similarity"], reverse=True)
    logger.debug(f"[MERGE] Merged results: {[(r['question'], r['similarity']) for r in results]}")
    return results

from langchain_google_genai import ChatGoogleGenerativeAI

# ---------- STEP 8: Format output ----------
def format_output(query: str, norm_query: str, symptoms: List[str], merged_results: List[Dict]) -> Dict:
    """
    Trả về JSON output với:
      - original_query
      - normalized_query
      - extracted_symptoms
      - results: list chứa idx, question, answer, context, similarity, matched_text
    """
    out_results = []
    for r in (merged_results or [])[: CONFIG.get("top_k", 5)]:
        out_results.append({
            "idx": r.get("idx"),
            "question": r.get("question"),
            "answer": r.get("answer"),
            "context": r.get("context"),
            "similarity": float(r.get("similarity", 0.0)),
            "matched_text": r.get("matched_text", "")
        })

    output = {
        "original_query": query,
        "normalized_query": norm_query,
        "extracted_symptoms": symptoms,
        "results": out_results
    }
    try:
        json.dumps(output, ensure_ascii=False)
        return output
    except Exception as e:
        logger.error(f"Invalid JSON output: {e}")
        return {"error": "Failed to generate valid JSON"}

# ---------- STEP 9: Format output ----------
def synthesize_answer(results):

    if not _gemini_model:
        return "Xin lỗi, hiện tại không thể sinh câu trả lời."

    # lấy top 3 kết quả có similarity cao nhất
    top3 = sorted(results, key=lambda x: x["similarity"], reverse=True)[:3]

    # Lấy tên bệnh & mô tả từ dataset
    diseases_info = []
    for r in top3:
        diseases_info.append(f"- {r['answer']}: {r.get('description', 'Không có mô tả')}")

    # ghép prompt cho Gemini
    prompt = f"""
    Người dùng đã mô tả một số triệu chứng. 
    Hệ thống đã tìm trong cơ sở dữ liệu và phát hiện ra 3 bệnh có mức độ liên quan cao nhất:

    {"".join(diseases_info)}.

    Hãy trả lời theo cấu trúc sau:
    - Đầu tiên: Liệt kê rõ ràng những bệnh này là gì .
    - Sau đó: Giải thích ngắn gọn tại sao các triệu chứng có thể liên quan đến chúng.
    - Cuối cùng: Đưa ra lời khuyên (nên đi khám bác sĩ, xét nghiệm, theo dõi).

    Lưu ý:
    - Trả lời bằng tiếng Việt.
    - Nhấn mạnh đây chỉ là gợi ý, không phải chẩn đoán chính thức.
    """

    response = _gemini_model.invoke(prompt)
    return response.content

# ---------- STEP 10: Call tool ----------

# ============== TOOL HOÀN CHỈNH ==============
from pydantic import BaseModel
from typing import Literal
class ResponseFormat(BaseModel):
    status: Literal['input_required', 'completed', 'error'] = 'completed'
    message: str
    data: dict = None  # cho phép chứa JSON chi tiết

    
from langchain_core.tools import tool
@tool
#@tool(return_direct=True)
def search_symptoms(user_query: str) -> Dict:
    """
    Tool chính: Nhận câu hỏi của user, chạy qua pipeline, 
    trả về JSON output + câu trả lời gợi ý của Gemini.
    """

    # 1. Init model + FAISS index nếu chưa sẵn sàng
    _init_models()
    _init_embedding_index()

    # 2. Chuẩn hóa + trích xuất triệu chứng
    norm_query = _normalize_text(user_query)
    symptoms = _extract_symptoms(user_query)

    # 3. Semantic search + fuzzy search
    sem_hits = _semantic_search(norm_query)
    fuzzy_hits = _fuzzy_search(norm_query)

    # 4. Merge kết quả
    merged_results = _merge_results(sem_hits, fuzzy_hits)

    # 5. Format JSON output
    output = format_output(user_query, norm_query, symptoms, merged_results)

    # 6. Sinh câu trả lời tự nhiên bằng Gemini (nếu có kết quả)
    if merged_results:
        try:
            output["synthesized_answer"] = synthesize_answer(merged_results)
        except Exception as e:
            logger.warning(f"Synthesize answer failed: {e}")
            output["synthesized_answer"] = None
    else:
        output["synthesized_answer"] = "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp."


    return ResponseFormat(
        status = "completed",
        message = output.get("synthesized_answer", "Không có câu trả lời"),
        data = output
    ).dict()

